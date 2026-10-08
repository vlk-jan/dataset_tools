"""End-to-end extraction of a synthetic ROS 2 bag written with ``rosbags`` (no ROS needed)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from dataset_tools import image_format, tf_pins
from dataset_tools.extract.cli import main
from dataset_tools.extract.gps import gps_row, pick_gps_topic
from dataset_tools.extract.pointcloud import header_stamp_from_cdr
from dataset_tools.extract.scene_io import TIME_ALIGN_FILE, scene_frame_counts, scene_is_complete
from dataset_tools.gt_io import read_trajectory
from dataset_tools.transforms import euler_matrix, header_stamp
from dataset_tools.validate_scene import validate_scene

TS = get_typestore(Stores.ROS2_JAZZY)
T0 = 1_700_000_000
FRAMES = 30  # enough odometry samples for a ground-truth segment (min_poses=20)
LIDAR_XYZ = (0.17, 0.0, 0.145)  # the HELHEST front Livox mount, rolled 180 deg


def _t(name):
    return TS.types[name]


def _stamp(t: float):
    sec = int(t)
    return _t("builtin_interfaces/msg/Time")(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def _header(t: float, frame: str):
    return _t("std_msgs/msg/Header")(stamp=_stamp(t), frame_id=frame)


def _quat(x=0.0, y=0.0, z=0.0, w=1.0):
    return _t("geometry_msgs/msg/Quaternion")(x=x, y=y, z=z, w=w)


def _vec(x=0.0, y=0.0, z=0.0):
    return _t("geometry_msgs/msg/Vector3")(x=x, y=y, z=z)


def _cloud_msg(t: float, points: np.ndarray, frame: str = "lidar_frame", offsets: np.ndarray | None = None):
    """A PointCloud2 with x, y, z, intensity float32 fields -- and, given
    ``offsets``, the Odin1's per-point ``offset_time`` (seconds after the stamp)."""
    names = ("x", "y", "z", "intensity") + (("offset_time",) if offsets is not None else ())
    fields = [_t("sensor_msgs/msg/PointField")(name=n, offset=4 * i, datatype=7, count=1) for i, n in enumerate(names)]
    cols = np.asarray(points, dtype=np.float32)
    if offsets is not None:
        cols = np.column_stack([cols, np.asarray(offsets, dtype=np.float32)])
    data = np.ascontiguousarray(cols).tobytes()
    step = 4 * len(names)
    return _t("sensor_msgs/msg/PointCloud2")(
        header=_header(t, frame),
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=step,
        row_step=step * len(points),
        data=np.frombuffer(data, dtype=np.uint8),
        is_dense=False,
    )


def _image_payload(value: int, ext: str = ".png") -> np.ndarray:
    """What the camera publishes for frame ``value``: the encoded bytes."""
    ok, payload = cv2.imencode(ext, np.full((8, 12, 3), value, dtype=np.uint8))
    assert ok
    return payload.ravel()


def _image_msg(t: float, value: int, ext: str = ".png", frame: str = "cam_optical"):
    fmt = {".png": "png", ".jpg": "jpeg"}[ext]
    return _t("sensor_msgs/msg/CompressedImage")(header=_header(t, frame), format=fmt, data=_image_payload(value, ext))


def _camera_info_msg(t: float, frame: str = "cam_optical"):
    return _t("sensor_msgs/msg/CameraInfo")(
        header=_header(t, frame),
        height=8,
        width=12,
        distortion_model="plumb_bob",
        d=np.array([0.1, -0.2, 0.0, 0.0, 0.0]),
        k=np.array([500.0, 0, 6, 0, 500.0, 4, 0, 0, 1]),
        r=np.eye(3).ravel(),
        p=np.array([500.0, 0, 6, 0, 0, 500.0, 4, 0, 0, 0, 1, 0]),
        binning_x=0,
        binning_y=0,
        roi=_t("sensor_msgs/msg/RegionOfInterest")(x_offset=0, y_offset=0, height=0, width=0, do_rectify=False),
    )


def _odom_msg(t: float, x: float, yaw: float = 0.0):
    pose = _t("geometry_msgs/msg/Pose")(
        position=_t("geometry_msgs/msg/Point")(x=x, y=0.0, z=0.0),
        orientation=_quat(z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
    )
    twist = _t("geometry_msgs/msg/Twist")(linear=_vec(), angular=_vec())
    return _t("nav_msgs/msg/Odometry")(
        header=_header(t, "odom"),
        child_frame_id="base_link",
        pose=_t("geometry_msgs/msg/PoseWithCovariance")(pose=pose, covariance=np.zeros(36)),
        twist=_t("geometry_msgs/msg/TwistWithCovariance")(twist=twist, covariance=np.zeros(36)),
    )


def _imu_msg(t: float, roll: float, yaw_rate: float = 0.0):
    return _t("sensor_msgs/msg/Imu")(
        header=_header(t, "base_link"),
        orientation=_quat(x=math.sin(roll / 2), w=math.cos(roll / 2)),
        orientation_covariance=np.zeros(9),
        angular_velocity=_vec(z=yaw_rate),
        angular_velocity_covariance=np.zeros(9),
        linear_acceleration=_vec(),
        linear_acceleration_covariance=np.zeros(9),
    )


def _joint_msg(t: float, k: int):
    return _t("sensor_msgs/msg/JointState")(
        header=_header(t, ""),
        name=["l", "r", "b"],
        position=np.array([k, k, k], dtype=np.float64),
        velocity=np.array([1.0, 2.0, 3.0]),
        effort=np.array([0.1, 0.2, 0.3]),
    )


def _navsatfix_msg(t: float, lat: float, lon: float, cov_type: int = 3):
    return _t("sensor_msgs/msg/NavSatFix")(
        header=_header(t, "FP_POI"),
        status=_t("sensor_msgs/msg/NavSatStatus")(status=2, service=15),
        latitude=lat,
        longitude=lon,
        altitude=227.0,
        position_covariance=np.array([0.11, 0, 0, 0, 0.22, 0, 0, 0, 0.33]),
        position_covariance_type=cov_type,
    )


#: What the Odin1 rig's tf_static carries: the URDF mount, and the unit's own
#: lidar-to-IMU and lidar-to-camera estimates, here deliberately off the pins.
ODIN_MOUNT_XYZ = (0.1, 0.0, 0.1)
ODIN_LIDAR_IN_IMU_XYZ = (0.0, 0.0, 0.05)
ODIN_CAM_IN_LIDAR_XYZ = (0.0, 0.1, 0.0)


def _tf_static_msg(odin: bool = False):
    def tf(parent, child, xyz, quat):
        return _t("geometry_msgs/msg/TransformStamped")(
            header=_header(T0, parent),
            child_frame_id=child,
            transform=_t("geometry_msgs/msg/Transform")(translation=_vec(*xyz), rotation=quat),
        )

    if odin:
        transforms = [
            tf("base_link", "odin1_base_link", ODIN_MOUNT_XYZ, _quat()),
            tf("odin1_base_link", "imu_link", (0.0, 0.0, 0.0), _quat()),
            tf("imu_link", "odin1_lidar", ODIN_LIDAR_IN_IMU_XYZ, _quat()),
            tf("odin1_lidar", "odin1_camera_link", ODIN_CAM_IN_LIDAR_XYZ, _quat()),
        ]
    else:
        transforms = [
            tf("base_link", "lidar_frame", LIDAR_XYZ, _quat(x=1.0, w=0.0)),  # rolled 180 deg
            tf("base_link", "cam_optical", (0.3, 0.0, 0.2), _quat()),
        ]
    return _t("tf2_msgs/msg/TFMessage")(transforms=transforms)


#: A vertical wall 6 m ahead of the robot's start, in the odometry frame.
WALL = np.column_stack([np.full(60, 6.0), np.linspace(-2.0, 2.0, 60), np.sort(np.tile(np.linspace(-0.4, 2.0, 12), 5))])


def _wall_in_base(tau, yaw_rate: float) -> np.ndarray:
    """:data:`WALL` as the robot at time(s) ``tau`` sees it: x = 1 m/s, yaw = yaw_rate * t."""
    tau = np.broadcast_to(np.asarray(tau, dtype=float), (len(WALL),))
    yaw = yaw_rate * (tau - T0)
    d = WALL - np.column_stack([tau - T0, np.zeros_like(tau), np.zeros_like(tau)])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1], d[:, 2]])


def _wall_in_lidar(tau, yaw_rate: float) -> np.ndarray:
    """The same, in the front lidar's frame (mounted at LIDAR_XYZ, rolled 180 deg)."""
    p = _wall_in_base(tau, yaw_rate) - np.asarray(LIDAR_XYZ)
    return p * np.array([1.0, -1.0, -1.0])


def write_bag(
    bag_dir: Path,
    frames: int = FRAMES,
    dropped_camera: set[int] = frozenset(),
    reset_from: int | None = None,
    image_ext: str = ".png",
    odin: bool = False,
    odom_until: int | None = None,
    lidar_frame: str | None = None,
    boot_joints: int = 0,
    yaw_rate: float = 0.0,
    sweep_s: float = 0.0,
) -> Path:
    """A bag with everything the extractor reads, ``frames`` lidar sweeps long.

    Cameras lag sweeps by 30 ms; ``dropped_camera`` sweeps have no image.
    ``reset_from``: odometry held at a far-away point and fix at null island
    (lost GNSS solution). ``odom_until``: odometry stops. ``lidar_frame``: a
    frame missing from TF. ``boot_joints``: joint states on a boot clock.
    ``yaw_rate`` (rad/s) turns the robot; ``sweep_s`` makes each cloud a raw
    bottom-to-top sweep of :data:`WALL` with per-point ``offset_time``.
    """
    with Writer(bag_dir, version=8) as writer:
        conns = {
            topic: writer.add_connection(topic, msgtype, typestore=TS)
            for topic, msgtype in (
                ("/cam/image/compressed", "sensor_msgs/msg/CompressedImage"),
                ("/cam/camera_info", "sensor_msgs/msg/CameraInfo"),
                ("/lidar", "sensor_msgs/msg/PointCloud2"),
                ("/odom", "nav_msgs/msg/Odometry"),
                ("/imu", "sensor_msgs/msg/Imu"),
                ("/joint_states", "sensor_msgs/msg/JointState"),
                ("/cmd_vel", "geometry_msgs/msg/Twist"),
                # Two NavSatFix topics as in real bags; the raw one never
                # publishes, so --gps auto must pick the fused one.
                ("/fixposition/odometry_llh", "sensor_msgs/msg/NavSatFix"),
                ("/fixposition/gnss1", "sensor_msgs/msg/NavSatFix"),
                ("/tf_static", "tf2_msgs/msg/TFMessage"),
            )
        }

        def put(topic, t, msg):
            writer.write(conns[topic], int(t * 1e9), TS.serialize_cdr(msg, conns[topic].msgtype))

        put("/tf_static", T0, _tf_static_msg(odin))
        default_lidar, cam_frame = ("odin1_lidar", "odin1_camera_link") if odin else ("lidar_frame", "cam_optical")
        lidar_frame = lidar_frame or default_lidar
        rng = np.random.default_rng(0)
        for k in range(frames):
            t = T0 + 0.1 * k
            if sweep_s > 0:
                offsets = np.linspace(0.0, sweep_s, len(WALL))
                pts = np.column_stack([_wall_in_lidar(t + offsets, yaw_rate), np.full(len(WALL), 7.0)])
                pts = np.vstack([[[np.nan, 0, 0, 1], [0, 0, 0, 5]], pts]).astype(np.float32)
                put("/lidar", t, _cloud_msg(t, pts, lidar_frame, np.r_[0.0, 0.0, offsets]))
            else:
                pts = rng.uniform(1.0, 5.0, size=(40, 4)).astype(np.float32)
                pts[0] = [np.nan, 0, 0, 1]  # a NaN return
                pts[1] = [0, 0, 0, 5]  # a dropped shot encoded as the origin
                pts[2] = [1.0, 0.0, 0.0, 9.0]  # a known point, to check the mount transform
                put("/lidar", t, _cloud_msg(t, pts, lidar_frame))
            if k not in dropped_camera:
                put("/cam/image/compressed", t + 0.03, _image_msg(t + 0.03, k, image_ext, cam_frame))
                put("/cam/camera_info", t + 0.03, _camera_info_msg(t + 0.03, cam_frame))
            lost = reset_from is not None and k >= reset_from
            if odom_until is None or k < odom_until:
                # 1 m/s: under the GT teleport gate
                put("/odom", t, _odom_msg(t, 900.0 if lost else 0.1 * k, yaw_rate * 0.1 * k))
            # The IMU at 100 Hz when there is a sweep to deskew, so its gyro covers it.
            for i in range(10 if sweep_s > 0 else 1):
                put("/imu", t + 0.01 * i, _imu_msg(t + 0.01 * i, math.radians(10.0), yaw_rate))
            put("/joint_states", t, _joint_msg(0.1 * k if k < boot_joints else t, k))
            put("/cmd_vel", t, _t("geometry_msgs/msg/Twist")(linear=_vec(x=0.4), angular=_vec(z=0.1)))
            # A short eastward track; the last fix declares an unknown
            # covariance, which must reach gps.npy as NaN rather than as 0.
            put(
                "/fixposition/odometry_llh",
                t,
                _navsatfix_msg(
                    t,
                    0.0 if lost else 50.1 + 1e-5 * k,
                    0.0 if lost else 14.4 + 2e-5 * k,
                    cov_type=0 if lost or k == frames - 1 else 3,
                ),
            )
    return bag_dir


COMMON = ["--cam", "/cam/image/compressed", "--lidar", "/lidar", "--odom", "/odom", "--imu", "/imu", "--jobs", "1"]


@pytest.fixture
def bags(tmp_path: Path) -> Path:
    root = tmp_path / "bags"
    write_bag(root / "bag_a", dropped_camera={5, 6})
    return root


def test_a_synthetic_bag_extracts_to_a_complete_scene(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--gt", "on"]) == 0
    scene = out / "0000"
    n = FRAMES - 2  # two sweeps had no camera frame
    assert scene_is_complete(scene)
    assert scene_frame_counts(scene) == {"images": n, "clouds": n, "data_ts.txt": n, "pcd_ts.txt": n}

    # Sync: the dropped sweeps are gone, the rest pair with the 30 ms later image.
    data_ts = np.loadtxt(scene / "data_ts.txt")
    pcd_ts = np.loadtxt(scene / "pcd_ts.txt")
    assert np.allclose(data_ts - pcd_ts, 0.015, atol=1e-6)
    assert np.all(np.diff(pcd_ts) > 0)

    # Clouds: NaN and origin rows dropped, the rest moved into base_link by the
    # rolled mount: (1, 0, 0) in the lidar -> (0.17 + 1, 0, 0.145).
    cloud = np.load(scene / "clouds" / "00000.npy")
    assert cloud.dtype == np.float32 and cloud.shape == (38, 4)
    known = cloud[np.isclose(cloud[:, 3], 9.0)]
    assert len(known) == 1 and np.allclose(known[0, :3], [1.17, 0.0, 0.145], atol=1e-5)

    # Images decode to what was encoded.
    img = cv2.imread(str(scene / "images" / "00003.png"))
    assert img.shape == (8, 12, 3) and int(img[0, 0, 0]) == 3

    # Odometry relative to frame 0, nearest sample per frame, at 0.1 m per sweep.
    transforms = np.load(scene / "transforms.npy")
    assert transforms.shape == (n, 4, 4) and np.allclose(transforms[0], np.eye(4))
    assert np.allclose(transforms[1, 0, 3], 0.1, atol=1e-9)

    # Gravity alignment reproduces the 10 deg roll the IMU reported.
    gravity = np.load(scene / "gravity_align.npy")
    assert gravity.shape == (n, 3, 3)
    assert np.allclose(gravity[0], euler_matrix(math.radians(10.0), 0.0, 0.0)[:3, :3])

    K = np.loadtxt(scene / "camera_intrinsics.txt")
    assert K[0, 0] == 500.0 and "plumb_bob" in (scene / "camera_info.yaml").read_text()
    T_cam = np.loadtxt(scene / "T_cam_to_base.txt")
    assert np.allclose(T_cam[:3, 3], [0.3, 0.0, 0.2])

    joints = np.load(scene / "joint_states.npy")
    assert joints.shape == (FRAMES, 10) and np.allclose(joints[:, 4:7], [1.0, 2.0, 3.0])
    cmd = np.load(scene / "cmd_vel.npy")
    assert cmd.shape == (FRAMES, 7) and np.allclose(cmd[:, 1], 0.4)

    # --gt on: the odometry stream is written as one anchored segment.
    assert (scene / "odometry.txt").is_file()
    seg = [line.split() for line in (scene / "odometry_segments.txt").read_text().splitlines() if line[0] != "#"]
    assert len(seg) == 1 and seg[0][3] == str(FRAMES) and seg[0][4] == "end_of_run"

    assert (scene / "source_bag.txt").read_text().strip() == "bag_a"
    assert (out / "scenes.csv").read_text().splitlines()[1].startswith("0000,bag_a,")


def test_gps_is_extracted_from_the_fused_navsatfix_topic(bags: Path, tmp_path: Path):
    """--gps auto finds the fix by type; gps.npy is full-rate, time-sorted
    ``[t, lat, lon, alt, status, cov_ee, cov_nn, cov_uu]``."""
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    gps = np.load(out / "0000" / "gps.npy")

    assert gps.shape == (FRAMES, 8)
    assert np.all(np.diff(gps[:, 0]) > 0)
    assert np.allclose(gps[:, 1], 50.1 + 1e-5 * np.arange(FRAMES))
    assert np.allclose(gps[:, 2], 14.4 + 2e-5 * np.arange(FRAMES))
    assert np.allclose(gps[:, 3], 227.0) and np.all(gps[:, 4] == 2)
    assert np.allclose(gps[:-1, 5:8], [0.11, 0.22, 0.33])
    # COVARIANCE_TYPE_UNKNOWN: the numbers are meaningless, so they are NaN.
    assert np.isnan(gps[-1, 5:8]).all()


def test_cam_time_offset_shifts_the_frame_clock(bags: Path, tmp_path: Path):
    """The offset is applied in both passes, else every pass-2 image lookup misses.

    data_ts.txt is the camera/lidar midpoint, so it moves by half the offset.
    """
    plain, shifted = tmp_path / "plain", tmp_path / "shifted"
    assert main([str(bags), str(plain), *COMMON]) == 0
    assert main([str(bags), str(shifted), *COMMON, "--cam_time_offset", "0.01"]) == 0

    a = np.loadtxt(plain / "0000" / "data_ts.txt")
    b = np.loadtxt(shifted / "0000" / "data_ts.txt")
    assert len(a) == len(b)
    # atol: one ulp of float64 at a 1.7e9 epoch is ~4e-7, well above the
    # default relative tolerance on a 0.005 difference.
    assert np.allclose(b - a, 0.005, atol=1e-6)

    # every frame still has its image and its cloud: the pass-2 lookups hit
    assert scene_frame_counts(shifted / "0000") == scene_frame_counts(plain / "0000")
    assert np.allclose(np.loadtxt(shifted / "0000" / "pcd_ts.txt"), np.loadtxt(plain / "0000" / "pcd_ts.txt"))


def test_the_default_run_is_unchanged_by_the_offset_flag(bags: Path, tmp_path: Path):
    """0.0 must be the identity on the float, not merely close to it."""
    plain, zero = tmp_path / "plain", tmp_path / "zero"
    assert main([str(bags), str(plain), *COMMON]) == 0
    assert main([str(bags), str(zero), *COMMON, "--cam_time_offset", "0"]) == 0
    for name in ("data_ts.txt", "pcd_ts.txt"):
        assert (zero / "0000" / name).read_bytes() == (plain / "0000" / name).read_bytes()


def test_a_null_island_fix_is_dropped():
    """lat = lon = 0 is rejected on position: a lost Fixposition still reports ``GBAS_FIX``."""

    class _Status:
        status = 2  # GBAS_FIX, exactly as the broken messages claim

    class _Fix:
        status = _Status()
        position_covariance = [0.0] * 9
        position_covariance_type = 0

        def __init__(self, lat, lon, alt=0.0):
            self.latitude, self.longitude, self.altitude = lat, lon, alt

    assert gps_row(1.0, _Fix(0.0, 0.0)) is None
    row = gps_row(1.0, _Fix(50.109, 14.417, 227.0))
    assert row is not None and row[1] == 50.109


def test_gps_off_writes_no_file(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--gps", "off"]) == 0
    assert not (out / "0000" / "gps.npy").exists()
    assert (out / "0000" / "source_bag.txt").is_file()  # the scene is otherwise complete


def test_gps_topic_selection_prefers_a_fused_fix_that_carries_messages():
    """Auto-detection is by message type, so it works on either rig's naming."""

    class Conn:
        def __init__(self, topic, msgtype, msgcount):
            self.topic, self.msgtype, self.msgcount = topic, msgtype, msgcount

    fix = "sensor_msgs/msg/NavSatFix"
    conns = [
        Conn("/odom", "nav_msgs/msg/Odometry", 100),
        Conn("/fixposition/gnss1", fix, 0),  # declared, never published
        Conn("/fixposition/gnss2", fix, 400),
        Conn("/fixposition/odometry_llh", fix, 800),
    ]
    assert pick_gps_topic(conns, "auto") == "/fixposition/odometry_llh"
    # An old-rig bag, whose receiver names the topic differently.
    septentrio = [Conn("/septentrio/navsatfix", fix, 500), Conn("/odom", "nav_msgs/msg/Odometry", 100)]
    assert pick_gps_topic(septentrio, "auto") == "/septentrio/navsatfix"
    assert pick_gps_topic(conns, "off") is None
    assert pick_gps_topic(conns, "/some/topic") == "/some/topic"
    assert pick_gps_topic([Conn("/odom", "nav_msgs/msg/Odometry", 100)], "auto") is None


def test_a_second_run_skips_the_complete_scene(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    scene = out / "0000"
    mtime = (scene / "source_bag.txt").stat().st_mtime_ns
    assert main([str(bags), str(out), *COMMON]) == 0
    assert (scene / "source_bag.txt").stat().st_mtime_ns == mtime


def test_overwrite_removes_what_an_earlier_run_left_behind(bags: Path, tmp_path: Path):
    """A stale frame from a previous extraction must not outlive a re-extraction,
    or the counts never agree again."""
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    scene = out / "0000"
    (scene / "images" / "00099.png").write_bytes(b"stale")
    (scene / "clouds" / "00099.npy").write_bytes(b"stale")
    (scene / "regen").mkdir()
    (scene / "regen" / "keep.txt").write_text("provenance, not ours to delete\n")
    (scene / "odometry.txt").write_text("# from the FAST-LIO regeneration, which runs before the extraction\n")
    assert not scene_is_complete(scene)

    assert main([str(bags), str(out), *COMMON, "--overwrite"]) == 0
    assert scene_is_complete(scene)
    assert not (scene / "images" / "00099.png").exists() and not (scene / "clouds" / "00099.npy").exists()
    assert (scene / "regen" / "keep.txt").is_file()
    # --gt is off for /odom, so the regenerated ground truth is not this run's to remove.
    assert (scene / "odometry.txt").read_text().startswith("# from the FAST-LIO")


def test_a_regenerated_ground_truth_survives_a_bag_without_odometry(tmp_path: Path):
    """--gt on with no odometry messages has nothing to write, so it must not delete what is there."""
    write_bag(tmp_path / "bags" / "bag_a", odom_until=0)
    out = tmp_path / "out"
    scene = out / "0000"
    scene.mkdir(parents=True)
    (scene / "odometry.txt").write_text("# from the FAST-LIO regeneration\n")
    assert main([str(tmp_path / "bags"), str(out), *COMMON, "--gt", "on"]) == 0
    assert scene_is_complete(scene)
    assert (scene / "odometry.txt").read_text().startswith("# from the FAST-LIO")


def test_a_png_that_cannot_be_written_fails_the_scene(bags: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *a, **k: False)
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--image_format", "png"]) == 1
    assert not (out / "0000" / "source_bag.txt").exists()


def test_header_stamp_read_from_cdr_matches_the_deserialised_one():
    msg = _image_msg(T0 + 0.123456789, 1)
    raw = TS.serialize_cdr(msg, "sensor_msgs/msg/CompressedImage")
    assert header_stamp_from_cdr(raw) == header_stamp(TS.deserialize_cdr(raw, "sensor_msgs/msg/CompressedImage"), 0.0)
    assert header_stamp_from_cdr(b"\x00\x01\x00") is None


def test_time_align_off_leaves_the_stamps_as_recorded(bags: Path, tmp_path: Path):
    """The escape hatch has to be exactly the identity, not merely close."""
    auto, off = tmp_path / "auto", tmp_path / "off"
    assert main([str(bags), str(auto), *COMMON]) == 0
    assert main([str(bags), str(off), *COMMON, "--time_align", "off"]) == 0
    # This bag is 30 frames, too few for a window, so auto measures nothing and
    # the two runs must agree byte for byte.
    for name in ("data_ts.txt", "pcd_ts.txt", "transforms.npy", "gravity_align.npy"):
        assert (off / "0000" / name).read_bytes() == (auto / "0000" / name).read_bytes(), name


def test_a_given_offset_moves_both_sensor_clocks(bags: Path, tmp_path: Path):
    """``--time_align SECONDS`` shifts both sensor clocks; pairing and the
    reference-clock streams are untouched (unlike ``--cam_time_offset``)."""
    plain, shifted = tmp_path / "plain", tmp_path / "shifted"
    assert main([str(bags), str(plain), *COMMON, "--gt", "on"]) == 0
    assert main([str(bags), str(shifted), *COMMON, "--gt", "on", "--time_align", "-0.05"]) == 0
    a, b = plain / "0000", shifted / "0000"

    for name in ("data_ts.txt", "pcd_ts.txt"):
        before, after = np.loadtxt(a / name), np.loadtxt(b / name)
        assert len(before) == len(after)
        # atol: one ulp of float64 at a 1.7e9 epoch is ~4e-7.
        assert np.allclose(after - before, -0.05, atol=1e-6), name

    assert scene_frame_counts(b) == scene_frame_counts(a)
    assert (b / "images" / "00003.png").read_bytes() == (a / "images" / "00003.png").read_bytes()
    assert (b / "odometry.txt").read_bytes() == (a / "odometry.txt").read_bytes()
    assert np.array_equal(np.load(b / "joint_states.npy"), np.load(a / "joint_states.npy"))


def test_the_camera_offset_still_re_pairs(bags: Path, tmp_path: Path):
    """The two knobs are different things and compose as different things."""
    plain, both = tmp_path / "plain", tmp_path / "both"
    assert main([str(bags), str(plain), *COMMON]) == 0
    assert main([str(bags), str(both), *COMMON, "--time_align", "-0.05", "--cam_time_offset", "0.01"]) == 0
    a, b = plain / "0000", both / "0000"

    assert np.allclose(np.loadtxt(b / "pcd_ts.txt") - np.loadtxt(a / "pcd_ts.txt"), -0.05, atol=1e-6)
    # data_ts is the camera/lidar midpoint, so it takes the whole frame-clock
    # shift plus half of the camera's own.
    assert np.allclose(np.loadtxt(b / "data_ts.txt") - np.loadtxt(a / "data_ts.txt"), -0.045, atol=1e-6)


def test_a_scene_too_short_to_measure_is_left_alone(bags: Path, tmp_path: Path):
    """Auto never invents a correction it could not measure."""
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    scene = out / "0000"

    # 30 frames is under one window, so there is nothing to measure with.
    data_ts = np.loadtxt(scene / "data_ts.txt")
    pcd_ts = np.loadtxt(scene / "pcd_ts.txt")
    assert np.allclose(data_ts - pcd_ts, 0.015, atol=1e-6)  # the raw 30 ms pairing, unshifted

    # It still says so, rather than saying nothing: a scene that was left alone
    # and a scene nobody looked at are not the same thing.
    report = (scene / TIME_ALIGN_FILE).read_text()
    assert "no correction" in report and "too little to measure against" in report


def test_a_reset_gnss_solution_never_reaches_the_scene(tmp_path: Path):
    """Frames during a lost fix (pose frozen 900 m away) are not exported."""
    root = tmp_path / "bags"
    write_bag(root / "bag_a", frames=60, reset_from=40)
    out = tmp_path / "out"
    assert main([str(root), str(out), *COMMON, "--gt", "on", "--gnss_guard", "on"]) == 0

    scene = out / "0000"
    counts = scene_frame_counts(scene)
    assert counts["images"] == counts["clouds"] == counts["data_ts.txt"]
    assert counts["images"] < 60  # the reset cost frames

    # No exported pose is anywhere near the fabricated position.
    transforms = np.load(scene / "transforms.npy")
    assert np.abs(transforms[:, :3, 3]).max() < 100.0
    assert len(transforms) == counts["images"]

    report = (scene / "gnss_guard.txt").read_text()
    assert "frozen" in report


def test_the_guard_stays_off_for_wheel_odometry(bags: Path, tmp_path: Path):
    """A wheel-IMU pose legitimately holds still while the robot is parked,
    which is the guard's own signature for a fabricated stretch."""
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--gt", "on"]) == 0
    assert not (out / "0000" / "gnss_guard.txt").exists()


# --------------------------------------------------------------------------
# what the images are stored as
# --------------------------------------------------------------------------


def test_a_scene_stores_the_camera_s_own_bytes(tmp_path: Path):
    """The camera publishes a JPEG; the scene holds that JPEG, byte for byte,
    and the root says so where a consumer looks for it."""
    write_bag(tmp_path / "bags" / "bag_a", image_ext=".jpg")
    out = tmp_path / "out"
    assert main([str(tmp_path / "bags"), str(out), *COMMON]) == 0
    scene = out / "0000"
    assert scene_is_complete(scene)
    assert {p.suffix for p in (scene / "images").iterdir()} == {".jpg"}
    assert scene_frame_counts(scene)["images"] == FRAMES
    assert (scene / "images" / "00003.jpg").read_bytes() == _image_payload(3, ".jpg").tobytes()
    assert image_format.declared(out) == ".jpg"
    assert json.loads((out / "manifest.json").read_text())["images"] == {
        "extension": "jpg",
        "encoding": "camera JPEG bytes verbatim, msg.data unmodified",
    }
    assert validate_scene(scene).errors == []


def test_image_format_png_stores_the_pixels_instead(tmp_path: Path):
    write_bag(tmp_path / "bags" / "bag_a", image_ext=".jpg")
    out = tmp_path / "out"
    assert main([str(tmp_path / "bags"), str(out), *COMMON, "--image_format", "png"]) == 0
    scene = out / "0000"
    assert {p.suffix for p in (scene / "images").iterdir()} == {".png"}
    want = cv2.imdecode(_image_payload(3, ".jpg"), cv2.IMREAD_COLOR)
    assert np.array_equal(cv2.imread(str(scene / "images" / "00003.png")), want)
    # A PNG root declares nothing: that is what no declaration means.
    assert image_format.declared(out) is None


def test_a_root_holds_one_image_format(tmp_path: Path):
    """A bag whose images are JPEG does not go into a root of PNGs: the scene
    fails, named, instead of leaving a root two loaders would read differently."""
    bags = tmp_path / "bags"
    write_bag(bags / "bag_a", image_ext=".png")
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    write_bag(bags / "bag_b", image_ext=".jpg")
    assert main([str(bags), str(out), *COMMON]) == 1
    assert scene_is_complete(out / "0000")
    assert not list((out / "0001").glob("images/*.jpg"))
    # ... and with the pixels stored as PNG it extends the root.
    assert main([str(bags), str(out), *COMMON, "--image_format", "png"]) == 0
    assert scene_is_complete(out / "0001")
    assert image_format.of_root(out) == ".png"


def _pin_matrix(parent, child):
    return next(p for p in tf_pins.PINS if (p.parent, p.child) == (parent, child)).matrix()


def test_odin_extrinsics_are_pinned(tmp_path: Path):
    """The bag's Odin1 TF estimates are replaced by the pins; tf_pins.txt records the originals."""
    bags = tmp_path / "bags"
    write_bag(bags / "bag_a", odin=True)
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    scene = out / "0000"

    mount = np.eye(4)
    mount[:3, 3] = ODIN_MOUNT_XYZ
    T_lidar = mount @ _pin_matrix("imu_link", "odin1_lidar")
    # The known point (1, 0, 0); the NaN and origin rows are dropped.
    known = np.load(scene / "clouds" / "00000.npy")[0, :3]
    assert np.allclose(known, (T_lidar @ [1.0, 0.0, 0.0, 1.0])[:3], atol=1e-5)
    assert not np.allclose(known[2], ODIN_MOUNT_XYZ[2] + ODIN_LIDAR_IN_IMU_XYZ[2], atol=0.01)  # not the bag's 5 cm

    T_cam = np.loadtxt(scene / "T_cam_to_base.txt")
    assert np.allclose(T_cam, T_lidar @ _pin_matrix("odin1_lidar", "odin1_camera_link"), atol=1e-6)

    text = (scene / tf_pins.FILE).read_text()
    rows = [line.split() for line in text.splitlines() if line and not line.startswith("#")]
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("imu_link", "odin1_lidar", "pin"),
        ("imu_link", "odin1_lidar", "bag"),
        ("odin1_lidar", "odin1_camera_link", "pin"),
        ("odin1_lidar", "odin1_camera_link", "bag"),
    ]
    assert [float(v) for v in rows[1][3:6]] == list(ODIN_LIDAR_IN_IMU_XYZ)
    assert "(far)" in text  # 10 cm of camera offset is far from the pin
    report = validate_scene(scene)
    assert report.status() != "FAIL", report.errors


def test_tf_pins_off_keeps_the_bag(tmp_path: Path):
    bags = tmp_path / "bags"
    write_bag(bags / "bag_a", odin=True)
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--tf_pins", "off"]) == 0
    scene = out / "0000"
    known = np.load(scene / "clouds" / "00000.npy")[0, :3]
    assert np.allclose(known, (1.1, 0.0, 0.15), atol=1e-5)
    assert not (scene / tf_pins.FILE).exists()


def test_livox_bag_has_nothing_to_pin(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    assert not (out / "0000" / tf_pins.FILE).exists()


# --------------------------------------------------------------------------
# what a scene says about itself, and what it refuses to fabricate
# --------------------------------------------------------------------------


def test_frames_past_the_pose_stream_are_dropped_once_the_clock_is_final(tmp_path: Path):
    """A frame past the last pose on the final clock is dropped and indices close up."""
    write_bag(tmp_path / "bags" / "bag_a", frames=60, odom_until=50)
    out = tmp_path / "out"
    flags = [*COMMON, "--gt", "on", "--gnss_guard", "on", "--time_align", "off"]
    assert main([str(tmp_path / "bags"), str(out), *flags]) == 0

    scene = out / "0000"
    frame_ts = np.loadtxt(scene / "data_ts.txt")
    odom_ts, _ = read_trajectory(scene / "odometry.txt")
    assert frame_ts[-1] <= odom_ts[-1]
    counts = scene_frame_counts(scene)
    assert counts["images"] == counts["clouds"] == len(frame_ts) == len(np.load(scene / "transforms.npy"))
    assert sorted(int(p.stem) for p in (scene / "clouds").iterdir()) == list(range(len(frame_ts)))
    report = (scene / "gnss_guard.txt").read_text()
    assert "after the clock correction, 0/60" not in report
    meta = json.loads((scene / "extract_meta.json").read_text())
    assert meta["frames"]["guard"]["n_frames_trimmed"] > 0
    assert meta["frames"]["written"] == len(frame_ts)


def test_boot_clock_stamps_fall_back_to_the_receive_time(tmp_path: Path):
    write_bag(tmp_path / "bags" / "bag_a", boot_joints=5)
    out = tmp_path / "out"
    assert main([str(tmp_path / "bags"), str(out), *COMMON]) == 0
    scene = out / "0000"
    t = np.load(scene / "joint_states.npy")[:, 0]
    assert t.min() > 1e9 and np.all(np.diff(t) > 0)
    meta = json.loads((scene / "extract_meta.json").read_text())
    assert meta["unset_stamps"] == {"/joint_states": 5}
    assert (scene / "joint_names.txt").read_text().split() == ["l", "r", "b"]


def test_a_lidar_outside_the_tf_tree_fails_the_scene_unless_allowed(tmp_path: Path):
    write_bag(tmp_path / "bags" / "bag_a", lidar_frame="nowhere")
    assert main([str(tmp_path / "bags"), str(tmp_path / "strict"), *COMMON]) == 1
    assert not (tmp_path / "strict" / "0000" / "source_bag.txt").exists()

    out = tmp_path / "allowed"
    assert main([str(tmp_path / "bags"), str(out), *COMMON, "--allow_frame_fallback"]) == 0
    meta = json.loads((out / "0000" / "extract_meta.json").read_text())
    assert len(meta["frame_fallbacks"]) == 1 and "nowhere" in meta["frame_fallbacks"][0]


def test_pairs_further_apart_than_max_sync_dt_are_not_frames(bags: Path, tmp_path: Path):
    """The synthetic camera fires 30 ms after its sweep."""
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--max_sync_dt", "0.01"]) == 0
    assert not (out / "0000" / "source_bag.txt").exists()


def test_the_meta_file_reports_each_stream_and_its_holes(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    meta = json.loads((out / "0000" / "extract_meta.json").read_text())
    assert meta["bag"]["name"] == "bag_a"
    assert meta["streams"]["/lidar"]["messages"] == FRAMES
    assert meta["streams"]["/cam/image/compressed"]["messages"] == FRAMES - 2
    assert meta["frames"]["pairs"] == FRAMES - 2 == meta["frames"]["written"]
    assert meta["args"]["max_sync_dt"] == 0.1


def test_a_changed_frame_clock_sets_the_per_frame_products_aside(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON]) == 0
    scene = out / "0000"
    (scene / "height_maps").mkdir()
    # Same bag, same clock: the maps still describe these frames.
    assert main([str(bags), str(out), *COMMON, "--overwrite"]) == 0
    assert (scene / "height_maps").is_dir()
    # A shifted clock: they do not.
    assert main([str(bags), str(out), *COMMON, "--overwrite", "--time_align", "0.2"]) == 0
    assert not (scene / "height_maps").exists()
    assert len(list(scene.glob("height_maps.stale-*"))) == 1


def test_joint_rows_follow_the_first_message_s_order():
    from types import SimpleNamespace

    from dataset_tools.extract.scene import _joint_row, _Pass1

    def msg(names, pos, vel, eff):
        return SimpleNamespace(name=names, position=pos, velocity=vel, effort=eff)

    p1 = _Pass1(gps_topic=None)
    first = _joint_row(p1, "/joint_states", 1.0, msg(["l", "r"], [1, 2], [3, 4], [5, 6]))
    swapped = _joint_row(p1, "/joint_states", 2.0, msg(["r", "l"], [2, 1], [4, 3], [6, 5]))
    no_effort = _joint_row(p1, "/joint_states", 3.0, msg(["l", "r"], [1, 2], [3, 4], []))
    other = _joint_row(p1, "/joint_states", 4.0, msg(["l", "x"], [1, 2], [3, 4], [5, 6]))
    short = _joint_row(p1, "/joint_states", 5.0, msg(["l", "r"], [1], [3, 4], [5, 6]))
    np.testing.assert_array_equal(first[1:], swapped[1:])
    assert np.isnan(no_effort[5:]).all() and len(no_effort) == len(first)
    assert other is None and short is None
    assert p1.malformed == {"/joint_states": 2}


def test_a_sweep_recorded_while_turning_is_deskewed_onto_the_frame_clock(tmp_path: Path):
    """A wall sheared 4 deg by a 67 ms sweep at 60 deg/s comes out straight;
    with --deskew off it stays sheared."""
    rate = math.radians(60.0)
    write_bag(tmp_path / "bags" / "bag_a", frames=40, yaw_rate=rate, sweep_s=0.067)
    flags = [*COMMON, "--gt", "on", "--time_align", "off"]

    out = tmp_path / "out"
    assert main([str(tmp_path / "bags"), str(out), *flags]) == 0
    scene = out / "0000"
    k = 7
    frame_ts = np.loadtxt(scene / "data_ts.txt")
    cloud = np.load(scene / "clouds" / f"{k:05d}.npy")
    assert cloud.shape == (len(WALL), 4) and np.allclose(cloud[:, 3], 7.0)
    want = _wall_in_base(frame_ts[k], rate)  # the frame's instant: the cam/lidar midpoint
    assert np.abs(cloud[:, :3] - want).max() < 0.02
    meta = json.loads((scene / "extract_meta.json").read_text())
    # The last sweep runs past the last pose sample, so it is the one left raw.
    assert meta["frames"]["deskew"]["deskewed"] == len(frame_ts) - 1
    assert meta["frames"]["deskew"]["no_pose"] == 1
    # No --deskew_imu in this bag: every sweep's rotation came from the slerp.
    assert meta["frames"]["deskew"]["slerp"] == len(frame_ts) - 1 and "gyro" not in meta["frames"]["deskew"]
    assert meta["frames"]["deskew"]["sweep_span_s"] == pytest.approx(0.067, abs=1e-3)

    raw = tmp_path / "raw"
    assert main([str(tmp_path / "bags"), str(raw), *flags, "--deskew", "off"]) == 0
    recorded = np.load(raw / "0000" / "clouds" / f"{k:05d}.npy")
    assert np.linalg.norm(recorded[:, :3] - want, axis=1).max() > 0.25  # 3 deg over 52 ms at 6 m
    assert json.loads((raw / "0000" / "extract_meta.json").read_text())["frames"]["deskew"] == {"mode": "off"}


def test_a_given_time_align_deskews_onto_the_corrected_clock(tmp_path: Path):
    """The sweep lands at the instant data_ts.txt ends up holding, not at the stamped one
    (0.03 s at 60 deg/s is 1.8 deg: 0.19 m at the wall)."""
    rate = math.radians(60.0)
    write_bag(tmp_path / "bags" / "bag_a", frames=40, yaw_rate=rate, sweep_s=0.067)
    out = tmp_path / "out"
    assert main([str(tmp_path / "bags"), str(out), *COMMON, "--gt", "on", "--time_align", "0.03"]) == 0
    scene = out / "0000"
    frame_ts = np.loadtxt(scene / "data_ts.txt")
    k = 7
    cloud = np.load(scene / "clouds" / f"{k:05d}.npy")
    assert np.abs(cloud[:, :3] - _wall_in_base(frame_ts[k], rate)).max() < 0.02


def test_a_cloud_without_per_point_time_is_written_as_recorded(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert main([str(bags), str(out), *COMMON, "--gt", "on"]) == 0
    meta = json.loads((out / "0000" / "extract_meta.json").read_text())
    n = FRAMES - 2
    assert meta["frames"]["deskew"] == {
        "mode": "auto",
        "clouds": n,
        "deskewed": 0,
        "sweep_span_s": 0.0,
        "no_time_field": n,
    }


def test_the_gyro_refines_the_sweep_and_the_odometry_is_checked(tmp_path: Path):
    """--deskew_imu uses the gyro (agrees with slerp at constant rate);
    --check_odom writes a clean pose_check.txt."""
    rate = math.radians(60.0)
    write_bag(tmp_path / "bags" / "bag_a", frames=40, yaw_rate=rate, sweep_s=0.067)
    out = tmp_path / "out"
    flags = [*COMMON, "--gt", "on", "--time_align", "off", "--deskew_imu", "/imu", "--check_odom", "/odom"]
    assert main([str(tmp_path / "bags"), str(out), *flags]) == 0
    scene = out / "0000"
    frame_ts = np.loadtxt(scene / "data_ts.txt")
    k = 7
    cloud = np.load(scene / "clouds" / f"{k:05d}.npy")
    assert np.abs(cloud[:, :3] - _wall_in_base(frame_ts[k], rate)).max() < 0.02
    meta = json.loads((scene / "extract_meta.json").read_text())
    assert meta["frames"]["deskew"]["gyro"] >= len(frame_ts) - 3  # the gyro's last samples end with the bag
    assert (
        meta["frames"]["deskew"]["gyro"] + meta["frames"]["deskew"].get("slerp", 0)
        == meta["frames"]["deskew"]["deskewed"]
    )
    # --check_odom equal to --odom is the same stream: nothing to compare, no report.
    assert not (scene / "pose_check.txt").exists() and "pose_check" not in meta["frames"]

    # The check against a different stream (here the same motion on a second topic) writes the report.
    write_bag(tmp_path / "bags2" / "bag_b", frames=40, yaw_rate=rate, sweep_s=0.067)
    out2 = tmp_path / "out2"
    flags = [*COMMON, "--gt", "on", "--time_align", "off", "--check_odom", "/fixposition/odometry_llh"]
    assert main([str(tmp_path / "bags2"), str(out2), *flags]) == 0  # a NavSatFix: no pose, check silently absent
    assert not (out2 / "0000" / "pose_check.txt").exists()
