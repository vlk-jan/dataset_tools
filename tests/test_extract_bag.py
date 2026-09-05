"""End-to-end extraction of a synthetic ROS 2 bag.

``rosbags`` can write bags as well as read them, so the whole bag-reading
path -- stamp collection, synchronisation, cloud parsing, the static-TF chain,
odometry and orientation matching, ground truth, resumption -- runs here on a
bag built in a temporary directory. No recording, no ROS installation.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

from dataset_tools import extract_data as ed

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


def _cloud_msg(t: float, points: np.ndarray):
    """A PointCloud2 with x, y, z, intensity float32 fields."""
    fields = [
        _t("sensor_msgs/msg/PointField")(name=n, offset=4 * i, datatype=7, count=1)
        for i, n in enumerate(("x", "y", "z", "intensity"))
    ]
    data = np.ascontiguousarray(points, dtype=np.float32).tobytes()
    return _t("sensor_msgs/msg/PointCloud2")(
        header=_header(t, "lidar_frame"),
        height=1,
        width=len(points),
        fields=fields,
        is_bigendian=False,
        point_step=16,
        row_step=16 * len(points),
        data=np.frombuffer(data, dtype=np.uint8),
        is_dense=False,
    )


def _image_msg(t: float, value: int):
    img = np.full((8, 12, 3), value, dtype=np.uint8)
    ok, png = cv2.imencode(".png", img)
    assert ok
    return _t("sensor_msgs/msg/CompressedImage")(header=_header(t, "cam_optical"), format="png", data=png.ravel())


def _camera_info_msg(t: float):
    return _t("sensor_msgs/msg/CameraInfo")(
        header=_header(t, "cam_optical"),
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


def _odom_msg(t: float, x: float):
    pose = _t("geometry_msgs/msg/Pose")(position=_t("geometry_msgs/msg/Point")(x=x, y=0.0, z=0.0), orientation=_quat())
    twist = _t("geometry_msgs/msg/Twist")(linear=_vec(), angular=_vec())
    return _t("nav_msgs/msg/Odometry")(
        header=_header(t, "odom"),
        child_frame_id="base_link",
        pose=_t("geometry_msgs/msg/PoseWithCovariance")(pose=pose, covariance=np.zeros(36)),
        twist=_t("geometry_msgs/msg/TwistWithCovariance")(twist=twist, covariance=np.zeros(36)),
    )


def _imu_msg(t: float, roll: float):
    return _t("sensor_msgs/msg/Imu")(
        header=_header(t, "base_link"),
        orientation=_quat(x=math.sin(roll / 2), w=math.cos(roll / 2)),
        orientation_covariance=np.zeros(9),
        angular_velocity=_vec(),
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


def _tf_static_msg():
    def tf(parent, child, xyz, quat):
        return _t("geometry_msgs/msg/TransformStamped")(
            header=_header(T0, parent),
            child_frame_id=child,
            transform=_t("geometry_msgs/msg/Transform")(translation=_vec(*xyz), rotation=quat),
        )

    return _t("tf2_msgs/msg/TFMessage")(
        transforms=[
            tf("base_link", "lidar_frame", LIDAR_XYZ, _quat(x=1.0, w=0.0)),  # rolled 180 deg
            tf("base_link", "cam_optical", (0.3, 0.0, 0.2), _quat()),
        ]
    )


def write_bag(bag_dir: Path, frames: int = FRAMES, dropped_camera: set[int] = frozenset()) -> Path:
    """A bag with everything the extractor reads, ``frames`` lidar sweeps long.

    Camera frames come 30 ms after their sweep; ``dropped_camera`` sweeps have
    no image at all and must therefore not become frames.
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
                ("/tf_static", "tf2_msgs/msg/TFMessage"),
            )
        }

        def put(topic, t, msg):
            writer.write(conns[topic], int(t * 1e9), TS.serialize_cdr(msg, conns[topic].msgtype))

        put("/tf_static", T0, _tf_static_msg())
        rng = np.random.default_rng(0)
        for k in range(frames):
            t = T0 + 0.1 * k
            pts = rng.uniform(1.0, 5.0, size=(40, 4)).astype(np.float32)
            pts[0] = [np.nan, 0, 0, 1]  # a NaN return
            pts[1] = [0, 0, 0, 5]  # a dropped shot encoded as the origin
            pts[2] = [1.0, 0.0, 0.0, 9.0]  # a known point, to check the mount transform
            put("/lidar", t, _cloud_msg(t, pts))
            if k not in dropped_camera:
                put("/cam/image/compressed", t + 0.03, _image_msg(t + 0.03, k))
                put("/cam/camera_info", t + 0.03, _camera_info_msg(t + 0.03))
            put("/odom", t, _odom_msg(t, 0.1 * k))  # 1 m/s: under the ground-truth teleport gate
            put("/imu", t, _imu_msg(t, math.radians(10.0)))
            put("/joint_states", t, _joint_msg(t, k))
            put("/cmd_vel", t, _t("geometry_msgs/msg/Twist")(linear=_vec(x=0.4), angular=_vec(z=0.1)))
    return bag_dir


COMMON = ["--cam", "/cam/image/compressed", "--lidar", "/lidar", "--odom", "/odom", "--imu", "/imu", "--jobs", "1"]


@pytest.fixture
def bags(tmp_path: Path) -> Path:
    root = tmp_path / "bags"
    write_bag(root / "bag_a", dropped_camera={5, 6})
    return root


def test_a_synthetic_bag_extracts_to_a_complete_scene(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert ed.main([str(bags), str(out), *COMMON, "--gt", "on"]) == 0
    scene = out / "0000"
    n = FRAMES - 2  # two sweeps had no camera frame
    assert ed.scene_is_complete(scene)
    assert ed.scene_frame_counts(scene) == {"images": n, "clouds": n, "data_ts.txt": n, "pcd_ts.txt": n}

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
    assert np.allclose(gravity[0], ed.euler_matrix(math.radians(10.0), 0.0, 0.0)[:3, :3])

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


def test_a_second_run_skips_the_complete_scene(bags: Path, tmp_path: Path):
    out = tmp_path / "out"
    assert ed.main([str(bags), str(out), *COMMON]) == 0
    marker = scene = out / "0000"
    mtime = (marker / "source_bag.txt").stat().st_mtime_ns
    assert ed.main([str(bags), str(out), *COMMON]) == 0
    assert (scene / "source_bag.txt").stat().st_mtime_ns == mtime


def test_overwrite_removes_what_an_earlier_run_left_behind(bags: Path, tmp_path: Path):
    """A stale frame from a previous extraction must not outlive a re-extraction.

    Otherwise the counts never agree again: the scene is 'incomplete' forever
    and the validator reports a mismatch on every run.
    """
    out = tmp_path / "out"
    assert ed.main([str(bags), str(out), *COMMON]) == 0
    scene = out / "0000"
    (scene / "images" / "00099.png").write_bytes(b"stale")
    (scene / "clouds" / "00099.npy").write_bytes(b"stale")
    (scene / "regen").mkdir()
    (scene / "regen" / "keep.txt").write_text("provenance, not ours to delete\n")
    (scene / "odometry.txt").write_text("# from the FAST-LIO regeneration, which runs before the extraction\n")
    assert not ed.scene_is_complete(scene)

    assert ed.main([str(bags), str(out), *COMMON, "--overwrite"]) == 0
    assert ed.scene_is_complete(scene)
    assert not (scene / "images" / "00099.png").exists() and not (scene / "clouds" / "00099.npy").exists()
    assert (scene / "regen" / "keep.txt").is_file()
    # --gt is off for /odom, so the regenerated ground truth is not this run's to remove.
    assert (scene / "odometry.txt").read_text().startswith("# from the FAST-LIO")


def test_header_stamp_read_from_cdr_matches_the_deserialised_one():
    msg = _image_msg(T0 + 0.123456789, 1)
    raw = TS.serialize_cdr(msg, "sensor_msgs/msg/CompressedImage")
    assert ed.header_stamp_from_cdr(raw) == ed.header_stamp(
        TS.deserialize_cdr(raw, "sensor_msgs/msg/CompressedImage"), 0.0
    )
    assert ed.header_stamp_from_cdr(b"\x00\x01\x00") is None
