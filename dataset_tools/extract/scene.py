"""One bag into one scene directory."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np

from dataset_tools import (  # noqa: F401  (rosbags_compat patches rosbags)
    gnss_guard,
    rosbags_compat,
    tf_pins,
    time_align,
)
from dataset_tools.extract import deskew
from dataset_tools.extract.camera import IMAGE_MSGTYPES, get_first_frame, save_camera_extrinsic, save_intrinsics
from dataset_tools.extract.gps import GPS_MSGTYPE, gps_row, pick_gps_topic
from dataset_tools.extract.ground_truth import _auto, write_ground_truth
from dataset_tools.extract.pointcloud import cloud_to_xyzit, header_stamp_from_cdr, valid_returns_mask
from dataset_tools.extract.scene_io import (
    DERIVED_DIRS,
    GNSS_GUARD_FILE,
    JOINT_NAMES_FILE,
    META_FILE,
    SENTINEL_FILE,
    STATUS_EMPTY,
    STATUS_OK,
    TIME_ALIGN_FILE,
    SceneResult,
    clear_scene_outputs,
    image_extension,
    root_image_extension,
)
from dataset_tools.extract.sync import interpolate_poses, sync_timestamps
from dataset_tools.extract.tf_graph import build_tf_graph, get_transform
from dataset_tools.gt_io import anchor_to_first
from dataset_tools.transforms import (
    euler_matrix,
    header_stamp,
    pose_to_matrix,
    quaternion_matrix,
    roll_pitch_from_matrix,
)

LOG = logging.getLogger("helhest.extract")


def align_frame_clock(scene_dir: Path, frame_ts: np.ndarray, odom_data, args, bag: str):
    """Take the sensor unit's capture-to-publish latency off the frame clock.

    ``--time_align auto`` measures it against the clouds just written (see
    :mod:`dataset_tools.time_align`); a number applies that constant instead,
    for a rig already characterised or a scene too poor in geometry to measure;
    ``off`` leaves the stamps alone. Returns the corrected clock and the profile
    to record, or ``(frame_ts, None)`` when nothing was applied.
    """
    mode = args.time_align
    if mode == "off":
        return frame_ts, None

    if mode != "auto":
        constant = float(mode)
        LOG.info("%s: frame clock shifted by %+.3f s (--time_align %s)", bag, constant, mode)
        profile = time_align.Profile([], np.array([frame_ts[0]]), np.array([constant]), "given on the command line")
        shifted, _ = time_align.apply_to_clock(frame_ts, profile)
        return shifted, profile

    if not odom_data:
        LOG.info("%s: no odometry to align the frame clock against; stamps left as recorded", bag)
        return frame_ts, None

    odom = sorted(odom_data, key=lambda x: x[0])
    profile = time_align.estimate(
        scene_dir,
        frame_ts,
        np.array([x[0] for x in odom]),
        np.array([x[1] for x in odom]),
    )
    LOG.info("%s: frame clock -- %s", bag, profile.summary)
    if profile.is_identity:
        return frame_ts, profile
    shifted, forced = time_align.apply_to_clock(frame_ts, profile)
    if forced:
        LOG.warning("%s: the measured offset ran backwards faster than the clock; forced monotonic", bag)
    return shifted, profile


@dataclass
class _Pass1:
    """What the first pass over the bag collects: the stamps of the bulky
    streams, and the small streams whole."""

    gps_topic: str | None
    cam_ts: list = field(default_factory=list)
    lidar_ts: list = field(default_factory=list)
    odom: list = field(default_factory=list)  # (ts, transform of odom child frame in odom fixed frame)
    odom_child_frame: str | None = None
    imu: list = field(default_factory=list)  # (ts, rotation matrix of imu frame in gravity-referenced frame)
    imu_frame: str | None = None
    joint_states: list = field(default_factory=list)
    joint_setpoints: list = field(default_factory=list)
    cmd_vel: list = field(default_factory=list)
    gps_rows: list = field(default_factory=list)  # (t, lat, lon, alt, status, cov diagonal) of a NavSatFix
    tf_msgs: list = field(default_factory=list)
    #: Connections whose stamps are read straight out of the raw bytes.
    verified: set[int] = field(default_factory=set)
    #: The joint order of every joint_states / joint_setpoints row: the first message's.
    joint_names: list | None = None
    #: Every stamp, by topic: what extract_meta.json reports coverage and gaps from.
    stamps: dict = field(default_factory=lambda: defaultdict(list))
    #: Messages whose header stamp was never set, by topic; their receive time stands in.
    unset_stamps: Counter = field(default_factory=Counter)
    #: Joint messages that could not be lined up with ``joint_names``, by topic; skipped.
    malformed: Counter = field(default_factory=Counter)
    #: Missing TF paths the run went on without (--allow_frame_fallback).
    frame_fallbacks: list = field(default_factory=list)
    #: Frame bookkeeping for extract_meta.json.
    counts: dict = field(default_factory=dict)


def process_scene(reader, bag_path: Path, scene_dir: Path, args) -> SceneResult:
    bag = bag_path.name
    result = SceneResult(index=-1, bag=bag, status=STATUS_OK)

    conns = {c.topic: c for c in reader.connections}
    cam_conn = conns.get(args.cam)
    lidar_conn = conns.get(args.lidar)

    if not cam_conn or not lidar_conn:
        missing = [topic for topic, conn in ((args.cam, cam_conn), (args.lidar, lidar_conn)) if conn is None]
        LOG.warning("%s: missing topic(s) %s; scene skipped", bag, missing)
        result.status = STATUS_EMPTY
        result.message = f"missing topic(s): {', '.join(missing)}"
        return result
    # Pass 2 hands the payload to cv2.imdecode, which cannot read a raw
    # sensor_msgs/Image; said here, once, instead of as an OpenCV assertion on
    # the first frame.
    if cam_conn.msgtype not in IMAGE_MSGTYPES:
        raise ValueError(f"--cam {args.cam} is {cam_conn.msgtype}; expected one of {', '.join(IMAGE_MSGTYPES)}")

    # The two 'auto' modes are decided by the odometry topic, which is worth a
    # line in the log: a rig this tool has not met gets neither ground truth
    # nor the pose guard, and would otherwise not be told why.
    if args.gt == "auto" or args.gnss_guard == "auto":
        LOG.info(
            "%s: from --odom %s: ground truth %s, GNSS guard %s",
            bag,
            args.odom,
            "on" if _auto(args.gt, args.odom) else "off (--gt on to force)",
            "on" if _auto(args.gnss_guard, args.odom) else "off (--gnss_guard on to force)",
        )

    # A scene is extracted from scratch every time, so whatever an earlier run
    # left behind must go first (see clear_scene_outputs). Its frame clock is
    # remembered, to tell whether the per-frame products built on it still hold.
    old_clock = _digest(scene_dir / "data_ts.txt")
    clear_scene_outputs(scene_dir, ground_truth=_auto(args.gt, args.odom))

    p1 = _collect(reader, conns, cam_conn, lidar_conn, args)
    selected = _select_frames(p1, args, bag)
    if isinstance(selected, str):
        result.status = STATUS_EMPTY
        result.message = selected
        return result
    cam_ts, pcd_ts, frame_ts, guard_verdict = selected

    (scene_dir / "images").mkdir(parents=True, exist_ok=True)
    (scene_dir / "clouds").mkdir(parents=True, exist_ok=True)
    _write_small_streams(p1, scene_dir, bag)
    tf_graph, T_lidar_to_base = _resolve_frames(reader, lidar_conn, p1, scene_dir, args, bag)
    _write_frames(
        reader,
        conns,
        cam_conn,
        lidar_conn,
        p1,
        cam_ts,
        pcd_ts,
        frame_ts,
        tf_graph,
        T_lidar_to_base,
        scene_dir,
        args,
        bag,
    )
    frame_ts = _write_derived(scene_dir, frame_ts, pcd_ts, p1, guard_verdict, args, bag)
    _set_aside_stale(scene_dir, old_clock, bag)
    _write_meta(scene_dir, reader, bag_path, p1, guard_verdict, len(frame_ts), args)

    # Provenance; written last, so it doubles as the "scene finished" marker.
    (scene_dir / SENTINEL_FILE).write_text(f"{bag}\n")

    result.frames = int(len(frame_ts))
    result.start_time = float(frame_ts[0])
    return result


def _collect(reader, conns, cam_conn, lidar_conn, args) -> _Pass1:
    """Pass 1: every stamp of the camera and the lidar, and the small streams whole."""
    # Added to the camera stamp in both passes, in one expression, so the two
    # produce bit-identical floats and pass 2's set lookups hit. This is the
    # camera's stamping against the LIDAR's, and nothing else: the offset both
    # of them share against the pose clock is a latency, not a stamp error, and
    # is measured after the clouds exist (see --time_align).
    shift_cam = args.cam_time_offset
    # Found by message type rather than by name, so the same run works on both
    # rig generations; None when the bag carries no fix at all.
    p1 = _Pass1(gps_topic=pick_gps_topic(reader.connections, args.gps))

    relevant_conns = [
        cam_conn,
        lidar_conn,
        conns.get(args.odom),
        conns.get(args.imu),
        conns.get("/joint_states"),
        conns.get("/joint_setpoints"),
        conns.get("/cmd_vel"),
        conns.get(p1.gps_topic),
        conns.get("/tf_static"),
    ]
    # Deduplicate (e.g. when --odom and --imu are the same topic)
    relevant_conns = list({id(c): c for c in relevant_conns if c is not None}.values())

    # Images and clouds are the bulk of the bag and this pass needs only their
    # stamps, which are read out of the raw bytes (header_stamp_from_cdr). The
    # shortcut is checked against a full decode of each connection's first
    # message and dropped for that connection if the two disagree.
    stamp_only = {id(cam_conn), id(lidar_conn)}
    verified = p1.verified

    def stamp(msg, recv: float, topic: str) -> float:
        # The header stamp, or the bag's receive time for a message without one
        # -- in seconds, as pass 2 computes it -- counting the headers whose
        # stamp was never set (transforms.MIN_VALID_STAMP).
        t = header_stamp(msg, None)
        if t is None:
            if hasattr(msg, "header"):
                p1.unset_stamps[topic] += 1
            t = recv
        return t

    for conn, timestamp, raw in reader.messages(relevant_conns):
        recv = timestamp * 1e-9
        if id(conn) in stamp_only and id(conn) in verified:
            t = header_stamp_from_cdr(raw)
            if t is None:
                t = stamp(reader.deserialize(raw, conn.msgtype), recv, conn.topic)
        else:
            msg = reader.deserialize(raw, conn.msgtype)
            t = stamp(msg, recv, conn.topic)
            if id(conn) in stamp_only:
                if header_stamp_from_cdr(raw) == t:
                    verified.add(id(conn))
                else:
                    stamp_only.discard(id(conn))
        if conn.topic != "/tf_static":
            p1.stamps[conn.topic].append(t)

        if conn.topic == args.cam:
            p1.cam_ts.append(t + shift_cam)
        elif conn.topic == args.lidar:
            p1.lidar_ts.append(t)
        elif conn.topic in (args.odom, args.imu):
            # The same topic may serve as both odometry and orientation source
            if conn.topic == args.odom:
                p1.odom_child_frame = msg.child_frame_id
                p1.odom.append((t, pose_to_matrix(msg.pose.pose)))
            if conn.topic == args.imu:
                # nav_msgs/Odometry also works as an orientation source
                if hasattr(msg, "orientation"):  # sensor_msgs/Imu
                    q = msg.orientation
                    p1.imu_frame = msg.header.frame_id
                else:
                    q = msg.pose.pose.orientation
                    p1.imu_frame = msg.child_frame_id
                p1.imu.append((t, quaternion_matrix([q.x, q.y, q.z, q.w])))
        elif conn.topic in ("/joint_states", "/joint_setpoints"):
            row = _joint_row(p1, conn.topic, t, msg)
            if row is not None:
                (p1.joint_states if conn.topic == "/joint_states" else p1.joint_setpoints).append(row)
        elif conn.topic == "/cmd_vel":
            tw = msg.twist if hasattr(msg, "twist") else msg
            p1.cmd_vel.append(
                np.r_[
                    t,
                    tw.linear.x,
                    tw.linear.y,
                    tw.linear.z,
                    tw.angular.x,
                    tw.angular.y,
                    tw.angular.z,
                ]
            )
        elif conn.topic == p1.gps_topic:
            row = gps_row(t, msg)
            if row is not None:
                p1.gps_rows.append(row)
        elif conn.topic == "/tf_static":
            p1.tf_msgs.extend(msg.transforms)
    return p1


def _joint_row(p1: _Pass1, topic: str, t: float, msg):
    """One joint_states / joint_setpoints row, its columns in ``p1.joint_names`` order.

    The order is the first message's, and is written to joint_names.txt. A later
    message naming its joints in another order is reordered; one naming other
    joints, or carrying a column of the wrong length, is skipped and counted. A
    column a message leaves empty (some drivers never fill effort) becomes NaN
    rather than a short row that would make the whole array fail to stack.
    """
    names = list(msg.name) or [f"joint{i}" for i in range(len(msg.position))]
    if p1.joint_names is None:
        p1.joint_names = names
    want = p1.joint_names
    if sorted(names) != sorted(want):
        p1.malformed[topic] += 1
        return None
    order = [names.index(j) for j in want]
    cols = []
    for values in (msg.position, msg.velocity, msg.effort):
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            cols.append(np.full(len(want), np.nan))
        elif len(values) == len(names):
            cols.append(values[order])
        else:
            p1.malformed[topic] += 1
            return None
    return np.r_[(t, *cols)]


def _select_frames(p1: _Pass1, args, bag: str):
    """Which camera/lidar pairs become frames: ``(cam_ts, pcd_ts, frame_ts,
    guard_verdict)``, or why the scene is empty, as a string."""
    # Sync timestamps (mutual nearest neighbour)
    valid_cam_ts, valid_pcd_ts = sync_timestamps(p1.cam_ts, p1.lidar_ts)
    p1.counts.update(camera_messages=len(p1.cam_ts), lidar_messages=len(p1.lidar_ts), pairs=len(valid_cam_ts))
    # Mutual nearest neighbours can still be far apart: across a camera stall,
    # the image either side pairs with whichever sweep is left.
    close = np.abs(valid_cam_ts - valid_pcd_ts) <= args.max_sync_dt
    p1.counts["pairs_over_max_sync_dt"] = int(np.count_nonzero(~close))
    if not close.all():
        LOG.warning(
            "%s: %d camera/lidar pairs more than %.3f s apart; dropped (--max_sync_dt)",
            bag,
            np.count_nonzero(~close),
            args.max_sync_dt,
        )
        valid_cam_ts, valid_pcd_ts = valid_cam_ts[close], valid_pcd_ts[close]

    if len(valid_cam_ts) == 0:
        LOG.warning(
            "%s: no synchronised camera/lidar pairs (%d camera, %d lidar messages); scene left empty",
            bag,
            len(p1.cam_ts),
            len(p1.lidar_ts),
        )
        return f"no synced pairs ({len(p1.cam_ts)} cam, {len(p1.lidar_ts)} lidar)"

    # One clock per frame. transforms.npy and gravity_align.npy used to be
    # sampled at the camera and the lidar stamps respectively, while data_ts.txt
    # was their midpoint -- three clocks behind one frame index, which is not
    # what a consumer reading "frame k" from all three expects.
    #
    # This is the clock as the sensors stamped it. Where the unit stamps at
    # publish rather than at capture it is late by the pipeline's latency, which
    # is measured and taken off below, once there are clouds to measure with.
    frame_ts = (valid_cam_ts + valid_pcd_ts) / 2

    # Whatever the GNSS/INS fabricated goes now, before a single image or cloud
    # is written, so the frames that survive are exactly the frames the dataset
    # can put a pose on and the indices below stay contiguous. The poses are
    # still the receiver's own output here, not yet re-expressed at the base
    # frame: a fabricated stretch is a property of what the receiver published,
    # and moving it down a lever arm only lets a fast rotation masquerade as a
    # jump. The same stage drops frames the pose stream never covered at all,
    # before it starts, after it ends or inside a hole too wide to interpolate
    # across: those would be exported carrying the nearest sample, which is a
    # fabricated pose with a different author. See dataset_tools.gnss_guard.
    guard_verdict = None
    if _auto(args.gnss_guard, args.odom) and p1.odom:
        p1.odom.sort(key=lambda x: x[0])
        # The stamps of the fixes that carried a position. The gaps in them are
        # when the receiver had no solution, whether it said so by publishing
        # null island (gps_row drops those) or by publishing nothing at all.
        # A rig with a fix topic but not one usable fix is an empty list, not
        # None: the receiver never had a solution, which is evidence too.
        fix_ts = [float(r[0]) for r in p1.gps_rows] if p1.gps_topic else None
        p1.odom, keep, guard_verdict = gnss_guard.guard(p1.odom, frame_ts, fix_ts)
        if not guard_verdict.clean:
            LOG.warning("%s: %s", bag, guard_verdict.summary(float(frame_ts[0])))
            valid_cam_ts, valid_pcd_ts, frame_ts = valid_cam_ts[keep], valid_pcd_ts[keep], frame_ts[keep]
            if len(valid_cam_ts) == 0:
                LOG.warning("%s: no frame of this bag has a trustworthy pose; scene left empty", bag)
                return "no frame of this bag has a pose the dataset can stand behind"
        else:
            LOG.info("%s: pose guard -- %s", bag, guard_verdict.summary(float(frame_ts[0])))
    return valid_cam_ts, valid_pcd_ts, frame_ts, guard_verdict


def _write_small_streams(p1: _Pass1, scene_dir: Path, bag: str) -> None:
    """Proprioception and GNSS, whole and in time order."""
    for name, rows in [
        ("joint_states", p1.joint_states),
        ("joint_setpoints", p1.joint_setpoints),
        ("cmd_vel", p1.cmd_vel),
    ]:
        if rows:
            arr = np.vstack(rows)
            np.save(scene_dir / f"{name}.npy", arr[np.argsort(arr[:, 0])])
        else:
            LOG.info("%s: no messages on /%s; %s.npy not written", bag, name, name)
    if p1.joint_names is not None:
        (scene_dir / JOINT_NAMES_FILE).write_text("".join(f"{n}\n" for n in p1.joint_names))
    for topic, n in sorted(p1.malformed.items()):
        LOG.warning("%s: %d messages on %s did not match the joints %s; skipped", bag, n, topic, p1.joint_names)
    for topic, n in sorted(p1.unset_stamps.items()):
        LOG.warning("%s: %d messages on %s carry no wall-clock stamp; their receive time is used", bag, n, topic)

    # GPS is an optional extra, not part of the scene contract: a bag without a
    # fix is a scene without gps.npy, exactly as one without /cmd_vel is.
    if p1.gps_rows:
        arr = np.vstack(p1.gps_rows)
        np.save(scene_dir / "gps.npy", arr[np.argsort(arr[:, 0])])
        LOG.info("%s: %d GNSS fixes from '%s'", bag, len(p1.gps_rows), p1.gps_topic)
    elif p1.gps_topic:
        LOG.info("%s: no usable fixes on '%s'; gps.npy not written", bag, p1.gps_topic)
    else:
        LOG.info("%s: no %s topic in the bag; gps.npy not written", bag, GPS_MSGTYPE)


def _resolve_frames(reader, lidar_conn, p1: _Pass1, scene_dir: Path, args, bag: str):
    """The static TF tree (pinned where the pins apply), the lidar mount, and
    the pose and orientation streams re-expressed at the base frame (in place
    on ``p1``). Returns ``(tf_graph, T_lidar_to_base)``."""
    tf_graph = build_tf_graph(p1.tf_msgs)
    if args.tf_pins == "on":
        pinned = tf_pins.apply(tf_graph)
        for rec in pinned:
            (LOG.warning if rec.far else LOG.info)(
                "%s: %s -> %s pinned; the bag's own value is %.0f mm, %.2f deg from the pin%s",
                bag,
                rec.pin.parent,
                rec.pin.child,
                rec.translation_m * 1000,
                rec.rotation_deg,
                " (FAR: check the clouds of this scene against its neighbours)" if rec.far else "",
            )
        if pinned:
            tf_pins.write_report(scene_dir / tf_pins.FILE, pinned)

    lidar_frame = get_first_frame(reader, lidar_conn)
    T_lidar_to_base = get_transform(tf_graph, lidar_frame, args.base_frame)
    if T_lidar_to_base is None:
        if args.lidar_extrinsic is not None:
            x, y, z, roll, pitch, yaw = args.lidar_extrinsic
            T_lidar_to_base = euler_matrix(roll, pitch, yaw)
            T_lidar_to_base[:3, 3] = [x, y, z]
        else:
            _frame_fallback(
                p1,
                args,
                bag,
                f"no TF path from '{lidar_frame}' to '{args.base_frame}' and no --lidar_extrinsic given: "
                "clouds stay in the lidar frame",
            )
            T_lidar_to_base = np.eye(4)

    # Odom poses track the odometry child frame (e.g. FP_POI for Fixposition);
    # re-express them as poses of the base frame when TF allows it.
    child = p1.odom_child_frame
    T_child_to_base = get_transform(tf_graph, child, args.base_frame) if child else None
    if T_child_to_base is not None and not np.allclose(T_child_to_base, np.eye(4)):
        T_base_in_child = np.linalg.inv(T_child_to_base)
        p1.odom = [(t, T @ T_base_in_child) for t, T in p1.odom]
    elif child and T_child_to_base is None:
        _frame_fallback(
            p1,
            args,
            bag,
            f"no TF path from odom child '{child}' to '{args.base_frame}': poses stay those of '{child}'",
        )

    # Re-express the orientation source at the base frame too, so gravity
    # alignment reflects the base tilt rather than the sensor mounting.
    imu_frame = p1.imu_frame
    T_imu_to_base = get_transform(tf_graph, imu_frame, args.base_frame) if imu_frame else None
    if T_imu_to_base is not None and not np.allclose(T_imu_to_base, np.eye(4)):
        T_base_in_imu = np.linalg.inv(T_imu_to_base)
        p1.imu = [(t, R @ T_base_in_imu) for t, R in p1.imu]
    elif imu_frame and T_imu_to_base is None:
        _frame_fallback(
            p1,
            args,
            bag,
            f"no TF path from orientation frame '{imu_frame}' to '{args.base_frame}': "
            "gravity_align.npy is the sensor's tilt, not the base's",
        )
    return tf_graph, T_lidar_to_base


def _frame_fallback(p1: _Pass1, args, bag: str, what: str) -> None:
    """A frame the run cannot put where it belongs: an error, unless asked to go on."""
    if not args.allow_frame_fallback:
        raise ValueError(f"{what} (--allow_frame_fallback to extract it like that anyway)")
    LOG.warning("%s: %s", bag, what)
    p1.frame_fallbacks.append(what)


def _write_frames(
    reader,
    conns,
    cam_conn,
    lidar_conn,
    p1: _Pass1,
    valid_cam_ts,
    valid_pcd_ts,
    frame_ts,
    tf_graph,
    T_lidar_to_base,
    scene_dir,
    args,
    bag: str,
) -> None:
    """Pass 2: the image and the cloud of every frame, under its chronological index.

    Each cloud is written in the base frame and, where the sweep carries
    per-point times and the pose stream covers it, moved onto the frame's own
    instant ``frame_ts[k]`` (see :mod:`dataset_tools.extract.deskew`).
    """
    camera_topic, lidar_topic = args.cam, args.lidar
    # The poses are at the base frame by now (_resolve_frames), which is the
    # frame the clouds are about to be written in -- the one deskewing needs.
    track = deskew.build_track(p1.odom) if args.deskew != "off" else None
    sweeps = Counter()
    span = 0.0
    shift_cam = args.cam_time_offset
    verified = p1.verified
    cam_ts_to_idx = {ts: i for i, ts in enumerate(valid_cam_ts)}
    pcd_ts_to_idx = {ts: i for i, ts in enumerate(valid_pcd_ts)}
    cam_set = set(valid_cam_ts)
    pcd_set = set(valid_pcd_ts)

    intrinsics_saved = False
    image_how = args.image_format
    image_ext = None  # one format per scene, and per root: decided by the first frame

    for conn, timestamp, raw in reader.messages([cam_conn, lidar_conn]):
        # Same stamp arithmetic as pass 1, so the set lookups below hit exactly
        # -- including the fallback to the bag's own timestamp for a message
        # type without a header.
        is_cam = conn.topic == camera_topic
        shift = shift_cam if is_cam else 0.0
        t = header_stamp_from_cdr(raw) if id(conn) in verified else None
        if t is None:
            msg = reader.deserialize(raw, conn.msgtype)
            t = header_stamp(msg, timestamp * 1e-9) + shift
        else:
            t += shift
            if t not in (cam_set if is_cam else pcd_set):
                continue  # not a synchronised frame: never decoded at all
            msg = reader.deserialize(raw, conn.msgtype)

        if conn.topic == camera_topic and t in cam_set:
            img_idx = cam_ts_to_idx[t]  # Get correct chronological index

            # Use OpenCV directly
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if cv_img is None:
                # A frame nothing can be written for leaves a hole in an
                # index-aligned scene, so the scene fails -- but with the frame
                # named, not with an OpenCV assertion from cv2.imwrite.
                raise ValueError(f"image {img_idx} (stamp {t:.6f}, {len(np_arr)} bytes) could not be decoded")
            ext = image_extension(np_arr, image_how, img_idx)
            if image_ext is None:
                held = root_image_extension(scene_dir)
                if held is not None and held != ext:
                    raise ValueError(
                        f"{scene_dir.parent} holds {held} images and this bag would add {ext}: a root has one "
                        f"image format. Extract into a new root"
                        + (", or pass --image_format png to extend this one" if held == ".png" else "")
                    )
                image_ext = ext
            elif ext != image_ext:
                raise ValueError(f"image {img_idx} is {ext} where the scene so far is {image_ext}")
            image_path = scene_dir / "images" / f"{img_idx:05d}{ext}"
            if image_how == "png":
                cv2.imwrite(str(image_path), cv_img)
            else:
                # The camera's own bytes. Decoded above all the same, because a
                # frame that cannot be read is better found here than in a loader.
                image_path.write_bytes(np_arr.tobytes())

            # Extract Intrinsics once
            if not intrinsics_saved:
                save_intrinsics(reader, conns, camera_topic, scene_dir)
                save_camera_extrinsic(tf_graph, msg.header.frame_id, args.base_frame, scene_dir)
                intrinsics_saved = True

            cam_set.remove(t)  # Handle duplicates

        elif conn.topic == lidar_topic and t in pcd_set:
            pcd_idx = pcd_ts_to_idx[t]  # Get correct chronological index
            # Before T_lidar_to_base, never after -- see drop_invalid_returns.
            xyzi, offsets = cloud_to_xyzit(msg, t)
            keep = valid_returns_mask(xyzi)
            xyzi = xyzi[keep]
            pcd = np.hstack((xyzi[:, :3], np.ones((len(xyzi), 1), dtype=np.float32)))
            pcd = (pcd @ T_lidar_to_base.T.astype(np.float32))[:, :3]
            if track is not None:
                if offsets is None:
                    sweeps["no_time_field"] += 1
                else:
                    pcd, status = deskew.deskew(pcd, offsets[keep], t, float(frame_ts[pcd_idx]), track)
                    sweeps[status] += 1
                    if keep.any():
                        span = max(span, float(np.ptp(offsets[keep])))
            np.save(
                scene_dir / "clouds" / f"{pcd_idx:05d}.npy",
                np.hstack((pcd, xyzi[:, 3:4])),
            )
            pcd_set.remove(t)

    _report_deskew(p1, track, sweeps, span, len(valid_pcd_ts), args, bag)


def _report_deskew(p1: _Pass1, track, sweeps: Counter, span: float, clouds: int, args, bag: str) -> None:
    """One line in the log and one entry in extract_meta.json about what the
    sweeps were moved by -- or why they were not."""
    if args.deskew == "off":
        p1.counts["deskew"] = {"mode": "off"}
        return
    if track is None:
        p1.counts["deskew"] = {"mode": "auto", "clouds": clouds, "deskewed": 0, "reason": "no pose stream"}
        LOG.info("%s: no pose stream to deskew the sweeps against; clouds written as recorded", bag)
        return
    entry = {"mode": "auto", "clouds": clouds, "deskewed": int(sweeps["deskewed"]), "sweep_span_s": round(span, 4)}
    entry.update({k: int(v) for k, v in sorted(sweeps.items()) if k != "deskewed" and v})
    p1.counts["deskew"] = entry
    if sweeps["deskewed"]:
        LOG.info(
            "%s: %d/%d sweeps deskewed onto the frame clock (span up to %.3f s)%s",
            bag,
            sweeps["deskewed"],
            clouds,
            span,
            "".join(f"; {v} {k.replace('_', ' ')}" for k, v in sorted(sweeps.items()) if k != "deskewed" and v),
        )
    elif sweeps["no_time_field"]:
        LOG.info("%s: the clouds carry no per-point time; written as recorded (nothing to deskew)", bag)
    else:
        LOG.warning("%s: no sweep could be deskewed: %s", bag, dict(sweeps))


def _write_derived(scene_dir: Path, frame_ts, valid_pcd_ts, p1: _Pass1, guard_verdict, args, bag: str):
    """Everything sampled onto the frame clock once the clouds exist: the
    corrected clock itself, the guard's report, ground truth, transforms and
    gravity. Returns the corrected frame clock."""
    # The frame clock, corrected now that there are clouds to measure against.
    # The lidar's own clock takes the same correction -- it is the same unit's
    # latency, and pcd_ts.txt has to stay comparable with data_ts.txt -- so the
    # camera/lidar offset within a frame survives it untouched. Everything
    # sampled onto the clock below follows from here.
    corrected, align = align_frame_clock(scene_dir, frame_ts, p1.odom, args, bag)
    applied = corrected - frame_ts
    frame_ts = corrected
    pcd_ts = valid_pcd_ts + applied

    # The guard kept frames within FRAME_PAD of a pose, because the clock was
    # not corrected yet. Now it is: a frame that still has no pose bracketing it
    # would be handed a copy of the nearest one (interpolate_poses), so it goes.
    if guard_verdict is not None and p1.odom:
        keep = gnss_guard.frame_support(frame_ts, [t for t, _ in p1.odom], pad=0.0)
        if not keep.all():
            guard_verdict.n_frames_trimmed = int(np.count_nonzero(~keep))
            LOG.warning(
                "%s: %d frames have no pose bracketing them on the corrected clock; dropped",
                bag,
                guard_verdict.n_frames_trimmed,
            )
            if not keep.any():
                raise ValueError("no frame keeps a pose once the frame clock is corrected")
            _drop_frames(scene_dir, keep)
            frame_ts, pcd_ts = frame_ts[keep], pcd_ts[keep]

    np.savetxt(scene_dir / "data_ts.txt", frame_ts)
    np.savetxt(scene_dir / "pcd_ts.txt", pcd_ts)
    if align is not None:
        time_align.write_report(scene_dir / TIME_ALIGN_FILE, align, float(frame_ts[0]))

    if guard_verdict is not None:
        gnss_guard.write_report(scene_dir / GNSS_GUARD_FILE, guard_verdict, float(frame_ts[0]))

    # Process Odom Transforms (relative to first frame)
    if _auto(args.gt, args.odom) and p1.odom:
        if write_ground_truth(p1.odom, scene_dir, spans=guard_verdict.spans if guard_verdict else None):
            LOG.info("%s: ground truth written (odometry.txt + segments)", bag)
        else:
            LOG.warning("%s: too few odometry samples for ground truth", bag)

    if p1.odom:
        # The odometry interpolated onto the frame clock, relative to frame 0.
        p1.odom.sort(key=lambda x: x[0])
        odom_ts = np.array([x[0] for x in p1.odom])
        odom_poses = np.array([x[1] for x in p1.odom])
        matched, n_interp = interpolate_poses(frame_ts, odom_ts, odom_poses)
        LOG.info(
            "%s: transforms.npy -- %d/%d interpolated, %d fell back to the nearest pose (gap or teleport)",
            bag,
            n_interp,
            len(frame_ts),
            len(frame_ts) - n_interp,
        )
        rel_transforms = anchor_to_first(matched)
        rel_transforms[0] = np.eye(4)
        np.save(scene_dir / "transforms.npy", rel_transforms)
    else:
        LOG.warning(
            "%s: no odometry messages on '%s'; transforms.npy not written",
            bag,
            args.odom,
        )

    # Process gravity alignment
    if p1.imu:
        # The orientation interpolated onto the frame clock; yaw is dropped.
        p1.imu.sort(key=lambda x: x[0])
        imu_ts = np.array([x[0] for x in p1.imu])
        imu_mats = np.array([x[1] for x in p1.imu])
        oriented, _ = interpolate_poses(frame_ts, imu_ts, imu_mats)
        gravity_aligns = np.empty((len(frame_ts), 3, 3))
        for k, M in enumerate(oriented):
            roll, pitch = roll_pitch_from_matrix(M)
            gravity_aligns[k] = euler_matrix(roll, pitch, 0.0)[:3, :3]
        np.save(scene_dir / "gravity_align.npy", gravity_aligns)
    else:
        LOG.warning(
            "%s: no orientation messages on '%s'; gravity_align.npy not written",
            bag,
            args.imu,
        )
    return frame_ts


def _drop_frames(scene_dir: Path, keep: np.ndarray) -> None:
    """Delete the frames ``keep`` rejects from images/ and clouds/ and renumber
    the rest, so frame k is again the k-th file of each.

    Ascending order makes the renames safe: a frame only ever moves down, onto
    an index whose own file has already moved or gone.
    """
    for sub in ("images", "clouds"):
        files = {int(p.stem): p for p in (scene_dir / sub).iterdir()}
        new = 0
        for old, kept in enumerate(keep):
            path = files.get(old)
            if not kept:
                if path is not None:
                    path.unlink()
                continue
            if path is not None and new != old:
                path.rename(path.with_name(f"{new:05d}{path.suffix}"))
            new += 1


def _digest(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _set_aside_stale(scene_dir: Path, old_clock: str | None, bag: str) -> None:
    """Move the per-frame products of an earlier run aside when the frame clock changed.

    They are indexed by frame, so once the frames are different moments they
    describe the wrong ones -- silently. Unchanged clock, same frames: they stay.
    """
    present = [d for d in DERIVED_DIRS if (scene_dir / d).is_dir()]
    if not present or old_clock == _digest(scene_dir / "data_ts.txt"):
        return
    suffix = time.strftime("stale-%Y%m%d-%H%M%S")
    for d in present:
        (scene_dir / d).rename(scene_dir / f"{d}.{suffix}")
    LOG.warning("%s: the frame clock changed; %s moved aside as *.%s -- rebuild them", bag, ", ".join(present), suffix)


@functools.cache
def _tool() -> dict:
    """Which code this is: the checkout's commit, and the versions that decode the bag."""
    here = Path(__file__).resolve().parent

    def git(*argv):
        try:
            run = subprocess.run(["git", "-C", str(here), *argv], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return run.stdout.strip() if run.returncode == 0 else None

    status = git("status", "--porcelain")
    try:
        rosbags = version("rosbags")
    except PackageNotFoundError:
        rosbags = None
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "rosbags": rosbags,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }


def _coverage(ts, t0: float, max_gap: float = 1.0, max_listed: int = 100) -> dict:
    """Count, extent and holes of one topic's stamps, in seconds since the bag began."""
    t = np.sort(np.asarray(ts, dtype=float))
    gaps = np.flatnonzero(np.diff(t) > max_gap)
    return {
        "messages": len(t),
        "first_s": round(float(t[0] - t0), 3),
        "last_s": round(float(t[-1] - t0), 3),
        "gaps_over_1s": len(gaps),
        "gaps": [[round(float(t[i] - t0), 3), round(float(t[i + 1] - t0), 3)] for i in gaps[:max_listed]],
    }


def _write_meta(scene_dir: Path, reader, bag_path: Path, p1: _Pass1, guard_verdict, frames: int, args) -> None:
    """extract_meta.json: the run, the bag, and what the run made of it.

    ``streams`` gives each topic the run read its coverage and its holes over a
    second, in seconds since the bag began -- the first place to look when a
    scene has a hole, to see which sensor went quiet.
    """
    t0 = reader.start_time * 1e-9
    topics: dict = {}
    for c in reader.connections:
        entry = topics.setdefault(c.topic, {"msgtype": c.msgtype, "messages": 0})
        entry["messages"] += int(getattr(c, "msgcount", 0) or 0)
    guard = None
    if guard_verdict is not None:
        guard = {k: v for k, v in vars(guard_verdict).items() if k != "spans"}
        guard["spans"] = [[round(s.start - t0, 3), round(s.end - t0, 3), s.reason] for s in guard_verdict.spans]
    meta = {
        "tool": _tool(),
        "argv": sys.argv,
        "args": vars(args),
        "bag": {
            "name": bag_path.name,
            "path": str(bag_path.resolve()),
            "start": t0,
            "duration_s": round(reader.duration * 1e-9, 3),
            "topics": topics,
        },
        "decisions": {
            "gps_topic": p1.gps_topic,
            "ground_truth": _auto(args.gt, args.odom),
            "gnss_guard": _auto(args.gnss_guard, args.odom),
        },
        "streams": {topic: _coverage(ts, t0) for topic, ts in sorted(p1.stamps.items()) if ts},
        "frames": {**p1.counts, "guard": guard, "written": frames},
        "unset_stamps": dict(p1.unset_stamps),
        "malformed_joint_messages": dict(p1.malformed),
        "frame_fallbacks": p1.frame_fallbacks,
    }
    (scene_dir / META_FILE).write_text(json.dumps(meta, indent=1) + "\n")
