"""One bag into one scene directory."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import cv2
import numpy as np

from dataset_tools import (  # noqa: F401  (rosbags_compat patches rosbags)
    gnss_guard,
    idle_trim,
    rosbags_compat,
    tf_pins,
    time_align,
)
from dataset_tools.extract import deskew, pose_check
from dataset_tools.extract.camera import IMAGE_MSGTYPES, get_first_frame, save_camera_extrinsic, save_intrinsics
from dataset_tools.extract.gps import GPS_MSGTYPE, gps_row, pick_gps_topic
from dataset_tools.extract.ground_truth import _auto, write_ground_truth
from dataset_tools.extract.pointcloud import cloud_to_xyzit, header_stamp_from_cdr, valid_returns_mask
from dataset_tools.extract.scene_io import (
    DERIVED_DIRS,
    DERIVED_FILES,
    GNSS_GUARD_FILE,
    GROUND_TRUTH_FILES,
    JOINT_NAMES_FILE,
    META_FILE,
    POSE_CHECK_FILE,
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
    """Take the sensor's capture-to-publish latency off the frame clock (``--time_align``).

    Returns the corrected clock and the profile to record, or ``(frame_ts, None)``.
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

    profile = time_align.estimate(scene_dir, frame_ts, *_columns(odom_data))
    LOG.info("%s: frame clock -- %s", bag, profile.summary)
    if profile.is_identity:
        return frame_ts, profile
    shifted, forced = time_align.apply_to_clock(frame_ts, profile)
    if forced:
        LOG.warning("%s: the measured offset ran backwards faster than the clock; forced monotonic", bag)
    return shifted, profile


def _given_shift(args) -> float:
    """A constant ``--time_align``, known before anything is written; 0 for ``auto``/``off``.

    ``auto`` is measured from the clouds this run writes, so it can only follow them:
    deskew and the idle trim then run on the stamped clock, exact for the Odin1
    odometry (same unit, same latency).
    """
    return 0.0 if args.time_align in ("auto", "off") else float(args.time_align)


def _columns(rows) -> tuple[np.ndarray, np.ndarray]:
    """``[(t, M), ...]`` as two arrays: the stamps and the matrices."""
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows])


@dataclass
class _Pass1:
    """Pass 1 output: stamps of the bulky streams, the small streams whole."""

    gps_topic: str | None
    cam_ts: list = field(default_factory=list)
    lidar_ts: list = field(default_factory=list)
    odom: list = field(default_factory=list)  # (ts, transform of odom child frame in odom fixed frame)
    odom_child_frame: str | None = None
    imu: list = field(default_factory=list)  # (ts, rotation matrix of imu frame in gravity-referenced frame)
    imu_frame: str | None = None
    #: (ts, wx, wy, wz) of the gyro that refines the rotation inside a sweep (--deskew_imu), and its frame.
    gyro: list = field(default_factory=list)
    gyro_frame: str | None = None
    gyro_rot: np.ndarray | None = None
    #: (ts, pose) of the stream the odometry is checked against (--check_odom), and its child frame.
    check_odom: list = field(default_factory=list)
    check_child: str | None = None
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
    # Pass 2 uses cv2.imdecode, which cannot read a raw sensor_msgs/Image.
    if cam_conn.msgtype not in IMAGE_MSGTYPES:
        raise ValueError(f"--cam {args.cam} is {cam_conn.msgtype}; expected one of {', '.join(IMAGE_MSGTYPES)}")

    # The 'auto' modes depend on the odometry topic; log the decision.
    if args.gt == "auto" or args.gnss_guard == "auto":
        LOG.info(
            "%s: from --odom %s: ground truth %s, GNSS guard %s",
            bag,
            args.odom,
            "on" if _auto(args.gt, args.odom) else "off (--gt on to force)",
            "on" if _auto(args.gnss_guard, args.odom) else "off (--gnss_guard on to force)",
        )

    # Extraction starts from scratch; the old clock decides whether derived products still hold.
    old_clock = _digest(scene_dir / "data_ts.txt")
    clear_scene_outputs(scene_dir, ground_truth=False)

    p1 = _collect(reader, conns, cam_conn, lidar_conn, args)
    # The ground truth goes only once pass 1 shows this run will rewrite it: with no
    # odometry in the bag a regenerated odometry.txt would be deleted and not replaced.
    if _auto(args.gt, args.odom) and p1.odom:
        for name in GROUND_TRUTH_FILES:
            (scene_dir / name).unlink(missing_ok=True)
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
        frame_ts + _given_shift(args),  # deskew onto the clock the scene will carry
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
    # Camera-vs-lidar stamp offset, added identically in both passes so pass 2's lookups hit exactly.
    shift_cam = args.cam_time_offset
    # Found by message type, so both rig generations work; None without a fix topic.
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
        conns.get(args.deskew_imu),
        conns.get(args.check_odom),
    ]
    # Deduplicate (e.g. when --odom and --imu are the same topic)
    relevant_conns = list({id(c): c for c in relevant_conns if c is not None}.values())

    # Image/cloud stamps are read from raw bytes, verified against a full decode of the first message.
    stamp_only = {id(cam_conn), id(lidar_conn)}
    verified = p1.verified

    def stamp(msg, recv: float, topic: str) -> float:
        # Header stamp, else receive time; unset header stamps are counted.
        t = header_stamp(msg, None)
        if t is None:
            if hasattr(msg, "header"):
                p1.unset_stamps[topic] += 1
            t = recv
        return t

    for conn, timestamp, raw in reader.messages(relevant_conns):
        recv = timestamp * 1e-9
        msg = None  # stays None on the stamp-only path, so the hasattr checks below see no stale message
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

        # Taken before the elif chain: these may double as --imu or --odom.
        if conn.topic == args.deskew_imu and hasattr(msg, "angular_velocity"):
            w = msg.angular_velocity
            p1.gyro.append((t, w.x, w.y, w.z))
            p1.gyro_frame = msg.header.frame_id
        if conn.topic == args.check_odom and conn.topic != args.odom and hasattr(msg, "pose"):
            p1.check_odom.append((t, pose_to_matrix(msg.pose.pose)))
            p1.check_child = msg.child_frame_id

        if conn.topic == args.cam:
            p1.cam_ts.append(t + shift_cam)
        elif conn.topic == args.lidar:
            p1.lidar_ts.append(t)
        elif conn.topic in (args.odom, args.imu):
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
    # Bags may hold out-of-order messages.
    for rows in (p1.odom, p1.imu, p1.check_odom):
        rows.sort(key=lambda x: x[0])
    return p1


def _joint_row(p1: _Pass1, topic: str, t: float, msg):
    """One joint row in ``p1.joint_names`` order (the first message's).

    Other joint sets or wrong-length columns are skipped and counted; an empty
    column (some drivers never fill effort) becomes NaN.
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
    valid_cam_ts, valid_pcd_ts = sync_timestamps(p1.cam_ts, p1.lidar_ts)
    p1.counts.update(camera_messages=len(p1.cam_ts), lidar_messages=len(p1.lidar_ts), pairs=len(valid_cam_ts))
    # Mutual nearest neighbours can still be far apart across a camera stall.
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

    # One clock per frame, as stamped; publish latency is taken off later (--time_align).
    frame_ts = (valid_cam_ts + valid_pcd_ts) / 2

    # Drop fabricated or uncovered poses before any frame is written, so indices stay
    # contiguous. Runs on the receiver's own frame: a lever arm would turn rotation into jumps.
    guard_verdict = None
    if _auto(args.gnss_guard, args.odom) and p1.odom:
        if args.gnss_guard == "on" or "fixposition" in args.odom:
            # Full guard. Gaps in fix stamps are no-solution periods; a fix topic
            # with no usable fix is [] (evidence), not None.
            fix_ts = [float(r[0]) for r in p1.gps_rows] if p1.gps_topic else None
            p1.odom, keep, guard_verdict = gnss_guard.guard(p1.odom, frame_ts, fix_ts)
        else:
            # Odin1 fabricates nothing (it jumps, handled by the segment gate): coverage only.
            p1.odom, keep, guard_verdict = gnss_guard.coverage_guard(p1.odom, frame_ts)
        if not guard_verdict.clean:
            LOG.warning("%s: %s", bag, guard_verdict.summary(float(frame_ts[0])))
            valid_cam_ts, valid_pcd_ts, frame_ts = valid_cam_ts[keep], valid_pcd_ts[keep], frame_ts[keep]
            if len(valid_cam_ts) == 0:
                LOG.warning("%s: no frame of this bag has a trustworthy pose; scene left empty", bag)
                return "no frame of this bag has a pose the dataset can stand behind"
        else:
            LOG.info("%s: pose guard -- %s", bag, guard_verdict.summary(float(frame_ts[0])))

    # --trim_idle, before any frame is written for the same reason. On the clock the scene
    # will carry where that is known (a constant --time_align), else as stamped.
    if getattr(args, "trim_idle", None) and p1.odom:
        max_s, keep_s = args.trim_idle
        clock = frame_ts + _given_shift(args)
        keep, idle = idle_trim.trim(clock, sorted(p1.odom, key=lambda x: x[0]), max_s, keep_s)
        p1.counts["idle_trim"] = idle.report(float(clock[0]))
        LOG.info("%s: idle trim -- %s", bag, idle_trim.describe(idle, float(clock[0])))
        if not keep.all():
            valid_cam_ts, valid_pcd_ts, frame_ts = valid_cam_ts[keep], valid_pcd_ts[keep], frame_ts[keep]
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

    # GPS is optional, like /cmd_vel.
    if p1.gps_rows:
        arr = np.vstack(p1.gps_rows)
        np.save(scene_dir / "gps.npy", arr[np.argsort(arr[:, 0])])
        LOG.info("%s: %d GNSS fixes from '%s'", bag, len(p1.gps_rows), p1.gps_topic)
    elif p1.gps_topic and p1.gps_topic not in p1.stamps:
        LOG.info("%s: no messages on '%s' (topic absent from the bag); gps.npy not written", bag, p1.gps_topic)
    elif p1.gps_topic:
        LOG.info(
            "%s: no usable fix among %d on '%s'; gps.npy not written", bag, len(p1.stamps[p1.gps_topic]), p1.gps_topic
        )
    else:
        LOG.info("%s: no %s topic in the bag; gps.npy not written", bag, GPS_MSGTYPE)


def _resolve_frames(reader, lidar_conn, p1: _Pass1, scene_dir: Path, args, bag: str):
    """Build the (pinned) TF tree and re-express ``p1``'s streams at the base frame in place.

    Returns ``(tf_graph, T_lidar_to_base)``."""
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

    # Odom poses track the child frame (e.g. FP_POI); re-express at the base frame.
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

    # So gravity alignment is the base's tilt, not the sensor mounting's.
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

    # Without a gyro->base path, deskew falls back to the pose track's slerp.
    if p1.gyro:
        T_gyro = get_transform(tf_graph, p1.gyro_frame, args.base_frame) if p1.gyro_frame else None
        if T_gyro is None:
            LOG.warning(
                "%s: no TF path from gyro frame '%s' to '%s'; sweeps are deskewed from the pose track alone",
                bag,
                p1.gyro_frame,
                args.base_frame,
            )
            p1.gyro = []
        else:
            p1.gyro_rot = T_gyro[:3, :3]

    # Compare at the base frame, or the lever arm reads as disagreement.
    child = p1.check_child
    T_check = get_transform(tf_graph, child, args.base_frame) if child else None
    if T_check is not None and not np.allclose(T_check, np.eye(4)):
        T_base_in_check = np.linalg.inv(T_check)
        p1.check_odom = [(t, T @ T_base_in_check) for t, T in p1.check_odom]
    elif child and T_check is None:
        LOG.warning(
            "%s: no TF path from '%s' to '%s'; the pose check compares at '%s'", bag, child, args.base_frame, child
        )
    return tf_graph, T_lidar_to_base


def _frame_fallback(p1: _Pass1, args, bag: str, what: str) -> None:
    """A frame the run cannot put where it belongs: an error, unless asked to go on."""
    if not args.allow_frame_fallback:
        raise ValueError(f"{what} (--allow_frame_fallback to extract it like that anyway)")
    LOG.warning("%s: %s", bag, what)
    p1.frame_fallbacks.append(what)


def _write_image(np_arr: np.ndarray, image_path: Path, png: bool, img_idx: int, t: float) -> tuple[tuple, float]:
    cv_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if cv_img is None:
        # A missing frame would break index alignment: fail, naming the frame.
        raise ValueError(f"image {img_idx} (stamp {t:.6f}, {len(np_arr)} bytes) could not be decoded")
    if png:
        if not cv2.imwrite(str(image_path), cv_img):
            raise ValueError(f"image {img_idx} could not be written to {image_path}")
    else:
        # The camera's own bytes (decoded above only to validate).
        image_path.write_bytes(np_arr.tobytes())
    return (), 0.0


def _write_cloud(msg, t: float, frame_t: float, T_lidar_to_base, track, gyro, path: Path) -> tuple[tuple, float]:
    """Write one base-frame cloud; returns the deskew counters to add and the sweep span."""
    counted, span = (), 0.0
    # Before T_lidar_to_base, never after -- see valid_returns_mask.
    xyzi, offsets = cloud_to_xyzit(msg, t)
    keep = valid_returns_mask(xyzi, offsets)
    xyzi = xyzi[keep]
    pcd = np.hstack((xyzi[:, :3], np.ones((len(xyzi), 1), dtype=np.float32)))
    pcd = (pcd @ T_lidar_to_base.T.astype(np.float32))[:, :3]
    if track is not None:
        if offsets is None:
            counted = ("no_time_field",)
        else:
            pcd, status, method = deskew.deskew(pcd, offsets[keep], t, frame_t, track, gyro)
            counted = (status, method) if method else (status,)
            if keep.any():
                span = float(np.ptp(offsets[keep]))
    np.save(path, np.hstack((pcd, xyzi[:, 3:4])))
    return counted, span


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
    """Pass 2: write each frame's image and base-frame cloud, deskewed onto ``frame_ts[k]`` where possible.

    The bag is read and every per-scene decision made here, in order; decoding
    and writing a frame runs on a thread pool (cv2, the cloudini shim and most
    of numpy drop the GIL). Results are taken back in submission order, so the
    first frame to fail is the one that raises.
    """
    track = time_align.PoseTrack.build(*_columns(p1.odom)) if args.deskew != "off" and len(p1.odom) >= 2 else None
    gyro = deskew.GyroTrack.build(p1.gyro, p1.gyro_rot) if track is not None and p1.gyro else None
    sweeps = Counter()
    span = 0.0
    shift_cam = args.cam_time_offset
    verified = p1.verified
    # Stamp -> frame index; popped once written, so duplicates are ignored.
    pending = {
        args.cam: {ts: i for i, ts in enumerate(valid_cam_ts)},
        args.lidar: {ts: i for i, ts in enumerate(valid_pcd_ts)},
    }

    intrinsics_saved = False
    image_how = args.image_format
    image_ext = None  # one format per scene, and per root: decided by the first frame

    workers = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    inflight = deque()

    def drain(keep: int) -> None:
        nonlocal span
        while len(inflight) > keep:
            counted, s = inflight.popleft().result()
            sweeps.update(counted)
            span = max(span, s)

    with ThreadPoolExecutor(workers) as pool:
        try:
            for conn, timestamp, raw in reader.messages([cam_conn, lidar_conn]):
                # Same stamp arithmetic as pass 1, so the lookups hit exactly.
                is_cam = conn.topic == args.cam
                shift = shift_cam if is_cam else 0.0
                msg = None
                t = header_stamp_from_cdr(raw) if id(conn) in verified else None
                if t is None:
                    msg = reader.deserialize(raw, conn.msgtype)
                    t = header_stamp(msg, timestamp * 1e-9)
                idx = pending[conn.topic].pop(t + shift, None)
                if idx is None:
                    continue  # not a synchronised frame: never decoded at all
                if msg is None:
                    msg = reader.deserialize(raw, conn.msgtype)
                t += shift

                if is_cam:
                    np_arr = np.frombuffer(msg.data, np.uint8)
                    ext = image_extension(np_arr, image_how, idx)
                    if image_ext is None:
                        held = root_image_extension(scene_dir)
                        if held is not None and held != ext:
                            raise ValueError(
                                f"{scene_dir.parent} holds {held} images and this bag would add {ext}: a root has "
                                f"one image format. Extract into a new root"
                                + (", or pass --image_format png to extend this one" if held == ".png" else "")
                            )
                        image_ext = ext
                    elif ext != image_ext:
                        raise ValueError(f"image {idx} is {ext} where the scene so far is {image_ext}")
                    image_path = scene_dir / "images" / f"{idx:05d}{ext}"
                    job = pool.submit(_write_image, np_arr, image_path, image_how == "png", idx, t)

                    if not intrinsics_saved:
                        save_intrinsics(reader, conns, args.cam, scene_dir)
                        save_camera_extrinsic(tf_graph, msg.header.frame_id, args.base_frame, scene_dir)
                        intrinsics_saved = True
                else:
                    path = scene_dir / "clouds" / f"{idx:05d}.npy"
                    job = pool.submit(_write_cloud, msg, t, float(frame_ts[idx]), T_lidar_to_base, track, gyro, path)
                inflight.append(job)
                drain(2 * workers)  # bounded: a frame in flight holds its decoded message
        finally:
            # An earlier frame's failure outranks whatever stopped the loop.
            drain(0)

    _report_deskew(p1, track, sweeps, span, len(valid_pcd_ts), args, bag)


def _report_deskew(p1: _Pass1, track, sweeps: Counter, span: float, clouds: int, args, bag: str) -> None:
    """Log and record in extract_meta.json how the sweeps were deskewed, or why not."""
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
            "%s: %d/%d sweeps deskewed onto the frame clock (span up to %.3f s; rotation from the gyro in %d, "
            "slerp in %d)%s",
            bag,
            sweeps["deskewed"],
            clouds,
            span,
            sweeps["gyro"],
            sweeps["slerp"],
            "".join(
                f"; {v} {k.replace('_', ' ')}"
                for k, v in sorted(sweeps.items())
                if k not in ("deskewed", "gyro", "slerp") and v
            ),
        )
    elif sweeps["no_time_field"]:
        LOG.info("%s: the clouds carry no per-point time; written as recorded (nothing to deskew)", bag)
    else:
        LOG.warning("%s: no sweep could be deskewed: %s", bag, dict(sweeps))


def _write_derived(scene_dir: Path, frame_ts, valid_pcd_ts, p1: _Pass1, guard_verdict, args, bag: str):
    """Correct the frame clock, then write everything sampled onto it. Returns the corrected clock."""
    # pcd_ts takes the same correction (same unit's latency), keeping it comparable to data_ts.
    corrected, align = align_frame_clock(scene_dir, frame_ts, p1.odom, args, bag)
    applied = corrected - frame_ts
    frame_ts = corrected
    pcd_ts = valid_pcd_ts + applied

    # The guard padded by FRAME_PAD before correction; now drop frames without a bracketing pose.
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

    if _auto(args.gt, args.odom) and p1.odom:
        if write_ground_truth(p1.odom, scene_dir, spans=guard_verdict.spans if guard_verdict else None):
            LOG.info("%s: ground truth written (odometry.txt + segments)", bag)
        else:
            LOG.warning("%s: too few odometry samples for ground truth", bag)

    if p1.odom:
        matched, n_interp = interpolate_poses(frame_ts, *_columns(p1.odom))
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

    # --check_odom: compare against the rig's other pose stream.
    if p1.odom and p1.check_odom:
        result = pose_check.compare(*_columns(p1.odom), *_columns(p1.check_odom), gnss_guard.MAX_SPEED)
        pose_check.write_report(scene_dir / POSE_CHECK_FILE, result, args.odom, args.check_odom)
        p1.counts["pose_check"] = {"reference": args.check_odom, **result}
        five = (result.get("displacement") or {}).get("5s") or {}
        LOG.info(
            "%s: pose check against %s -- clock lag %s s, 5 s displacement median %s cm, %s jumps in the odometry",
            bag,
            args.check_odom,
            result.get("clock_lag_s"),
            five.get("median_cm"),
            (result.get("primary_jumps") or {}).get("steps"),
        )
        for w in result.get("warnings", []):
            LOG.warning("%s: pose check -- %s", bag, w)

    if p1.imu:
        # Orientation on the frame clock, yaw dropped.
        oriented, _ = interpolate_poses(frame_ts, *_columns(p1.imu))
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
    """Delete rejected frames from images/ and clouds/ and renumber the rest.

    Ascending order keeps renames safe: files only move down onto freed indices.
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
    """Move an earlier run's per-frame products aside if the frame clock changed."""
    present = [d for d in DERIVED_DIRS if (scene_dir / d).is_dir()]
    if not present or old_clock == _digest(scene_dir / "data_ts.txt"):
        return
    present += [f for f in DERIVED_FILES if (scene_dir / f).is_file()]
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

    # --no-optional-locks: N workers at once must not race for index.lock.
    status = git("--no-optional-locks", "status", "--porcelain")
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


def _coverage(ts, t0: float) -> dict:
    """Count, extent and holes over 1 s (the first 100) of one topic's stamps, in seconds since the bag began."""
    t = np.sort(np.asarray(ts, dtype=float))
    gaps = np.flatnonzero(np.diff(t) > 1.0)
    return {
        "messages": len(t),
        "first_s": round(float(t[0] - t0), 3),
        "last_s": round(float(t[-1] - t0), 3),
        "gaps_over_1s": len(gaps),
        "gaps": [[round(float(t[i] - t0), 3), round(float(t[i + 1] - t0), 3)] for i in gaps[:100]],
    }


def _write_meta(scene_dir: Path, reader, bag_path: Path, p1: _Pass1, guard_verdict, frames: int, args) -> None:
    """extract_meta.json: the run, the bag, and per-topic coverage/gaps (seconds since bag start)."""
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
