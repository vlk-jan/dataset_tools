#!/usr/bin/env python3
"""Survey one v2-rig bag without extracting it.

Writes summary.json (topics, tf_static, Odin1 behaviour, motion, GNSS, clouds),
a 1 Hz track.npy and a few thumbnails: ~1 MB per bag to choose scenes from.

    scripts/survey_v2_bag.py <bag_dir> <out_dir> [--thumbs 12] [--clouds 60]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from rosbags.highlevel import AnyReader

from dataset_tools import rosbags_compat  # noqa: F401  (patches rosbags)
from dataset_tools.extract.pointcloud import cloud_to_xyzit, header_stamp_from_cdr, valid_returns_mask
from dataset_tools.extract.tf_graph import build_tf_graph, get_transform
from dataset_tools.gnss_guard import MAX_SPEED
from dataset_tools.tf_pins import rpy_deg
from dataset_tools.transforms import quaternion_matrix

CAM = "/odin1/image/compressed"
CLOUD = "/odin1/cloud_filtered/cloudini"
ODOM = "/fixposition/odometry_enu"
LLH = "/fixposition/odometry_llh"
JS = "/joint_states"
CMD = "/cmd_vel"
TF_STATIC = "/tf_static"
DIAG = "/diagnostics"
CAMINFO = "/odin1/camera_info"
BASE = "base_link"
RANGE_BINS = np.arange(0.0, 6.5, 0.5)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--thumbs", type=int, default=12)
    ap.add_argument("--clouds", type=int, default=60, help="clouds to sample across the bag")
    ap.add_argument("--thumb_width", type=int, default=480)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    with AnyReader([args.bag]) as reader:
        conns = {c.topic: c for c in reader.connections}
        counts = {c.topic: c.msgcount for c in reader.connections}
        want = [t for t in (CAM, CLOUD, ODOM, LLH, JS, CMD, TF_STATIC, DIAG, CAMINFO) if t in conns]
        n_img, n_cloud = counts.get(CAM, 0), counts.get(CLOUD, 0)
        img_every = max(1, n_img // max(1, args.thumbs))
        cloud_every = max(1, n_cloud // max(1, args.clouds))

        lat = defaultdict(list)  # topic -> recv - header
        stamps = defaultdict(list)
        thumbs, clouds, cloud_t = [], [], []
        odom, llh, js_nonzero, js_rows = [], [], 0, 0
        tf_msgs, diag = [], Counter()
        diag_values = {}
        img_shape = None
        cloud_frame = None
        caminfo = None
        k_img = k_cloud = 0
        for conn, recv_ns, raw in reader.messages([conns[t] for t in want]):
            recv = recv_ns * 1e-9
            t = conn.topic
            if t in (CAM, CLOUD, ODOM, JS):
                s = header_stamp_from_cdr(raw)
                if s is not None and s > 0:
                    lat[t].append(recv - s)
                    stamps[t].append(s)
            if t == CAM:
                if k_img % img_every == 0 and len(thumbs) < args.thumbs:
                    msg = reader.deserialize(raw, conn.msgtype)
                    im = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                    if im is not None:
                        img_shape = im.shape
                        h = int(im.shape[0] * args.thumb_width / im.shape[1])
                        thumbs.append((recv, cv2.resize(im, (args.thumb_width, h), interpolation=cv2.INTER_AREA)))
                k_img += 1
            elif t == CLOUD:
                if k_cloud % cloud_every == 0 and len(clouds) < args.clouds:
                    msg = reader.deserialize(raw, conn.msgtype)
                    cloud_frame = msg.header.frame_id
                    stamp = stamps[t][-1] if stamps[t] else recv
                    xyzi, _ = cloud_to_xyzit(msg, stamp)
                    clouds.append(xyzi[valid_returns_mask(xyzi), :3])
                    cloud_t.append(stamp)
                k_cloud += 1
            elif t == ODOM:
                msg = reader.deserialize(raw, conn.msgtype)
                p, q = msg.pose.pose.position, msg.pose.pose.orientation
                odom.append((stamps[t][-1] if stamps[t] else recv, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
            elif t == LLH:
                msg = reader.deserialize(raw, conn.msgtype)
                cov = msg.position_covariance
                llh.append((recv, msg.latitude, msg.longitude, msg.altitude, msg.status.status, cov[0], cov[4]))
            elif t == JS:
                if js_rows % 10 == 0:
                    msg = reader.deserialize(raw, conn.msgtype)
                    v = np.asarray(msg.velocity, dtype=float)
                    pos = np.asarray(msg.position, dtype=float)
                    if (len(v) and np.any(v != 0)) or (len(pos) and np.any(pos != 0)):
                        js_nonzero += 1
                js_rows += 1
            elif t == TF_STATIC:
                if len(tf_msgs) < 200:
                    tf_msgs.extend(reader.deserialize(raw, conn.msgtype).transforms)
            elif t == DIAG:
                if sum(diag.values()) < 2000:
                    msg = reader.deserialize(raw, conn.msgtype)
                    for st in msg.status:
                        diag[(st.name, st.hardware_id)] += 1
                        if st.name not in diag_values:
                            diag_values[st.name] = {kv.key: kv.value for kv in st.values}
                            diag_values[st.name]["_message"] = st.message
            elif t == CAMINFO and caminfo is None:
                msg = reader.deserialize(raw, conn.msgtype)
                caminfo = dict(
                    frame=msg.header.frame_id,
                    w=int(msg.width),
                    h=int(msg.height),
                    model=msg.distortion_model,
                    K=[float(x) for x in msg.k],
                    D=[float(x) for x in msg.d],
                )

    out: dict = dict(bag=str(args.bag), counts=counts, n_topics=len(counts))
    # tf_static: does the Odin mount exist, and where are the wheels?
    G = build_tf_graph(tf_msgs)
    edges = sorted((a, b) for a in G for b in G[a])
    X = get_transform(G, cloud_frame, BASE) if cloud_frame else None
    out["tf_static"] = dict(
        n_edges=len(edges) // 2,
        frames=sorted(G),
        cloud_frame=cloud_frame,
        odin_mount=None if X is None else [round(float(v), 4) for v in X[:3, 3]],
        odin_mount_rpy_deg=None if X is None else [round(v, 3) for v in rpy_deg(X)],
        cam_in_base=(
            None
            if caminfo is None or get_transform(G, caminfo["frame"], BASE) is None
            else [round(float(v), 4) for v in get_transform(G, caminfo["frame"], BASE)[:3, 3]]
        ),
    )
    out["camera"] = dict(
        shape=list(img_shape) if img_shape else None,
        info=caminfo,
        per_lidar=round(n_img / n_cloud, 3) if n_cloud else None,
    )
    out["latency_s"] = {
        t: dict(p50=float(np.median(v)), p05=float(np.percentile(v, 5)), p95=float(np.percentile(v, 95)))
        for t, v in lat.items()
        if len(v)
    }
    # Frame clock: holes where the Odin stopped publishing.
    for t in (CAM, CLOUD):
        s = np.asarray(stamps.get(t, []))
        if len(s) > 2:
            d = np.diff(s)
            out.setdefault("holes", {})[t] = dict(
                period_p50=float(np.median(d)), n_over_1s=int((d > 1).sum()), longest_s=float(d.max()), n_msgs=len(s)
            )
    # Diagnostics: anything that names a version or firmware.
    version_words = ("version", "firmware", "fw", "sdk", "driver", "serial", "build")
    version_like = {
        n: {k: v for k, v in kv.items() if any(w in f"{k} {v}".lower() for w in version_words)}
        for n, kv in diag_values.items()
    }
    out["diagnostics"] = dict(
        status_names=sorted({n for n, _ in diag}),
        hardware_ids=sorted({h for _, h in diag if h}),
        version_like={n: kv for n, kv in version_like.items() if kv},
    )
    # Odometry: path, speed, tilt.
    if odom:
        o = np.asarray(odom)
        o = o[np.argsort(o[:, 0])]
        d = np.linalg.norm(np.diff(o[:, 1:4], axis=0), axis=1)
        dt = np.diff(o[:, 0])
        # Same teleport gate as the extractor.
        ok = (dt > 0) & (dt < 1.0) & (d / np.maximum(dt, 1e-6) < MAX_SPEED)
        v = d[ok] / dt[ok]
        tilt = np.degrees(np.arccos(np.clip([quaternion_matrix(q)[2, 2] for q in o[:, 4:8]], -1, 1)))
        lvl = tilt < 60
        out["motion"] = dict(
            poses=len(o),
            span_s=float(o[-1, 0] - o[0, 0]),
            path_m=float(d[ok].sum()),
            teleports=int(((d / np.maximum(dt, 1e-6)) >= MAX_SPEED).sum()),
            v_p50=float(np.median(v)) if len(v) else None,
            v_p90=float(np.percentile(v, 90)) if len(v) else None,
            stationary_frac=float((v < 0.05).mean()) if len(v) else None,
            tilt_p50=float(np.median(tilt)),
            tilt_p95=float(np.percentile(tilt, 95)),
            tilt_max=float(tilt.max()),
            tilt_std_level=float(tilt[lvl].std()) if lvl.any() else None,
            inverted_frac=float((tilt > 90).mean()),
        )
    if llh:
        g = np.asarray(llh)
        g = g[(g[:, 1] != 0) | (g[:, 2] != 0)]
        if len(g):
            sig = np.sqrt(np.maximum(g[:, 5] + g[:, 6], 0))
            out["gnss"] = dict(
                fixes=len(g),
                rtk_frac=float((g[:, 4] == 2).mean()),
                nofix_frac=float((g[:, 4] < 0).mean()),
                h_sigma_p50=float(np.median(sig)),
            )
            step = max(1, int(len(g) / max(1.0, g[-1, 0] - g[0, 0])))
            np.save(args.out / "track.npy", g[::step, :5])
    out["proprio"] = dict(
        joint_states=counts.get(JS, 0),
        joint_states_nonzero_frac=(js_nonzero / max(1, js_rows // 10 + 1)),
        cmd_vel=counts.get(CMD, 0),
        joint_setpoints=counts.get("/joint_setpoints", 0),
    )
    # Clouds: self-returns, ground profile by range, extent.
    if clouds:
        near, npts, r95, prof, ground = [], [], [], [], []
        for c in clouds:
            if X is not None:
                c = (X[:3, :3] @ c.T).T + X[:3, 3]
            r = np.hypot(c[:, 0], c[:, 1])
            near.append(int((r < 0.5).sum()))
            npts.append(len(c))
            r95.append(float(np.percentile(r, 95)))
            fwd = np.abs(np.arctan2(c[:, 1], c[:, 0])) < np.radians(20)
            row = []
            for lo, hi in zip(RANGE_BINS[:-1], RANGE_BINS[1:], strict=True):
                m = fwd & (r >= lo) & (r < hi)
                row.append(float(np.percentile(c[m, 2], 20)) if m.sum() > 30 else np.nan)
            prof.append(row)
            m = fwd & (r > 1.0) & (r < 2.5)
            if m.sum() > 50:
                ground.append(float(np.median(c[m, 2])))
        prof = np.asarray(prof)
        out["clouds"] = dict(
            sampled=len(clouds),
            in_base_link=X is not None,
            npts_p50=float(np.median(npts)),
            near_r05_p50=float(np.median(near)),
            r95_p50=float(np.median(r95)),
            ground_z_p50=float(np.median(ground)) if ground else None,
            ground_profile_low20_by_range={
                f"{lo:.1f}-{hi:.1f}": (None if np.isnan(v) else round(float(v), 3))
                for lo, hi, v in zip(RANGE_BINS[:-1], RANGE_BINS[1:], np.nanmedian(prof, axis=0), strict=True)
            },
        )
    for i, (_, im) in enumerate(thumbs):
        cv2.imwrite(str(args.out / f"thumb_{i:02d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    out["thumbs"] = [round(t, 3) for t, _ in thumbs]
    (args.out / "summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(
        json.dumps(
            {k: v for k, v in out.items() if k not in ("counts", "tf_static", "diagnostics")}, indent=1, default=str
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
