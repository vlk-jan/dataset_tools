#!/usr/bin/env python3
"""Build HELHEST_v1_3_fill: a symlink tree of HELHEST_v1_3 whose odometry.txt /
odometry_segments.txt / manifest.json are real files carrying three changes, each
tagged in the segment table so nothing is silent:

  coverage_guard   frozen runs outside encoder coverage (the refiner's ZUPT held the
                   first/last joint_states sample) replaced by v1.2's own relative motion
  v11_fill         v1.1 odometry in the remaining gaps, through the v1.2 quality gate
  regen_fill       FAST-LIO offset re-runs of silent-stall holes (scratch pilots)

Every appended run passes the SAME acceptance: filter_segments.bad_poses (2 s windows vs
/odom, > 1 m), gt_gaps seams (gap / wheel_gap), gt_map.window_offset on measured cells
(> 0.3 m over 4 s cut with 0.5 s margin), >= 20 poses, re-anchored at identity.

    build_fill_root.py <src_root> <dst_root> <v11_root> [--regen scene:pilot_dir:t0:t1 ...]
"""
import argparse
import datetime
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the package, when it is not installed
from filter_segments import bad_poses  # noqa: E402
from merge_v11_fill import fill_segments, load_segment_rows  # noqa: E402

from dataset_tools.gt_gaps import ReferencePath, gap_seams, split_runs  # noqa: E402
from dataset_tools.gt_io import (  # noqa: E402
    SegmentRow,
    anchor_to_first,
    read_segments,
    read_trajectory,
    write_segments,
    write_trajectory,
)
from dataset_tools.gt_map import Mismatch, mismatch_intervals, window_offset  # noqa: E402

WINDOW, MAX_ERR, MIN_POSES, MAX_GAP = 2.0, 1.0, 20, 1.0
MAP_HORIZON, MAP_MAX_OFF, MAP_MARGIN = 4.0, 0.3, 0.5
COVER_MARGIN = 0.15
FROZEN_MIN_S, FROZEN_MIN_MOVE = 1.0, 0.3


# ---------------------------------------------------------------------------- tree
def link_tree(src: Path, dst: Path):
    dst.mkdir(exist_ok=True)
    for e in sorted(src.iterdir()):
        d = dst / e.name
        if d.exists() or d.is_symlink():
            continue
        if e.is_dir() and e.name.isdigit():
            d.mkdir()
            for f in sorted(e.iterdir()):
                t = d / f.name
                if f.name in ("odometry.txt", "odometry_segments.txt"):
                    shutil.copy(f, t)
                elif f.name == "regen":
                    t.mkdir()
                    for g in sorted(f.iterdir()):
                        (t / g.name).symlink_to(g.absolute())
                else:
                    t.symlink_to(f.absolute())
        elif e.name in ("manifest.json", "README.md"):
            shutil.copy(e, d)
        else:
            d.symlink_to(e.absolute())


# ---------------------------------------------------------------------------- gate
class SceneCtx:
    def __init__(self, root: Path, scene: str):
        self.d = root / scene
        self.scene = scene
        self.ts = np.loadtxt(self.d / "data_ts.txt")
        self.pts = np.loadtxt(self.d / "pcd_ts.txt")
        self.ga = np.load(self.d / "gravity_align.npy")
        rt, rT = read_trajectory(self.d / "regen" / "odom_ref.txt")
        self.rt, self.rT = rt, rT
        self.ref = ReferencePath.from_trajectory(rt, rT)

    def map_hits(self, tt, TT):
        hits = []
        for f in np.where((self.pts >= tt[0]) & (self.pts + MAP_HORIZON <= tt[-1] + 1e-3))[0]:
            if f >= len(self.ga):
                break
            mp, mk = self.d / "height_maps" / f"{f:05d}.npz", self.d / "height_masks" / f"{f:05d}.npz"
            if not (mp.exists() and mk.exists()):
                continue
            t0 = float(self.pts[f])
            m = (tt >= t0) & (tt <= t0 + MAP_HORIZON)
            if m.sum() < 5:
                continue
            res = window_offset(TT[m], self.ga[f], np.load(mp)["height_map"], np.load(mk)["validity"])
            if res is not None and abs(res[0]) > MAP_MAX_OFF:
                hits.append(Mismatch(int(f), t0, res[0], res[1]))
        return hits

    def gate(self, tt, TT, stats):
        """Full acceptance for one candidate run -> list of (ts, anchored poses)."""
        out = []
        bad = bad_poses(tt, TT, self.rt, self.rT, WINDOW, MAX_ERR)
        stats["odom_bad"] += int(bad.sum())
        seam, tags = gap_seams(tt, TT, self.ref, MAX_GAP, MAX_ERR)
        stats["seams"] += int(seam.sum())
        for r0, r1, _cut in split_runs(len(tt), seam, tags, bad):
            if r1 - r0 < MIN_POSES:
                continue
            rt_, rT_ = tt[r0:r1], TT[r0:r1]
            hits = self.map_hits(rt_, rT_)
            keep = np.ones(len(rt_), bool)
            if hits:
                stats["map_hits"] += len(hits)
                for a, b in mismatch_intervals(hits, MAP_HORIZON, MAP_MARGIN):
                    keep &= ~((rt_ >= a) & (rt_ <= b))
            edges = np.diff(np.concatenate([[0], keep.astype(int), [0]]))
            for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0], strict=True):
                if b - a >= MIN_POSES:
                    out.append((rt_[a:b], anchor_to_first(rT_[a:b])))
        return out


def rewrite(d: Path, pieces):
    pieces.sort(key=lambda p: p[0][0])
    write_trajectory(d / "odometry.txt", np.concatenate([p[0] for p in pieces]), np.concatenate([p[1] for p in pieces]))
    write_segments(
        d / "odometry_segments.txt", [SegmentRow(k, tt[0], tt[-1], len(tt), r) for k, (tt, _T, r) in enumerate(pieces)]
    )


def load_pieces(d: Path):
    t, T = read_trajectory(d / "odometry.txt")
    out = []
    for s in read_segments(d / "odometry_segments.txt"):
        m = (t >= s.start_ts) & (t <= s.end_ts)
        out.append((t[m], T[m], s.reason))
    return out


def gps_witness(ctx, tt, TT):
    """Rigid 2-D fit of a run against gps_odom (comparison only): median residual, n poses used."""
    gp = ctx.d / "gps_odom.npy"
    if not gp.exists() or ctx.scene == "0007":  # 0007's GPS is unusable (manifest gps block)
        return None
    g = np.load(gp)
    ok = (g[:, 8] < 0.3) & (g[:, 9] < 0.3)
    g = g[ok]
    if len(g) < 10 or tt[0] < g[0, 0] or tt[-1] > g[-1, 0]:
        return None
    gx = np.interp(tt, g[:, 0], g[:, 1])
    gy = np.interp(tt, g[:, 0], g[:, 2])
    P = TT[:, :2, 3]
    Q = np.stack([gx, gy], 1)
    if np.linalg.norm(P[-1] - P[0]) < 3.0:
        return None
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    U, _S, Vt = np.linalg.svd(Pc.T @ Qc)
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    res = np.linalg.norm((R @ Pc.T).T - Qc, axis=1)
    return dict(
        n=int(len(tt)),
        path_m=float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()),
        rigid_res_median_m=float(np.median(res)),
        rigid_res_p90_m=float(np.percentile(res, 90)),
    )


# ---------------------------------------------------------------------------- 1. coverage guard
def coverage_guard(dst: Path, v12: Path, scene: str):
    d = dst / scene
    pieces = load_pieces(d)
    js = np.load(d / "joint_states.npy")
    jt0, jt1 = js[0, 0], js[-1, 0]
    t2, T2 = read_trajectory(v12 / scene / "odometry.txt")
    fixed = []
    new_pieces = []
    for tt, TT, reason in pieces:
        n_before = len(fixed)
        p = TT[:, :3, 3]
        same = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) < 1e-9, [False]])
        TT = TT.copy()
        i = 0
        while i < len(same):
            if not same[i]:
                i += 1
                continue
            j = i
            while j < len(same) and same[j]:
                j += 1
            # frozen run i..j (j inclusive as the last identical pose)
            if tt[j] - tt[i] >= FROZEN_MIN_S and (tt[i] < jt0 or tt[j] > jt1):
                k2 = np.searchsorted(t2, tt[i : j + 1])
                k2 = np.clip(k2, 0, len(t2) - 1)
                if np.all(np.abs(t2[k2] - tt[i : j + 1]) < 5e-3):
                    move = np.linalg.norm(T2[k2[-1]][:3, 3] - T2[k2[0]][:3, 3])
                    if move > FROZEN_MIN_MOVE:
                        base = TT[i] @ np.linalg.inv(T2[k2[0]])
                        new = np.einsum("ij,njk->nik", base, T2[k2])
                        D = new[-1] @ np.linalg.inv(TT[j])
                        TT[i : j + 1] = new
                        TT[j + 1 :] = np.einsum("ij,njk->nik", D, TT[j + 1 :])
                        fixed.append(
                            dict(
                                t0=float(tt[i] - np.loadtxt(d / "data_ts.txt")[0]),
                                dur=float(tt[j] - tt[i]),
                                poses=int(j - i + 1),
                                v12_move_m=float(move),
                            )
                        )
            i = j
        if len(fixed) > n_before:
            TT = anchor_to_first(TT)
            reason = reason + "+coverage_guard"
        new_pieces.append((tt, TT, reason))
    if fixed:
        shutil.copy(d / "odometry.txt", d / "regen" / "odometry_precoverageguard.txt")
        rewrite(d, new_pieces)
    return fixed


# ---------------------------------------------------------------------------- 2. v1.1 fill
def v11_fill(dst: Path, v11: Path, scene: str, ctx: SceneCtx):
    d = dst / scene
    p11 = v11 / scene / "odometry.txt"
    segs = load_segment_rows(d / "odometry_segments.txt")
    if not p11.exists() or any("v11_fill" in r[3] for r in segs):
        return None
    t1, T1 = read_trajectory(p11)
    cands = fill_segments(t1, T1, segs, ctx.rt, ctx.rT)  # bad_poses gate + >=20 poses
    stats = dict(odom_bad=0, seams=0, map_hits=0)
    runs = [r for tt, TT in cands for r in ctx.gate(tt, TT, stats)]
    if not runs:
        return dict(runs=0, span_s=0.0, **stats)
    pieces = load_pieces(d) + [(tt, TT, "v11_fill") for tt, TT in runs]
    rewrite(d, pieces)
    return dict(
        runs=len(runs),
        span_s=float(sum(tt[-1] - tt[0] for tt, _ in runs)),
        poses=int(sum(len(tt) for tt, _ in runs)),
        **stats,
    )


# ---------------------------------------------------------------------------- 3. regen fill
def regen_fill(dst: Path, scene: str, pilot: Path, a: float, b: float, ctx: SceneCtx, tag: str):
    d = dst / scene
    T0 = ctx.ts[0]
    a, b = T0 + a, T0 + b
    t, T = read_trajectory(pilot / "odometry.txt")
    psegs = read_segments(pilot / "odometry_segments.txt")
    segs = load_segment_rows(d / "odometry_segments.txt")
    covered = np.zeros(len(t), bool)
    for s0, s1, *_ in segs:
        covered |= (t >= s0 - COVER_MARGIN) & (t <= s1 + COVER_MARGIN)
    stats = dict(odom_bad=0, seams=0, map_hits=0, raw_poses=int(len(t)))
    runs = []
    for s in psegs:
        idx = np.where((t >= s.start_ts) & (t <= s.end_ts) & ~covered & (t >= a) & (t <= b))[0]
        if len(idx) < MIN_POSES:
            continue
        # a pilot segment may straddle removed (covered) poses: split at time holes
        for run in np.split(idx, np.where(np.diff(t[idx]) > MAX_GAP)[0] + 1):
            if len(run) >= MIN_POSES:
                runs += ctx.gate(t[run], T[run], stats)
    if not runs:
        return dict(runs=0, span_s=0.0, **stats)
    pieces = load_pieces(d) + [(tt, TT, tag) for tt, TT in runs]
    rewrite(d, pieces)
    gps = [w for w in (gps_witness(ctx, tt, TT) for tt, TT in runs) if w]
    return dict(
        runs=len(runs),
        span_s=float(sum(tt[-1] - tt[0] for tt, _ in runs)),
        poses=int(sum(len(tt) for tt, _ in runs)),
        hole_s=float(b - a),
        gps_witness=gps,
        **stats,
    )


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("v11")
    ap.add_argument("--v12", required=True, help="the HELHEST v1.2 root the coverage guard compares against")
    ap.add_argument("--regen", nargs="*", default=[], help="scene:pilot_dir:t0_rel:t1_rel")
    ap.add_argument("--no-v11", action="store_true")
    a = ap.parse_args()
    src, dst, v11, v12 = Path(a.src), Path(a.dst), Path(a.v11), Path(a.v12)
    link_tree(src, dst)
    scenes = sorted(p.name for p in dst.iterdir() if p.is_dir() and p.name.isdigit())
    report = dict(
        built=datetime.datetime.now().isoformat(timespec="seconds"),
        source_root=str(src),
        coverage_guard={},
        v11_fill={},
        regen_fill={},
    )
    for s in scenes:
        fx = coverage_guard(dst, v12, s)
        if fx:
            report["coverage_guard"][s] = fx
            print(f"{s}: coverage_guard {fx}")
    ctxs = {}
    for spec in a.regen:
        s, pilot, t0, t1 = spec.split(":")
        ctxs.setdefault(s, SceneCtx(dst, s))
        tag = f"regen_fill:t{float(t0):.0f}"
        r = regen_fill(dst, s, Path(pilot), float(t0), float(t1), ctxs[s], tag)
        report["regen_fill"].setdefault(s, []).append(dict(pilot=pilot, t0=float(t0), t1=float(t1), **r))
        print(
            f"{s}: {tag} +{r['runs']} runs {r['span_s']:.1f} s of {r.get('hole_s', 0):.1f} (odom_bad {r['odom_bad']}, seams {r['seams']}, map_hits {r['map_hits']})"
        )
    if not a.no_v11:
        for s in scenes:
            ctxs.setdefault(s, SceneCtx(dst, s))
            r = v11_fill(dst, v11, s, ctxs[s])
            if r and r["runs"]:
                report["v11_fill"][s] = r
                print(
                    f"{s}: v11_fill +{r['runs']} runs {r['span_s']:.1f} s (odom_bad {r['odom_bad']}, seams {r['seams']}, map_hits {r['map_hits']})"
                )

    # effect over the whole root, before/after
    def totals(root):
        segs = poses = span = 0
        for sc in scenes:
            rows = read_segments(root / sc / "odometry_segments.txt")
            segs += len(rows)
            poses += sum(r.n_poses for r in rows)
            span += sum(r.end_ts - r.start_ts for r in rows)
        return segs, poses, span

    si, pi, spi = totals(src)
    so, po, spo = totals(dst)
    per_scene = {}
    for sc in scenes:
        e = {}
        if sc in report["coverage_guard"]:
            e["coverage_guard_runs"] = len(report["coverage_guard"][sc])
        if sc in report["regen_fill"]:
            e["regen_fill_s"] = round(sum(x["span_s"] for x in report["regen_fill"][sc]), 1)
        if sc in report["v11_fill"]:
            e["v11_fill_s"] = round(report["v11_fill"][sc]["span_s"], 1)
        if e:
            per_scene[sc] = e
    mp = dst / "manifest.json"
    m = json.loads(mp.read_text())
    m["dataset_version"] = str(m.get("dataset_version", "")) + "+fill"
    m["odometry_fill"] = dict(
        status=f"built {report['built']} as a SEPARATE symlink root (dgx07); NOT merged into HELHEST_v1_3",
        generator="DRIFT session scratch build_fill_root.py (gate = dataset_tools filter_segments.bad_poses + gt_gaps + gt_map.window_offset)",
        why="30% of v1.3 frame time had no GT. 5308 s of it were FAST-LIO silent stalls over complete bags (lidar 10 Hz, Livox IMU ~195 Hz, /odom and frames continuous inside every hole; the recorder writes every pose it receives), recoverable by replaying the hole from a bag offset with the watchdog runner. v1.1 covers further gaps at gate quality. The refiner's ZUPT froze real motion outside joint_states coverage in three segments.",
        rule="(1) coverage_guard: within a segment, a run of identical positions >= 1 s that starts before the first / ends after the last joint_states sample, where v1.2 moved > 0.3 m over the same poses, is replaced by v1.2's relative motion re-expressed at the run's first pose; the rest of the segment is shifted rigidly to stay continuous; the segment is re-anchored and its reason suffixed '+coverage_guard'. (2) regen_fill:t<hole>: poses of an offset FAST-LIO replay (rate 0.5, mid360_v2.yaml, --restart, START_OFFSET/PLAY_DURATION window) that lie inside the hole and outside every existing segment (+-0.15 s). (3) v11_fill: v1.1 poses inside remaining gaps (merge_v11_fill.fill_segments). Every candidate run of (2) and (3) passes: 2 s windows vs /odom > 1 m dropped (filter_segments.bad_poses), gt_gaps seams (gap > 1 s, wheel_gap > 1 m), gt_map.window_offset on MEASURED cells only (> 0.3 m over 4 s cut with 0.5 s margin), >= 20 poses, re-anchored at identity. FAST-LIO's own roll/pitch is kept; nothing is derived from the IMU.",
        parameters=dict(
            window_s=WINDOW,
            max_err_m=MAX_ERR,
            min_poses=MIN_POSES,
            max_gap_s=MAX_GAP,
            map_horizon_s=MAP_HORIZON,
            map_max_off_m=MAP_MAX_OFF,
            map_margin_s=MAP_MARGIN,
            cover_margin_s=COVER_MARGIN,
            frozen_min_s=FROZEN_MIN_S,
            frozen_min_move_m=FROZEN_MIN_MOVE,
            replay_rate=0.5,
            replay_config="mid360_v2.yaml",
        ),
        effect=dict(
            segments_in=si,
            segments_out=so,
            poses_in=pi,
            poses_out=po,
            span_s_in=round(spi),
            span_s_out=round(spo),
            coverage_guard_runs=sum(len(v) for v in report["coverage_guard"].values()),
            regen_fill_s=round(sum(x["span_s"] for v in report["regen_fill"].values() for x in v)),
            regen_fill_runs=sum(x["runs"] for v in report["regen_fill"].values() for x in v),
            v11_fill_s=round(sum(v["span_s"] for v in report["v11_fill"].values())),
            v11_fill_runs=sum(v["runs"] for v in report["v11_fill"].values()),
        ),
        per_scene=per_scene,
        tags=["<reason>+coverage_guard", "regen_fill:t<hole start, s>", "v11_fill"],
        backups="source root untouched; regen/odometry_precoverageguard.txt where the guard fired; per-run detail in FILL_REPORT.json (odom_bad / seams / map_hits per fill, gps_witness = rigid 2-D fit vs gps_odom, comparison only)",
        replay_outputs="<pilot_dir>/<scene>_hole_t<t0>/ as passed in --regen (raw run_scene.sh outputs, prefilter)",
        merge_note="downstream guards (gen_proprio / validate batches) read odometry_map_cut.effect.segments_out; on merge point them at odometry_fill.effect.segments_out",
    )
    m.setdefault("corrections", []).append("odometry_fill")
    tmp = mp.with_suffix(".tmp")
    tmp.write_text(json.dumps(m, indent=1))
    os.replace(tmp, mp)
    (dst / "FILL_REPORT.json").write_text(json.dumps(report, indent=1))
    rd = dst / "README.md"
    rd.write_text(
        "# HELHEST v1.3 + fill (separate root, NOT merged)\n\nSee manifest.json -> odometry_fill and FILL_REPORT.json. "
        "Everything except odometry.txt / odometry_segments.txt / manifest.json is a symlink into HELHEST_v1_3.\n\n---\n\n"
        + rd.read_text()
    )
    print("done ->", dst)


if __name__ == "__main__":
    main()
