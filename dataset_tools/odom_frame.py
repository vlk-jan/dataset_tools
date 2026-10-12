"""The Odin1 odometry, rotated onto the frame its own clouds are in.

The Odin1 computes odometry with its own running lidar-to-IMU estimate, not the
pinned constant the clouds go through, so the two frames differ by a few tenths
of a degree per scene (enough to fail DRIFT's path-vs-map slope guard).
:func:`measure` (open3d) finds the cloud->pose rotation ``C`` from ICP pairs
(pitch from travel elevation, roll from a Kabsch fit when tight, no yaw);
:func:`apply` re-expresses poses as ``X^T T X`` and re-levels gravity, once
(recorded in :data:`RECORD`). Derived products built before are stale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dataset_tools.gt_io import _write_trajectory, read_trajectory, write_tmp
from dataset_tools.transforms import euler_matrix, roll_pitch_from_matrix, rotation_angle_deg, rpy_deg

RECORD = "odom_frame.txt"
FILES = ("odometry.txt", "transforms.npy", "gravity_align.npy")
#: The Kabsch roll is used only when its bootstrap spread is below this.
ROLL_MAX_SD_DEG = 0.15
#: Pairs: clouds this many frames apart (14.5 Hz), moved at least this far or turned this much.
PAIR_GAP = (20, 45)
MIN_TRAVEL_M = 1.5
MIN_TURN_DEG = 15.0
#: Translation and rotation-axis vectors to collect for the fit.
N_VECTORS = 600


def _kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """The rotation C minimising sum |P_i - C Q_i|^2."""
    U, _, Vt = np.linalg.svd(Q.T @ P)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _axis(R: np.ndarray) -> np.ndarray:
    return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def measure(scene_dir: Path) -> dict:
    """ICP between cloud pairs against the poses; returns the fit and the correction ``C``."""
    import open3d as o3d

    scene_dir = Path(scene_dir)
    t, T = read_trajectory(scene_dir / "odometry.txt")
    pts = np.loadtxt(scene_dir / "pcd_ts.txt", ndmin=1)
    pts = pts if pts.ndim == 1 else pts[:, -1]
    seg = np.loadtxt(scene_dir / "odometry_segments.txt", ndmin=2, usecols=(1, 2))
    rng = np.random.default_rng(0)
    reg = o3d.pipelines.registration

    def pose(f: int):
        i = int(np.argmin(np.abs(t - pts[f])))
        return T[i] if abs(t[i] - pts[f]) < 0.04 else None

    def cloud(f: int):
        c = np.load(scene_dir / "clouds" / f"{f:05d}.npy")[:, :3].astype(np.float64)
        r = np.linalg.norm(c[:, :2], axis=1)
        p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(c[(r > 0.8) & (r < 25)])).voxel_down_sample(0.08)
        p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.4, max_nn=20))
        return p

    P, Q, elev = [], [], []
    tries = 0
    while len(P) < N_VECTORS and tries < 8000:
        tries += 1
        a = int(rng.integers(len(pts)))
        b = a + int(rng.integers(*PAIR_GAP))
        inside = (seg[:, 0] <= pts[a]) & (pts[a] <= seg[:, 1])
        if b >= len(pts) or not inside.any() or pts[b] > seg[inside][0][1]:
            continue
        Ta, Tb = pose(a), pose(b)
        if Ta is None or Tb is None:
            continue
        Tp = np.linalg.inv(Ta) @ Tb
        dist = float(np.linalg.norm(Tp[:3, 3]))
        turn = rotation_angle_deg(Tp[:3, :3])
        if dist < MIN_TRAVEL_M and turn < MIN_TURN_DEG:
            continue
        ca, cb = cloud(a), cloud(b)
        if len(ca.points) < 2000 or len(cb.points) < 2000:
            continue
        est, crit = reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=60)
        r = reg.registration_icp(cb, ca, 0.3, Tp, est, crit)
        r = reg.registration_icp(cb, ca, 0.12, r.transformation, est, crit)
        Tc = np.asarray(r.transformation)
        if r.fitness < 0.5 or np.linalg.norm(Tc[:3, 3] - Tp[:3, 3]) > 0.4:
            continue
        if dist > 1.0:
            P.append(Tp[:3, 3]), Q.append(Tc[:3, 3])
            if np.linalg.norm(Tp[:2, 3]) > 1.0:
                el = lambda v: math.degrees(math.atan2(v[2], np.linalg.norm(v[:2])))  # noqa: E731
                elev.append(el(Tc[:3, 3]) - el(Tp[:3, 3]))
        if turn > 10:
            P.append(4 * _axis(Tp[:3, :3])), Q.append(4 * _axis(Tc[:3, :3]))
    if len(elev) < 30:
        raise RuntimeError(f"{scene_dir}: only {len(elev)} registered translations, too few to measure")
    P, Q = np.array(P), np.array(Q)
    K = _kabsch(P, Q)
    boot = np.array([rpy_deg(_kabsch(P[i], Q[i])) for i in (rng.integers(len(P), size=len(P)) for _ in range(50))])
    sd = boot.std(axis=0)
    # Elevation of the travel direction: cloud frame minus pose frame. C (cloud -> pose)
    # carries a cloud-frame vector up by -elev, i.e. pitch = +elev in 'sxyz' (nose-down positive).
    pitch = float(np.median(elev))
    roll = rpy_deg(K)[0] if sd[0] < ROLL_MAX_SD_DEG else 0.0
    C = euler_matrix(math.radians(roll), math.radians(pitch), 0.0)[:3, :3]
    return {
        "C": C.tolist(),
        "roll_deg": roll,
        "pitch_deg": pitch,
        "kabsch_rpy_deg": rpy_deg(K),
        "kabsch_sd_deg": sd.tolist(),
        "n_vectors": len(P),
        "n_translations": len(elev),
        "elev_p25_p75_deg": np.percentile(elev, [25, 75]).tolist(),
    }


def corrected(poses: np.ndarray, C: np.ndarray) -> np.ndarray:
    """``X^T T X`` for every pose, X the rotation C."""
    X = np.eye(4)
    X[:3, :3] = C
    return np.einsum("ji,njk,kl->nil", X, poses, X)


def corrected_gravity(gravity: np.ndarray, C: np.ndarray) -> np.ndarray:
    out = np.empty_like(gravity)
    for k, G in enumerate(gravity):
        roll, pitch = roll_pitch_from_matrix(G @ C)
        out[k] = euler_matrix(roll, pitch, 0.0)[:3, :3]
    return out


def apply(scene_dir: Path, fit: dict, note: str = "") -> None:
    """Rotate the poses and levelling of ``scene_dir`` by ``fit["C"]``, once.

    All three files are built as temp files first, then :data:`RECORD` is written
    and the temp files replace the originals in a row, so a crash never leaves a
    half-rotated scene that a re-run would rotate again: the record refuses it.
    """
    scene_dir = Path(scene_dir)
    if (scene_dir / RECORD).exists():
        raise RuntimeError(f"{scene_dir}: {RECORD} exists, the rotation is already applied")
    for name in FILES:
        if (scene_dir / name).is_symlink() or not (scene_dir / name).is_file():
            raise RuntimeError(f"{scene_dir / name}: missing or a symlink; apply where the file lives")
    C = np.asarray(fit["C"], dtype=np.float64)
    if rotation_angle_deg(C) > 3.0:
        raise RuntimeError(f"{scene_dir}: a {rpy_deg(C)} deg correction is not a calibration residual")
    ts, poses = read_trajectory(scene_dir / "odometry.txt")
    rel = np.load(scene_dir / "transforms.npy")
    gravity = np.load(scene_dir / "gravity_align.npy")
    tmp = {
        "odometry.txt": write_tmp(
            scene_dir / "odometry.txt", lambda fh: _write_trajectory(fh, ts, corrected(poses, C)), "w"
        ),
        "transforms.npy": write_tmp(scene_dir / "transforms.npy", lambda fh: np.save(fh, corrected(rel, C)), "wb"),
        "gravity_align.npy": write_tmp(
            scene_dir / "gravity_align.npy", lambda fh: np.save(fh, corrected_gravity(gravity, C)), "wb"
        ),
    }
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# The odometry of this scene re-expressed at the frame of its clouds (dataset_tools.odom_frame).",
        f"# {stamp}: odometry.txt, transforms.npy and gravity_align.npy rotated by C (cloud frame -> Odin1 pose frame).",
        "# Height maps, masks, labels and keyframe lists built before this are stale.",
    ]
    if note:
        lines.append(f"# {note}")
    lines.append(json.dumps({"applied_utc": stamp, **fit}))
    (scene_dir / RECORD).write_text("\n".join(lines) + "\n")
    for name in FILES:
        os.replace(tmp[name], scene_dir / name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("action", choices=("measure", "apply"))
    p.add_argument("root", type=Path)
    p.add_argument(
        "--scenes", nargs="+", default=None, help="default: every scene with clouds and no odom_frame.txt yet"
    )
    p.add_argument("--fits", type=Path, required=True, help="JSON {scene: fit}: written by measure, read by apply")
    p.add_argument("--note", default="")
    a = p.parse_args(argv)
    if a.scenes is None:
        a.scenes = sorted(
            d.name
            for d in a.root.iterdir()
            if d.name.isdigit() and (d / "clouds").is_dir() and not (d / RECORD).exists()
        )
    if a.action == "measure":
        fits = json.loads(a.fits.read_text()) if a.fits.exists() else {}
        for s in a.scenes:
            fits[s] = measure(a.root / s)
            f = fits[s]
            print(
                f"{s} roll {f['roll_deg']:+.2f} pitch {f['pitch_deg']:+.2f} deg | kabsch {np.round(f['kabsch_rpy_deg'], 2)} "
                f"sd {np.round(f['kabsch_sd_deg'], 2)} | {f['n_translations']} translations",
                flush=True,
            )
            a.fits.write_text(json.dumps(fits, indent=1))
        return 0
    fits = json.loads(a.fits.read_text())
    for s in a.scenes:
        apply(a.root / s, fits[s], a.note)
        print(f"{s} applied roll {fits[s]['roll_deg']:+.2f} pitch {fits[s]['pitch_deg']:+.2f} deg", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
