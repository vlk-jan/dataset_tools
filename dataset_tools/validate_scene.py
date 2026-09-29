#!/usr/bin/env python3
"""Completeness and consistency checks for extracted HELHEST scenes.

Usage:
    helhest-validate <scene-dir> [<scene-dir> ...]
    helhest-validate <dataset-root>            # every NNNN scene below it

Structural problems (missing files, frame-count mismatches, wrong dtypes or
shapes, non-monotonic timestamps) are reported as errors and make the tool exit
non-zero. Data-quality observations (NaNs, odometry jumps, extreme tilt) are
warnings; ``--strict`` turns them into failures too.

The older HELHEST layout is accepted too; its extra files (``trajectory.txt``,
``T_cam_to_lidar.txt``, ``<scene>_full.npz``, ...) count as optional extras.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    from dataset_tools.gnss_guard import MAX_SPEED
    from dataset_tools.gt_io import read_segments, read_trajectory
except ImportError:  # when the file is run directly from dataset_tools/
    from gnss_guard import MAX_SPEED
    from gt_io import read_segments, read_trajectory

SCENE_DIR_RE = re.compile(r"^\d{4,}$")

#: Files every extracted scene must have.
REQUIRED_FILES = (
    "data_ts.txt",
    "pcd_ts.txt",
    "transforms.npy",
    "gravity_align.npy",
    "camera_intrinsics.txt",
)
#: Files that may legitimately be absent (topic missing, or old layout only).
OPTIONAL_FILES = (
    "joint_states.npy",
    "joint_setpoints.npy",
    "cmd_vel.npy",
    "gps.npy",
    "source_bag.txt",
    "camera_info.yaml",
    "T_cam_to_base.txt",
    "odometry.txt",
    "odometry_segments.txt",
    "trajectory.txt",
    "T_cam_to_lidar.txt",
)
OPTIONAL_DIRS = ("height_maps", "height_masks")

ERROR = "ERROR"
WARNING = "WARN"


@dataclass
class SceneReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)
    extras: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def status(self, strict: bool = False) -> str:
        if self.errors:
            return "FAIL"
        if self.warnings:
            return "FAIL" if strict else "WARN"
        return "OK"


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------
def looks_like_scene(path: Path) -> bool:
    return path.is_dir() and (
        (path / "images").is_dir() or (path / "clouds").is_dir() or (path / "data_ts.txt").is_file()
    )


def find_scenes(paths: Sequence[Path]) -> list[Path]:
    """Expand dataset roots into scene directories, keeping explicit ones."""
    scenes: list[Path] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"path does not exist: {path}")
        if looks_like_scene(path):
            scenes.append(path)
            continue
        children = sorted(child for child in path.iterdir() if child.is_dir() and SCENE_DIR_RE.match(child.name))
        found = [child for child in children if looks_like_scene(child)]
        if not found:
            raise FileNotFoundError(f"no scenes found in {path}")
        scenes.extend(found)
    return scenes


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _indexed_files(directory: Path, suffix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix == suffix)


def _check_index_continuity(report: SceneReport, files: Sequence[Path], label: str) -> None:
    stems = [p.stem for p in files]
    expected = [f"{i:05d}" for i in range(len(files))]
    if stems != expected:
        missing = sorted(set(expected) - set(stems))[:5]
        report.error(
            f"{label}: filenames are not a contiguous 00000..{len(files) - 1:05d} run "
            f"(first gaps: {missing or 'unexpected names'})"
        )


def _check_timestamps(report: SceneReport, ts: np.ndarray, label: str) -> None:
    if ts.ndim != 1:
        report.error(f"{label}: expected one timestamp per line, got shape {ts.shape}")
        return
    if not np.all(np.isfinite(ts)):
        report.error(f"{label}: contains non-finite values")
        return
    diffs = np.diff(ts)
    if np.any(diffs < 0):
        report.error(f"{label}: not monotonic ({int(np.sum(diffs < 0))} decreasing steps)")
    n_dup = int(np.sum(diffs == 0))
    if n_dup:
        report.warn(f"{label}: {n_dup} duplicate timestamps")


def _is_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    if R.shape[-2:] != (3, 3):
        return False
    identity = np.einsum("...ij,...kj->...ik", R, R)
    return bool(np.allclose(identity, np.eye(3), atol=tol) and np.all(np.abs(np.linalg.det(R) - 1.0) < tol))


def check_clouds(report: SceneReport, clouds: Sequence[Path], n_sample: int) -> None:
    if not clouds:
        return
    idx = np.unique(np.linspace(0, len(clouds) - 1, min(n_sample, len(clouds))).astype(int))
    n_points: list[int] = []
    nan_counts = 0
    total_values = 0
    dtypes: set[str] = set()
    for i in idx:
        path = clouds[int(i)]
        try:
            cloud = np.load(path)
        except Exception as exc:  # noqa: BLE001
            report.error(f"clouds/{path.name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        dtypes.add(str(cloud.dtype))
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            report.error(f"clouds/{path.name}: expected (N, 4), got {cloud.shape}")
            continue
        if cloud.dtype.kind != "f":
            report.error(f"clouds/{path.name}: expected a float dtype, got {cloud.dtype}")
            continue
        n_points.append(int(cloud.shape[0]))
        if cloud.size:
            nan_counts += int(np.count_nonzero(~np.isfinite(cloud)))
            total_values += int(cloud.size)
        else:
            report.warn(f"clouds/{path.name}: empty cloud")

    if dtypes:
        report.stats["cloud_dtype"] = "/".join(sorted(dtypes))
        if len(dtypes) > 1:
            report.warn(f"clouds: mixed dtypes {sorted(dtypes)}")
        elif not dtypes <= {"float32", "float64"}:
            # float32 is what the extractor writes; the old dataset is float64.
            report.warn(f"clouds: unexpected dtype {sorted(dtypes)[0]}")
    if n_points:
        report.stats["points_mean"] = int(np.mean(n_points))
        report.stats["points_min"] = int(np.min(n_points))
    if total_values:
        report.stats["nan_frac"] = nan_counts / total_values
    report.stats["clouds_sampled"] = int(len(idx))


def check_odometry(
    report: SceneReport,
    transforms: np.ndarray,
    max_jump: float,
    ts: np.ndarray | None = None,
    max_speed: float = MAX_SPEED,
) -> None:
    positions = transforms[:, :3, 3]
    if len(positions) < 2:
        return
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    report.stats["max_step_m"] = float(np.max(steps))
    report.stats["path_len_m"] = float(np.sum(steps))
    n_jumps = int(np.sum(steps > max_jump))
    if n_jumps:
        report.warn(f"transforms.npy: {n_jumps} frame-to-frame jumps > {max_jump} m " f"(max {np.max(steps):.2f} m)")

    # Distance alone lets a short step past that no robot could have taken: at a
    # 16 Hz frame rate 0.9 m is 14 m/s. Divide by the interval the frames
    # actually claim and the question becomes the one that matters -- is this
    # motion, or is it a pose that belongs to a different moment
    # (dataset_tools.gnss_guard). The gate is the one write_ground_truth breaks
    # segments on, so the two agree about which steps are real.
    if ts is None or len(ts) != len(positions) or not np.isfinite(ts).all():
        return
    dt = np.diff(ts)
    speed = steps / np.maximum(dt, 1e-6)
    report.stats["max_speed_mps"] = float(np.max(speed))
    n_fast = int(np.sum((dt > 0) & (speed > max_speed)))
    if n_fast:
        report.warn(
            f"transforms.npy: {n_fast} frame-to-frame steps imply > {max_speed} m/s "
            f"(max {np.max(speed):.1f} m/s) -- a pose from another moment, not motion"
        )


def check_tilt(report: SceneReport, gravity: np.ndarray, max_tilt_deg: float) -> None:
    cos_tilt = np.clip(gravity[:, 2, 2], -1.0, 1.0)
    tilt = np.degrees(np.arccos(cos_tilt))
    report.stats["tilt_mean_deg"] = float(np.mean(tilt))
    report.stats["tilt_max_deg"] = float(np.max(tilt))
    n_extreme = int(np.sum(tilt > max_tilt_deg))
    if n_extreme:
        report.warn(
            f"gravity_align.npy: {n_extreme} frames tilted > {max_tilt_deg} deg " f"(max {np.max(tilt):.1f} deg)"
        )


def check_extrinsic(report: SceneReport, path: Path) -> None:
    """``T_cam_to_base.txt`` / ``T_cam_to_lidar.txt``: one finite SE(3) matrix."""
    try:
        T = np.loadtxt(path)
    except Exception as exc:  # noqa: BLE001
        report.error(f"{path.name}: unreadable ({type(exc).__name__}: {exc})")
        return
    if T.shape != (4, 4):
        report.error(f"{path.name}: expected 4x4, got {T.shape}")
    elif not np.isfinite(T).all():
        report.error(f"{path.name}: contains non-finite values")
    elif not _is_rotation(T[:3, :3]) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        report.error(f"{path.name}: not a rigid transform")


def check_ground_truth(report: SceneReport, scene_dir: Path, atol: float = 1e-6) -> None:
    """``odometry.txt`` and ``odometry_segments.txt`` agree with each other.

    Every segment's pose count must match the poses in its span, segments may
    not overlap, and each segment's first pose should be the identity (they
    are re-anchored). Poses outside every segment are a warning: consumers
    following the contract never read them, so they are dead weight rather
    than a defect.
    """
    odometry = scene_dir / "odometry.txt"
    segments = scene_dir / "odometry_segments.txt"
    if not odometry.is_file():
        if segments.is_file():
            report.error("odometry_segments.txt is present but odometry.txt is not")
        return
    try:
        ts, poses = read_trajectory(odometry)
    except (OSError, ValueError) as exc:
        report.error(f"odometry.txt: unreadable ({type(exc).__name__}: {exc})")
        return
    if ts.size == 0:
        report.warn("odometry.txt: no poses")
        return
    report.stats["gt_poses"] = int(ts.size)
    if not np.isfinite(poses).all():
        report.error("odometry.txt: contains non-finite values")
        return
    if not _is_rotation(poses[:, :3, :3]):
        report.error("odometry.txt: rotation blocks are not orthonormal")
    if not segments.is_file():
        return

    try:
        rows = read_segments(segments)
    except OSError as exc:
        report.error(f"odometry_segments.txt: unreadable ({type(exc).__name__}: {exc})")
        return
    if not rows:
        report.warn("odometry_segments.txt: no segments")
        return
    report.stats["gt_segments"] = len(rows)
    ordered = sorted(rows, key=lambda r: r.start_ts)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur.start_ts < prev.end_ts - atol:
            report.error(f"odometry_segments.txt: segments {prev.index} and {cur.index} overlap")
    covered = np.zeros(ts.size, dtype=bool)
    for row in rows:
        inside = (ts >= row.start_ts - atol) & (ts <= row.end_ts + atol)
        covered |= inside
        n = int(inside.sum())
        if row.n_poses and n != row.n_poses:
            report.error(f"segment {row.index}: n_poses is {row.n_poses} but odometry.txt has {n} poses in its span")
        if n and not np.allclose(poses[inside][0], np.eye(4), atol=1e-4):
            report.warn(f"segment {row.index}: first pose is not the identity (segments are re-anchored)")
    outside = int((~covered).sum())
    if outside:
        report.warn(f"odometry.txt: {outside} poses lie outside every segment")


def check_gps(report: SceneReport, scene_dir: Path, data_ts: np.ndarray | None) -> None:
    """``gps.npy`` is well-formed, when the scene has one.

    Everything here is a warning. GPS is an optional extra: it is not part of
    the scene contract, nothing else is computed from it, and it is not
    frame-aligned. What is worth saying is that a file which *is* there cannot
    be used -- too few columns, an unsorted clock, coordinates off the globe,
    or a track that does not overlap the frames it is supposed to describe.
    """
    path = scene_dir / "gps.npy"
    if not path.is_file():
        return
    try:
        gps = np.load(path)
    except Exception as exc:  # noqa: BLE001
        report.warn(f"gps.npy: unreadable ({type(exc).__name__}: {exc})")
        return
    if gps.ndim != 2 or gps.shape[0] == 0 or gps.shape[1] < 4:
        report.warn(f"gps.npy: expected (M, >=4) [t, lat, lon, alt, ...], got {gps.shape}")
        return

    report.stats["gps_fixes"] = int(gps.shape[0])
    ts, lat, lon = gps[:, 0], gps[:, 1], gps[:, 2]
    if not np.all(np.diff(ts) >= 0):
        report.warn("gps.npy: timestamps are not sorted")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        report.warn("gps.npy: contains non-finite coordinates")
    elif np.abs(lat).max() > 90.0 or np.abs(lon).max() > 180.0:
        report.warn("gps.npy: coordinates outside [-90, 90] / [-180, 180] degrees")
    if gps.shape[1] >= 5:
        no_fix = float((gps[:, 4] < 0).mean())
        report.stats["gps_no_fix_frac"] = no_fix
        if no_fix > 0.5:
            report.warn(f"gps.npy: {no_fix:.0%} of the fixes report no fix at all")
    if data_ts is not None and data_ts.size and np.isfinite(ts).all():
        overlap = min(ts[-1], data_ts[-1]) - max(ts[0], data_ts[0])
        if overlap <= 0:
            report.warn("gps.npy: its time span does not overlap data_ts.txt")


# ---------------------------------------------------------------------------
# Scene validation
# ---------------------------------------------------------------------------
def validate_scene(
    scene_dir: Path,
    n_sample: int = 8,
    max_jump: float = 1.0,
    max_tilt_deg: float = 45.0,
    max_nan_frac: float = 0.01,
    max_sync_offset: float = 0.1,
) -> SceneReport:
    report = SceneReport(path=scene_dir)
    counts: dict[str, int] = {}

    images = _indexed_files(scene_dir / "images", ".png")
    clouds = _indexed_files(scene_dir / "clouds", ".npy")
    for label, files, directory in (
        ("images", images, scene_dir / "images"),
        ("clouds", clouds, scene_dir / "clouds"),
    ):
        if not directory.is_dir():
            report.error(f"{label}/ is missing")
            continue
        if not files:
            report.error(f"{label}/ is empty")
            continue
        counts[label] = len(files)
        _check_index_continuity(report, files, label)

    for name in REQUIRED_FILES:
        if not (scene_dir / name).is_file():
            report.error(f"{name} is missing")

    for name in OPTIONAL_FILES:
        if (scene_dir / name).is_file():
            report.extras.append(name)
    for name in OPTIONAL_DIRS:
        if (scene_dir / name).is_dir():
            report.extras.append(f"{name}/")
    full_npz = scene_dir / f"{scene_dir.name}_full.npz"
    if full_npz.is_file():
        report.extras.append(full_npz.name)

    timestamps: dict[str, np.ndarray] = {}
    for name in ("data_ts.txt", "pcd_ts.txt"):
        path = scene_dir / name
        if not path.is_file():
            continue
        try:
            ts = np.atleast_1d(np.loadtxt(path))
        except Exception as exc:  # noqa: BLE001
            report.error(f"{name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        timestamps[name] = ts
        counts[name] = int(ts.shape[0])
        _check_timestamps(report, ts, name)

    if {"data_ts.txt", "pcd_ts.txt"} <= timestamps.keys():
        data_ts, pcd_ts = timestamps["data_ts.txt"], timestamps["pcd_ts.txt"]
        if data_ts.shape == pcd_ts.shape and data_ts.size:
            # data_ts is the midpoint of the camera and lidar stamps, so the
            # camera/lidar offset is twice the difference to pcd_ts.
            offset = 2.0 * np.max(np.abs(data_ts - pcd_ts))
            report.stats["max_sync_offset_s"] = float(offset)
            if offset > max_sync_offset:
                report.warn(
                    f"camera/lidar sync offset up to {offset * 1e3:.0f} ms " f"(> {max_sync_offset * 1e3:.0f} ms)"
                )
            report.stats["duration_s"] = float(data_ts[-1] - data_ts[0])

    transforms = None
    path = scene_dir / "transforms.npy"
    if path.is_file():
        try:
            transforms = np.load(path)
        except Exception as exc:  # noqa: BLE001
            report.error(f"transforms.npy: unreadable ({type(exc).__name__}: {exc})")
            transforms = None
    if transforms is not None:
        if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
            report.error(f"transforms.npy: expected (N, 4, 4), got {transforms.shape}")
            transforms = None
        else:
            counts["transforms.npy"] = int(transforms.shape[0])
            if not np.all(np.isfinite(transforms)):
                report.error("transforms.npy: contains non-finite values")
            elif not _is_rotation(transforms[:, :3, :3]):
                report.error("transforms.npy: rotation blocks are not orthonormal")
            bottom = transforms[:, 3, :]
            if not np.allclose(bottom, [0.0, 0.0, 0.0, 1.0], atol=1e-6):
                report.error("transforms.npy: bottom row is not [0 0 0 1]")

    gravity = None
    path = scene_dir / "gravity_align.npy"
    if path.is_file():
        try:
            gravity = np.load(path)
        except Exception as exc:  # noqa: BLE001
            report.error(f"gravity_align.npy: unreadable ({type(exc).__name__}: {exc})")
            gravity = None
    if gravity is not None:
        if gravity.ndim != 3 or gravity.shape[1:] != (3, 3):
            report.error(f"gravity_align.npy: expected (N, 3, 3), got {gravity.shape}")
            gravity = None
        else:
            counts["gravity_align.npy"] = int(gravity.shape[0])
            if not np.all(np.isfinite(gravity)):
                report.error("gravity_align.npy: contains non-finite values")
                gravity = None
            elif not _is_rotation(gravity):
                report.error("gravity_align.npy: not a stack of rotation matrices")

    path = scene_dir / "camera_intrinsics.txt"
    if path.is_file():
        try:
            K = np.loadtxt(path)
            if K.shape != (3, 3):
                report.error(f"camera_intrinsics.txt: expected 3x3, got {K.shape}")
            elif not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                report.error("camera_intrinsics.txt: implausible focal lengths")
        except Exception as exc:  # noqa: BLE001
            report.error(f"camera_intrinsics.txt: unreadable ({type(exc).__name__}: {exc})")

    for name in ("T_cam_to_base.txt", "T_cam_to_lidar.txt"):
        if (scene_dir / name).is_file():
            check_extrinsic(report, scene_dir / name)

    check_gps(report, scene_dir, timestamps.get("data_ts.txt"))
    check_ground_truth(report, scene_dir)

    if counts:
        distinct = set(counts.values())
        if len(distinct) > 1:
            report.error("frame counts disagree: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        report.stats["frames"] = counts.get("images", min(counts.values()))

    check_clouds(report, clouds, n_sample)
    if transforms is not None:
        check_odometry(report, transforms, max_jump, timestamps.get("data_ts.txt"))
    if gravity is not None:
        check_tilt(report, gravity, max_tilt_deg)

    nan_frac = report.stats.get("nan_frac")
    if isinstance(nan_frac, float) and nan_frac > max_nan_frac:
        report.warn(f"clouds: {nan_frac:.2%} non-finite values (> {max_nan_frac:.2%})")

    return report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "scene",
    "status",
    "frames",
    "duration_s",
    "points_mean",
    "cloud_dtype",
    "nan_frac",
    "max_step_m",
    "max_speed_mps",
    "path_len_m",
    "tilt_mean_deg",
    "tilt_max_deg",
    "max_sync_offset_s",
    "errors",
    "warnings",
]


def _cell(report: SceneReport, column: str, strict: bool) -> str:
    if column == "scene":
        return report.name
    if column == "status":
        return report.status(strict)
    if column == "errors":
        return str(len(report.errors))
    if column == "warnings":
        return str(len(report.warnings))
    value = report.stats.get(column)
    if value is None:
        return ""
    if isinstance(value, float):
        if column == "nan_frac":
            return f"{value:.2e}"
        return f"{value:.3f}"
    return str(value)


def print_report(report: SceneReport, strict: bool, verbose: bool) -> None:
    stats = report.stats
    bits = [f"{stats.get('frames', 0)} frames"]
    if "duration_s" in stats:
        bits.append(f"{stats['duration_s'] / 60.0:.1f} min")
    if "points_mean" in stats:
        bits.append(f"{stats['points_mean'] / 1000.0:.1f}k pts")
    if "nan_frac" in stats:
        bits.append(f"nan {stats['nan_frac']:.2%}")
    if "max_step_m" in stats:
        step = f"max step {stats['max_step_m']:.2f} m"
        if "max_speed_mps" in stats:
            step += f" ({stats['max_speed_mps']:.1f} m/s)"
        bits.append(step)
    if "tilt_max_deg" in stats:
        bits.append(f"tilt <= {stats['tilt_max_deg']:.1f} deg")
    if "gt_poses" in stats:
        bits.append(
            f"GT {stats['gt_poses']} poses" + (f" / {stats['gt_segments']} seg" if "gt_segments" in stats else "")
        )
    print(f"{report.name:<10} {report.status(strict):<5} " + "  ".join(bits))
    for message in report.errors:
        print(f"    {ERROR}: {message}")
    for message in report.warnings:
        print(f"    {WARNING}: {message}")
    if verbose and report.extras:
        print(f"    extras: {', '.join(report.extras)}")


def write_table(reports: Sequence[SceneReport], path: str, strict: bool, markdown: bool) -> None:
    rows = [[_cell(r, c, strict) for c in CSV_COLUMNS] for r in reports]
    if markdown:
        lines = [
            "| " + " | ".join(CSV_COLUMNS) + " |",
            "| " + " | ".join("---" for _ in CSV_COLUMNS) + " |",
        ]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        text = "\n".join(lines) + "\n"
        if path == "-":
            sys.stdout.write(text)
        else:
            Path(path).write_text(text)
        return

    if path == "-":
        writer = csv.writer(sys.stdout)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate extracted HELHEST scenes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("paths", nargs="+", type=str, help="Scene directories, or a dataset root")
    parser.add_argument(
        "--sample-clouds",
        type=int,
        default=8,
        help="Point clouds to open per scene (0 disables the cloud checks)",
    )
    parser.add_argument(
        "--max-jump",
        type=float,
        default=1.0,
        help="Odometry step warning threshold [m]",
    )
    parser.add_argument("--max-tilt", type=float, default=45.0, help="Tilt warning threshold [deg]")
    parser.add_argument("--max-nan", type=float, default=0.01, help="Non-finite cloud value fraction")
    parser.add_argument(
        "--max-sync",
        type=float,
        default=0.1,
        help="Camera/lidar sync offset warning threshold [s]",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        metavar="PATH",
        help="Write a table ('-' = stdout)",
    )
    parser.add_argument("--markdown", action="store_true", help="Write the --csv table as markdown")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print the summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also list extra files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenes = find_scenes([Path(p) for p in args.paths])
    except FileNotFoundError as exc:
        print(f"{ERROR}: {exc}", file=sys.stderr)
        return 2

    reports = []
    for scene in scenes:
        report = validate_scene(
            scene,
            n_sample=args.sample_clouds,
            max_jump=args.max_jump,
            max_tilt_deg=args.max_tilt,
            max_nan_frac=args.max_nan,
            max_sync_offset=args.max_sync,
        )
        reports.append(report)
        if not args.quiet:
            print_report(report, args.strict, args.verbose)

    if args.csv:
        write_table(reports, args.csv, args.strict, args.markdown)

    failed = [r for r in reports if r.status(args.strict) == "FAIL"]
    warned = [r for r in reports if r.status(args.strict) == "WARN"]
    total_frames = sum(int(r.stats.get("frames", 0) or 0) for r in reports)
    print(
        f"\n{len(reports)} scene(s), {total_frames} frames: "
        f"{len(reports) - len(failed) - len(warned)} ok, {len(warned)} with warnings, "
        f"{len(failed)} failed"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
