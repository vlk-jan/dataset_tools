#!/usr/bin/env python3
"""Completeness and consistency checks for extracted HELHEST scenes.

Usage:
    helhest-validate <scene-dir> [<scene-dir> ...]
    helhest-validate <dataset-root>            # every NNNN scene below it

Structural problems are errors (non-zero exit), data-quality findings are
warnings (failures under ``--strict``), and notable non-defects are INFO.
Each message has a stable code; findings listed in a root's
``validation_expect.json`` (``{"expected": [["0008", "TILT"], ["*", "CLOUD_EMPTY"]]}``)
are downgraded to INFO.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset_tools import image_format
from dataset_tools.extract.scene_io import DERIVED_DIRS, source_bag
from dataset_tools.gnss_guard import MAX_SPEED
from dataset_tools.gt_io import read_segments, read_trajectory
from dataset_tools.keyframes import LISTS
from dataset_tools.transforms import MIN_VALID_STAMP, tilt_deg

#: Files every extracted scene must have.
REQUIRED_FILES = (
    "data_ts.txt",
    "pcd_ts.txt",
    "transforms.npy",
    "gravity_align.npy",
    "camera_intrinsics.txt",
)
#: Files that may legitimately be absent (topic missing, or not asked for).
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
    "tf_pins.txt",
    "extract_meta.json",
    "joint_names.txt",
    "gnss_guard.txt",
    "time_align.txt",
)

ERROR = "ERROR"
WARNING = "WARN"
INFO = "INFO"

#: Clouds under MIN_POINTS are sparse; frame or proprio holes over MAX_GAP_S are listed; a proprio stream
#: starting or ending over MAX_SHORT_S inside the frames is short; mounts over MAX_EXTRINSIC_M from the median differ.
MIN_POINTS, MAX_GAP_S, MAX_SHORT_S, MAX_EXTRINSIC_M = 100, 1.0, 2.0, 0.005
#: Slack, s, on segment bounds.
TS_TOL = 1e-6

#: The file a root lists its known findings in.
EXPECT_FILE = "validation_expect.json"


@dataclass
class Message:
    severity: str
    code: str
    text: str
    #: Seconds into the scene (from the first frame), when the finding has a place.
    t0: float | None = None
    t1: float | None = None
    #: Listed in validation_expect.json: reported, but no longer counted.
    expected: bool = False

    def as_dict(self, scene: str | None) -> dict:
        out = {"scene": scene, "code": self.code, "severity": self.severity, "msg": self.text}
        if self.t0 is not None:
            out["t0"] = round(self.t0, 3)
            out["t1"] = round(self.t1 if self.t1 is not None else self.t0, 3)
        if self.expected:
            out["expected"] = True
        return out


@dataclass
class SceneReport:
    path: Path
    messages: list[Message] = field(default_factory=list)
    stats: dict[str, object] = field(default_factory=dict)
    extras: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.name

    def add(self, severity: str, code: str, text: str, t0: float | None = None, t1: float | None = None) -> None:
        self.messages.append(Message(severity, code, text, t0, t1))

    def _texts(self, severity: str) -> list[str]:
        return [m.text for m in self.messages if m.severity == severity and not m.expected]

    @property
    def errors(self) -> list[str]:
        return self._texts(ERROR)

    @property
    def warnings(self) -> list[str]:
        return self._texts(WARNING)

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
        found = [child for child in image_format.scenes(path) if looks_like_scene(child)]
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
        report.add(
            ERROR,
            "INDEX_GAP",
            f"{label}: filenames are not a contiguous 00000..{len(files) - 1:05d} run "
            f"(first gaps: {missing or 'unexpected names'})",
        )


def _check_timestamps(report: SceneReport, ts: np.ndarray, label: str) -> None:
    if ts.ndim != 1:
        report.add(ERROR, "TS_SHAPE", f"{label}: expected one timestamp per line, got shape {ts.shape}")
        return
    if not np.all(np.isfinite(ts)):
        report.add(ERROR, "TS_NONFINITE", f"{label}: contains non-finite values")
        return
    diffs = np.diff(ts)
    if np.any(diffs < 0):
        report.add(ERROR, "TS_ORDER", f"{label}: not monotonic ({int(np.sum(diffs < 0))} decreasing steps)")
    n_dup = int(np.sum(diffs == 0))
    if n_dup:
        report.add(WARNING, "TS_DUPLICATE", f"{label}: {n_dup} duplicate timestamps")


def _is_rotation(R: np.ndarray, tol: float = 1e-3) -> bool:
    if R.shape[-2:] != (3, 3):
        return False
    identity = np.einsum("...ij,...kj->...ik", R, R)
    return bool(np.allclose(identity, np.eye(3), atol=tol) and np.all(np.abs(np.linalg.det(R) - 1.0) < tol))


def _load_stack(
    report: SceneReport, scene_dir: Path, name: str, shape: tuple[int, int], counts: dict[str, int]
) -> np.ndarray | None:
    """``name`` as an ``(N, *shape)`` stack, counted; None when absent, unreadable or misshapen.

    Non-finite values are reported here but the stack is still returned.
    """
    path = scene_dir / name
    if not path.is_file():
        return None
    try:
        stack = np.load(path)
    except Exception as exc:  # noqa: BLE001
        report.add(ERROR, "UNREADABLE", f"{name}: unreadable ({type(exc).__name__}: {exc})")
        return None
    if stack.ndim != 3 or stack.shape[1:] != shape:
        report.add(ERROR, "SHAPE", f"{name}: expected (N, {shape[0]}, {shape[1]}), got {stack.shape}")
        return None
    counts[name] = int(stack.shape[0])
    if not np.all(np.isfinite(stack)):
        report.add(ERROR, "NONFINITE", f"{name}: contains non-finite values")
    return stack


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
            report.add(ERROR, "CLOUD_FORMAT", f"clouds/{path.name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        dtypes.add(str(cloud.dtype))
        if cloud.ndim != 2 or cloud.shape[1] != 4:
            report.add(ERROR, "CLOUD_FORMAT", f"clouds/{path.name}: expected (N, 4), got {cloud.shape}")
            continue
        if cloud.dtype.kind != "f":
            report.add(ERROR, "CLOUD_FORMAT", f"clouds/{path.name}: expected a float dtype, got {cloud.dtype}")
            continue
        n_points.append(int(cloud.shape[0]))
        if cloud.size:
            nan_counts += int(np.count_nonzero(~np.isfinite(cloud)))
            total_values += int(cloud.size)

    if dtypes:
        report.stats["cloud_dtype"] = "/".join(sorted(dtypes))
        if len(dtypes) > 1:
            report.add(WARNING, "CLOUD_DTYPE", f"clouds: mixed dtypes {sorted(dtypes)}")
        elif not dtypes <= {"float32", "float64"}:
            # float32 is what the extractor writes; float64 is accepted too.
            report.add(WARNING, "CLOUD_DTYPE", f"clouds: unexpected dtype {sorted(dtypes)[0]}")
    if n_points:
        report.stats["points_mean"] = int(np.mean(n_points))
        report.stats["points_min"] = int(np.min(n_points))
    if total_values:
        report.stats["nan_frac"] = nan_counts / total_values
    report.stats["clouds_sampled"] = int(len(idx))


def _npy_rows(path: Path) -> int | None:
    """Rows of a ``.npy`` file, without reading its data; None if unreadable (or truncated)."""
    try:
        shape = np.load(path, mmap_mode="r").shape
    except (OSError, ValueError):
        return None
    return int(shape[0]) if shape else None


def check_cloud_sizes(report: SceneReport, clouds: Sequence[Path]) -> None:
    """Every cloud's point count, read from headers only (sampling misses the odd empty frame)."""
    rows = [_npy_rows(p) for p in clouds]
    bad = [p.name for p, n in zip(clouds, rows, strict=True) if n is None]
    if bad:
        report.add(
            ERROR, "CLOUD_FORMAT", f"clouds: {len(bad)} files with no readable .npy header ({', '.join(bad[:5])})"
        )
    sizes = np.array([n for n in rows if n is not None])
    if not len(sizes):
        return
    report.stats["points_min_all"] = int(sizes.min())
    empty = [p.stem for p, n in zip(clouds, rows, strict=True) if n == 0]
    sparse = int(np.count_nonzero((sizes > 0) & (sizes < MIN_POINTS)))
    report.stats["clouds_empty"] = len(empty)
    if empty:
        report.add(WARNING, "CLOUD_EMPTY", f"clouds: {len(empty)} empty ({', '.join(empty[:10])})")
    if sparse:
        report.add(INFO, "CLOUD_SPARSE", f"clouds: {sparse} with fewer than {MIN_POINTS} points")


def check_frame_gaps(report: SceneReport, data_ts: np.ndarray) -> None:
    """Holes in the frame clock (INFO: a sensor went quiet, or the guard dropped frames)."""
    if data_ts.size < 2:
        return
    dt = np.diff(data_ts)
    gaps = np.flatnonzero(dt > MAX_GAP_S)
    report.stats["max_frame_gap_s"] = float(dt.max())
    report.stats["hole_time_s"] = float(dt[gaps].sum())
    for i in gaps:
        a, b = float(data_ts[i] - data_ts[0]), float(data_ts[i + 1] - data_ts[0])
        report.add(INFO, "FRAME_GAP", f"data_ts.txt: no frames for {b - a:.1f} s, from {a:.1f} s", a, b)


#: Proprioception streams, and how much a stream that does not cover the frames matters.
PROPRIO = (("joint_states.npy", WARNING), ("joint_setpoints.npy", WARNING), ("cmd_vel.npy", INFO))


def check_proprio(report: SceneReport, scene_dir: Path, data_ts: np.ndarray | None) -> None:
    """Proprioception streams are wall-clock stamped, sorted, and cover the frames.

    ``cmd_vel`` exists only while autonomy drives, so its coverage findings are INFO.
    """
    for name, severity in PROPRIO:
        path = scene_dir / name
        if not path.is_file():
            continue
        try:
            arr = np.load(path)
        except Exception as exc:  # noqa: BLE001
            report.add(ERROR, "UNREADABLE", f"{name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        if arr.ndim != 2 or arr.shape[0] == 0:
            report.add(ERROR, "SHAPE", f"{name}: expected (M, K) rows, got {arr.shape}")
            continue
        t = arr[:, 0]
        unset = t < MIN_VALID_STAMP
        if unset.any():
            report.add(
                WARNING,
                "PROPRIO_T0",
                f"{name}: {int(unset.sum())} rows stamped before 2001 (never-set or boot-clock stamps, "
                f"first {t[unset].min():.3f} s)",
            )
        t = t[~unset]
        if len(t) and np.any(np.diff(t) < 0):
            report.add(WARNING, "PROPRIO_ORDER", f"{name}: timestamps are not sorted")
        if data_ts is None or not data_ts.size or not len(t):
            continue
        f0, f1 = float(data_ts[0]), float(data_ts[-1])
        late, early = t[0] - f0, f1 - t[-1]
        # The guard keeps frames up to a second past the pose stream, so allow some slack.
        if late > MAX_SHORT_S or early > MAX_SHORT_S:
            parts = ([f"starts {late:.1f} s after the first frame"] if late > MAX_SHORT_S else []) + (
                [f"ends {early:.1f} s before the last"] if early > MAX_SHORT_S else []
            )
            report.add(severity, "PROPRIO_COVERAGE", f"{name}: {', '.join(parts)}")
        inside = t[(t >= f0) & (t <= f1)]
        dt = np.diff(inside)
        for i in np.flatnonzero(dt > MAX_GAP_S):
            a, b = float(inside[i] - f0), float(inside[i + 1] - f0)
            n = int(np.count_nonzero((data_ts > inside[i]) & (data_ts < inside[i + 1])))
            if n:
                report.add(
                    severity,
                    "PROPRIO_GAP",
                    f"{name}: {b - a:.1f} s without a row from {a:.1f} s, under {n} frames",
                    a,
                    b,
                )


def check_derived(report: SceneReport, scene_dir: Path, n_frames: int | None) -> None:
    """Derived per-frame products name existing frames and are not stale."""
    for name in DERIVED_DIRS:
        directory = scene_dir / name
        if not directory.is_dir():
            continue
        files = sorted(p for p in directory.iterdir() if p.is_file())
        stems = [p.stem for p in files]
        bad = [s for s in stems if not s.isdigit() or (n_frames is not None and int(s) >= n_frames)]
        report.stats[f"{name}_files"] = len(files)
        if bad:
            report.add(
                ERROR, "DERIVED_INDEX", f"{name}/: {len(bad)} files name no frame of this scene ({', '.join(bad[:5])})"
            )
        if files:
            sample = files[len(files) // 2]
            try:
                loaded = np.load(sample)
                if hasattr(loaded, "close"):  # an .npz: np.load reads only the zip directory
                    for key in loaded.files:
                        loaded[key]
                    loaded.close()
            except Exception as exc:  # noqa: BLE001
                report.add(ERROR, "UNREADABLE", f"{name}/{sample.name}: unreadable ({type(exc).__name__}: {exc})")
    stale = sorted(p.name for p in scene_dir.glob("*.stale-*") if p.is_dir())
    if stale:
        report.add(INFO, "DERIVED_STALE", f"set aside by a re-extraction, rebuild them: {', '.join(stale)}")


def check_provenance(report: SceneReport, scene_dir: Path) -> None:
    """Stats from gnss_guard.txt and time_align.txt."""
    guard = scene_dir / "gnss_guard.txt"
    if guard.is_file():
        rows = [ln for ln in guard.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
        report.stats["guard_spans"] = len(rows)
    align = scene_dir / "time_align.txt"
    if not align.is_file():
        if (scene_dir / "odometry.txt").is_file() or (scene_dir / "extract_meta.json").is_file():
            report.add(INFO, "TIME_ALIGN", "time_align.txt is absent: the frame clock was left as recorded")
        return
    deltas = []
    for line in align.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and not line.startswith("#"):
            with contextlib.suppress(ValueError):
                deltas.append(float(fields[1]))
    if deltas:
        report.stats["time_align_s"] = float(np.median(deltas))


def check_odometry(
    report: SceneReport,
    transforms: np.ndarray,
    max_jump: float,
    ts: np.ndarray | None = None,
) -> None:
    positions = transforms[:, :3, 3]
    if len(positions) < 2:
        return
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    report.stats["max_step_m"] = float(np.max(steps))
    report.stats["path_len_m"] = float(np.sum(steps))
    n_jumps = int(np.sum(steps > max_jump))
    if n_jumps:
        report.add(
            INFO,
            "ODO_JUMP",
            f"transforms.npy: {n_jumps} frame-to-frame jumps > {max_jump} m " f"(max {np.max(steps):.2f} m)",
        )

    # Speed, not distance, flags poses from another moment (0.9 m at 16 Hz is 14 m/s).
    # Same gate as write_ground_truth's segment breaks.
    if ts is None or len(ts) != len(positions) or not np.isfinite(ts).all():
        return
    dt = np.diff(ts)
    moving = dt > 0  # a repeated stamp is no speed at all
    if not moving.any():
        return
    speed = steps[moving] / dt[moving]
    report.stats["max_speed_mps"] = float(np.max(speed))
    n_fast = int(np.sum(speed > MAX_SPEED))
    if n_fast:
        report.add(
            WARNING,
            "ODO_SPEED",
            f"transforms.npy: {n_fast} frame-to-frame steps imply > {MAX_SPEED} m/s "
            f"(max {np.max(speed):.1f} m/s) -- a pose from another moment, not motion",
        )


def check_tilt(report: SceneReport, gravity: np.ndarray, max_tilt_deg: float) -> None:
    tilt = tilt_deg(gravity)
    report.stats["tilt_mean_deg"] = float(np.mean(tilt))
    report.stats["tilt_max_deg"] = float(np.max(tilt))
    n_extreme = int(np.sum(tilt > max_tilt_deg))
    if n_extreme:
        report.add(
            WARNING,
            "TILT",
            f"gravity_align.npy: {n_extreme} frames tilted > {max_tilt_deg} deg " f"(max {np.max(tilt):.1f} deg)",
        )


def check_extrinsic(report: SceneReport, path: Path) -> None:
    """``T_cam_to_base.txt``: one finite SE(3) matrix."""
    try:
        T = np.loadtxt(path)
    except Exception as exc:  # noqa: BLE001
        report.add(ERROR, "EXTRINSIC", f"{path.name}: unreadable ({type(exc).__name__}: {exc})")
        return
    if T.shape != (4, 4):
        report.add(ERROR, "EXTRINSIC", f"{path.name}: expected 4x4, got {T.shape}")
    elif not np.isfinite(T).all():
        report.add(ERROR, "EXTRINSIC", f"{path.name}: contains non-finite values")
    elif not _is_rotation(T[:3, :3]) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        report.add(ERROR, "EXTRINSIC", f"{path.name}: not a rigid transform")


def check_ground_truth(report: SceneReport, scene_dir: Path) -> None:
    """``odometry.txt`` and ``odometry_segments.txt`` agree: counts match, no
    overlap, each segment starts at identity; poses outside segments are a warning."""
    odometry = scene_dir / "odometry.txt"
    segments = scene_dir / "odometry_segments.txt"
    if not odometry.is_file():
        if segments.is_file():
            report.add(ERROR, "GT_MISSING", "odometry_segments.txt is present but odometry.txt is not")
        return
    try:
        ts, poses = read_trajectory(odometry)
    except (OSError, ValueError) as exc:
        report.add(ERROR, "UNREADABLE", f"odometry.txt: unreadable ({type(exc).__name__}: {exc})")
        return
    if ts.size == 0:
        report.add(WARNING, "GT_EMPTY", "odometry.txt: no poses")
        return
    report.stats["gt_poses"] = int(ts.size)
    if not np.isfinite(poses).all():
        report.add(ERROR, "NONFINITE", "odometry.txt: contains non-finite values")
        return
    if not _is_rotation(poses[:, :3, :3]):
        report.add(ERROR, "NOT_ROTATION", "odometry.txt: rotation blocks are not orthonormal")
    if not segments.is_file():
        return

    try:
        rows = read_segments(segments)
    except OSError as exc:
        report.add(ERROR, "UNREADABLE", f"odometry_segments.txt: unreadable ({type(exc).__name__}: {exc})")
        return
    if not rows:
        report.add(WARNING, "GT_EMPTY", "odometry_segments.txt: no segments")
        return
    report.stats["gt_segments"] = len(rows)
    if len({r.index for r in rows}) != len(rows):
        report.add(ERROR, "GT_SEG_DUP", "odometry_segments.txt: segment numbers are not unique")
    ordered = sorted(rows, key=lambda r: r.start_ts)
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        if cur.start_ts < prev.end_ts - TS_TOL:
            report.add(ERROR, "GT_SEG_OVERLAP", f"odometry_segments.txt: segments {prev.index} and {cur.index} overlap")
    covered = np.zeros(ts.size, dtype=bool)
    for row in rows:
        inside = (ts >= row.start_ts - TS_TOL) & (ts <= row.end_ts + TS_TOL)
        covered |= inside
        n = int(inside.sum())
        if row.n_poses and n != row.n_poses:
            report.add(
                ERROR,
                "GT_SEG_COUNT",
                f"segment {row.index}: n_poses is {row.n_poses} but odometry.txt has {n} poses in its span",
            )
        if n and not np.allclose(poses[inside][0], np.eye(4), atol=1e-4):
            report.add(
                WARNING,
                "GT_SEG_ANCHOR",
                f"segment {row.index}: first pose is not the identity (segments are re-anchored)",
            )
    outside = int((~covered).sum())
    if outside:
        report.add(WARNING, "GT_OUTSIDE", f"odometry.txt: {outside} poses lie outside every segment")


def check_gps(report: SceneReport, scene_dir: Path, data_ts: np.ndarray | None) -> None:
    """``gps.npy`` is well-formed, if present. Warnings only: GPS is an optional extra."""
    path = scene_dir / "gps.npy"
    if not path.is_file():
        return
    try:
        gps = np.load(path)
    except Exception as exc:  # noqa: BLE001
        report.add(WARNING, "GPS", f"gps.npy: unreadable ({type(exc).__name__}: {exc})")
        return
    if gps.ndim != 2 or gps.shape[0] == 0 or gps.shape[1] < 4:
        report.add(WARNING, "GPS", f"gps.npy: expected (M, >=4) [t, lat, lon, alt, ...], got {gps.shape}")
        return

    report.stats["gps_fixes"] = int(gps.shape[0])
    ts, lat, lon = gps[:, 0], gps[:, 1], gps[:, 2]
    if not np.all(np.diff(ts) >= 0):
        report.add(WARNING, "GPS", "gps.npy: timestamps are not sorted")
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        report.add(WARNING, "GPS", "gps.npy: contains non-finite coordinates")
    elif np.abs(lat).max() > 90.0 or np.abs(lon).max() > 180.0:
        report.add(WARNING, "GPS", "gps.npy: coordinates outside [-90, 90] / [-180, 180] degrees")
    if gps.shape[1] >= 5:
        no_fix = float((gps[:, 4] < 0).mean())
        report.stats["gps_no_fix_frac"] = no_fix
        if no_fix > 0.5:
            report.add(WARNING, "GPS", f"gps.npy: {no_fix:.0%} of the fixes report no fix at all")
    if data_ts is not None and data_ts.size and np.isfinite(ts).all():
        overlap = min(ts[-1], data_ts[-1]) - max(ts[0], data_ts[0])
        if overlap <= 0:
            report.add(WARNING, "GPS", "gps.npy: its time span does not overlap data_ts.txt")


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

    # One image format per scene, the root's.
    try:
        image_ext = image_format.of_scene(scene_dir)
    except image_format.ImageFormatError as exc:
        report.add(ERROR, "IMAGE_FORMAT", str(exc))
        have = image_format.held(scene_dir)
        image_ext = max(have, key=have.get) if have else image_format.DEFAULT
    report.stats["image_format"] = image_ext.lstrip(".")
    images = _indexed_files(scene_dir / "images", image_ext)
    clouds = _indexed_files(scene_dir / "clouds", ".npy")
    for label, files, directory in (
        ("images", images, scene_dir / "images"),
        ("clouds", clouds, scene_dir / "clouds"),
    ):
        if not directory.is_dir():
            report.add(ERROR, "MISSING_DIR", f"{label}/ is missing")
            continue
        if not files:
            report.add(ERROR, "EMPTY_DIR", f"{label}/ is empty")
            continue
        counts[label] = len(files)
        _check_index_continuity(report, files, label)

    for name in REQUIRED_FILES:
        if not (scene_dir / name).is_file():
            report.add(ERROR, "MISSING_FILE", f"{name} is missing")

    for name in OPTIONAL_FILES:
        if (scene_dir / name).is_file():
            report.extras.append(name)
    for name in DERIVED_DIRS:
        if (scene_dir / name).is_dir():
            report.extras.append(f"{name}/")

    timestamps: dict[str, np.ndarray] = {}
    for name in ("data_ts.txt", "pcd_ts.txt"):
        path = scene_dir / name
        if not path.is_file():
            continue
        try:
            ts = np.atleast_1d(np.loadtxt(path))
        except Exception as exc:  # noqa: BLE001
            report.add(ERROR, "UNREADABLE", f"{name}: unreadable ({type(exc).__name__}: {exc})")
            continue
        timestamps[name] = ts
        counts[name] = int(ts.shape[0])
        _check_timestamps(report, ts, name)

    if {"data_ts.txt", "pcd_ts.txt"} <= timestamps.keys():
        data_ts, pcd_ts = timestamps["data_ts.txt"], timestamps["pcd_ts.txt"]
        if data_ts.shape == pcd_ts.shape and data_ts.size:
            # data_ts is the cam/lidar midpoint, so the offset is twice the difference.
            offset = 2.0 * np.max(np.abs(data_ts - pcd_ts))
            report.stats["max_sync_offset_s"] = float(offset)
            if offset > max_sync_offset:
                report.add(
                    WARNING,
                    "SYNC_OFFSET",
                    f"camera/lidar sync offset up to {offset * 1e3:.0f} ms " f"(> {max_sync_offset * 1e3:.0f} ms)",
                )
            report.stats["duration_s"] = float(data_ts[-1] - data_ts[0])

    transforms = _load_stack(report, scene_dir, "transforms.npy", (4, 4), counts)
    if transforms is not None:
        if np.all(np.isfinite(transforms)) and not _is_rotation(transforms[:, :3, :3]):
            report.add(ERROR, "NOT_ROTATION", "transforms.npy: rotation blocks are not orthonormal")
        if not np.allclose(transforms[:, 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
            report.add(ERROR, "NOT_ROTATION", "transforms.npy: bottom row is not [0 0 0 1]")

    gravity = _load_stack(report, scene_dir, "gravity_align.npy", (3, 3), counts)
    if gravity is not None and not np.all(np.isfinite(gravity)):
        gravity = None
    elif gravity is not None and not _is_rotation(gravity):
        report.add(ERROR, "NOT_ROTATION", "gravity_align.npy: not a stack of rotation matrices")

    path = scene_dir / "camera_intrinsics.txt"
    if path.is_file():
        try:
            K = np.loadtxt(path)
            if K.shape != (3, 3):
                report.add(ERROR, "INTRINSICS", f"camera_intrinsics.txt: expected 3x3, got {K.shape}")
            elif not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
                report.add(ERROR, "INTRINSICS", "camera_intrinsics.txt: implausible focal lengths")
        except Exception as exc:  # noqa: BLE001
            report.add(ERROR, "UNREADABLE", f"camera_intrinsics.txt: unreadable ({type(exc).__name__}: {exc})")

    if (scene_dir / "T_cam_to_base.txt").is_file():
        check_extrinsic(report, scene_dir / "T_cam_to_base.txt")

    check_gps(report, scene_dir, timestamps.get("data_ts.txt"))
    check_ground_truth(report, scene_dir)
    check_proprio(report, scene_dir, timestamps.get("data_ts.txt"))
    if "data_ts.txt" in timestamps and np.isfinite(timestamps["data_ts.txt"]).all():
        check_frame_gaps(report, timestamps["data_ts.txt"])
    check_provenance(report, scene_dir)

    if counts:
        distinct = set(counts.values())
        if len(distinct) > 1:
            report.add(
                ERROR,
                "FRAME_COUNT",
                "frame counts disagree: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
            )
        report.stats["frames"] = counts.get("images", min(counts.values()))

    check_clouds(report, clouds, n_sample)
    if n_sample:
        check_cloud_sizes(report, clouds)
    check_derived(report, scene_dir, counts.get("images", counts.get("data_ts.txt")))
    if transforms is not None:
        check_odometry(report, transforms, max_jump, timestamps.get("data_ts.txt"))
    if gravity is not None:
        check_tilt(report, gravity, max_tilt_deg)

    nan_frac = report.stats.get("nan_frac")
    if isinstance(nan_frac, float) and nan_frac > max_nan_frac:
        report.add(WARNING, "CLOUD_NAN", f"clouds: {nan_frac:.2%} non-finite values (> {max_nan_frac:.2%})")

    return report


# ---------------------------------------------------------------------------
# Across the scenes of a root
# ---------------------------------------------------------------------------
def validate_root(root: Path, reports: Sequence[SceneReport]) -> list[Message]:
    """Cross-scene checks: duplicate bags, manifest drift, keyframe lists, camera mount consistency."""
    out: list[Message] = []
    by_bag: dict[str, list[str]] = {}
    for r in reports:
        if (bag := source_bag(r.path)) is not None:
            by_bag.setdefault(bag, []).append(r.name)
    for bag, scenes in sorted(by_bag.items()):
        if len(scenes) > 1:
            out.append(Message(ERROR, "ROOT_DUP_BAG", f"{bag} was extracted into {', '.join(scenes)}"))

    manifest = root / "manifest.json"
    if manifest.is_file():
        try:
            described = json.loads(manifest.read_text()).get("scenes") or {}
        except (OSError, ValueError) as exc:
            out.append(Message(WARNING, "ROOT_MANIFEST", f"manifest.json: unreadable ({exc})"))
            described = {}
        if isinstance(described, dict) and described:
            on_disk = {r.name: r for r in reports}
            for name in sorted(set(described) - set(on_disk)):
                out.append(Message(WARNING, "ROOT_MANIFEST", f"manifest.json describes {name}, which is not on disk"))
            for name in sorted(set(on_disk) - set(described)):
                out.append(Message(WARNING, "ROOT_MANIFEST", f"{name} is on disk but not in manifest.json"))
            for name in sorted(set(described) & set(on_disk)):
                entry, r = described[name], on_disk[name]
                frames = entry.get("frames") if isinstance(entry, dict) else None
                if frames is not None and r.stats.get("frames") is not None and frames != r.stats["frames"]:
                    out.append(
                        Message(
                            WARNING,
                            "ROOT_MANIFEST",
                            f"{name}: manifest.json says {frames} frames, disk has {r.stats['frames']}",
                        )
                    )
                bag = entry.get("bag") if isinstance(entry, dict) else None
                held = next((b for b, sc in by_bag.items() if name in sc), None)
                if bag and held and bag != held:
                    out.append(
                        Message(
                            WARNING, "ROOT_MANIFEST", f"{name}: manifest.json says bag {bag}, source_bag.txt {held}"
                        )
                    )

    # Keyframe lists vs the manifest's hold-out: no held-out scene in training.
    held_out: set[str] = set()
    if manifest.is_file():
        with contextlib.suppress(OSError, ValueError):
            held = json.loads(manifest.read_text()).get("held_out")
            scenes = held.get("scenes") if isinstance(held, dict) else held if isinstance(held, list) else []
            held_out = {str(x).zfill(4) if str(x).isdigit() else str(x) for x in scenes or []}
    on_disk_names = {r.name for r in reports}
    for name in LISTS:
        path = root / name
        if not path.is_file():
            continue
        try:
            listed = {line.split("/", 1)[0] for line in path.read_text().splitlines() if line.strip()}
        except OSError as exc:
            out.append(Message(WARNING, "ROOT_KEYFRAMES", f"{name}: unreadable ({exc})"))
            continue
        unknown = sorted(listed - on_disk_names)
        if unknown:
            out.append(Message(WARNING, "ROOT_KEYFRAMES", f"{name} names scene(s) not on disk: {', '.join(unknown)}"))
        if name == "keyframes_train.lst" and (leak := sorted(listed & held_out)):
            out.append(
                Message(
                    ERROR,
                    "ROOT_KEYFRAMES",
                    f"{name} lists held-out scene(s) {', '.join(leak)} (manifest.json held_out.scenes); a "
                    "training run on this list would see the held-out session",
                )
            )
        if name == "keyframes_heldout.lst" and (extra := sorted(listed - held_out)):
            out.append(
                Message(
                    WARNING,
                    "ROOT_KEYFRAMES",
                    f"{name} lists scene(s) the manifest does not hold out: {', '.join(extra)}",
                )
            )

    mounts = {}
    for r in reports:
        path = r.path / "T_cam_to_base.txt"
        if path.is_file():
            with contextlib.suppress(ValueError, OSError):
                T = np.loadtxt(path)
                if T.shape == (4, 4) and np.isfinite(T).all():
                    mounts[r.name] = T[:3, 3]
    if len(mounts) > 1:
        names = sorted(mounts)
        xyz = np.array([mounts[n] for n in names])
        off = np.linalg.norm(xyz - np.median(xyz, axis=0), axis=1)
        if off.max() > MAX_EXTRINSIC_M:
            worst = names[int(off.argmax())]
            out.append(
                Message(
                    WARNING,
                    "ROOT_EXTRINSIC",
                    f"T_cam_to_base.txt differs between scenes by up to {off.max() * 1000:.1f} mm from the median "
                    f"(worst {worst}); the camera mount is one physical thing",
                )
            )
    return out


def load_expected(path: Path) -> set[tuple[str, str]]:
    """``validation_expect.json``: ``{"expected": [[scene or "*", code], ...]}``."""
    data = json.loads(path.read_text())
    return {(str(scene), str(code)) for scene, code in data.get("expected", [])}


def apply_expected(reports: Sequence[SceneReport], expected: set[tuple[str, str]]) -> list[Message]:
    """Mark the listed findings expected; return INFO for entries nothing matched."""
    used: set[tuple[str, str]] = set()
    for r in reports:
        for m in r.messages:
            if m.severity == INFO:
                continue
            for key in ((r.name, m.code), ("*", m.code)):
                if key in expected:
                    m.expected = True
                    used.add(key)
    return [
        Message(INFO, "EXPECT_UNUSED", f"{EXPECT_FILE}: [{scene}, {code}] matched nothing")
        for scene, code in sorted(expected - used)
    ]


def write_json(path: str, reports: Sequence[SceneReport], root_messages: Sequence[Message], strict: bool) -> None:
    doc = {
        "scenes": {
            r.name: {
                "status": r.status(strict),
                "stats": r.stats,
                "messages": [m.as_dict(r.name) for m in r.messages],
            }
            for r in reports
        },
        "root": [m.as_dict(None) for m in root_messages],
    }
    text = json.dumps(doc, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n"
    if path == "-":
        sys.stdout.write(text)
    else:
        Path(path).write_text(text)


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
    for m in report.messages:
        if m.expected:
            if verbose:
                print(f"    {INFO}: {m.text}  [{m.code}, expected {m.severity}]")
        elif m.severity != INFO or verbose:
            print(f"    {m.severity}: {m.text}  [{m.code}]")
    if verbose and report.extras:
        print(f"    extras: {', '.join(report.extras)}")


def write_table(reports: Sequence[SceneReport], path: str, strict: bool, markdown: bool) -> None:
    rows = [[_cell(r, c, strict) for c in CSV_COLUMNS] for r in reports]
    with contextlib.nullcontext(sys.stdout) if path == "-" else Path(path).open("w", newline="") as handle:
        if markdown:
            handle.write("| " + " | ".join(CSV_COLUMNS) + " |\n")
            handle.write("| " + " | ".join("---" for _ in CSV_COLUMNS) + " |\n")
            handle.writelines("| " + " | ".join(row) + " |\n" for row in rows)
        else:
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
        help="Odometry step reported (as INFO) above this [m]",
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
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        metavar="PATH",
        help="Write every message with its code, severity and time span, and every scene's stats ('-' = stdout)",
    )
    parser.add_argument(
        "--expect",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Findings known and accepted, as [[scene or '*', code], ...] under 'expected'; they are reported as "
        f"INFO. Default: {EXPECT_FILE} in a root given on the command line, if there is one",
    )
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only print the summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Also list INFO messages and extra files")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        scenes = find_scenes([Path(p) for p in args.paths])
    except FileNotFoundError as exc:
        print(f"{ERROR}: {exc}", file=sys.stderr)
        return 2

    reports = [
        validate_scene(
            scene,
            n_sample=args.sample_clouds,
            max_jump=args.max_jump,
            max_tilt_deg=args.max_tilt,
            max_nan_frac=args.max_nan,
            max_sync_offset=args.max_sync,
        )
        for scene in scenes
    ]

    roots = [Path(p) for p in args.paths if not looks_like_scene(Path(p))]
    root_messages: list[Message] = []
    for root in roots:
        root_messages += validate_root(root, [r for r in reports if r.path.parent == root])
    expect = args.expect or next((r / EXPECT_FILE for r in roots if (r / EXPECT_FILE).is_file()), None)
    if expect is not None:
        try:
            root_messages += apply_expected(reports, load_expected(expect))
        except (OSError, ValueError, TypeError) as exc:
            print(f"{ERROR}: {expect}: {exc}", file=sys.stderr)
            return 2

    if not args.quiet:
        for report in reports:
            print_report(report, args.strict, args.verbose)
        for m in root_messages:
            if m.severity != INFO or args.verbose:
                print(f"root       {m.severity}: {m.text}  [{m.code}]")
    if args.csv:
        write_table(reports, args.csv, args.strict, args.markdown)
    if args.json:
        write_json(args.json, reports, root_messages, args.strict)

    failed = [r for r in reports if r.status(args.strict) == "FAIL"]
    warned = [r for r in reports if r.status(args.strict) == "WARN"]
    total_frames = sum(int(r.stats.get("frames", 0) or 0) for r in reports)
    print(
        f"\n{len(reports)} scene(s), {total_frames} frames: "
        f"{len(reports) - len(failed) - len(warned)} ok, {len(warned)} with warnings, "
        f"{len(failed)} failed"
    )
    root_failed = any(m.severity == ERROR or (args.strict and m.severity == WARNING) for m in root_messages)
    return 1 if failed or root_failed else 0


if __name__ == "__main__":
    sys.exit(main())
