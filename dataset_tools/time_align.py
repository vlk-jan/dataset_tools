"""Put a scene's frames on the same clock as its poses.

Sensors that stamp at publish time are late by their pipeline latency (~0.69 s
on the 2026-09 Odin1 rig), and that latency varies with load, so the correction
is a piecewise-linear profile measured from the written clouds: per window, the
offset ``delta`` that minimises the voxels filled by the clouds placed at
``frame_ts + delta``. Unlike ICP it cannot slide, and it does not scale with
speed, which separates a latency from a mount error. Only windows with a deep
enough minimum (``MIN_DEPTH``, which also accepts a sharp minimum at zero)
become knots; with none, the clock is left alone.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_tools.transforms import interpolate_poses, mat_to_quat, poses_from_quat

#: Voxel edge, m: separates a 0.1 s error at walking pace, coarser than lidar noise.
VOXEL = 0.12
#: Frames per window (~6 s at 14 Hz). The velocity must change within it:
#: at constant velocity a clock shift moves the map without blurring it.
WINDOW = 80
#: Range band, m: nearer is the robot itself, further is too sparse.
R_MIN, R_MAX = 1.0, 15.0
#: Height band, m, relative to the sensor.
Z_MIN, Z_MAX = -0.5, 3.0
#: A window must move and turn to constrain the offset.
MIN_SPEED, MIN_TURN, MAX_SPEED = 0.4, 0.05, 5.0
#: Minimum depth (%) against its own shoulders for a window to be believed.
#: Unlike gain, this credits a sharp minimum at zero (an already-aligned stretch).
MIN_DEPTH = 18.0
#: Gain (% of the count at delta = 0 removed) above which a window is "strong"; reported only.
STRONG_GAIN = 25.0
#: Knot median within this of zero is noise about an aligned clock (half the fine step).
ZERO_TOL = 0.010
#: Larger offsets are failed measurements, not latencies.
MAX_ABS_OFFSET = 2.0
#: Window spacing, s, and the per-scene cap.
TARGET_STEP_S, MAX_WINDOWS = 3.0, 120
#: Frames in the retry of a too-shallow window: more travel, more velocity variation.
LONG_WINDOW = 280


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------
@dataclass
class PoseTrack:
    """A pose track prepared once, so a sweep does not re-derive it per delta."""

    ts: np.ndarray
    poses: np.ndarray
    quat: np.ndarray

    @classmethod
    def build(cls, ts: np.ndarray, poses: np.ndarray) -> PoseTrack:
        order = np.argsort(ts)
        ts, poses = np.asarray(ts, float)[order], np.asarray(poses, float)[order]
        return cls(ts, poses, mat_to_quat(poses[:, :3, :3]))

    def at(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pose at each query time: lerp in translation, SLERP in rotation."""
        pos, quat, _ = interpolate_poses(self.ts, self.poses, query, quats=self.quat)
        return pos, poses_from_quat(pos, quat)[:, :3, :3]


def load_clouds(scene_dir: Path, frames: np.ndarray) -> list[np.ndarray]:
    """The window's clouds, cropped to the band that carries the geometry, every second point."""
    out = []
    for f in frames:
        path = scene_dir / "clouds" / f"{int(f):05d}.npy"
        if not path.is_file():
            continue
        c = np.load(path)[:, :3].astype(np.float64)
        r = np.hypot(c[:, 0], c[:, 1])
        keep = (r > R_MIN) & (r < R_MAX) & (c[:, 2] > Z_MIN) & (c[:, 2] < Z_MAX)
        out.append(c[keep][::2])
    return out


def occupied(clouds: list[np.ndarray], pos: np.ndarray, rot: np.ndarray) -> int:
    """Voxels the window fills once every cloud is put in the world frame."""
    pts = np.vstack([c @ rot[k].T + pos[k] for k, c in enumerate(clouds)])
    g = np.floor(pts / VOXEL).astype(np.int64)
    g -= g.min(axis=0)
    key = (g[:, 0] * np.int64(1 << 21) + g[:, 1]) * np.int64(1 << 21) + g[:, 2]
    key.sort()  # np.unique, without the copy and the mask it does not need here
    return int(1 + np.count_nonzero(key[1:] != key[:-1]))


def sweep(clouds: list[np.ndarray], track: PoseTrack, query: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    counts = np.empty(len(deltas))
    for n, d in enumerate(deltas):
        pos, rot = track.at(query + d)
        counts[n] = occupied(clouds, pos, rot)
    return counts


def refine(deltas: np.ndarray, counts: np.ndarray) -> float:
    """The minimum to sub-step precision, by a parabola through its neighbours."""
    i = int(np.argmin(counts))
    if 0 < i < len(counts) - 1:
        a, b, c = counts[i - 1], counts[i], counts[i + 1]
        den = a - 2 * b + c
        if den > 0:
            return float(deltas[i] + 0.5 * (deltas[1] - deltas[0]) * (a - c) / den)
    return float(deltas[i])


def measure_window(
    scene_dir: Path,
    frames: np.ndarray,
    track: PoseTrack,
    frame_ts: np.ndarray,
    span: float = 1.5,
) -> dict | None:
    """One window's offset: a coarse sweep, then a fine one around its minimum."""
    clouds = load_clouds(scene_dir, frames)
    if len(clouds) < len(frames) // 2:
        return None
    query = frame_ts[frames]
    coarse = np.arange(-span, span + 1e-9, 0.10)
    cc = sweep(clouds, track, query, coarse)
    centre = coarse[int(np.argmin(cc))]
    fine = np.arange(centre - 0.12, centre + 0.121, 0.02)
    fc = sweep(clouds, track, query, fine)
    at_zero = float(cc[int(np.argmin(np.abs(coarse)))])
    best = refine(fine, fc)
    return {
        "frame_start": int(frames[0]),
        "t": float(frame_ts[frames].mean()),
        "delta": best,
        "gain": float(100.0 * (1.0 - fc.min() / at_zero)) if at_zero > 0 else 0.0,
        "depth": _depth(coarse, cc, fine, fc, best),
    }


def _depth(coarse: np.ndarray, cc: np.ndarray, fine: np.ndarray, fc: np.ndarray, best: float) -> float:
    """How far (%) the count climbs 0.3-0.6 s away from the minimum, wherever it sits."""
    off = np.abs(coarse - best)
    shoulder = cc[(off >= 0.3) & (off <= 0.6)]
    if not len(shoulder) or fc.min() <= 0:
        return 0.0
    return float(100.0 * (1.0 - fc.min() / np.median(shoulder)))


def pick_windows(frame_ts: np.ndarray, track: PoseTrack, limit: int) -> list[np.ndarray]:
    """Windows every ``TARGET_STEP_S`` seconds where the robot moved and turned enough, at most ``limit``."""
    pos, rot = track.at(frame_ts)
    yaw = np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
    step = np.r_[0.0, np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 1]))]
    dt = float(np.median(np.diff(frame_ts))) if len(frame_ts) > 1 else 0.1
    stride = max(1, int(round(TARGET_STEP_S / max(dt, 1e-3))))

    scored = []
    for start in range(0, len(frame_ts) - WINDOW, stride):
        span = frame_ts[start + WINDOW - 1] - frame_ts[start]
        if span <= 0:
            continue
        speed = step[start : start + WINDOW].sum() / span
        turn = abs(yaw[start + WINDOW - 1] - yaw[start])
        if MIN_SPEED <= speed <= MAX_SPEED and turn >= MIN_TURN:
            scored.append(start)
    if not scored:
        return []
    # Thin evenly across the whole scene, not by truncation (that leaves the tail unmeasured).
    if len(scored) > limit:
        scored = [scored[i] for i in np.unique(np.linspace(0, len(scored) - 1, limit).astype(int))]
    return [np.arange(s, s + WINDOW) for s in scored]


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------
@dataclass
class Profile:
    """The offset to add to a scene's frame clock, as a function of that clock."""

    windows: list[dict]
    knot_ts: np.ndarray
    knot_delta: np.ndarray
    note: str

    @property
    def is_identity(self) -> bool:
        return len(self.knot_ts) == 0

    @property
    def summary(self) -> str:
        if self.is_identity:
            return f"no correction ({self.note})"
        lo, hi = self.knot_delta.min(), self.knot_delta.max()
        mid = float(np.median(self.knot_delta))
        shape = f"{mid:+.3f} s" if hi - lo < 0.05 else f"{lo:+.3f} .. {hi:+.3f} s, varying"
        return f"{shape} from {len(self.knot_ts)}/{len(self.windows)} windows ({self.note})"

    def at(self, ts: np.ndarray) -> np.ndarray:
        """The offset at each time; flat beyond the outermost knot."""
        if self.is_identity:
            return np.zeros(len(ts))
        return np.interp(ts, self.knot_ts, self.knot_delta)

    def rows(self, t0: float) -> list[tuple[float, float, float, float]]:
        """``(seconds into the scene, delta, gain, depth)`` for every window."""
        return [(w["t"] - t0, w["delta"], w["gain"], w["depth"]) for w in self.windows]


def _retry_if_shallow(
    scene_dir: Path, w: dict, track: PoseTrack, frame_ts: np.ndarray, span: float, n_frames: int
) -> dict:
    """Re-measure a too-shallow window over ``LONG_WINDOW`` frames; keep it only if deeper."""
    if w["depth"] >= MIN_DEPTH or n_frames < LONG_WINDOW:
        return w
    centre = w["frame_start"] + WINDOW // 2
    start = int(np.clip(centre - LONG_WINDOW // 2, 0, n_frames - LONG_WINDOW))
    wide = measure_window(scene_dir, np.arange(start, start + LONG_WINDOW), track, frame_ts, span)
    if wide is None or wide["depth"] <= w["depth"]:
        return w
    wide["t"] = w["t"]  # it still speaks for the moment it was centred on
    wide["frame_start"] = w["frame_start"]
    return wide


def _knots(windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """The believable windows, median-filtered against single-window outliers."""
    strong = [w for w in windows if w["depth"] >= MIN_DEPTH and abs(w["delta"]) <= MAX_ABS_OFFSET]
    if not strong:
        return np.zeros(0), np.zeros(0)
    ts = np.array([w["t"] for w in strong])
    d = np.array([w["delta"] for w in strong])
    order = np.argsort(ts)
    ts, d = ts[order], d[order]
    if len(d) >= 3:  # a 3-tap median: one wild window cannot bend the curve
        d = np.array([np.median(d[max(0, k - 1) : k + 2]) for k in range(len(d))])
    return ts, d


def estimate(
    scene_dir: Path,
    frame_ts: np.ndarray,
    odom_ts: np.ndarray,
    odom_poses: np.ndarray,
    limit: int = MAX_WINDOWS,
    span: float = 1.5,
) -> Profile:
    """Measure the frame clock's offset from the pose clock; identity when nothing localises."""
    if len(odom_ts) < 2 or len(frame_ts) < WINDOW:
        return Profile([], np.zeros(0), np.zeros(0), "too little to measure against")
    track = PoseTrack.build(odom_ts, odom_poses)
    frames = pick_windows(frame_ts, track, limit)
    if not frames:
        return Profile([], np.zeros(0), np.zeros(0), "no window with enough motion")

    def one(f):
        w = measure_window(scene_dir, f, track, frame_ts, span)
        return w and _retry_if_shallow(scene_dir, w, track, frame_ts, span, len(frame_ts))

    # The windows are independent and numpy drops the GIL in the heavy parts
    # (threads matched a process pool); map keeps their order, so the result
    # does not depend on the thread count.
    with ThreadPoolExecutor(getattr(os, "process_cpu_count", os.cpu_count)()) as pool:
        windows = [w for w in pool.map(one, frames) if w]
    ts, d = _knots(windows)
    if not len(ts):
        typical = float(np.median([w["gain"] for w in windows])) if windows else 0.0
        return Profile(windows, ts, d, f"no window localises a minimum, median window gain {typical:.1f}%")
    if already_aligned(windows, d):
        return Profile(windows, np.zeros(0), np.zeros(0), "already aligned; the windows' minima sit at zero")
    return Profile(windows, ts, d, "measured")


def already_aligned(windows: list[dict], knot_delta: np.ndarray) -> bool:
    """Whether the knots scatter *around* zero (median within ``ZERO_TOL``).

    A small but consistent offset is still a measurement.
    """
    if not len(knot_delta) or any(w["gain"] >= STRONG_GAIN for w in windows):
        return False
    return abs(float(np.median(knot_delta))) < ZERO_TOL and bool(np.all(np.abs(knot_delta) < 0.05))


def apply_to_clock(frame_ts: np.ndarray, profile: Profile) -> tuple[np.ndarray, bool]:
    """The corrected clock (profile evaluated at the uncorrected stamps), and whether
    a bad measurement forced it back to monotonic."""
    out = np.asarray(frame_ts, dtype=float) + profile.at(frame_ts)
    forced = bool(len(out) > 1 and np.any(np.diff(out) <= 0))
    if forced:
        # nextafter: at a 1.7e9 epoch a harmless epsilon is below one ulp.
        out = np.maximum.accumulate(out)
        for k in range(1, len(out)):
            if out[k] <= out[k - 1]:
                out[k] = np.nextafter(out[k - 1], np.inf)
    return out, forced


def write_report(path: Path, profile: Profile, t0: float) -> None:
    """Write every measured window, for auditing."""
    lines = [
        "# Frame-clock offset measured against the odometry (dataset_tools.time_align).",
        f"# {profile.summary}",
        "# t_into_scene_s  delta_s  gain_pct  depth_pct  (a window becomes a knot "
        f"when depth >= {MIN_DEPTH:.0f} and |delta| <= {MAX_ABS_OFFSET:.1f}; depth is",
        "# how far the count climbs away from the minimum, gain how much the minimum beats delta = 0)",
    ]
    lines += [f"{t:12.3f} {d:+9.4f} {g:8.2f} {p:9.2f}" for t, d, g, p in profile.rows(t0)]
    path.write_text("\n".join(lines) + "\n")
