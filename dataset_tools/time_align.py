"""Putting a scene's frames on the same clock as its poses.

A sensor unit that stamps at *publish* time hands out headers that are correct
about when the message was sent and wrong about when the light and the returns
were captured, by however long its own pipeline took. Nothing in the bag
records that: ``recv - header`` measures publish-to-receive, so it stays a
tidy few tens of milliseconds while the pixels are half a second old. Measured
on the 2026-09 Odin1 rig, the pipeline runs ~0.69 s behind -- and, because a
latency is a property of how loaded the unit is rather than of how a clock is
disciplined, it does **not** hold still: scene 0000 of HELHEST v2 beta spends
80 of its 290 seconds at ~0.05 s and the rest at ~0.69 s.

So the correction is a curve, not a constant, and this module measures it from
the data the extractor has just written.

What it measures against
------------------------
Put a window's clouds in the world frame with the pose read at
``frame_ts + delta`` and count the voxels they fill. A wrong ``delta``
displaces each cloud along its own direction of travel, which differs from
cloud to cloud whenever the velocity changes, so surfaces separate and the
count rises. The minimum is the offset.

This fits no transform, so unlike ICP it cannot slide -- which matters, because
these clouds are dominated by ground and low vegetation and point-to-point
slides to zero rotation. And the answer does not scale with speed, which is
what separates a latency from a spatial error such as a wrong lidar mount: a
mount 0.5 m out would come back as ``distance / speed`` and disagree between a
slow window and a fast one.

What it does with the measurements
----------------------------------
Only windows that actually localise a minimum are believed (see
``MIN_DEPTH``); the rest are silent, not zero. The believed ones become knots
of a piecewise-linear profile, median-filtered against single-window outliers
and held flat beyond the outermost knot.

The distinction that matters here is between a window with nothing to correct
and a window that cannot tell: both leave the count at ``delta = 0`` looking
fine, and they mean opposite things. A window already on the right clock has a
sharp minimum sitting at zero -- it is evidence *for* zero, and has to enter
the profile as such, or an aligned stretch contributes nothing and quietly
inherits the offset of the stretch beside it. So what qualifies a window is the
depth of its minimum against its own shoulders, which reads the same wherever
that minimum sits, and not how far the minimum beats zero. A scene where no
window localises anything is left alone rather than guessed at.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset_tools.transforms import mat_to_quat, poses_from_quat, quat_slerp

#: Voxel edge, metres. Fine enough that a 0.1 s error at walking pace separates
#: two surfaces, coarse enough that the lidar's own noise does not.
VOXEL = 0.12
#: Frames per window: ~6 s at the 2026-09 rig's 14 Hz. Long enough for the
#: velocity to change within it -- a window driven at a constant velocity
#: cannot see the offset at all, since shifting the clock then displaces every
#: cloud by the same vector, which moves the map without blurring it -- and
#: short enough to resolve a latency that moves over tens of seconds.
WINDOW = 80
#: Points nearer than this are the robot itself; further than this are too
#: sparse to land repeatedly in one voxel.
R_MIN, R_MAX = 1.0, 15.0
#: Above the sensor there is sky, below it the ground the robot stands on.
Z_MIN, Z_MAX = -0.5, 3.0
#: A stationary window has nothing to say and costs the same to measure, so
#: motion is worth requiring up front. Turning is not: it helps, but how much a
#: window constrains the offset is read directly off its own curve afterwards
#: (``MIN_DEPTH``), and screening on turn instead throws away the straight
#: stretches -- which then inherit a neighbour's offset rather than their own.
MIN_SPEED, MIN_TURN, MAX_SPEED = 0.4, 0.05, 5.0
#: How deep the minimum must be *relative to its own neighbourhood* for a
#: window to be believed. This, not the gain below, is what says a window
#: constrains the offset -- and the two say different things. A window that is
#: already on the right clock has a sharp minimum sitting at zero, so it has
#: nothing to gain and everything to tell; a window with no geometry or no
#: change of velocity has a flat curve, so it has nothing to gain and nothing
#: to tell. Judging by gain confuses the two, and then an aligned stretch
#: contributes no knot and silently inherits its neighbour's offset -- which is
#: how a scene that is right for 200 of its 290 seconds gets moved anyway.
#: Set from what the real minima look like: on HELHEST v2 beta a window that
#: agrees with its neighbours sits at 16-49 %, while the ones that wander sit
#: below 16 %. Windows that fall short are retried over more ground
#: (``LONG_WINDOW``) rather than written off, so the bar can afford to be here.
MIN_DEPTH = 18.0
#: The gain -- percent of the voxel count at ``delta = 0`` that the best delta
#: removes -- is what the correction is worth at that window, and is reported
#: rather than used as the criterion. Above this a window is called strong in
#: the summary.
STRONG_GAIN = 25.0
#: Knots whose median lies within this of zero are noise about an aligned clock,
#: not an offset: half the fine sweep's 0.02 s step.
ZERO_TOL = 0.010
#: No rig's pipeline is further behind than this; anything larger is a failed
#: measurement, not a latency, and is dropped rather than applied.
MAX_ABS_OFFSET = 2.0
#: Resolution to aim for when placing windows, and the ceiling on how many to
#: measure in one scene. The 2026-09 transitions ramp over ~18 s, so a few
#: seconds resolves them; the cap keeps a 20-minute scene affordable.
TARGET_STEP_S, MAX_WINDOWS = 3.0, 120
#: Frames in the second, longer attempt at a window that came out too shallow
#: to believe. More travel means more variation in the velocity, which is the
#: only thing that constrains the offset at all -- measured on scene 0002, a
#: stretch that read -0.333 s at depth 8.7 % over 6 s reads -0.040 s at depth
#: 16.3 % over 20 s, and the longer answer is the one the neighbours agree
#: with. Only shallow windows are retried, so the cost falls where it buys
#: something.
LONG_WINDOW = 280


# ---------------------------------------------------------------------------
# The objective
# ---------------------------------------------------------------------------
@dataclass
class PoseTrack:
    """A pose track prepared once, so a sweep does not re-derive it per delta."""

    ts: np.ndarray
    xyz: np.ndarray
    quat: np.ndarray

    @classmethod
    def build(cls, ts: np.ndarray, poses: np.ndarray) -> PoseTrack:
        order = np.argsort(ts)
        ts, poses = np.asarray(ts, float)[order], np.asarray(poses, float)[order]
        return cls(ts, poses[:, :3, 3].copy(), mat_to_quat(poses[:, :3, :3]))

    def at(self, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pose at each query time: lerp in translation, SLERP in rotation."""
        j = np.clip(np.searchsorted(self.ts, query), 1, len(self.ts) - 1)
        i = j - 1
        span = self.ts[j] - self.ts[i]
        u = np.clip(np.where(span > 0, (query - self.ts[i]) / np.where(span > 0, span, 1.0), 0.0), 0.0, 1.0)
        pos = self.xyz[i] + u[:, None] * (self.xyz[j] - self.xyz[i])
        return pos, poses_from_quat(pos, quat_slerp(self.quat[i], self.quat[j], u))[:, :3, :3]


def load_clouds(scene_dir: Path, frames: np.ndarray, stride: int = 2) -> list[np.ndarray]:
    """The window's clouds, cropped to the band that carries the geometry."""
    out = []
    for f in frames:
        path = scene_dir / "clouds" / f"{int(f):05d}.npy"
        if not path.is_file():
            continue
        c = np.load(path)[:, :3].astype(np.float64)
        r = np.hypot(c[:, 0], c[:, 1])
        keep = (r > R_MIN) & (r < R_MAX) & (c[:, 2] > Z_MIN) & (c[:, 2] < Z_MAX)
        out.append(c[keep][::stride])
    return out


def occupied(clouds: list[np.ndarray], pos: np.ndarray, rot: np.ndarray, voxel: float = VOXEL) -> int:
    """Voxels the window fills once every cloud is put in the world frame."""
    pts = np.vstack([c @ rot[k].T + pos[k] for k, c in enumerate(clouds)])
    g = np.floor(pts / voxel).astype(np.int64)
    g -= g.min(axis=0)
    key = (g[:, 0] * np.int64(1 << 21) + g[:, 1]) * np.int64(1 << 21) + g[:, 2]
    return int(np.unique(key).size)


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
    """How far the count climbs on stepping away from the minimum, in percent.

    Measured against the shoulder 0.3-0.6 s to either side rather than against
    ``delta = 0``, so it says the same thing wherever the minimum happens to
    sit -- which is the whole point of it.
    """
    off = np.abs(coarse - best)
    shoulder = cc[(off >= 0.3) & (off <= 0.6)]
    if not len(shoulder) or fc.min() <= 0:
        return 0.0
    return float(100.0 * (1.0 - fc.min() / np.median(shoulder)))


def pick_windows(frame_ts: np.ndarray, track: PoseTrack, limit: int, step_s: float = TARGET_STEP_S) -> list[np.ndarray]:
    """Windows where the robot moved and turned enough to constrain the offset.

    Placed every ``step_s`` seconds rather than back to back, so a latency that
    moves during the scene is resolved rather than averaged over; thinned to
    ``limit`` when that would be more windows than a scene is worth.
    """
    pos, rot = track.at(frame_ts)
    yaw = np.unwrap(np.arctan2(rot[:, 1, 0], rot[:, 0, 0]))
    step = np.r_[0.0, np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 1]))]
    dt = float(np.median(np.diff(frame_ts))) if len(frame_ts) > 1 else 0.1
    stride = max(1, int(round(step_s / max(dt, 1e-3))))

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
    # Thin to `limit` across the WHOLE scene. Subsampling by a stride and then
    # truncating to the first `limit` looks equivalent and is not: it walks off
    # the end of a long scene and leaves its tail unmeasured, to inherit
    # whatever the last knot happened to say. That cost HELHEST v2 beta's two
    # longest scenes ~0.6 s over their closing minutes.
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
        shape = f"{lo:+.3f} s" if hi - lo < 0.05 else f"{lo:+.3f} .. {hi:+.3f} s, varying"
        return f"{shape} from {len(self.knot_ts)}/{len(self.windows)} windows ({self.note})"

    def at(self, ts: np.ndarray) -> np.ndarray:
        """The offset at each time; flat beyond the outermost knot."""
        if self.is_identity:
            return np.zeros(len(ts))
        return np.interp(ts, self.knot_ts, self.knot_delta)

    def rows(self, t0: float) -> list[tuple[float, float, float, float]]:
        """``(seconds into the scene, delta, gain, depth)`` for every window."""
        return [(w["t"] - t0, w["delta"], w["gain"], w.get("depth", 0.0)) for w in self.windows]


def _retry_if_shallow(
    scene_dir: Path, w: dict, track: PoseTrack, frame_ts: np.ndarray, span: float, n_frames: int
) -> dict:
    """A window too shallow to believe, measured again over more ground.

    A short window that happens to be driven at a near-constant velocity says
    almost nothing, and what little it says is noise -- but the stretch is not
    unmeasurable, only under-sampled. Widening it brings in more variation in
    the velocity, which is the only thing the objective has to work with. The
    wider answer is kept only if it is actually better conditioned, so this can
    add a measurement but never replace a good one with a worse one.
    """
    if w["depth"] >= MIN_DEPTH or n_frames < LONG_WINDOW:
        return w
    centre = w["frame_start"] + WINDOW // 2
    start = int(np.clip(centre - LONG_WINDOW // 2, 0, n_frames - LONG_WINDOW))
    wide = measure_window(scene_dir, np.arange(start, start + LONG_WINDOW), track, frame_ts, span)
    if wide is None or wide["depth"] <= w["depth"]:
        return w
    wide["t"] = w["t"]  # it still speaks for the moment it was centred on
    wide["frame_start"] = w["frame_start"]
    wide["wide"] = True
    return wide


def _knots(windows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """The believable windows, median-filtered against single-window outliers."""
    strong = [w for w in windows if w.get("depth", 0.0) >= MIN_DEPTH and abs(w["delta"]) <= MAX_ABS_OFFSET]
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
    """Measure the frame clock's offset from the pose clock, over the scene.

    Returns a profile that is the identity when nothing could be measured --
    no poses, no window with enough motion, or no window whose voxel count
    moved enough to place a minimum. Correcting by a number that was never
    measured is worse than not correcting.
    """
    if len(odom_ts) < 2 or len(frame_ts) < WINDOW:
        return Profile([], np.zeros(0), np.zeros(0), "too little to measure against")
    track = PoseTrack.build(odom_ts, odom_poses)
    frames = pick_windows(frame_ts, track, limit)
    if not frames:
        return Profile([], np.zeros(0), np.zeros(0), "no window with enough motion")

    windows = [w for w in (measure_window(scene_dir, f, track, frame_ts, span) for f in frames) if w]
    windows = [_retry_if_shallow(scene_dir, w, track, frame_ts, span, len(frame_ts)) for w in windows]
    ts, d = _knots(windows)
    if not len(ts):
        typical = float(np.median([w["gain"] for w in windows])) if windows else 0.0
        return Profile(windows, ts, d, f"no window localises a minimum, median window gain {typical:.1f}%")
    if already_aligned(windows, d):
        return Profile(windows, np.zeros(0), np.zeros(0), "already aligned; the windows' minima sit at zero")
    return Profile(windows, ts, d, "measured")


def already_aligned(windows: list[dict], knot_delta: np.ndarray) -> bool:
    """Whether the believed windows say the clock is right as it stands.

    Right means the knots scatter *around* zero -- their median within
    ``ZERO_TOL`` -- not merely near it. Until 2026-10-01 this accepted any set
    of knots all below 0.05 s, which swallowed scene 0011 of HELHEST v2.2:
    four windows at +0.040..+0.048 s, each as deep as its neighbours' in other
    scenes, left uncorrected while every other scene got its +0.04. A
    consistent offset is a measurement, however small; only noise about zero
    is nothing to correct.
    """
    if not len(knot_delta) or any(w["gain"] >= STRONG_GAIN for w in windows):
        return False
    return abs(float(np.median(knot_delta))) < ZERO_TOL and bool(np.all(np.abs(knot_delta) < 0.05))


def apply_to_clock(frame_ts: np.ndarray, profile: Profile) -> tuple[np.ndarray, bool]:
    """The corrected clock, and whether it had to be forced back into order.

    The profile is evaluated at the *uncorrected* stamps, which is where it was
    measured. A latency that fell faster than time passes would fold the clock
    over; that never happens on a real one -- the 2026-09 ramps run at 0.03 s
    per second against the 1.0 that would be needed -- but a bad measurement
    could, and a frame clock that runs backwards would break every consumer
    downstream, so it is forced monotonic and the caller is told.
    """
    out = np.asarray(frame_ts, dtype=float) + profile.at(frame_ts)
    forced = bool(len(out) > 1 and np.any(np.diff(out) <= 0))
    if forced:
        # nextafter, not a fixed epsilon: one ulp at a 1.7e9 epoch is ~4e-7, so
        # any epsilon small enough to be harmless is too small to be added.
        out = np.maximum.accumulate(out)
        for k in range(1, len(out)):
            if out[k] <= out[k - 1]:
                out[k] = np.nextafter(out[k - 1], np.inf)
    return out, forced


def write_report(path: Path, profile: Profile, t0: float) -> None:
    """Every window that was measured, so the correction can be audited."""
    lines = [
        "# Frame-clock offset measured against the odometry (dataset_tools.time_align).",
        f"# {profile.summary}",
        "# t_into_scene_s  delta_s  gain_pct  depth_pct  (a window becomes a knot "
        f"when depth >= {MIN_DEPTH:.0f} and |delta| <= {MAX_ABS_OFFSET:.1f}; depth is",
        "# how far the count climbs away from the minimum, gain how much the minimum beats delta = 0)",
    ]
    lines += [f"{t:12.3f} {d:+9.4f} {g:8.2f} {p:9.2f}" for t, d, g, p in profile.rows(t0)]
    path.write_text("\n".join(lines) + "\n")
