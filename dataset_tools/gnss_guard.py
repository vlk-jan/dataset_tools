"""Keeping a reset GNSS/INS solution out of the exported poses.

When a Fixposition loses its fix -- and losing it is exactly what a roll-over
does, because the antennas end up pointing at the ground -- it does not stop
publishing. It keeps the odometry topic running at full rate and fills it with
the reset state: a position hundreds of metres from where the robot is, held
perfectly still, with the attitude snapped back to identity. The recording has
no flag on those messages. They look like poses.

Left alone they reach the export twice over. ``odometry.txt`` breaks a segment
at the jump, which is the right shape but still carries the fabricated side as
a segment of its own; ``transforms.npy`` samples the raw stream and has no
segment table at all, so a frame lands 1.2 km from the park. That is how scene
0010 of HELHEST v2 beta came to report a 1249 m step between two frames 64 ms
apart, and 0014 a pair of 1130 m round trips.

What gives it away
------------------
Four signatures, none of which is attitude. Only ``island`` convicts on its
own; the rest have to agree with each other, because each of them has an
innocent explanation the others do not share.

``teleport``
    A step implying more than ``MAX_SPEED`` across a sample interval short
    enough to be a real interval (``MAX_GAP``). This marks the *edge* -- it
    says one of the two sides is fabricated without saying which, so it is
    never a span on its own; it fences the ``island`` rule below.

``island``
    A run fenced in by a teleport at both ends and over in a few seconds.
    Nothing leaves a place and returns to it at four figures of metres per
    second; scene 0014 does it twice, 1130 m out and back inside 3 s.

``frozen``
    A run of samples whose translation does not move at all, for at least
    ``MIN_FREEZE``. This is the discriminator for *which side* of a jump is the
    fabricated one -- the reset state is a single position repeated, and the
    two fabricated stretches of HELHEST v2 beta hold still for 2.9 s and
    46.5 s.

    It is never enough on its own. A robot can be genuinely stopped, and a
    Fixposition quantises to a micrometre, so a stopped robot publishes the
    same number for seconds too: scene 0009 does it three times while the
    driver leans on a robot that will not move, with a continuous fix, the
    scene's best covariance and a GNSS latitude and longitude every bit as
    still as the odometry. So a frozen run is believed only where a
    ``teleport`` or a ``no_fix`` agrees with it.

``no_fix``
    The receiver's own admission, read as a hole in the stream of fixes that
    carried a position. A Fixposition with no solution may publish
    ``NavSatFix`` at null island, which
    :func:`~dataset_tools.extract.gps.gps_row` drops, or it may simply stop
    publishing; 0014 does both, 3 s of null island around one of its jumps and
    a 12.7 s silence around the other. Reading the hole rather than the flavour
    of the hole catches them with one rule.

    Not a span by itself: an inertial solution coasts through a short outage
    perfectly well, and throwing away a dead-reckoned stretch that is merely
    drifting would cost good data for nothing. It becomes one when the position
    also jumped during it. Neither half is enough -- an outage is coasting, a
    lone jump says one of its two sides is wrong without saying which -- but
    together they are conclusive, and that pairing is the only thing that finds
    0014's remaining 3.29 m step at 53 m/s, which nothing froze and no second
    jump fenced. An outage also widens a span the other rules found, out to the
    full length of what the receiver admits it missed.

Deliberately not attitude
-------------------------
The tempting fourth signature is the 180 deg flip, and it is a trap: the robot
really does roll over, and when it does the flip is the measurement, not the
fault. Scene 0010 spends 56 s upside down *before* the solution resets, and
that stretch is good data. Nothing here reads the rotation.

What happens to a span
----------------------
Its samples are dropped before anything downstream sees them, so
``write_ground_truth`` and ``interpolate_poses`` agree about what exists, and
the hole surfaces as a segment break rather than as a pose. Frames whose stamps
fall inside it are not exported at all: a frame the dataset cannot put a pose
on has nothing to offer a consumer of a pose-annotated dataset, and the
alternative -- nearest-neighbour onto the last good sample -- is a frozen robot
under a moving image, which is worse for being plausible.

Frames with no pose under them
------------------------------
That last sentence is not only true of a span. ``interpolate_poses`` falls back
to the nearest sample wherever it cannot interpolate -- outside the stream's
range, or across a hole longer than its bracket gate -- and the result is the
same fabrication by a different author: a stationary pose under a frame the
robot was driving through. It reaches ``transforms.npy`` looking exactly like a
measurement. Scene 0014 exports a 3.29 m step between two frames 62 ms apart,
53 m/s, with no fabricated pose anywhere near it; both frames simply sit in a
12 s hole in the odometry and borrow the same distant sample.

So :func:`frame_support` applies the same test to the whole frame clock: a
frame is exported only where real samples bracket it closely enough to
interpolate between. This costs whatever the pose stream does not cover --
1,127 frames of HELHEST v2 beta, 1.1%, two thirds of them 0016's last 57 s,
where the receiver stops publishing while the camera runs on. Those frames are
not so much lost as revealed: they never had a pose, only a plausible one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Speed, m/s, above which a step between two samples is not motion. The one
#: definition: :func:`~dataset_tools.extract.ground_truth.write_ground_truth`,
#: :func:`~dataset_tools.extract.sync.interpolate_poses` and
#: :mod:`dataset_tools.validate_scene` take it from here, so they all agree
#: about which steps are real.
#:
#: It was 4.0 until the 2026-09-28 recordings, where the robot is sprinted at
#: up to 5.35 m/s -- the receiver's own twist, the wheels and the Odin1's
#: odometry all agree -- and every sample of a sprint read as a teleport. The
#: slowest fabricated step on record is 53 m/s, so there is room above.
MAX_SPEED = 6.0

#: Sample interval, s, beyond which a step is a hole in the stream rather than
#: a jump in the solution. Driving through a 60 s dropout covers ground
#: honestly; that is not a teleport and must not read as one.
MAX_GAP = 1.0

#: Translation, m, below which two consecutive samples count as the same
#: position. The reset state repeats bit-identically; this leaves room for a
#: float32 round trip without admitting any real dither.
FREEZE_STEP = 1e-6

#: Seconds a position must hold perfectly still to be the reset state. Three
#: times the longest genuine still run measured across HELHEST v2 beta.
MIN_FREEZE = 1.0

#: Seconds. A run of samples fenced in by a teleport at *both* ends and shorter
#: than this is an excursion, not a visit: nothing leaves a place and returns
#: to it at four figures of metres per second.
MIN_ISLAND = 5.0

#: Seconds without a usable fix that count as the receiver having lost its
#: solution. The fix rate is 10 Hz, so this is two orders of magnitude clear of
#: a dropped message.
MIN_OUTAGE = 2.0

#: Seconds. The widest bracket :func:`~dataset_tools.extract.sync.interpolate_poses`
#: will blend across; wider than that it copies the nearest sample instead, and
#: what it copies is not a measurement of the frame it lands on. This has to
#: track that function's ``max_gap`` default, which is why it is not MAX_GAP --
#: that one is about what counts as a hole in the stream, this one about what
#: counts as a pose for a frame.
SUPPORT_GAP = 0.5

#: Seconds of margin around a span when deciding which frames it costs. The
#: frame clock has not been corrected for capture-to-publish latency yet at the
#: point this runs (see :mod:`dataset_tools.time_align`), so a frame stamp may
#: sit up to a latency away from the pose clock the span is measured in.
FRAME_PAD = 1.0


@dataclass
class Span:
    """A stretch of the pose stream that the receiver fabricated."""

    start: float
    end: float
    reason: str

    def __contains__(self, t: float) -> bool:
        return self.start <= t <= self.end


@dataclass
class Verdict:
    """What the guard found, and what it cost."""

    spans: list[Span] = field(default_factory=list)
    n_poses_dropped: int = 0
    n_frames_dropped: int = 0
    n_poses: int = 0
    n_frames: int = 0
    #: Frames dropped for having no pose under them, over and above the ones a
    #: span cost. Kept apart from ``n_frames_dropped`` because the two have
    #: different causes and a scene can have either without the other.
    n_frames_unsupported: int = 0
    #: ``n_frames_unsupported`` by where the frames lay: before the first
    #: surviving pose, after the last, or inside a hole. A sensor that kept
    #: recording after the robot side stopped shows up as ``after``; a hole in
    #: the frames themselves never shows up here at all (there are no frames).
    n_unsupported_before: int = 0
    n_unsupported_after: int = 0
    n_unsupported_inside: int = 0
    #: Frames dropped after the clock correction, when the corrected stamp
    #: turned out to have no pose bracketing it (see :func:`frame_support`).
    n_frames_trimmed: int = 0

    @property
    def clean(self) -> bool:
        return not self.spans and not self.n_frames_unsupported

    def summary(self, t0: float) -> str:
        parts = []
        if self.spans:
            where = ", ".join(f"{s.start - t0:.1f}-{s.end - t0:.1f}s ({s.reason})" for s in self.spans)
            parts.append(
                f"{len(self.spans)} fabricated stretch(es) dropped -- {where}; "
                f"{self.n_poses_dropped}/{self.n_poses} poses and "
                f"{self.n_frames_dropped}/{self.n_frames} frames removed"
            )
        if self.n_frames_unsupported:
            parts.append(f"{self.n_frames_unsupported}/{self.n_frames} frames dropped for having no pose under them")
        return "; ".join(parts) if parts else "no fabricated poses"


# ---------------------------------------------------------------------------
# The four signatures
# ---------------------------------------------------------------------------
def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal half-open index ranges where ``flags`` is true."""
    edges = np.flatnonzero(np.diff(np.r_[0, flags.astype(np.int8), 0]))
    return list(zip(edges[0::2], edges[1::2], strict=True))


def teleport_edges(ts: np.ndarray, pos: np.ndarray):
    """Indices ``i`` whose step to ``i + 1`` is faster than anything can move.

    A step across a hole longer than ``MAX_GAP`` is excluded: the stream said
    nothing for that long, and whatever distance the robot covered meanwhile is
    its own business.
    """
    if len(ts) < 2:
        return np.empty(0, dtype=int)
    dt = np.diff(ts)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.flatnonzero((dt > 0) & (dt <= MAX_GAP) & (step > MAX_SPEED * dt))


def frozen_spans(ts: np.ndarray, pos: np.ndarray) -> list[Span]:
    """Stretches where the translation does not move at all for long enough."""
    if len(ts) < 2:
        return []
    still = np.linalg.norm(np.diff(pos, axis=0), axis=1) <= FREEZE_STEP
    out = []
    for a, b in _runs(still):
        # `still[k]` describes the step from sample k to k + 1, so the run of
        # samples holding that position is a..b inclusive.
        if ts[b] - ts[a] >= MIN_FREEZE:
            out.append(Span(float(ts[a]), float(ts[b]), "frozen"))
    return out


def island_spans(ts: np.ndarray, edges: np.ndarray) -> list[Span]:
    """Short stretches fenced in by a teleport at both ends."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        a, b = int(lo) + 1, int(hi)  # samples strictly between the two jumps
        if a <= b and ts[b] - ts[a] < MIN_ISLAND:
            out.append(Span(float(ts[a]), float(ts[b]), "island"))
    return out


def outage_spans(fix_ts) -> list[Span]:
    """Stretches with no usable fix, read off the stamps of the usable ones.

    ``fix_ts`` are the stamps of the fixes that carried a position -- what
    reaches ``gps.npy``. Everything the receiver does when it has no solution
    shows up here as the same thing, a hole: it may publish ``NavSatFix`` at
    null island, which :func:`~dataset_tools.extract.gps.gps_row` drops, or it
    may simply stop publishing. Scene 0014 of HELHEST v2 beta does both -- 3 s
    of null island around one of its jumps and a 12.7 s silence around the
    other -- so reading the hole rather than the flavour of the hole is what
    catches them with one rule.

    Only holes *between* fixes count. A stream that starts late or stops early
    is the ordinary shape of a recording -- 0009 waits 11.7 s for its first fix
    and 0016 spends its last 57 s without one -- and an inertial solution
    coasting through the end of a scene is degraded, not fabricated.
    """
    t = np.sort(np.asarray(fix_ts, dtype=float))
    if len(t) < 2:
        return []
    return [Span(float(t[i]), float(t[i + 1]), "no_fix") for i in np.flatnonzero(np.diff(t) > MIN_OUTAGE)]


# ---------------------------------------------------------------------------
# Putting them together
# ---------------------------------------------------------------------------
def _merge(spans: list[Span]) -> list[Span]:
    """Union of overlapping spans, keeping every reason that contributed."""
    if not spans:
        return []
    out: list[Span] = []
    for s in sorted(spans, key=lambda s: s.start):
        if out and s.start <= out[-1].end:
            last = out[-1]
            reasons = sorted({*last.reason.split("+"), *s.reason.split("+")})
            out[-1] = Span(last.start, max(last.end, s.end), "+".join(reasons))
        else:
            out.append(Span(s.start, s.end, s.reason))
    return out


def _outages_with_a_jump(ts: np.ndarray, edges: np.ndarray, outages: list[Span]) -> list[Span]:
    """Outages during which the position also did something impossible.

    Neither half is enough alone. An outage on its own is an inertial solution
    coasting, and it coasts well. A jump on its own says one of its two sides
    is wrong without saying which, and after it the stream may look perfectly
    ordinary -- scene 0014 leaves a 3.29 m step at 53 m/s sitting inside an
    otherwise unremarkable stretch, with nothing frozen and no second jump to
    fence it. Together they are conclusive: the receiver has said it had no
    solution, and the position has moved further than anything can move.
    """
    if not len(edges) or not outages:
        return []
    out = []
    for o in outages:
        jumped = any(o.start <= ts[int(i)] <= o.end or o.start <= ts[int(i) + 1] <= o.end for i in edges)
        if jumped:
            out.append(Span(o.start, o.end, "no_fix+teleport"))
    return out


def _corroborated(span: Span, ts: np.ndarray, edges: np.ndarray, outages: list[Span], fix_ts=None) -> bool:
    """Is there a second reason to believe this frozen run is fabricated?

    On its own a frozen position is not enough, because a robot can be
    genuinely stopped -- and a Fixposition quantises to a micrometre, so a
    robot that is stopped really does publish the same number for seconds.
    Scene 0009 of HELHEST v2 beta does exactly that for three runs of 1.3 to
    2.5 s while the driver is commanding 1.3 m/s into a robot that is not
    moving: the fix is continuous, the covariance is its best of the scene
    (0.0002 m^2) and the GNSS latitude and longitude are just as still as the
    odometry. Nothing was fabricated there; the robot was stuck.

    So a frozen run counts only when something independent says the reported
    position is not where the robot is: a jump into or out of it that nothing
    can move that fast, or the receiver's own admission that it had no
    solution. Both hold for every stretch this was written for -- 0010's
    freeze opens with a 1249 m jump, 0014's is fenced by a pair of 1130 m
    jumps, and the receiver publishes null island through both.
    """
    if any(span.start <= o.end and o.start <= span.end for o in outages):
        return True
    # No usable fix anywhere under the run: the receiver never claimed a
    # position there. ``outages`` cannot say so when the run sits before the
    # first fix, after the last, or in a scene with none at all -- the 2026-09-28
    # bag robotour_merged_2026_09_28-16_52_16 publishes the origin with an
    # identity attitude for all of its 147 s, and null island on every fix.
    if fix_ts is not None and not np.any((fix_ts >= span.start) & (fix_ts <= span.end)):
        return True
    inside = np.flatnonzero((ts >= span.start) & (ts <= span.end))
    if not len(inside):
        return False
    first, last = int(inside[0]), int(inside[-1])
    return bool((first - 1) in edges or last in edges)


def find_spans(ts, poses, fix_ts=None) -> list[Span]:
    """Every stretch of a pose stream that the receiver fabricated.

    ``ts`` and ``poses`` are the sorted odometry stream; ``fix_ts`` are the
    stamps of the GNSS fixes that carried a position, and may be None on a rig
    that publishes none -- the guard then has only the pose stream to go on.

    The guard fails towards keeping data. A fabricated stretch that arrives
    without a jump and that the receiver never admits to is not found here --
    but a position cannot reach a reset origin hundreds of metres away without
    jumping there, so in practice that combination does not arise, and the
    alternative is deleting the frames of a robot that was merely stuck.
    """
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return []
    pos = np.asarray(poses, dtype=float)[:, :3, 3]
    edges = teleport_edges(ts, pos)
    outages = [] if fix_ts is None else outage_spans(fix_ts)
    fixes = None if fix_ts is None else np.asarray(fix_ts, dtype=float)

    frozen = [s for s in frozen_spans(ts, pos) if _corroborated(s, ts, edges, outages, fixes)]
    found = frozen + island_spans(ts, edges) + _outages_with_a_jump(ts, edges, outages)
    if not found:
        return []
    spans = _merge(found)

    # The receiver's own no-solution stretches, but only where a fabricated
    # span already reaches into one: an outage on its own is a solution
    # coasting, which is worth keeping.
    for outage in outages:
        if any(s.start <= outage.end and outage.start <= s.end for s in spans):
            spans = _merge([*spans, outage])
    return spans


def drop_poses(odom_data, spans: list[Span]):
    """``odom_data`` with every sample inside a span removed."""
    return [row for row in odom_data if not any(row[0] in s for s in spans)]


def frame_mask(frame_ts: np.ndarray, spans: list[Span]) -> np.ndarray:
    """True for the frames that keep a trustworthy pose."""
    keep = np.ones(len(frame_ts), dtype=bool)
    for s in spans:
        keep &= ~((frame_ts >= s.start - FRAME_PAD) & (frame_ts <= s.end + FRAME_PAD))
    return keep


def frame_support(frame_ts, odom_ts, pad: float = FRAME_PAD) -> np.ndarray:
    """True for the frames that real pose samples bracket closely enough.

    False where :func:`~dataset_tools.extract.sync.interpolate_poses` would
    have no choice but to copy the nearest sample: before the stream starts,
    after it ends, or inside a hole wider than ``SUPPORT_GAP``. ``FRAME_PAD`` is
    the same allowance :func:`frame_mask` makes for the frame clock not yet being
    corrected for capture-to-publish latency, and it is spent towards keeping
    the frame -- a frame is dropped only when no stamp it could correct to has
    a pose either, so a hole has to exceed ``SUPPORT_GAP + 2 * FRAME_PAD`` to
    cost anything.

    Only the stamps are read. The other way ``interpolate_poses`` falls back --
    a bracket implying an impossible speed -- is left alone here: after the
    spans are gone what remains of it is rare, and the pose on either side is a
    real one.

    With ``pad=0`` the stamps are taken as final: run on the corrected clock,
    it finds the frames that would still be handed a copied pose.
    """
    t = np.asarray(frame_ts, dtype=float)
    o = np.sort(np.asarray(odom_ts, dtype=float))
    if not len(o):
        return np.zeros(len(t), dtype=bool)
    lo, hi = t - pad, t + pad
    keep = (hi >= o[0]) & (lo <= o[-1])
    for i in np.flatnonzero(np.diff(o) > SUPPORT_GAP):
        keep &= ~((lo > o[i]) & (hi < o[i + 1]))
    return keep


def guard(odom_data, frame_ts: np.ndarray, fix_ts=None) -> tuple[list, np.ndarray, Verdict]:
    """Drop the fabricated poses, and the frames they leave unsupported.

    Returns the surviving odometry, the keep-mask over ``frame_ts``, and what
    it cost. The odometry must be sorted by time.
    """
    verdict = Verdict(n_poses=len(odom_data), n_frames=len(frame_ts))
    if not odom_data:
        return list(odom_data), np.ones(len(frame_ts), dtype=bool), verdict
    ts = np.array([row[0] for row in odom_data], dtype=float)
    poses = np.array([row[1] for row in odom_data], dtype=float)
    frames = np.asarray(frame_ts, dtype=float)

    verdict.spans = find_spans(ts, poses, fix_ts)
    kept = drop_poses(odom_data, verdict.spans)
    mask = frame_mask(frames, verdict.spans)
    verdict.n_poses_dropped = len(odom_data) - len(kept)
    verdict.n_frames_dropped = int(np.count_nonzero(~mask))

    # Support is measured against what survives, so the holes the spans just
    # opened are covered by the same rule as the ones the receiver left.
    support = _support(verdict, frames, [row[0] for row in kept], mask)
    return kept, mask & support, verdict


def _support(verdict: Verdict, frames: np.ndarray, kept_ts: list, mask: np.ndarray) -> np.ndarray:
    """:func:`frame_support` over ``kept_ts``, with the frames it costs (among
    those ``mask`` still keeps) counted into ``verdict`` by where they lay."""
    support = frame_support(frames, kept_ts)
    lost = mask & ~support
    verdict.n_frames_unsupported = int(np.count_nonzero(lost))
    if kept_ts:
        verdict.n_unsupported_before = int(np.count_nonzero(lost & (frames < kept_ts[0])))
        verdict.n_unsupported_after = int(np.count_nonzero(lost & (frames > kept_ts[-1])))
    else:
        verdict.n_unsupported_before = verdict.n_frames_unsupported
    verdict.n_unsupported_inside = (
        verdict.n_frames_unsupported - verdict.n_unsupported_before - verdict.n_unsupported_after
    )
    return support


def coverage_guard(odom_data, frame_ts: np.ndarray) -> tuple[list, np.ndarray, Verdict]:
    """The support half of :func:`guard` alone, for a pose source that has no
    reset solution to detect (the Odin1's odometry relocalises in jumps, which
    the segment gate turns into breaks; nothing in it is fabricated). Frames
    before the stream, after it or inside a hole still go, for the same
    reason as under the full guard."""
    verdict = Verdict(n_poses=len(odom_data), n_frames=len(frame_ts))
    frames = np.asarray(frame_ts, dtype=float)
    if not odom_data:
        return list(odom_data), np.ones(len(frames), dtype=bool), verdict
    support = _support(verdict, frames, [row[0] for row in odom_data], np.ones(len(frames), dtype=bool))
    return list(odom_data), support, verdict


def write_report(path, verdict: Verdict, t0: float) -> None:
    """Record what the guard did, so a scene can be audited after the fact.

    One line per span: when it started and ended relative to the first frame,
    and which signature caught it.
    """
    lines = [
        "# stretches of fabricated GNSS/INS pose removed from this scene",
        "# start_s end_s reason   (seconds into the scene)",
        f"# {verdict.n_poses_dropped}/{verdict.n_poses} poses, "
        f"{verdict.n_frames_dropped}/{verdict.n_frames} frames removed",
        f"# {verdict.n_frames_unsupported}/{verdict.n_frames} further frames had no pose under them "
        f"({verdict.n_unsupported_before} before the pose stream, {verdict.n_unsupported_after} after it, "
        f"{verdict.n_unsupported_inside} inside holes)",
        f"# after the clock correction, {verdict.n_frames_trimmed}/{verdict.n_frames} more frames "
        "had no pose bracketing them, dropped as well",
    ]
    lines += [f"{s.start - t0:.3f} {s.end - t0:.3f} {s.reason}" for s in verdict.spans]
    Path(path).write_text("\n".join(lines) + "\n")
