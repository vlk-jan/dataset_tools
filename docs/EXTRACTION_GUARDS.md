# What the extractor corrects for

Three things in a raw bag would put a wrong pose under a frame without any
error being raised: a sensor unit that stamps its data late, a GNSS/INS that
keeps publishing after it has lost its fix, and frames the pose stream never
covers. `helhest-extract` handles each during extraction (`--time_align auto`,
`--gnss_guard auto`, both the default; on the default `/odin1/odometry` the
guard runs only its coverage part, the reset-span part is for Fixposition
topics); this is what they look for, why, and
where they fall short. The README's *Known caveats* summarise them.

## Publish latency (`--time_align`)

**The sensor unit may stamp at publish, not at capture.** On the 2026-09
Odin1 recordings the poses run ahead of the imagery *and* of the clouds by
**~0.69 s** -- 0.5 m of smear at walking pace, and the reason the viewer
showed the odometry arriving somewhere before the camera did.

It is not a clock error, and no clock on the robot is wrong:
`scripts/probe_clock_skew.py` reads `recv - header` for every message and
gets +0.049 s for the camera, +0.020 s for the lidar, -0.004 s for the
Fixposition and +0.001 s for `/joint_states`, none of them stepping anywhere
in the bag. What that difference cannot see is the pipeline *behind* the
stamp: the unit stamps when it publishes, and its pixels and points are older
than that by however long its own processing took.

A latency follows load, so it does not hold still -- one 290 s recording
spends 80 s at ~0.05 s and the rest at ~0.69 s, and others have none at all. There is no per-rig or even per-bag number to configure, so
`--time_align auto` measures it **during extraction**, as a curve, and takes
it off. Every scene carries the measurement it was corrected by in
`time_align.txt`.

What measures it is the sharpness of the aggregated cloud, not scan matching:
ICP could not do it, because these clouds are dominated by ground and low
vegetation and point-to-point slides to zero rotation. Counting the voxels a
window's clouds fill fits no transform at all, so it cannot slide, and the
answer it gives does not change between a 0.3 m/s window and a 1.7 m/s one --
which is what separates a latency from a spatial error such as a wrong lidar
mount. A window is believed on the depth of its minimum against its own
shoulders rather than on how far that minimum beats zero, so a stretch that
is *already* right counts as evidence for zero instead of falling silent and
inheriting its neighbour's offset.

## Reset GNSS/INS solutions (`--gnss_guard`)

**A GNSS/INS with no fix keeps publishing poses.** When the Fixposition loses
its solution -- and rolling the robot over is exactly how it loses it, with
the antennas pointing at the ground -- the odometry topic does not stop. It
fills with the reset state: a position hundreds of metres from where the
robot is, held perfectly still, attitude snapped back to identity, and no
flag on any of it. One recording reported a **1249 m step between two
frames 64 ms apart** that way, another a pair of 1130 m round trips.

`--gnss_guard auto` removes those stretches during extraction, before an
image or a cloud is written, so the frames that survive are the frames the
dataset can put a pose on and the indices stay contiguous. What marks the
fabricated side is that the translation stops moving *at all* -- the reset
state is one position repeated, held for 2.9 s and 46.5 s in those two
cases. That is never enough on its own, because a robot can be genuinely
stopped and a Fixposition quantises to a micrometre, so a stopped robot
publishes the same number for seconds too: one recording does it three times
while the driver leans on a robot that will not move, with a continuous fix,
the scene's best covariance and a GNSS latitude and longitude every bit as
still as the odometry. A freeze is therefore believed only where something
independent agrees -- a jump faster than 6 m/s into or out of it (the real
ones imply four figures of metres per second), a hole in the stream of fixes
that carried a position, or no usable fix under it at all. A
short island fenced in by a jump at both ends is an excursion rather than a
visit, and stands on its own. So does an outage with a jump inside it: an
outage alone is a solution coasting, a lone jump says one of its two sides is
wrong without saying which, but together they are conclusive. That last rule
has had no catch of its own -- the one span seen was already convicted by
its island -- but it is the only one that would reach a fabricated stretch
which neither held still long enough nor came back.

The outage is read as a *hole*, not as a flag, because the receiver reports
losing its fix two different ways and one recording uses both: 3 s of null-island
`NavSatFix` around one jump, and 12.7 s of publishing nothing at all around
the other. Only holes between fixes count -- a stream that starts late or
stops early is the ordinary shape of a recording (one waits 11.7 s for its
first fix, another spends its last 57 s without one), and coasting is degraded,
not fabricated.

The guard fails towards keeping data: a fabricated stretch that arrives
without a jump and that the receiver never admits to would not be found. A
position cannot reach a reset origin hundreds of metres away without jumping
there, so that combination does not arise in practice, and the alternative is
deleting the frames of a robot that was merely stuck.

What the guard deliberately does *not* read is the attitude. The 180 deg flip
is the tempting signature and it is a trap: the robot really does roll over,
and when it does, the flip is the measurement. One recording spends 56 s upside
down **before** its solution resets, and that stretch is good data.

Each scene records what it lost in `gnss_guard.txt`, and the hole reads as
`fix_lost` in `odometry_segments.txt`.

## Frames with no pose under them (`--gnss_guard`)

**Frames with no pose under them.** A fabricated pose does not need a
fabricating receiver. `interpolate_poses` falls back to the nearest sample
wherever it cannot interpolate -- before the pose stream starts, after it
ends, or across a hole wider than its 0.5 s bracket -- and what comes out is a
stationary pose under a frame the robot was driving through, written into
`transforms.npy` looking exactly like a measurement. One recording
exported a **3.29 m step between two frames 62 ms apart, 53 m/s**, with no
fabricated pose anywhere near it: both frames sat in a 12 s hole in the
odometry and borrowed the same distant sample.

The same `--gnss_guard` stage drops them, on the same principle that governs
a span -- a frame the dataset cannot put a real pose on has nothing to offer
a consumer of a pose-annotated dataset. The margin is spent the other way
round from a span, though: a span drops a frame that *might* fall inside it,
support keeps a frame that *might* sit on a pose, so a hole has to exceed
2 s (1 s of margin each side) before it costs anything. In one measured set of recordings this was 1,127 frames, 1.1%,
two thirds of them one recording's last 57 s, where the receiver stops publishing while
the camera runs on. With it, no scene in the root exports a step above
3.2 m/s, and every one of those is driving.

That margin is there because the guard runs before the clock correction. Once
the clock is final, a frame that still has no pose on either side of it is
dropped too and the indices close up behind it (`gnss_guard.txt`: "after the
clock correction").
