# New rig — pre-extraction readiness check

What the 2026-09-02 HELHEST recordings contain, measured against the
[HELHEST v1.3](DATA.md) scene contract, **before** anything is extracted for
keeps. Written 2026-09-05 from nine test bags; the outing was a rig shakedown,
not dataset material, so nothing here was generated permanently.

Short version: the bags carry everything a v1.3-shaped dataset needs, the two
August blockers are gone, and one smoke extraction passes `helhest-validate`
with no warnings and no manual extrinsic. What is still missing is the **robot
model**, not sensor data.

---

## 1. The recordings

Nine `_merged` mcap bags (`subtdata:/data/helhest/26-09-02-stromovka-roads`),
2026-09-02 13:29–15:30 UTC, one city-park outing (asphalt and gravel paths,
lawns, mature trees, a viaduct). 106 topics per bag, rosbag2 metadata v9.

| Bag (`helhest_2026_09_02-`) | Length | Camera | LiDAR | FP poses | Path | RTK_FIXED | NO_FIX |
|---|---|---|---|---|---|---|---|
| `15_29_17_merged` | 7.2 min | 7 775 | 3 819 | 4 301 | 321 m | 43 % | 0.0 % |
| `15_44_42_merged` | 10.5 min | 16 809 | 8 209 | 6 296 | 369 m | 52 % | 0.0 % |
| `16_03_26_merged` | 3.5 min | 5 535 | 2 733 | 2 083 | 101 m | 0 % | 0.0 % |
| `16_16_52_merged` | 6.6 min | 10 791 | 5 311 | 3 973 | 253 m | 35 % | 0.9 % |
| `16_25_23_merged` | 1.4 min | 2 277 | 1 122 | 858 | 37 m | 37 % | 0.0 % |
| `16_27_10_merged` | 16.4 min | 27 346 | 13 482 | 9 853 | 449 m | 33 % | 2.8 % |
| `16_59_32_merged` | 9.7 min | 16 460 | 8 164 | 5 819 | 84 m | 50 % | 0.0 % |
| `17_21_39_merged` | 3.4 min | 5 261 | 2 614 | 2 037 | 140 m | 20 % | 52 % |
| `17_25_05_merged` | 5.1 min | 8 402 | 4 179 | 3 040 | 221 m | 30 % | 13 % |
| **total** | **63.9 min** | **100 656** | **49 633** | — | **1 976 m** | | |

132 GB of bags. Fix shares come from the four most common
`/fixposition/fpa/odomstatus` combinations per bag, so rows short of 100 % had a
tail of rarer ones. `init_status` is `GLOBAL_INIT` in **100 % of poses in every
bag** — the fusion was initialised throughout, which is what actually matters
(§4).

Expected extraction: **~48 950 frames** (98.6 % of LiDAR scans survive mutual
nearest-neighbour sync, measured), **~222 GB** on disk.

## 2. Topic map — old stack → v1 → new rig

| Role | v1 (Livox + OAK) | New rig (2026-09-02) | Rate |
|---|---|---|---|
| LiDAR | `/livox/lidar_front` | `/odin1/cloud_filtered/cloudini` | 13–14 Hz |
| Camera | `/luxonis/oak/rgb/image_raw/compressed` | `/odin1/image/compressed` | 27–29 Hz |
| Camera info | OAK `camera_info` | `/odin1/camera_info` | 27–29 Hz |
| Ground truth | FAST-LIO, offline | `/fixposition/odometry_enu` | 10 Hz |
| Gravity | `/ekf/imu/data` | `/fixposition/odometry_enu` | 10 Hz |
| Live odometry | `/odom` | `/odom_2d`, `/odin1/odometry` | 30 / 14 Hz |
| Proprioception | `/joint_states` | `/joint_states`, `/joint_setpoints`, `/cmd_vel` | 100 / 100 / var |

Absent by design: there is no `/odom` and no `/ekf/imu/data`. `/imu/data`
exists but its frame (`imu`) still has no TF to `base_link` — do not use it for
gravity alignment, exactly as on the August bags.

## 3. v1.3 scene contract, file by file

Measured on `16_25_23_merged` → 1106 frames, `helhest-validate`: **OK, 0
warnings**.

| v1.3 file | New rig | Detail |
|---|---|---|
| `images/` | yes | 1600×1296 BGR PNG (v1.3: 1280×800) |
| `clouds/` | yes | 25–39 k pts/frame float32 in `base_link` (v1.3: ~9 k) |
| `data_ts.txt`, `pcd_ts.txt` | yes | sync median **3.9 ms**, p95 12.3 ms, max 34.7 ms |
| `odometry.txt` | yes | 10 Hz, `--gt auto` fires on the Fixposition topic |
| `odometry_segments.txt` | yes | **one segment per scene** — no gap >1 s and no teleport in any of the nine bags |
| `transforms.npy` | see §5 | currently a copy of the GT |
| `gravity_align.npy` | yes | det = 1.000000, roll/pitch ±3° on this flat outing |
| `joint_states.npy` | yes | 100 Hz, positions **and** velocities real |
| `joint_setpoints.npy` | yes | velocity real; position/effort `±inf` **by design** (§6) |
| `cmd_vel.npy` | yes | `TwistStamped`, already handled by the extractor |
| `camera_intrinsics.txt`, `camera_info.yaml` | yes | `equidistant` fisheye, fx 730.7, cx 835.1 |
| `T_cam_to_base.txt` | yes | correct optical convention — but see the naming trap in §6 |
| `source_bag.txt`, `../scenes.csv` | yes | |
| `height_maps/`, `height_masks/` | derivable | DRIFT `build_heightmaps.py`; every input present |
| `proprio_labels/` | **blocked** | needs wheel geometry (§6) |
| `atlas/`, `manifest.json`, `README.md` | hand-authored | per dataset, as in v1.3 |
| `regen/` | n/a | FAST-LIO provenance; there is no FAST-LIO stage on this rig |

Working invocation — the defaults are already right, no `--lidar_extrinsic`:

```bash
helhest-extract <bags> <out> --jobs 8
```

## 4. What the new rig fixed

Both blockers named in the README's *Known caveats* for the 2026-08 bags are
resolved.

- **The Odin1 mount extrinsic is in `tf_static`.** `base_link → odin1_base_link
  = (0.10, 0, 0.10)`, so clouds land in `base_link` natively. The August bags
  had no such edge and fell back to the identity.
- **`/joint_states` carries positions and velocities**, not efforts alone.
- **The Fixposition child frame is reachable**: `FP_POI → FP_VRTK → base_link`,
  so ENU poses re-express at the base without a manual chain.
- **RTK_FLOAT is not a defect.** The unit needs an RTK fix once; after that it
  publishes a fusion solution over IMU, wheel speed and vision that stays stable
  and precise. `init_status = GLOBAL_INIT` holds in every pose of every bag.
  `16_03_26_merged` never sees RTK_FIXED inside its own span, but the session
  ran continuously from `15_44_42_merged`, which did. Worth recording the status
  alongside the poses for provenance (§7), not worth gating on.

### Measured sanity checks (`16_25_23_merged`, 1106 frames)

| Check | Result | Reference |
|---|---|---|
| Yaw rate vs wheel differential | **r = +0.926** | v1's published odometry was *negative* — the mirroring bug does not recur |
| Metres per radian of wheel rotation | 0.336 | v1 firm ground ~0.38 → smaller wheels |
| Ground plane, `base_link` frame | z = −0.317 m | v1 wheel radius 0.36 m |
| Camera/LiDAR sync | 3.9 ms median | v1.3 ~10 Hz frames; this rig 13.6 Hz |
| Cloud FOV | az ±65°, el ±46°, 256×192 organised | forward-only, like v1 |
| Cloud range | median 1.84 m, p95 10.9 m | denser and shorter-ranged than the Livox |

The two independent estimates of wheel radius (0.336 from odometry scale, 0.317
from the ground plane) agree and both disagree with the URDF's 0.36 m.

## 5. `transforms.npy` — use Odin, with a teleport filter

`--odom` feeds both `odometry.txt` and `transforms.npy`, so on this rig they are
the same trajectory and `transforms.npy` is not the independent live estimate
v1.3 documents. Two candidates, both compared against Fixposition over sliding
windows (`16_25_23` / `16_03_26`):

| Source | 1 s median | 2 s median | 5 s median | Scale over moving 2 s windows | Raw path vs FP |
|---|---|---|---|---|---|
| `/odin1/odometry` | 4.5 / 6.9 cm | 8.6 / 13.2 cm | 23 / 32 cm | **0.971 / 0.986** | 69.3 vs 37.1 m; 194.5 vs 100.4 m |
| `/odom_2d` | 2.8 / 4.4 cm | 4.0 / 8.0 cm | 28 / 26 cm | 1.031 / 1.044 | 37.8 vs 37.1 m; 102.9 vs 100.4 m |

**Odin's odometry is locally precise, and its inflated path length is not
drift.** The scale is right to within 3 %; the excess is a handful of discrete
relocalisation jumps:

| Bag | Poses | Steps implying >2 m/s | Metres they carry | Path after dropping them | Fixposition |
|---|---|---|---|---|---|
| `16_25_23` | 1 132 | 34 (3.0 %) | 36.5 of 69.4 m | 33.0 m | 37.1 m |
| `16_03_26` | 2 747 | 10 (0.4 %) | 101.3 of 194.8 m | 93.5 m | 100.4 m |

Ten steps out of 2 747 carry more than half the trajectory. Worst single-window
disagreement is 16.6 m (1 s) and 48.4 m — those are the jumps, nothing else.

So `/odin1/odometry` is usable for `transforms.npy` **provided it is segmented
at the teleports**, which is exactly what `write_ground_truth()`'s existing
`max_speed=4.0` break already does. Between jumps it is as good as the wheel
odometry and it is 6-DoF, which `/odom_2d` is not (`z_ptp = 0.00`). No FAST-LIO
stage is needed on this rig.

### The Fixposition fabricates poses when it loses its fix

Segmenting at the teleports is the right shape but not the whole answer, because
the teleport is not the only thing the receiver does. When it loses its solution
it keeps publishing `/fixposition/odometry_enu` at full rate and fills it with
the reset state: a position hundreds of metres away, **held perfectly still**,
attitude snapped back to identity, and nothing in the message saying so. Scene
0010 of v2 beta reported a 1249 m step between two frames 64 ms apart that way,
and 0014 a pair of 1130 m round trips.

Losing the fix is not exotic on this rig: rolling the robot over puts the
antennas against the ground, and the robot does roll over. So `--gnss_guard
auto` removes those stretches during extraction, before an image or a cloud is
written, and the frames left without a pose go with them.

What marks the fabricated side of a jump is that the translation stops moving
*at all* — the reset state is one position repeated, held for 2.9 s and 46.5 s
in the two cases here.

That signature is never trusted alone. A robot can be genuinely stopped, and the
Fixposition quantises to a micrometre, so a stopped robot publishes the same
number for seconds as well. Scene 0009 does exactly that for three runs of 1.3
to 2.5 s while the driver is commanding 1.3 m/s into a robot that is not moving:
the fix is continuous, the covariance is the scene's best (0.0002 m², against a
scene median well above it) and the GNSS latitude and longitude are just as still
as the odometry. Nothing was fabricated; the robot was stuck, and deleting those
105 frames would have been the guard inventing a defect.

So a freeze counts only where something independent says the reported position
is not where the robot is: a jump implying four figures of metres per second
into or out of it, or a hole in the stream of fixes that carried a position.
Both hold for 0010 and 0014 — 0010's freeze opens with a 1249 m jump, 0014's is
fenced by a pair of 1130 m jumps, and the fix stream has a hole through both. An
outage on its own is not a span either: an inertial solution coasts through a
short one perfectly well, and throwing away a merely-drifting stretch would cost
good data for nothing. It becomes one when the position *also* jumped during it
— the receiver saying it had nothing, and the position moving further than
anything can move, are conclusive together where neither is alone. It has no
catch of its own in v2 beta — 0014's span is already convicted by its island,
and the rule only adds `no_fix+teleport` to the reason and widens the span to
the full 3 s the receiver admits it missed — but it is the only rule that would
reach a fabricated stretch that neither held still long enough nor came back,
and it costs nothing to carry.

The outage is read as a hole in the fix stream rather than as a flag, because
the receiver reports the loss two different ways and 0014 uses both: 3 s of
null-island `NavSatFix` around one jump and 12.7 s of publishing nothing at all
around the other. Only holes *between* fixes count — 0009 waits 11.7 s for its
first fix and 0016 spends its last 57 s without one, and neither is a defect.

Nothing in the guard reads the attitude, and that is deliberate. The 180° flip
is the tempting signature and it is a trap: scene 0010 spends 56 s upside down
**before** its solution resets, and that stretch is good data. See
`dataset_tools/gnss_guard.py`.

### A fabricated pose does not need a fabricating receiver

Removing the reset stretches left one step in the root that still could not be
driving: **3.29 m between two frames 62 ms apart in 0014's `transforms.npy`,
53 m/s**, with no span anywhere near it. It is not a receiver artefact at all.
`interpolate_poses` copies the nearest sample wherever it cannot interpolate —
outside the pose stream's range, or across a bracket wider than 0.5 s — and
both of those frames sit in a 12 s hole in the odometry and borrow the same
distant sample. The fabrication is ours, and it reaches `transforms.npy` in the
same shape and with the same plausibility as the receiver's.

So the guard applies its own principle to the whole frame clock: a frame is
exported only where real samples bracket it closely enough to interpolate
between (`gnss_guard.frame_support`). The margin runs the opposite way from a
span — a span drops a frame that *might* fall inside it, support keeps a frame
that *might* sit on a pose — so a hole has to exceed 2.5 s before it costs
anything.

It costs **1,127 frames of v2 beta, 1.1%**, two thirds of them 0016's last
57 s, where the Fixposition stops publishing while the camera runs on; the rest
is a scene's first second or two and 0014's 12 s hole. Those frames were never annotated,
only plausibly annotated. With them gone the worst step anywhere in the root is
3.2 m/s, and every one of those is a robot driving.

## 6. Open items before the real outing

**The Odin1 stamps at publish, not at capture — corrected automatically.** Its
images and clouds carry headers ~0.69 s later than the moment they show, so
reading a pose at `data_ts[k]` returned where the robot was 0.7 s *after* frame
`k` — 0.5 m at walking pace, which is what made the viewer show the odometry
arriving somewhere before the camera, and what smeared every aggregated cloud
into arcs (`img/v2_time_offset_0013.png`).

**No clock on the robot is wrong.** `scripts/probe_clock_skew.py` reads
`recv - header` for every message in a bag, which is the direct reading a
stepped clock cannot hide from:

| topic | recv − header | steps in the bag |
|---|---|---|
| `/odin1/image/compressed` | +0.049 s (0.024 … 0.066) | none |
| `/odin1/cloud_filtered/cloudini` | +0.020 s (0.014 … 0.028) | none |
| `/fixposition/odometry_enu` | −0.004 s | none |
| `/joint_states` | +0.001 s | none |

Every publisher agrees with the recorder to within 50 ms and none of them
moves. The poses are sound too: GT yaw rate tracks the wheel differential to
±0.1 s (|r| = 0.82–0.97) across all of scene 0000, with no drift. What
`recv - header` cannot see is the pipeline *behind* the stamp — publish to
receive is not capture to publish — and that is where the 0.69 s is.

**It does not hold still.** A latency follows load, so unlike a clock offset it
moves during a recording. Scene 0000 measured window by window, every window
driven at ~0.85 m/s:

| t into scene | latency |
|---|---|
| 0–45 s | ~0 |
| 45–52 s | 0.62 s |
| 57–134 s | ~0 |
| 140–216 s | 0.65 s |
| 222–290 s | ~0 |

Two independent modalities agree on that shape: the cloud objective, and visual
yaw rate from the camera cross-correlated against GT yaw rate (lag ≈ 0.00,
r = 0.82–0.86 in the flat stretches; −0.64, r = 0.945 across 157–216 s). The
alternatives are ruled out — GNSS covariance is at its *best* in the offset
stretch (0.010 vs 0.42 m² elsewhere), the frame intervals there are clean, and
the robot is moving throughout.

So there is no number to configure, per rig or per bag. `--time_align auto`
(the default) measures the curve during extraction against the clouds the run
has just written, and takes it off `data_ts.txt` and `pcd_ts.txt`; the pairing
and the clouds themselves are untouched. Each scene carries the measurement it
was corrected by in `time_align.txt`, one row per window, so the correction can
be audited after the fact — and `scripts/measure_time_offset.py` re-asks the
question of a root already on disk. See `dataset_tools/time_align.py` for what
the objective is and why a window is judged on the depth of its minimum rather
than on how far that minimum beats zero.

**Attitude, measured 2026-10-01 (v2.3).** Two readings rank the two
attitude sources against gravity. (1) On the paved scene 0000, the ground-plane
slope in the yaw-carrying levelled frame regressed on `[1, cos ψ, sin ψ]`
separates a body-fixed levelling error (the constant) from the world's slope
(the harmonic); injecting ±2° moves the constant by exactly ±2°. Levelled by
the Fixposition attitude (v2.2) the constant is +1.77° of pitch; by the Odin1
odometry's attitude (v2.3) it is +0.03°. (2) During standstills the Odin1
accelerometer (`/odin1/imu`, frame `imu_link`, identity to `base_link` in TF)
puts the Odin1 attitude within 0.3° of gravity and the Fixposition 1.45° off,
in two bags alike. The two attitude streams track each other frame by frame
(pitch correlation +0.995 in 0000 and 0006, difference sd 0.1–0.3°), so the
Fixposition error is a **rigid offset per scene**: +1.68° of pitch in 0000,
+1.16° in 0006 (and ~0.2° of roll, opposite signs in the two), varying between
sessions. So every `gravity_align.npy` of v2 beta–v2.2 tilts its scene's maps
and compression labels by a constant 1.2–1.7° of pitch (2.0–2.9 cm per metre,
13–19 cm at the map edge), the same angle for every frame of a scene -- a
known, constant, correctable distortion; the v2.3 manifest lists it per scene.
The "−2.2° plaza figure" above was an estimate of the same offset.
`/imu/data` is no reference at all: its accelerometer reads roll and pitch with
the opposite sign to the Odin1's, so the sensor is mounted turned about z and
the `base_link -> imu` identity in the 2026-09 bags' tf_static is wrong.

**Resolved 2026-10-01 — it stamps at capture; the frame is a sweep.** On the
2026-09 bags the publish latency is gone (recv − header steady at +0.12 s,
IMU and odometry on GNSS time within 5 ms). What remained, +27..+52 ms per
scene, is the cloud itself: a 67 ms bottom-to-top raster behind one stamp,
each point carrying its `offset_time`. The extractor now deskews it
(`--deskew`, see the README); on a deskewed root this correction reads as
already aligned, and the camera, exposed at the stamp, needs no offset.

**Robot model — blocks `proprio_labels/`.** No `/robot_description` was recorded
and `tf_static` has only nine edges, none of them the chassis. v1.3 sourced
wheel centres `(0, ±0.36, 0)`, `(−0.697, 0, 0)` and radius 0.36 m from the URDF
(DRIFT `models/simulator.py`). The URDF is published on the robot, it just isn't
in these bags. The wheels appear to have changed since it was written — see the
0.336 / 0.317 m measurements in §4. Wheelbase and track are expected unchanged
but unverified. **To do:** record `/robot_description` in the real outing, and
re-measure the wheel radius against the robot when it is next accessible.

**Camera — two properties that differ from v1.3.**

- The robot's own front wheels permanently occlude **~10 % of every frame**
  (two arcs, lower-left and lower-right), which is precisely the near-field
  terrain the wheels are about to cross — the DRIFT signal.
- The lens is a fisheye (`equidistant`, 4 coefficients). v1.3 was
  `rational_polynomial` on a pinhole. Anything downstream that unprojects has to
  change.

**TF naming trap.** `odin1_camera_link` is *already* optical-convention: its +Z
maps to base +X. The child `odin1_camera_link_optical_frame` applies the optical
rotation a **second** time and points 90° off (+Z → base −Y). The extractor is
correct because it takes the image's own `header.frame_id`, but nothing may ever
look up the `_optical_frame`. The extrinsic itself
`(0.1021, 0.0340, 0.1020)` is unvalidated — same open item as v1.3's
`camera_extrinsic.discrepancy_note`; a checkerboard at 1–2 m would settle it.

**`/cmd_vel` header stamps.** Non-monotonic in seven bags, and **zero in 35 128
of 40 971 messages (86 %) in `16_59_32_merged`**. Those rows sort to `t = 0` in
`cmd_vel.npy`. Plausible reading, to be confirmed with hardware: the joystick is
connected but idle and the robot is being driven by a ROS node, so the stamped
publisher is not the one in control. Same bag also publishes `/cmd_vel` at
~70 Hz against 11–25 Hz elsewhere, consistent with two publishers.

**`joint_setpoints` `±inf` is expected.** Only velocities are commanded;
position and effort are unused. `joint_setpoints.npy` stores them as `±inf` in
columns 1–3 and 7–9, and `helhest-validate` does not look inside optional files.
Consumers must read the velocity columns only — worth stating in the dataset
README rather than "fixing".

**Odometry tail.** `/fixposition/odometry_enu` stops 1.0–1.7 s before the camera
in six of nine bags. Nearest-neighbour lookup silently clamps those last frames
to the final pose. Stop the Fixposition recording last, or trim the tail.

**Storage.** 4.53 MB/frame (4.02 images + 0.51 clouds) against v1.3's 1.53. A
v1.3-sized outing would be ~830 GB. Worth deciding on PNG vs JPEG, or a
downscale, before recording rather than after.

**Content.** These nine bags are one park outing — asphalt and gravel path with
some lawn, members of the public in frame throughout (same non-anonymisation
caveat as v1). Fine as a rig test. As dataset material it is a single terrain
class, where v1.3's value was snow, silt, tussock, leaf litter and three
tip-overs.

## 7. Suggested tooling changes

None are required to extract; all three make the result better documented.

1. **Separate the live-odometry source from the ground-truth source.** A
   `--live_odom` flag feeding `transforms.npy` while `--odom` keeps feeding
   `odometry.txt`, so `/odin1/odometry` (segmented, §5) can be carried
   independently of the Fixposition GT.
2. **Carry the Fixposition status per pose.** `init_status` and `gnss1_status`
   from `/fixposition/fpa/odomstatus` alongside `odometry.txt`, as provenance.
   Cheap, and it makes "was fusion initialised here" answerable after the fact.
3. **Have `validate_scene` look inside the optional arrays** — a non-finite
   count for `joint_setpoints.npy` and `cmd_vel.npy`, and a zero-timestamp
   count, reported as observations rather than errors.

## 8. Reproducing this

```bash
# smoke extraction of one bag (2 min, 1106 frames)
.venv/bin/python -m dataset_tools.extract \
    new_rig_bags/helhest_2026_09_02-16_25_23_merged <out> --jobs 1 -v
.venv/bin/python -m dataset_tools.validate_scene <out>
```

Reading the small topics out of a 20–40 GB mcap takes several minutes per bag —
the chunks have to be decompressed regardless of how little of each is wanted.
The full nine-bag sweep took about 40 minutes.
