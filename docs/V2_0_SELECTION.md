# HELHEST v2.0 — what the new rig recorded, and what goes into the root

Written 2026-09-25 from a survey of every new-rig bag on
`subtdata:/data/helhest` (`docs/v2_inventory.csv`, 54 bags, 7 sessions,
1.14 TB), without extracting any of them: `slurm/survey_v2.batch` runs
`scripts/survey_v2_bag.py` over one bag per array task and leaves ~1 MB per
bag under `/mnt/personal/vlkjan6/HELHEST_v2_survey/<session>/<bag>/`
(summary.json, a 1 Hz GNSS track, 12 thumbnails). RCI job 11704468. The
sessions already in `HELHEST_v2_beta` were read from that root instead.

v2.0 is a **fresh extraction into its own root**, `HELHEST_v2_0`, with its own
bag list (`docs/v2_0_bags.csv`, row = scene number, numbered from 0000 again)
and **no symlink or hard link into the beta**: training is running on the
beta's images and clouds, and a v2.0 scene does not even have the same frame
set as its beta counterpart (different tf_static per session, the beta's
maps rebuilt since). The beta stays as it is.

## 1. The recordings

| Session | Bags | Size | Driving | Odin firmware | Camera : lidar | Wheel states | RTK |
|---|---|---|---|---|---|---|---|
| 26-09-02-stromovka-roads | 9 | 133 GB | 64 min | old | 2.03 | real | yes |
| 26-09-08-stromovka-robotour-prep | 8 (+2 robotour-stack, +1 Odin-dead) | 236 GB | 126 min | old | 2.03 | **all zero** | yes |
| 26-09-11-stromovka-robotour-prep | 4 | 132 GB | 62 min | old | 2.01 | real | mostly |
| 26-09-14-stromovka-robotour-prep | 9 | 239 GB | 139 min | new | 1.00 | real | 4 bags none |
| 26-09-16-stromovka-robotour-prep | 5 | 116 GB | 102 min | new | 1.00 | real | **none all day** (one bag 70 %) |
| 26-09-18-stromovka-robotour-prep | 9 | 92 GB | 81 min | new | 1.00 | real | yes (one bag 18 %) |
| 26-09-19-stromovka-robotour (finals) | 5 | 159 GB | 132 min | new | 1.00 | real | yes |
| 26-09-22-lipnik | 2 | 37 GB | 48 min | new | 1.00 | real | yes |

The Odin firmware was updated between 09-11 and 09-14 (recording team). The
data agrees on where the boundary is, three ways:

- **Self-returns.** Old firmware: ~4,000 points per frame inside 0.5 m on
  09-02/09-08 (the front wheels as two arms at r 0.5–0.57 m, plus a near-range
  ramp reading the ground 5–13 cm high), ~1,800 on 09-11. New firmware: 300–600.
- **Camera doubling.** Every image published twice up to 09-11; 1:1 at the
  nominal 15 Hz from 09-14.
- **Publish latency.** The cloud header trails the recorder's clock by 0.01–0.02 s
  on the old firmware and 0.12 s on the new — a different pipeline. (The 0.7 s
  stretches in the beta's 09-02 scenes are the old one under load.)

Two more things changed at the same boundary and are not firmware:

- **The Odin publishes a calibrated mount from 09-14 on.** `base_link →
  odin1_base_link` is exactly (0.10, 0, 0.10) with identity rotation up to
  09-11; from 09-14 every boot publishes its own value, translation within
  ±2 cm of (0.12, 0.03, 0.11) and rotation roll −0.1..−0.4°, pitch −0.4..−0.7°,
  yaw +0.1..+0.3°. The camera position in base_link moves with it (y 0.035 →
  0.063 m). The extractor reads whatever the bag carries, so this is handled;
  it is why a v2.0 scene is not frame-identical to its beta counterpart.
- **`robotour_finals_3` is the odd one out**: mount (0.071, 0.029, 0.135),
  rotation (+0.86, +0.39, +0.49)°, against (≈0.12, 0.03, 0.10) and (−0.3, −0.5,
  +0.1)° in the other 21 new-firmware bags — and 1,292 near points per frame
  where its neighbours have 300–400. Either the unit was re-seated during the
  finals and the self-calibration is right, or it is not. See §5.

## 2. Calibration findings that affect every scene

- **Fixposition attitude carries a ~2° pitch relative to base_link.** On the
  flat paved plaza (beta 0022, robot stationary) the raw lidar ground plane is
  level to 0.8°, but `gravity_align.npy` (from the Fixposition orientation)
  tilts it by −2.2°, so the *levelled* ground climbs 2.9° — 25 cm at 5 m.
  Smaller values (0.5–1°) in moving scenes are the same offset under motion.
  `FP_VRTK → base_link` in tf_static is identity; the mounting pitch is not
  there. Consequence today: the height-map anchors are shifted onto a ground
  that is itself tilted.
- **The lidar ground sits 7.6 cm above the wheel bottoms** at r = 0.36
  (unchanged finding; the rig rolls 0.336 m/rad). Odin mount z or wheel radius.
- **The lidar ground profile is not a plane.** 20th-percentile z of forward
  ground by range: old firmware −0.27 (1 m) → −0.31 (3 m) → −0.21 (5.5 m), a
  bowl; new firmware −0.28 → −0.28 → −0.22, a climb from 3 m out. Part of it
  is the Fixposition pitch above, part is in the sensor.
- **Lipník: a sloped plate in front of the chassis** (green in every image)
  returns ~900 points per frame at r 0.2–0.5 m, z −0.03 (at 0.25 m) down to
  −0.21 (at 0.5 m). The height maps cut everything inside 0.65 m, which clears
  it, but any consumer of the raw clouds sees a ramp in front of the robot.
- **No RTK is not a defect for DRIFT.** In the four 09-14 scenes without a fix
  the fused ground truth tracks the wheel odometry as well as with RTK (speed
  correlation 0.88–0.96, GT/wheel ratio 0.95–0.99, 1–3 % of moving 1 s windows
  off by more than 0.2 m/s, yaw-rate correlation 0.87–0.89 — the same figures
  as 0020/0021/0023). Absolute position is metres off (σ 5–22 m); the 2 s
  windows are not.

## 3. Content, session by session

Beta scenes (from the root) and unextracted bags (from the survey). `stat` =
share of the time stationary, `tilt95` = 95th percentile of body tilt, `tilt sd`
= its spread while upright, `novel` = share of the GNSS track more than 10 m
from every other bag's track.

**09-02** (0000–0008): asphalt and gravel paths, lawn edges, the viaduct. Flat
(tilt sd 0.7–2.3°). 0004 and 0006 are mostly stationary (48 %, 71 %). Routes
repeated by later sessions (novel ≤ 0.08 except 0000 at 0.40).

**09-08** (0009–0016): the only Stromovka session off the paths — meadow,
leaf litter, roots, wooded banks, two roll-overs (0010 31 % inverted, 0014),
tilt sd up to 9°. Novel routes (0009 0.95, 0015 0.59). **Wheel states are zero
in every bag** (efforts only); DRIFT's loader reads wheel velocities from
`joint_states.npy`, so these scenes drive the simulator with nothing unless
`joint_setpoints.npy` stands in. Old firmware.

**09-11** (4 bags): asphalt and gravel avenues, one 27-minute stationary test
(16_20_36, 82 % still). Novel 0.00 in all four. Old firmware.

**09-14** (0017–0025): cobbles, flagstones, gravel, asphalt, the plaza.
0017/0019/0022 are 64–82 % stationary. 0020 (1.06 km), 0023 (1.16 km) are long
path drives. 0021 is the plaza and the cobbled approach (695 m, RTK); 0025 is
cobbles and gravel on a route nothing else covers (627 m, novel 0.97, no RTK).

**09-16** (5 bags): gravel and asphalt paths again, 2.4 km of it at 0.85 m/s,
no RTK in four bags (σ 8–30 m). 16_34_03 is the one long new route (1.3 km,
novel 0.89, RTK 70 %). Tilt sd ≤ 2.5°.

**09-18** (9 bags): the setts and cobbles around the plaza and the pavilions,
some leaf litter and grass verges; 30–74 % stationary in six of nine bags;
12_24_40 has the most tilt (95th pct 10.9°). RTK in all but one.

**09-19** (finals): 5.8 km of autonomous driving on cobbles, asphalt and gravel
at 0.9 m/s, 4–18 % stationary, RTK 89–100 %, 1:1 camera. finals_1 (1.5 km) and
finals_3 (1.8 km, but see the mount above) are the longest continuous drives
in the whole collection. finals_4 is 41 % stationary (the start line, other
robots and the crowd in frame).

**09-22 Lipník** (0026–0027): the Robotour site — ploughed field, rutted mud,
grass mounds and banks, gravel and railway ballast, a felled log. Tilt 95th
pct 11.5°, sd 3.5–3.7°, the roughest upright driving in the collection; 0.46
m/s; RTK. 0027 has 525 s without any Fixposition pose (294 s of it one hole
inside the scene), so 7,557 of its frames are not exported.

## 4. Selection

Rule 1, from the firmware: **no old-firmware clouds in v2.0** — that removes
09-02, 09-08 and 09-11 entirely. 09-08 is the loss: it is the interesting
Stromovka terrain, but its clouds are the old ones and its wheel states are
zero, and it stays in the beta.

Rule 2, from the purpose: made surfaces are kept as a *sample*, not in bulk.

**Core (`docs/v2_0_bags.csv`, 5 scenes; built 2026-09-25/26 as RCI job 11706726:
89,202 frames, 3,876 m, 1 h 48 min, 347 GB extracted, `validate_scene` 0 failed):**

| Scene | Bag | Why |
|---|---|---|
| 0000 | 09-14 `15_31_12` (beta 0021) | cobbles into the open paved plaza; RTK; 695 m |
| 0001 | 09-14 `17_23_46` (beta 0025) | cobbles and gravel on a route nothing else drives; 627 m; no RTK (fine, §2) |
| 0002 | 09-19 `robotour_finals_1` | the longest clean autonomous drive: 1.5 km, cobbles/asphalt/gravel, 18 % stationary |
| 0003 | 09-22 `14_16_30` (beta 0026) | Lipník field, ruts, mounds, ballast |
| 0004 | 09-22 `15_27_08` (beta 0027) | Lipník banks, log, ballast; one 294 s pose hole |

**Optional, new firmware, moving, if volume on made surfaces is wanted**
(append rows in this order to keep scenes chronological): 09-14 `13_45_23`
(0020, 1.06 km gravel/asphalt), `16_37_57` (0023, 1.16 km), `17_16_34` (0024,
cobbles, no RTK), `13_29_16` (0018, no RTK); 09-16 `12_24_51`, `13_26_22`,
`16_01_17`, `16_34_03` (the last has RTK and a new route); 09-18 `12_09_30`
(cobbles, moving), `12_24_40` (setts, the most tilt), `11_48_22`, `15_51_47`;
09-19 `robotour_finals_2`, `robotour_finals_4`, `robotour_finals_3` (only after
the mount question is settled). All of it together is another ~20 scenes and
~750 GB extracted.

**Discard:** everything old-firmware (09-02, 09-08, 09-11); the stationary
bags (09-14 `10_41_58`/`13_37_17`/`16_28_23`, 09-11 `16_20_36`, 09-18
`12_56_00`/`16_11_48`/`16_23_23`, short 09-18 evening bags); the 09-08
robotour-stack bags (no proprioception) and `14_18_12` (no Odin), as before.

## 5. Build (done: `/mnt/personal/vlkjan6/HELHEST_v2_0`, finalized 2026-09-26; Temesvár added 2026-09-29, §7)

```bash
cd ~/projects/dataset_tools && mkdir -p slurm/slurm_out
CSV=docs/v2_0_bags.csv sbatch --array=0-4%4 slurm/extract_v2.batch /mnt/personal/$USER/HELHEST_v2_0
# when every task is COMPLETED, on a login node:
slurm/finalize_v2.sh /mnt/personal/$USER/HELHEST_v2_0 docs/v2_0_bags.csv
```

Height maps come from DRIFT's `slurm/build_heightmaps_v2_beta.batch` at its
current defaults (min_radius 0.65, p90, ground fill, wheel-contact anchors with
the shift relevel, wheel radius 0.36), the same settings the beta was rebuilt
with on 2026-09-25. `manifest.json`, `README.md` and `atlas/curation.json` are
written by hand afterwards, as for the beta.

## 6. Verified before submitting (answers 2026-09-25, build = RCI job 11706726)

1. **The keep list.** The core five, and which of the optional rows, if any.
   *Answer: the core five, nothing else.*
2. **09-08 out.** It is old firmware and has no wheel states, so it is out by
   the rule; say so if the terrain is worth a different rule (it would need
   `joint_setpoints` velocities to stand in for `joint_states`).
   *Answer: keep it out.*
3. **`robotour_finals_3`'s Odin mount** (0.071, 0.029, 0.135) / (+0.86, +0.39,
   +0.49)°: was the unit knocked or re-seated during the finals? If the
   published value is a genuine self-calibration the bag is fine; if not, its
   clouds are 1° off and it stays out.
   *Answer: not knocked as far as anyone knows; moot, the bag is not in the core.*
4. **The Fixposition pitch.** Measure the receiver's mounting pitch against
   base_link (or accept the plaza figure, −2.2°) and decide whether v2.0
   corrects `gravity_align.npy` for it. That is an extractor option I have not
   written; without it every levelled ground climbs ~2.5 cm per metre.
   *Answer: keep the gravity alignment as the receiver's INS reports it, no correction.*
5. **Wheel radius and the 7.6 cm.** The anchors use r = 0.36 and shift the
   contact plane onto the lidar ground; a measured radius (0.336?) or Odin
   mount z closes this. Not blocking, but it is a constant baked into every map.
   *Answer: the URDF's 0.36 m is a previous HELHEST's wheel; the current wheels are
   different and cannot be measured now (no access to the robot). The anchors stay at
   0.36 with the shift relevel, which puts them on the lidar ground regardless.*
6. **The Lipník front plate.** Confirm what it is and that it ends inside
   0.65 m of base_link (the clouds say 0.5 m); if the plate stays on the
   robot, it belongs in the robot model and in the occlusion note.
   *Answer: a Bluetooth antenna, mounted for follow-me scenarios only.*
7. **Storage.** `/mnt/personal` is at 87 % with 64 TB free; the core root is
   ~360 GB, the full optional set ~1.1 TB, alongside the beta's 995 GB.
   *Answer: `/mnt/personal/vlkjan6/HELHEST_v2_0`.*
8. **Numbering.** v2.0 scenes start at 0000 again and do not match the beta's
   ids; anything that names beta scenes must not be pointed at v2.0.
   *Answer: yes, from 0000.*

## 7. Added 2026-09-29: the Temesvár field test of 2026-09-28 (scenes 0005–0011)

Eleven bags, 36 GB, 32 min of recording in `/data/helhest/26-09-28-temesvar`,
surveyed as rows 54–64 of `docs/v2_inventory.csv` (RCI job 11717950). All on
the new firmware: clouds and images 1:1 at 15 Hz, cloud latency 0.11 s, the
Odin's per-boot mount inside the 09-14 envelope (the first boot of the day,
bags 16:12–16:52, sits at y = 0.003 instead of 0.03). Content the root did not
have: a meadow of knee-high grass, weeds and thistles, driven into a low sun,
with sprints at up to 5.4 m/s and two roll-overs; a forest gravel track between
pines; a reed bed taller than the robot.

| Bag (16:xx = 14:xx UTC) | s | Scene | Frames | Path | What to know |
|---|---|---|---|---|---|
| 16_12_11 | 226 | — | | | set-up at the meadow edge. The receiver's fusion was not initialised for the first 216 s (pose at the ENU origin, identity attitude); 7 s remain. Dropped (user, 2026-09-29) |
| 16_16_16 | 449 | 0005 | 6,472 | 363 m | the meadow, longest scene of the day; 278 blind frames |
| 16_25_23 | 307 | 0006 | 4,413 | 311 m | meadow into the sun, tilt 23°, RTK 87 % |
| 16_35_04 | 121 | 0007 | 1,665 | 17 m | meadow edge, people, half stationary |
| 16_40_50 | 191 | 0008 | 2,431 | 80 m | sprints to 5.1 m/s, tipping at t+128 s and a roll-over at t+167 s; the receiver reset after each (130.9–133.9 s and 169.6–182.7 s removed), 3 segments, 34 frames past 45° |
| 16_45_03 | 73 | 0009 | 957 | 148 m | sprints to 5.3 m/s and a roll-over kept in the scene (17 frames past 45°); one reset removed (67.1–72.4 s); RTK 54 % of fixes |
| 16_52_16 | 147 | — | | | fusion never initialised: the pose sits at the origin for the whole bag while the wheels turn ~190 m. Nothing left after the guard |
| 16_59_55 | 254 | 0010 | 3,645 | 93 m | forest track into a reed bed: the lidar ground is the reed tops 0.32 m above base_link; 629 frames without a map, 401 blind, one empty cloud |
| 17_10_48 | 103 | 0011 | 896 | 27 m | forest gravel track, one sprint at 4.6 m/s; every robot-side topic stops at 60.9 s (the on-board computer shut down after a flip), 586 frames without a pose |
| 18_02_56 | 18 | — | | | lawn at dusk; dropped as too short (user, 2026-09-29) |
| 18_03_45 | 33 | — | | | same; dropped |

None of the Temesvár scenes has a `cmd_vel.npy`: the session was driven
manually from the joystick, which does not publish `/cmd_vel` (only the
autonomy stack does). `joint_setpoints.npy` holds the commanded wheel rates.

**Two guard faults found on this session, fixed by the laptop session and
merged (dataset_tools `3b46139`, `3fa1e26`):**

- The speed gate was 4 m/s. The sprints read as teleports, two in a row fenced
  a one-sample island, and the guard removed each sprint pose by pose and
  tagged the hole `fix_lost` with RTK fixed throughout. At the first
  extraction (jobs 11717951, 11718014) 16_40_50 came out as 2,321 frames in 4
  segments, 16_45_03 as 571 in 4 and 17_10_48 as 857 in 2. The gate is now
  6 m/s, defined once in `gnss_guard.MAX_SPEED`; the slowest fabricated step
  on record is 53 m/s.
- A pose stream that is fabricated from its first message passed as a
  stationary robot, because a frozen run was believed only next to a teleport
  or an outage between two fixes. A frozen run with no usable fix under it now
  counts as fabricated.

Scenes 0005–0011 were re-extracted with both (RCI job 11718103) and agree
frame for frame with the root built on the laptop. 0000–0004 predate the fixes;
their fastest step is 2.9 m/s and no frozen run without a fix is known in them.

Vegetation is what these scenes are for, not a defect: the lidar sees the
canopy and DRIFT has to predict the supporting terrain under it. The height
maps map the canopy, not the soil (meadow lowest returns 0.10–0.20 m below base_link
instead of 0.26 m; reed bed +0.32 m), and the shift relevel puts the
wheel-contact anchors on that canopy too. Frames with no return beyond 1 m are
counted per scene in the manifest (769 in the root, 401 of them in the reed
bed).

Wheel geometry, settled with the DRIFT session the same day: the wheel centres
the anchors use, (0, ±0.36) and (−0.697, 0), are what the robot publishes in
`/robot_description` (continuous joints, hence not in tf_static). The radius
stays open: the URDF's 0.36 was written on 2025-04-02 (helhest-common
`04daac0`) and never edited, so it says nothing about the wheels of the 2026
rebuild.

The root is 12 scenes, 109,681 frames, 4,917 m of ground truth, 2 h 12 min,
108,244 height maps; `validate_scene` 0 failed, 4 with warnings (steps across
pose holes in 0001, 0004 and 0008, frames past 45° in 0008 and 0009).
Finalized 2026-09-29.

**DRIFT-side files, written the same day on the finalized root.**
`keyframes.lst` (DRIFT `b6738ca`): 108,244 lines, every frame that has a
cloud, an image, a map and a mask; no motion or tilt gate. `proprio_labels/`
(DRIFT `da55abe`, RCI job 11718296, 2 s horizon, `min_validity` 2): 80,502 of
108,244 keyframes labelled (74.4 %), with the robot model `helhest_v2` —
wheel centres from `/robot_description`, **radius 0.334 m, the measured
rolling scale** (path over integrated wheel rotation 0.334 on made surfaces,
0.327 pooled; ground-truth speed over wheel rate 0.336), not the URDF's 0.36.
By label count the root is 58 % made surfaces, 27.6 % Lipník and 14.4 %
Temesvár. The laptop mirror of the root leaves `proprio_labels/` out on
purpose.

## 8. HELHEST v2.1 (2026-09-29): v2.0 linked, plus the Temesvár session of 2026-09-29

`/mnt/personal/vlkjan6/HELHEST_v2_1`, bag list `docs/v2_1_bags.csv` (the twelve
rows of v2.0 followed by six new ones). v2.0 was not extended in place because
a training run reads it; it stays a separate, unchanged root.

**Storage.** Scenes 0000–0011 are directories of relative symlinks into
`../HELHEST_v2_0` (images, clouds, maps, masks and proprio labels as directory
links, the small files one by one). Scenes 0012–0017 are real files, extracted
by RCI job 11719242 with the code of v2.0's 0005–0011. A tool that writes into
an existing directory of a linked scene writes into v2.0, so every writer has
to be restricted to scenes ≥ 0012.

**The new session.** Six bags in `/data/helhest/26-09-29-temesvar` (inventory
rows 65–70), 23 GB, 27 min, the same meadow as 0005–0009 from late afternoon
into dusk, driven from the joystick almost without stopping (2–11 % stationary,
RTK ≥ 98 %). All six were kept; the pose guard removed nothing.

| Scene | Bag | Frames | Maps | Path | > 10 m from 09-28 | What to know |
|---|---|---|---|---|---|---|
| 0012 | 17_18_57 | 4,395 | 4,248 | 402 m | 100 % | lane and verge, tight loops in knee-high grass |
| 0013 | 17_25_16 | 3,498 | 3,414 | 283 m | 100 % | one tip onto the side: 129 frames past 45°, max 99° |
| 0014 | 17_33_15 | 3,500 | 3,425 | 225 m | 93 % | tall grass under old fruit trees |
| 0015 | 18_23_35 | 3,865 | 3,828 | 494 m | 100 % | long straight run, loops on open grass at dusk |
| 0016 | 18_28_39 | 2,614 | 2,558 | 220 m | 79 % | open meadow at dusk; one empty cloud (frame 01480) |
| 0017 | 18_32_30 | 5,507 | 5,142 | 321 m | 72 % | crosses the patch of the day before |

**Held out.** The session is validation only (user, 2026-09-29) and the split
of the root is deferred. DRIFT reads `<root>/keyframes.lst` by that fixed name,
so v2.1 deliberately has **no** `keyframes.lst`: a loader pointed at it fails
instead of training on the held-out session. `HELHEST_v2_1_heldout` beside it
holds only scenes 0012–0017, as links into v2.1 under the same numbers, with a
keyframe list of its own written by DRIFT. What a result on it supports is an
unseen session of a known site: 90 % of the new track lies more than 10 m from
anything driven on 2026-09-28 and 25 % more than 50 m, but the six bags overlap
each other heavily (0012–0014 for their whole length).

The root is 18 scenes, 133,060 frames, 6,863 m, 2 h 39 min, 130,859 height
maps, 75 GB of its own; `validate_scene` 0 failed, 5 with warnings. Proprio
labels exist for 0000–0011 (v2.0's, through the link); those of 0012–0017 are
DRIFT's to write.
