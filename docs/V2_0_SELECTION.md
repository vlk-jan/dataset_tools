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

**Core (`docs/v2_0_bags.csv`, 5 scenes, ~89k frames, ~145 GB of bags,
~360 GB extracted):**

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

## 5. Build

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

## 6. Verify before submitting

1. **The keep list.** The core five, and which of the optional rows, if any.
2. **09-08 out.** It is old firmware and has no wheel states, so it is out by
   the rule; say so if the terrain is worth a different rule (it would need
   `joint_setpoints` velocities to stand in for `joint_states`).
3. **`robotour_finals_3`'s Odin mount** (0.071, 0.029, 0.135) / (+0.86, +0.39,
   +0.49)°: was the unit knocked or re-seated during the finals? If the
   published value is a genuine self-calibration the bag is fine; if not, its
   clouds are 1° off and it stays out.
4. **The Fixposition pitch.** Measure the receiver's mounting pitch against
   base_link (or accept the plaza figure, −2.2°) and decide whether v2.0
   corrects `gravity_align.npy` for it. That is an extractor option I have not
   written; without it every levelled ground climbs ~2.5 cm per metre.
5. **Wheel radius and the 7.6 cm.** The anchors use r = 0.36 and shift the
   contact plane onto the lidar ground; a measured radius (0.336?) or Odin
   mount z closes this. Not blocking, but it is a constant baked into every map.
6. **The Lipník front plate.** Confirm what it is and that it ends inside
   0.65 m of base_link (the clouds say 0.5 m); if the plate stays on the
   robot, it belongs in the robot model and in the occlusion note.
7. **Storage.** `/mnt/personal` is at 87 % with 64 TB free; the core root is
   ~360 GB, the full optional set ~1.1 TB, alongside the beta's 995 GB.
8. **Numbering.** v2.0 scenes start at 0000 again and do not match the beta's
   ids; anything that names beta scenes must not be pointed at v2.0.
