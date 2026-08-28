# HELHEST dataset

Outdoor terrain-traversal recordings from **HELHEST**, a small three-wheeled
skid-steer field robot developed at CTU in Prague (VRAS group). The dataset
pairs forward-facing LiDAR scans and RGB images with the robot's proprioception
(wheel joint states, attitude) and, where available, a 6-DoF trajectory from
LiDAR-inertial odometry. It was gathered to study how terrain properties
(stiffness, friction, support surface) can be inferred from robot–terrain
interaction, so the driving is deliberately rough: tall vegetation, loose
banks, a drained reservoir bed, snow, and three tip-over events are all in.

| | |
|---|---|
| Scenes | 43 (`0000` – `0042`), one per raw recording |
| Synchronised frames | 182,913 LiDAR–camera pairs, 7.5 h |
| Sessions | 12–14 Sep 2025 (0000–0011), 9 Oct 2025 (0012–0029), 16 Feb 2026 (0030–0042) |
| Sites | Robotour 2025 venue — chateau landscape park & arboretum [LOCATION]; Temešvár area (Písek district) — village farmyard, pine-forest edge & fallow meadow, forested hills around a drained pond; Milíčovský háj (Prague) — snow-covered lowland deciduous forest |
| Odometry | present in 32 scenes (see table); absent in 11 |
| Size on disk | ~7 GB per 10k frames (PNG images dominate) |
| Location | `/mnt/data/vras/data/HELHEST/` (CTU VRAS storage); raw bags on `subtdata:/data/` (see [Raw recordings](#raw-recordings)) |

> This document describes the **first-generation** sensor setup (Livox + OAK-D,
> Sep 2025 – Feb 2026). The robot was rebuilt in summer 2026 with an Odin1
> LiDAR-camera unit and a Fixposition GNSS/INS; recordings from that setup are
> extracted with the same tooling but are **not** part of this dataset yet
> (see [New sensor generation](#new-sensor-generation)).

---

## 1. Platform and sensors

HELHEST is a ~1 m long, three-wheeled robot (two driven front wheels
`left_wheel_j`, `right_wheel_j` and a rear wheel `rear_wheel_j`) with a
sensor mast. Sensor suite of this dataset:

| Sensor | Details | In the dataset as |
|---|---|---|
| **Livox solid-state LiDAR**, front-mounted | ROS topic `/livox/lidar_front` (`sensor_msgs/PointCloud2`, fields x, y, z, intensity, tag, line, timestamp), ~10 Hz, non-repetitive scan pattern. **The dataset clouds come from the front LiDAR only**: there are zero returns beyond ±140° bearing in every frame — the rear of the robot is never observed. (The robot also has a rear-mounted Livox; it was recorded only in the September 2025 bags as `/livox/lidar_rear` and is not part of the extracted scenes.) | `clouds/*.npy`, transformed into the robot base frame |
| **Luxonis OAK-D PoE** RGB camera | `/luxonis/oak/rgb/image_raw/compressed`, 1280 × 800, pinhole with fx ≈ fy ≈ 761.6 px, cx = 633.9, cy = 407.0 (identical in all scenes). Mounted on the LiDAR, looking forward: `T_cam_to_lidar.txt` is the same in every scene (yaw 90° optical-frame convention, t = [−0.06, 0, −0.085] m). Horizontal half-FOV ≈ 40°. | `images/*.png`, `camera_intrinsics.txt`, `T_cam_to_lidar.txt` |
| **IMU** (EKF-filtered attitude, `/ekf/imu/data`) | Roll/pitch used for gravity alignment. | `gravity_align.npy` |
| **Wheel joint states** (`/joint_states`, 100 Hz) | Position, velocity, effort for the three wheels. | `joint_states.npy` |
| **LiDAR-inertial odometry** (Point-LIO, computed offline) | 6-DoF pose at 10 Hz in a local frame anchored at the first pose. Not georeferenced — there is no GNSS in this generation. | `odometry.txt` (+ `transforms.npy` per frame) |

Frames are sampled at the RGB frame rate: **~3 Hz** in the September 2025
recordings (camera-limited), **~10 Hz** in October 2025 and February 2026.

## 2. Directory layout and file formats

```
HELHEST/
├── 0000/ … 0042/                 one directory per scene
│   ├── images/00000.png …        RGB frames, 1280×800, 8-bit BGR PNG
│   ├── clouds/00000.npy …        LiDAR scans, float64 (N, 4): x y z [m] intensity, base frame
│   ├── pcd_ts.txt                LiDAR timestamp per frame (unix s, UTC)
│   ├── data_ts.txt               (camera + LiDAR stamp)/2 per frame
│   ├── camera_intrinsics.txt     3×3 K
│   ├── T_cam_to_lidar.txt        4×4 camera→LiDAR extrinsic
│   ├── gravity_align.npy         (F, 3, 3) rotation removing roll/pitch per frame
│   ├── transforms.npy            (F, 4, 4) LIVE /odom pose from the bag per frame, relative to frame 0 (NOT Point-LIO)
│   ├── joint_states.npy          (J, 10): t, pos×3, vel×3, effort×3 (wheel order: left, rear, right)
│   ├── odometry.txt              [optional] offline Point-LIO trajectory (the ground truth used by DRIFT), see below
│   ├── trajectory.txt            [some scenes] identical copy of odometry.txt
│   ├── height_maps/00000.npz …   [derived] per-frame BEV height map, key `height_map` (128,128) float64, 0.1 m cells, ±6.4 m
│   ├── height_masks/00000.npz …  [derived] key `validity` (128,128) uint8 (measured/extrapolated), `height_threshold`
│   └── <scene>_full.npz          [derived, 31 scenes] scene-wide aggregated height map: `height_map`, `side_range`, `fwd_range`, `res`
├── maps_full/<scene>.pcd         [derived, 31 scenes] scene-wide aggregated point cloud
├── keyframes.lst                 [derived] "scene/clouds/f.npy scene/images/f.png" pairs used for training
├── keyframes_filtered.lst        [derived] subset after quality filtering
└── keyframes_adaptation.lst      [derived] small subset for online-adaptation experiments
```

Files marked *derived* are produced by downstream processing (the DRIFT
project's `scripts/data/prep_dataset.py`), not by the extraction tool; they
are included for convenience and can be regenerated.

**Frame indices** are chronological and shared: `images/i.png`,
`clouds/i.npy`, `gravity_align[i]`, `transforms[i]`, `pcd_ts[i]` all refer to
the same synchronised pair. Pairs were formed by mutual nearest-neighbour
matching of camera and LiDAR header stamps, so unmatched frames were dropped.

**`odometry.txt`** — text, one row per odometry message:

```
# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1
```

i.e. the timestamp followed by the row-major 4×4 pose of the robot base in the
odometry frame — **note: the LiDAR frame, mirrored vs. `base_link`, see
§5**. It runs at 10 Hz with occasional 1–4 s dropouts; it may start
after the scene begins or stop before it ends (see table); its stamps are not
strictly monotonic (sort before interpolating). **`transforms.npy` is a
different trajectory**: it holds the robot's *live* `/odom` from the bag
(wheel/EKF odometry recorded on the robot), matched to each frame and
re-anchored so that `transforms[0] = I`. The two disagree by tens of metres
over a scene; use `odometry.txt` for ground truth and treat `transforms.npy`
as a rough live estimate.

**Coordinate frames.** Clouds are in the robot **base frame** (`base_link`: x
forward, y left, z up, origin at the robot base). `gravity_align[i]` is the
rotation R = Ry(pitch)·Rx(roll) that, applied to a cloud (`pts @ R.T`),
levels it so the ground is horizontal; yaw is untouched. Because the LiDAR is
forward-facing, ~73 % of raw points are robot self-returns within 0.5 m of the
origin — filter by range before use. Beyond 0.5 m a frame holds ~4–7 k returns
out to ~20–25 m (95th percentile).

## 3. Sessions and sites

**12–14 Sep 2025 — Robotour 2025, chateau landscape park and arboretum (0000–0011)** [LOCATION].
Fine pinkish-grey gravel paths, mown lawns, clipped hedges and topiary, tree
avenues and an arboretum of very large conifers with rhododendron banks; one
wing of the chateau is under scaffolding throughout. Late-summer foliage,
mixed overcast and low sun with lens flare. Park visitors, joggers and dog
walkers pass close to the robot in most scenes. Recorded at ~3 Hz. Odometry is
missing or truncated in several scenes.

**9 Oct 2025 — Temešvár (Písek district, South Bohemia), one day, four sites (0012–0029), ~10 Hz, uniformly overcast:**
- *Village farmyard and garden (0012–0015):* mown lawn, bare soil mounds,
  gravel yard, an ornamental rockery and shrubs the robot pushes into.
- *Pine-forest edge, field tracks, fallow meadow (0016–0020):* the only
  mostly-asphalt scene (0016, a straight road), gravel forest roads, a wooden
  hunting hide, and a tussock meadow with tall dry weeds where the robot
  ploughs through vegetation with heavy wheel slip (0018, 0019).
- *Forested hills around a drained reservoir (0021–0029):* gravel and wet
  asphalt forest roads, deep leaf litter, steep cut banks and eroded root
  banks, then the exposed reservoir bed — cracked silt, sand, wheel ruts,
  erosion gullies, a ~33° sand slope. **All three tip-over events are here**
  (0021, 0022, 0025). Light fades to dusk in 0028–0029.

**16 Feb 2026 — Milíčovský háj (Prague), snow-covered lowland deciduous forest (0030–0042), ~10 Hz.**
~5 cm of snow over leaf litter, wide forest rides with vehicle ruts, glades,
stacked timber piles, low embankments; 0042 is on adjacent open scrub with
bramble thickets. Flat overcast winter light; the survey crew, tripods and an
equipment sledge appear in several scenes. Three scenes are (nearly)
stationary.

Rough share of driven distance by surface: gravel park paths ~33 %, snow ~30 %,
gravel/dirt forest roads ~26 %, dry reservoir silt/sand ~5 %, grass ~4 %,
asphalt ~4 %, bare leaf litter/root bank <1 %. Grass is under-represented by
distance but over-represented by time (0018+0019 are 45 min of slow driving).

## 4. Scene table

Contact sheets (frames at 10 %, 50 %, 90 % of every scene) are in
[`docs/img/`](img/): [`sheet_0.jpg`](img/sheet_0.jpg) (0000–0010),
[`sheet_1.jpg`](img/sheet_1.jpg) (0011–0021), [`sheet_2.jpg`](img/sheet_2.jpg)
(0022–0032), [`sheet_3.jpg`](img/sheet_3.jpg) (0033–0042); the two terminal
rollovers frame by frame in [`flip_0022.jpg`](img/flip_0022.jpg) and
[`flip_0025.jpg`](img/flip_0025.jpg).


`Odometry` gives coverage relative to the scene (`starts +N s` = odometry
begins N s after the first frame; `ends N s early` = it stops N s before the
last frame; dropouts are gaps >1 s in the 10 Hz stream). `Path` is the 2-D
odometry path length. `Max tilt` is the largest roll/pitch deviation from
level in `gravity_align`.

| Scene | Date (UTC) | Frames | Length | Sync | Odometry | Path | Max tilt | Environment |
|---|---|---|---|---|---|---|---|---|
| 0000 | 2025-09-12 15:13 | 1962 | 10.9 min | 3 Hz | ends 283 s early | 369 m | 12° | Chateau park: gravel drive past scaffolded building, park paths, lawns, hedges |
| 0001 | 2025-09-12 15:33 | 3918 | 22.5 min | 3 Hz | **none** | – | 20° | Chateau park: paths under dense conifers/rhododendron, leaf-strewn path, ends at the chateau |
| 0002 | 2025-09-13 11:37 | 7426 | 41.3 min | 3 Hz | ends 1264 s early; 56 dropouts >1 s | 349 m | 17° | Chateau park: wall-lined alley, tree avenues, topiary garden; many visitors close to the robot |
| 0003 | 2025-09-13 12:19 | 4051 | 22.5 min | 3 Hz | 80 dropouts >1 s | 484 m | 16° | Chateau park: broad avenues, hedge-lined paths, chateau facade |
| 0004 | 2025-09-13 13:49 | 3074 | 17.1 min | 3 Hz | ends 798 s early | 63 m | 11° | Chateau park: path by scaffolded wing, hedge alley; mostly parked/manoeuvring, strong sun glare |
| 0005 | 2025-09-13 14:33 | 1551 | 8.7 min | 3 Hz | ends 286 s early; 13 dropouts >1 s | 196 m | 14° | Arboretum: paths under very tall conifers, rhododendron banks; sun flare |
| 0006 | 2025-09-13 14:42 | 1419 | 8.0 min | 3 Hz | ends 347 s early | 46 m | 18° | Arboretum: narrow leaf-covered path under big conifers; heavy motion blur |
| 0007 | 2025-09-13 14:52 | 1147 | 6.7 min | 3 Hz | **none** | – | 11° | Arboretum path junction under cedars; robot idle 83 % of the time; pedestrians |
| 0008 | 2025-09-13 15:06 | 678 | 3.9 min | 3 Hz | 12 dropouts >1 s | 140 m | 14° | Chateau park: curving path under conifers, bench, open lawn; motion blur |
| 0009 | 2025-09-13 15:10 | 1660 | 9.5 min | 3 Hz | ends 314 s early; 11 dropouts >1 s | 137 m | 25° | Chateau park: path junctions, lawns, chateau facade with crew/equipment |
| 0010 | 2025-09-14 08:17 | 1916 | 10.7 min | 3 Hz | 29 dropouts >1 s | 232 m | 19° | Chateau park: hedge alley by scaffolded wing, lawns, wide paths |
| 0011 | 2025-09-14 09:48 | 8342 | 46.6 min | 3 Hz | ends 1638 s early; 56 dropouts >1 s | 431 m | 33° | Chateau park + arboretum, red/pink gravel section, ends on the chateau forecourt |
| 0012 | 2025-10-09 10:49 | 1484 | 2.5 min | 10 Hz | full | 96 m | 19° | Village farmyard: mown lawn by a barn with solar roof, parked car, farmhouse |
| 0013 | 2025-10-09 10:53 | 393 | 0.7 min | 10 Hz | full | 43 m | 14° | Village farmyard: lawn, bare soil mounds, orchard trees |
| 0014 | 2025-10-09 10:54 | 968 | 1.6 min | 10 Hz | **none** | – | 45° | Farmyard and adjacent grass field; operators at the robot; two brief >30° tilts |
| 0015 | 2025-10-09 11:02 | 1203 | 2.0 min | 10 Hz | **none** | – | 24° | Village garden: loose-stone rockery, lawn, shrubs (robot pushes into them), house |
| 0016 | 2025-10-09 11:45 | 395 | 0.7 min | 10 Hz | full | 42 m | 4° | Straight paved minor road along a pine-forest edge (only mostly-asphalt scene) |
| 0017 | 2025-10-09 11:56 | 8537 | 14.5 min | 10 Hz | ends 259 s early; 18 dropouts >1 s | 467 m | 27° | Gravel forest road, field/dirt tracks, dry grass, wooden hunting hide |
| 0018 | 2025-10-09 12:11 | 6896 | 11.5 min | 10 Hz | ends 601 s early | 54 m | 40° | Fallow meadow with tussock grass and tall dry weeds; robot ploughs into vegetation (heavy slip); drone overhead |
| 0019 | 2025-10-09 12:24 | 20047 | 33.6 min | 10 Hz | **none** | – | 14° | Same fallow meadow, long low-speed run; idle 44 %; group of people at the end |
| 0020 | 2025-10-09 13:58 | 14198 | 23.7 min | 10 Hz | 64 dropouts >1 s | 837 m | 17° | Gravel forest roads through spruce/pine plantation; longest single-surface Oct drive |
| 0021 | 2025-10-09 14:22 | 6070 | 10.1 min | 10 Hz | **none** | – | 88° | Forested hillside: gravel + wet asphalt roads, deep leaf litter, steep cut banks, chalets; ~11 s on its side (88°) |
| 0022 | 2025-10-09 14:33 | 301 | 0.5 min | 10 Hz | full | 11 m | 174° | ROLLOVER: climbs a leaf-littered bank, tumbles, lies fully inverted (170°) for the last 58 % of frames |
| 0023 | 2025-10-09 14:34 | 2578 | 4.3 min | 10 Hz | ends 208 s early | 20 m | 44° | Forest: robot noses into an eroded root bank for ~48 s (30–44°), gravel road, cottage, reservoir shore |
| 0024 | 2025-10-09 14:38 | 2659 | 4.4 min | 10 Hz | full | 161 m | 26° | Descent from forest/meadow track onto the bed of a drained reservoir (rutted mud, dry silt) |
| 0025 | 2025-10-09 14:43 | 947 | 1.6 min | 10 Hz | full | 61 m | 88° | TIP-OVER: drained reservoir bed (cracked silt/sand, steep bare banks); rolls onto its side (84°) for the last 25 % |
| 0026 | 2025-10-09 14:45 | 3426 | 5.7 min | 10 Hz | **none** | – | 33° | Drained reservoir bed: silt flats, wheel ruts, erosion gullies, ~33° sand slope, weedy meadow with scrap |
| 0027 | 2025-10-09 14:52 | 867 | 1.5 min | 10 Hz | **none** | – | 26° | Reservoir shore and adjacent leaf-covered forest track |
| 0028 | 2025-10-09 14:54 | 1574 | 2.6 min | 10 Hz | starts +47 s | 88 m | 14° | Gravel forest road descending past a wooden chalet; dusk, vehicle headlights |
| 0029 | 2025-10-09 14:57 | 9532 | 15.9 min | 10 Hz | 31 dropouts >1 s | 955 m | 19° | Long forest ride: gravel/dirt roads, hillside cuttings, then asphalt across fields to a village car park (955 m, 45 m elevation) |
| 0030 | 2026-02-16 09:28 | 14231 | 23.8 min | 10 Hz | ends 519 s early; 50 dropouts >1 s | 360 m | 16° | Snow-covered mature deciduous forest: wide ride and open glades |
| 0031 | 2026-02-16 09:58 | 7290 | 12.2 min | 10 Hz | ends 31 s early; 15 dropouts >1 s | 285 m | 15° | Snowy forest ride and glade; person walking beside the robot |
| 0032 | 2026-02-16 10:15 | 1480 | 2.5 min | 10 Hz | 31 dropouts >1 s | 44 m | 7° | Snowy forest, short run next to the operator; idle 57 % |
| 0033 | 2026-02-16 10:20 | 1172 | 2.0 min | 10 Hz | **none** | – | 3° | Snowy forest edge by a dense thicket; near-static (idle 89 %) |
| 0034 | 2026-02-16 10:22 | 1199 | 2.0 min | 10 Hz | **none** | – | 5° | Snowy forest, very low speed near operators; near-static (idle 93 %) |
| 0035 | 2026-02-16 10:28 | 15434 | 25.8 min | 10 Hz | starts +207 s; 29 dropouts >1 s | 597 m | 17° | Snowy forest rides, glades, low embankments; longest winter drive |
| 0036 | 2026-02-16 10:58 | 5669 | 9.6 min | 10 Hz | ends 33 s early; 108 dropouts >1 s | 485 m | 24° | Snowy forest ride with timber piles and vehicle ruts |
| 0037 | 2026-02-16 11:17 | 1989 | 3.3 min | 10 Hz | **none** | – | 3° | STATIONARY: fixed view of a snowy ride with a timber pile; zero wheel motion, all frames identical |
| 0038 | 2026-02-16 11:32 | 2054 | 3.4 min | 10 Hz | full | 137 m | 19° | Snowy forest: timber pile at close range, wide ride, forest edge onto a field |
| 0039 | 2026-02-16 12:18 | 1864 | 3.1 min | 10 Hz | full | 77 m | 25° | Snowy forest ride beside a survey/measurement setup (tripod, cases, people) |
| 0040 | 2026-02-16 13:47 | 3139 | 5.2 min | 10 Hz | full | 171 m | 12° | Snowy forest ride with crew, equipment sledge, timber stacks, signpost |
| 0041 | 2026-02-16 14:01 | 6065 | 10.1 min | 10 Hz | full | 312 m | 25° | Snowy forest rides and glades with exposed leaf litter; 11.8 m elevation range |
| 0042 | 2026-02-16 14:14 | 2108 | 3.6 min | 10 Hz | full | 85 m | 11° | Snow-covered fallow scrub outside the forest: bramble thickets, snowy track, wayside shrine, parked vans |

## 5. Known issues and caveats

- **Clouds are offset by −0.34 m along x (extraction bug, found 2026-08-28).**
  The static TF chain `base_link → livox_front_frame` (x = +0.17 m,
  z = +0.145 m, rolled 180° — the Livox is mounted upside down) was composed
  in the wrong direction by the original extractor. Because R_x(π) is its own
  inverse, rotation and z came out right, but the LiDAR origin was placed at
  x = −0.17 m instead of +0.17 m: **every point in `clouds/*.npy` is 0.34 m
  too far back** relative to the true `base_link`. Verified by re-extracting
  scene 0013 from its bag with the fixed tool (per-point delta exactly
  (+0.34, 0, 0) m) and by better map self-consistency through turns. Fix on
  existing data: `pts[:, 0] += 0.34` before use (the `height_maps` were built
  from the offset clouds and inherit the shift; regenerate them). Odometry,
  gravity alignment and images are unaffected. `dataset_tools` ≥ 2026-08-28
  extracts correctly. Independent check: the no-return cluster (next item)
  sits at (−0.170, 0, +0.145) in every scene of all three sessions, exactly
  the mirror of the mount position.
- **`odometry.txt` is the LiDAR's pose, mirrored relative to `base_link`
  (found 2026-08-28).** Point-LIO was run on the raw Livox stream without the
  mount extrinsic, so `odometry.txt` gives the pose of `livox_front_frame` —
  which is mounted upside down — in a world frame whose z points *down*.
  Relative to the base frame of the clouds this negates y and z and inverts
  the yaw: a left turn in reality is a right turn in `odometry.txt`. Evidence:
  yaw rate from `odometry.txt` is negatively correlated with the wheel-speed
  differential (right − left; r = −0.54…−0.81) and with the robot's live
  `/odom` in `transforms.npy` (r = −0.68…−0.90) in every scene tested, and
  registering consecutive scans through turns with the mirrored poses
  improves from ~1–2 m to ~0.2–0.5 m misalignment. Convert to a base-frame
  pose in a z-up world with `T_base = X @ T @ inv(X)`, where
  `X = [[1,0,0,0.17],[0,-1,0,0],[0,0,-1,0.145],[0,0,0,1]]` (the
  `base_link → livox_front_frame` transform, R_x(π) with the +0.34 m x
  correction already implied). Translation magnitudes and the x/forward
  component are unaffected; consumers that only use forward speed are fine.
- **~10 k of the ~20 k points per frame are invalid.** Livox reports
  non-returns as (0, 0, 0); after the base transform they all sit at one
  point near the origin (x ≈ −0.17, z ≈ 0.145 in the published data). Filter by
  range (> 0.5 m) before use — a frame has ~4–7 k real returns. The current
  extractor drops them.
- **Odometry gaps are intentional.** Point-LIO was run offline after the
  recordings and stopped whenever it started to drift, so 11 scenes have no
  `odometry.txt` (0001, 0007, 0014, 0015, 0019, 0021, 0026, 0027, 0033, 0034,
  0037) and others are truncated (0002 and 0011 lose the last 21 and 27 min).
  Where present, the odometry is drift-free enough for local trajectory use
  but has no absolute reference; 0009 shows 15.7 m of vertical "relief" over a
  137 m path on flat ground — treat its z as drift.
- **Tip-overs are real and terminal.** 0022: the robot climbs a bank, tumbles
  and lies fully inverted (~170°) for the last 58 % of the scene (camera
  looking into the canopy). 0025: rolls onto its side (~84°) for the last
  25 %. 0021: ~11 s on its side (88°) around frames 1675–1785, then recovers.
  0023 holds 30–44° for ~48 s nosed into a root bank. Downstream code that
  assumes a level robot should filter on `gravity_align` (e.g. tilt > 45°).
- **One-frame glitch in 0040.** `transforms.npy` (live `/odom`) jumps 8.3 m
  between frames 721 and 722; `odometry.txt` is smooth there and frame
  indexing is intact. Irrelevant if you use `odometry.txt`.
- **Stationary/near-stationary scenes:** 0037 has zero wheel motion (1989
  identical frames); 0033 and 0034 are ≥89 % idle; 0004, 0007 and 0019 spend
  most of their time parked. Several long scenes are 30–50 % idle.
- **Forward-only LiDAR.** Nothing is ever observed behind the robot or beyond
  ±140°; any rear-sector terrain representation derived from these clouds
  (e.g. the `height_maps` cells outside the returns' convex hull) is
  extrapolation — `height_masks/validity` marks which cells were measured.
  About 20 % of the driving is in reverse, i.e. into unobserved terrain.
- **Camera artefacts:** persistent dark smudge/droplet blobs on the lens in
  0000, 0018, 0019; strong flare in 0005/0007; overexposure in 0004; heavy
  rotational motion blur on rough scenes (0006, 0008, 0022–0024, early 0029);
  a dark band along the top edge of many 0042 frames; strong magenta white
  balance under canopy in the September scenes.
- **People.** Members of the public appear (often close) in the park scenes
  and the operating crew in many others; the images have not been anonymised.
  Consider this before redistributing.
- **Wheel slip is large and unlabelled.** Odometry metres per radian of wheel
  rotation is ~0.38 on firm surfaces but collapses to 0.02–0.08 in 0018,
  0004, 0023 (vegetation / root bank / in-place turning) — useful as a signal,
  but there is no ground-truth slip.
- **Another copy exists.** The ScaLR/pretraining copy of this dataset shares
  scenes 0000–0029 but its 0030+ indices are *different* recordings (some with
  a rear-mounted camera and 360° LiDAR). Never exchange scene IDs between the
  two trees without naming the tree. Every scene in *this* tree is front
  camera.
- **Timestamps** are unix seconds (UTC) taken from message headers; a few
  scenes have sub-second out-of-order stamps that were sorted during
  extraction.

## 6. Raw recordings

All scenes were extracted from ROS 2 bags (rosbag2, **mcap** storage,
metadata version 9) stored on the VRAS data server `subtdata` under `/data/`.
Each scene corresponds to exactly one bag (no bag was split, no scene spans
two bags); a scene starts 0–20 s after its bag and ends a few seconds before
it. Bag directory names are local wall-clock time (CEST/CET), scene timestamps
are UTC.

| Scenes | Bags | Size |
|---|---|---|
| 0000–0001 | `/data/robotour2025/helhest_250912/helhest_2025_09_12-{17_13_12,17_33_47}_merged` | 73 GB |
| 0002–0009 | `/data/robotour2025/helhest_250913/helhest_2025_09_13-*_merged` (8 bags, chronological) | 284 GB |
| 0010–0011 | `/data/robotour2025/helhest_250914/helhest_2025_09_14-{10_17_39,11_48_15}_merged` | 135 GB |
| 0012–0029 | `/data/outdoor-dataset/25-10-09-temesvar-helhest/combined/helhest_2025_10_09-*` (18 bags, chronological; the sibling `nuc/` tree is a proprioception-only subset) | 194 GB |
| 0030–0042 | `/data/outdoor-dataset/26-02-16-milicovsky-haj/helhest_2026_02_16-*_xx` (13 of 14 bags; `14_53_xx` was skipped — it has no `/odom`/`/tf`) | 238 GB |

Scene ↔ bag mapping in detail: [`scene_bags.csv`](scene_bags.csv). A structural check of every scene with `helhest-validate` is in [`validation_v1.csv`](validation_v1.csv).

Bag contents (old generation, all standard ROS 2 message types — no custom
packages needed for the topics used here): `/livox/lidar_front`,
`/livox/lidar_both_filtered` (+ `/livox/lidar_rear` in Sep 2025),
`/livox/imu_*`, `/luxonis/oak/rgb/image_raw/compressed`, OAK left/right/stereo
streams, `/odom` (Point-LIO), `/ekf/imu/data`, `/joint_states`, `/tf`,
`/tf_static`, GNSS/nav auxiliaries (~125 topics per bag).

**Reading with `rosbags`.** Each mcap contains one schema with encoding
`unknown` (`action_msgs/msg/GoalStatusArray`), which makes stock `rosbags`
≤ 0.11 fail on open; `dataset_tools` patches this automatically
(`dataset_tools/rosbags_compat.py`).

**Viewing / replaying.** Use the ROS 2 Jazzy container
`/mnt/data/vras/data/helhest-jazzy.sif` (rviz2, rosbag2 with mcap support;
everything in the bags deserialises except the single
`compass_interfaces/msg/Azimuth` topic). There is no ROS 2 image on
`subtdata` itself — copy the bag or run through the container on a machine
that mounts both:

```bash
singularity exec --bind /data,/mnt/data /mnt/data/vras/data/helhest-jazzy.sif \
    bash -c 'source /opt/ros/jazzy/setup.bash && ros2 bag play <bag_dir> --clock'
```

**Not yet extracted (same sensor generation):**
`/data/temesvar_04_26/{long_term_navigation,road_following,astar_nav,helhest_taros,flying_icp}`
(22 bags, 735 GB, 20–24 Apr 2026) contain front-LiDAR + camera + odometry and
could be added with the same tooling.
`/data/outdoor-dataset/26-04-14-temesvar-junior` (Apr 2026) is the
HELHEST *junior* platform with a different sensor suite and is not directly
compatible. Earlier recordings
(`/data/alpine-trial/*`, Jun–Sep 2025) predate the LiDAR and are not usable
here.

## 7. New sensor generation

Since August 2026 HELHEST carries an **Odin1** integrated LiDAR–camera–IMU
unit (organised 256 × 192 clouds at ~14 Hz, cloudini-compressed; 1600 × 1296
camera at ~29 Hz; onboard SLAM odometry) and a **Fixposition** GNSS/INS
(georeferenced ENU poses at 10 Hz), with the same wheel/IMU proprioception.
The extraction tool in this repository handles both generations and writes the
same scene layout (plus `joint_setpoints.npy` and `cmd_vel.npy` when
available). The first recordings with the new stack
(`subtdata:/data/helhest/26-08-01-dobyvani_hradu`, `26-08-21-stromovka-roads`)
exposed integration bugs — missing LiDAR-to-base extrinsic, empty wheel
velocities, an unreferenced IMU — and are not part of the dataset; see the
repository README for details and the collection checklist.

## 8. Tooling

Extraction, validation and statistics scripts live in
[`dataset_tools`](https://github.com/vlk-jan/dataset_tools):

```bash
helhest-extract <bag_dir> <out_dir> --cam /luxonis/oak/rgb/image_raw/compressed \
    --lidar /livox/lidar_front --odom /odom --imu /ekf/imu/data      # old generation
helhest-validate /mnt/data/vras/data/HELHEST --csv scenes.csv        # completeness/consistency report
```

## 9. Citation, licence, contact

Data collected by the Vision for Robotics and Autonomous Systems group (VRAS),
Faculty of Electrical Engineering, Czech Technical University in Prague.
Contact: Jan Vlk (vlkjan6@fel.cvut.cz).

Licence: **[to be decided — the tooling is BSD-3-Clause; the data itself has no licence yet and contains identifiable people]**. If you use the dataset, please cite **[paper/report TBD]**.
