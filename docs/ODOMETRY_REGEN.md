# Regenerating the v1 ground-truth odometry ("level 2")

Status: **plan, nothing run yet** (2026-09-01). Owner of the run: the laptop that
generated v1 (`~/Documents/setprop/flio_ws`). Consumer: DRIFT, via a HELHEST v1.2
root on RCI.

## Why

`odometry.txt` in every v1 scene is the pose of the **Livox-front internal IMU**,
not of `base_link` (FAST-LIO publishes the IMU state and hard-codes
`child_frame_id = "base_link"`; see `docs/DATA.md`, caveats). The lidar is
mounted upside-down, so yaw is inverted and y/z mirrored relative to the clouds.
Level 1 (a derived fix, no bags) conjugates the existing files by the mount
extrinsic and is correct in rotation and to a few cm in translation — the
residual comes from `extrinsic_est_en: true`, which re-estimated the lidar↔IMU
offset online in every run. Level 2 re-runs the odometry from the raw bags and
is the only way to get:

1. an exact `base_link` trajectory (fixed extrinsics, no per-run drift);
2. ground truth for the scenes that lack it. **Only ~17 of 43 scenes are fully
   covered.** 11 have no `odometry.txt` (0001, 0007, 0014, 0015, 0019, 0021,
   0026, 0027, 0033, 0034, 0037 — 0001's is a bookkeeping loss: its file was
   moved to 0002 by hand), and in 15 more FAST-LIO was killed on drift before
   the scene ended, so the file covers only part of it (fraction of frames):
   0018 13 %, 0023 19 %, 0004 22 %, 0006 28 %, 0011 42 %, 0009 45 %, 0005 45 %,
   0002 49 %, 0000 57 %, 0030 64 %, 0017 70 %, 0028 70 % (starts 47 s late),
   0035 87 % (starts 207 s late), 0036 94 %, 0031 96 %. 0033/0034/0037 are
   mostly stationary and 0021 has a tip-over — some may be unrecoverable and
   should then be marked as such;
3. the camera distortion coefficients (v1 kept only `K`; `extract_data.py` now
   writes `camera_info.yaml` with `D`, `R`, `P` as well);
4. a recorded, reproducible pipeline instead of a USB-drive working copy.

## Inputs

Raw bags on `subtdata:/data` (rosbag2, 913 GB total), mapped 1:1 to scenes in
`docs/scene_bags.csv`:

| session dir on subtdata | bags | GB | scenes | without odometry |
|---|---|---|---|---|
| `/data/robotour2025/helhest_250912` | 2 | 73 | 0000–0001 | 0001 |
| `/data/robotour2025/helhest_250913` | 8 | 284 | 0002–0009 | 0007 |
| `/data/robotour2025/helhest_250914` | 2 | 135 | 0010–0011 | — |
| `/data/outdoor-dataset/25-10-09-temesvar-helhest/combined` | 18 | 194 | 0012–0029 | 0014 0015 0019 0021 0026 0027 |
| `/data/outdoor-dataset/26-02-16-milicovsky-haj` | 13 | 228 | 0030–0042 | 0033 0034 0037 |

The USB drive "VRAS 1TB" holds a partial copy (two milicovsky_haj bags were
deleted after extraction) — do not rely on it.

Tooling, all on the laptop:

- `~/Documents/setprop/flio_ws/src/fast_lio` — `vlk-jan/FAST_LIO_ROS2`. **The
  config in force during v1** (`git show e6cbec2^:config/mid360.yaml`, with the
  topic names edited locally at the time): `lid_topic: /livox/lidar_front`,
  `imu_topic: /livox/imu_front`, `lidar_type: 1` (Livox native),
  `extrinsic_est_en: true`, `extrinsic_T: [-0.011, -0.02329, 0.04412]`,
  `extrinsic_R: I`. The bags were replayed with
  `ros2 bag play … --topics /livox/lidar_front /livox/imu_front` — FAST-LIO
  consumed the raw front topics directly; there was **no body filter and no
  merge step** in the v1 loop. HEAD (`e6cbec2`, 2026-04-21, a month after the
  dataset) changed this to `/livox/lidar_both_filtered` and `lidar_type: 4`;
  that config never produced any v1 file and must not be used for the re-run.
- `flio_ws/src/dataset_tools` — second working copy with the uncommitted
  `--index 30` scene-offset patch; `launch/body_filter.launch.py` (two
  `pcl_ros::CropBox` nodes on `/livox/lidar_front` → `/livox/lidar_front_filtered`),
  `launch/odom_recorder.launch.py` + `dataset_tools/odom_node.py`
  (`OdomRecorder`: `/Odometry` → `trajectory.txt`, append mode, respawn).
  `launch/body_filter.launch.py` was an experiment; nothing consumed its output.
- `~/Work/helhest-singularity/helhest-jazzy.sif` (2026-02-18) — ROS Jazzy, the
  only image that can deserialize every v1 topic (`compass_interfaces`
  source under `/opt/helhest/src`). **Copy it aside before any rebuild**: the
  merged upstream def deletes that source dir and writes the same filename.

## Environment (laptop, checked 2026-09-01)

- **No merge step to reconstruct** (see Tooling); the fallback test is moot.
- `/Odometry` header = lidar scan-end time; keep it, it is what `data_ts.txt`
  was matched against.
- **Run inside the sif** (`~/Work/helhest-singularity/helhest-jazzy.sif`;
  backup `helhest-jazzy-2026-02-18.sif` is byte-identical). The host has no
  ROS. `flio_ws` was built in the container, but its `install/` symlinks dangle
  after the `src/FAST_LIO → src/fast_lio` rename — a `colcon build` inside the
  sif is required before any run.
- subtdata: use the FQDN `subtdata.felk.cvut.cz` (bare `subtdata` does not
  resolve from the laptop); key-based ssh works; 1.1 TB free there. Laptop has
  ~390 GB free → fetch one bag, process, delete, next.
- Wall time: 7.5 h of scene time at `--rate 1.0`, 15 h at `0.5`, plus fetch.
  Longest scenes 0011 (47 min), 0002 (41), 0019 (34), 0035 (26).
- **Determinism**: rosbag2 replay is timing-dependent. Fix `--rate` (0.5
  recommended), `--clock`, and log dropped-message counts; run scene 0013
  twice and report the relative-pose difference as the reproducibility floor.
- **Bag names are local time (CEST), `data_ts.txt` is UTC** — 2 h apart. Match
  by `docs/scene_bags.csv`, not by eye.

## Changes to make before running

FAST-LIO fork (`config/mid360.yaml`, `src/laserMapping.cpp`) — done on the
laptop branch `v1-odometry-regen` (0dcaceb), not pushed:

- `extrinsic_est_en: false`; keep `extrinsic_T/R` at the Mid-360 datasheet
  values above.
- Publish an honest frame: leave `/Odometry` as the IMU pose but label it
  `livox_front_imu` (all three hard-coded `base_link` labels in
  `laserMapping.cpp`: body cloud, odometry child, TF broadcast) and let the
  recorder do the transform — one formula, in Python, unit-testable,
  identical to level 1:

  ```
  X  = T^base_lidar : R = diag(1,-1,-1), t = (0.17, 0, 0.145)
  L  = T^lidar_imu  : R = I,             t = (+0.011, +0.02329, -0.04412)
  X' = X @ L        : R = diag(1,-1,-1), t = (0.181, -0.02329, 0.18912)
  T'(t)   = T_odom_imu(t) @ inv(X')          # base pose, IMU-anchored frame
  T_out(t) = inv(T'(t0)) @ T'(t)             # anchored at the first base pose
  ```

  Verify the mount numbers against the bag's `/tf_static`
  (`base_link → livox_front_frame`) with `dataset_tools.extract_data.get_transform`
  rather than trusting the constants.
  Beware: `/livox/imu_front` messages carry `frame_id = livox_frame`, and the
  bags publish a *separate* `base_link → livox_frame` static TF at
  t = (0.11, 0, 0.145) (driver default name, uncalibrated). Do not use it —
  the IMU sits inside the Mid-360 at the datasheet offset from the lidar
  origin, and the calibrated mount is `livox_front_frame` at x = 0.17.
- `gravity_align.npy` needs no change: it is roll/pitch of `/ekf/imu/data`,
  whose frame `imu` is `base_link` (identity static TF) — i.e. it is already in
  the frame the regenerated odometry will be in.
- Recorder (`odom_node.py`) — done on dataset_tools branch
  `odometry-regen-recorder` (2480429), not pushed: open the file once with truncation (not append per
  message), refuse to start if the output exists, write sorted, and keep the
  header/format byte-compatible:
  `# Timestamp(sec) r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz 0 0 0 1`,
  `%.6f` throughout. Also dump the raw IMU-frame pose alongside
  (`odometry_imu_raw.txt`) so the transform can be re-done without a re-run.

Extraction (`dataset_tools`, already done on `main`):

- `camera_info.yaml` with the full `CameraInfo` (`D`, `R`, `P`).
- Static-TF composition fixed (clouds land at the correct x); clouds should be
  re-extracted too so the root is self-consistent (clouds, `transforms.npy`,
  `camera_info.yaml`, `odometry.txt` from one pass). Zero-returns: filter
  `range == 0` points at extraction (they are 53–70 % of each frame).

## Why FAST-LIO diverged, and how to keep more of each scene

The v1 runs were killed at the first divergence, so one bad stretch cost the
whole remaining scene (0011 lost 27 min). Coverage correlates *positively*
with terrain roughness (r ≈ +0.57 over the 15 truncated scenes): FAST-LIO
failed on flat, featureless ground, not on hard ground. That is geometric
degeneracy — a ground plane constrains z/roll/pitch only — and the mount
maximises it: the Mid-360's −7°…+52° vertical FOV, upside-down at 0.145 m,
sees the ground out to the horizon plus 7° above it, with the rear blocked by
the body. Open ground with no vertical structure near the horizon gives
nothing to register against; trees, ruts and vegetation do. Two process
factors made it worse: real-time replay (dropped scans during turns) and the
single-run-per-scene policy.

Levers, in order of payoff per effort:

1. **Segment-wise restarts.** DRIFT needs relative motion over 1–2 s windows,
   not a globally consistent trajectory. Detect divergence (see below),
   re-initialise FAST-LIO at that time, and write `odometry.txt` as healthy
   segments with gaps. Coverage becomes "all but the degenerate seconds"
   instead of "everything before the first failure". The recorder tolerates
   gaps and the DRIFT loader already skips uncovered windows.
   Divergence detector (in the recorder, all from the bag): flag when the
   FAST-LIO speed differs from the live `/odom` speed by > 0.5 m/s for > 1 s,
   or yaw rate differs by > 0.5 rad/s, or the pose jumps > 0.5 m between
   consecutive scans, or FAST-LIO's own state covariance / ESKF iteration
   count blows up. Record the flag stream; do not silently drop.
2. **Replay slowly, tune for a sparse downward lidar.** `--rate 0.5 --clock`,
   `point_filter_num: 1` (only 4–7 k real returns per scan; the default 3
   leaves ~1.5 k), `filter_size_surf: 0.3`, `filter_size_map: 0.5`,
   `blind: 0.5` (also removes the zero-return blob), `max_iteration: 5`,
   `extrinsic_est_en: false`. Verify the deskew path on the Livox
   `PointCloud2` (`timestamp` field: absolute ns vs per-scan offset,
   `timestamp_unit`) — wrong deskew shows up as divergence on turns.
3. **Wheel-odometry prior for flat ground: Kinematic-ICP** (KISS-ICP group;
   ICP with a wheel-odometry motion model and a ground-robot constraint).
   Targets exactly the flat-open-ground failure; `/odom` and `/joint_states`
   are in every bag. Evaluate on the three worst scenes (0018, 0004, 0011) if
   levers 1–2 do not recover them.
4. **Fallback for residual gaps:** the live `/odom` (already in
   `transforms.npy`), written flagged as such and never used as ground truth
   for slip/traction — wheel odometry as GT is circular there.
5. Not levers: Point-LIO (helps aggressive motion such as the 0021 tip-over,
   not degeneracy); reverse-time replay. Stationary scenes (0033, 0034, 0037)
   need no LIO — a near-identity trajectory is the correct GT.

## Dry run (revised)

Scenes: **0018** (worst truncated, 13 % covered; 12 min) and **0013** (fully
covered, 0.7 min; the regression reference). Order: 0013 first (fast, proves
the pipeline), then 0018.

1. `colcon build` in the sif; fetch the two bags from
   `subtdata.felk.cvut.cz` (`docs/scene_bags.csv`, bag names are CEST).
2. Run A — v1 recipe on 0013 (`extrinsic_est_en: true`, default filters,
   `--rate 1.0`): confirm the regenerated raw IMU pose matches the published
   `odometry.txt` (relative-pose error over 10 s windows) — that pins the
   frame provenance beyond doubt.
3. Run B — tuned recipe (lever 2) on 0013, twice: reproducibility floor.
4. Run C — tuned recipe on 0018 with the divergence detector logging but NOT
   restarting: where and how does it fail (timestamp of first flag, which
   test fired, what the terrain looks like in the image at that time)?
5. Run D — tuned recipe + segment restarts on 0018: coverage achieved
   (fraction of `data_ts.txt` frames inside some segment) vs the 13 % of v1;
   per-segment length histogram.
6. Report: coverage for 0018 under C and D, the failure signature, the
   0013 reproducibility floor and the level-1-vs-level-2 relative-pose error;
   then a recommendation on whether Kinematic-ICP (lever 3) is needed.

## Per-scene run

For each row of `docs/scene_bags.csv`:

1. fetch the bag from subtdata (`rsync`, one at a time);
2. `ros2 bag play --rate 0.5 --clock --topics /livox/lidar_front /livox/imu_front /odom`
   + FAST-LIO (tuned) + recorder with segment restarts inside the sif; keep
   the divergence-flag stream per scene;
3. `extract_data.py` for clouds/images/`camera_info.yaml`/`transforms.npy`
   (`--start_index` gives the scene number; `--frame_fix` is NOT needed — the
   new odometry is already `base_link`);
4. `validate_scene.py`;
5. push the scene to RCI (`/mnt/personal/vlkjan6/HELHEST_v1_2/NNNN`), delete
   the local bag.

## Acceptance

- For the 32 scenes with existing odometry: relative-pose error between the
  new file and the level-1 conjugated file — expect mean translation error at
  the cm level and rotation at the noise level; anything larger points at a
  wrong extrinsic, not at FAST-LIO.
- Sign checks that failed on v1: yaw rate vs wheel-speed differential
  (`joint_states.npy`) positively correlated in every scene; DRIFT's
  simulator-vs-GT yaw test agrees on strong turns.
- Driven cells in the rebuilt height maps are ground (valid-cell IoU of the
  proprio labels, DRIFT side).
- `validate_scene.py` clean on every scene; `scene_bags.csv` gains a column
  with the FAST-LIO run id/date; DATA.md gets a v1.2 changelog entry.
