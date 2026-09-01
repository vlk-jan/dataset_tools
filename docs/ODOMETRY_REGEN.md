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
2. ground truth for the **11 scenes that have none** — 0001, 0007, 0014, 0015,
   0019, 0021, 0026, 0027, 0033, 0034, 0037 (FAST-LIO was killed on drift
   during the original runs; 0033/0034/0037 are mostly stationary, 0021 has a
   tip-over — some may be unrecoverable and should then be marked as such);
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

- `~/Documents/setprop/flio_ws/src/fast_lio` — `vlk-jan/FAST_LIO_ROS2` at
  `e6cbec2`, config `config/mid360.yaml` (`lid_topic: /livox/lidar_both_filtered`,
  `imu_topic: /livox/imu_front`, `extrinsic_est_en: true`,
  `extrinsic_T: [-0.011, -0.02329, 0.04412]`, `extrinsic_R: I`).
- `flio_ws/src/dataset_tools` — second working copy with the uncommitted
  `--index 30` scene-offset patch; `launch/body_filter.launch.py` (two
  `pcl_ros::CropBox` nodes on `/livox/lidar_front` → `/livox/lidar_front_filtered`),
  `launch/odom_recorder.launch.py` + `dataset_tools/odom_node.py`
  (`OdomRecorder`: `/Odometry` → `trajectory.txt`, append mode, respawn).
- `~/Work/helhest-singularity/helhest-jazzy.sif` (2026-02-18) — ROS Jazzy, the
  only image that can deserialize every v1 topic (`compass_interfaces`
  source under `/opt/helhest/src`). **Copy it aside before any rebuild**: the
  merged upstream def deletes that source dir and writes the same filename.

## Unknowns to resolve first (laptop)

1. **What produced `/livox/lidar_both_filtered`?** The committed launch emits
   `_front_filtered`; the config consumes `_both_filtered`. Look in the
   flio_ws working copy (launch files, `git diff`, `git stash list`), shell
   history, and the USB drive. Hypothesis: a merge of front + rear CropBox
   outputs in the front-lidar frame (the rear Livox returned zero points in
   every v1 bag, so the merge would be a no-op). Test: run FAST-LIO on
   `_front_filtered` for scene 0013 and compare with the published
   `odometry.txt` — relative-pose agreement at the noise level (< 2 cm, < 0.5°
   over 10 s windows) means the merge step can be dropped.
2. **Timestamp source**: `/Odometry` header = lidar scan-end time; keep it, it is
   what `data_ts.txt` was matched against.
3. **Where to run**: laptop natively (needs `subtdata` reachable and ~50 GB free
   per bag, streamed one bag at a time) or RCI cpufast (needs `fast_lio` +
   `livox_ros_driver2` built inside the sif; `flio_ws/install` is a native
   build and will not run in the container as-is). Report disk/network
   reality; the laptop-native route is the one that already worked.
4. **Determinism**: rosbag2 replay is timing-dependent. Fix `--rate` (0.5 or
   lower), `--clock`, and log dropped-message counts; run scene 0013 twice and
   report the relative-pose difference between the two runs as the
   reproducibility floor.

## Changes to make before running

FAST-LIO fork (`config/mid360.yaml`, `src/laserMapping.cpp`):

- `extrinsic_est_en: false`; keep `extrinsic_T/R` at the Mid-360 datasheet
  values above.
- Publish an honest frame: either (a) leave `/Odometry` as the IMU pose but set
  `child_frame_id = "livox_front_imu"` and let the recorder do the transform,
  or (b) compose in the node. Option (a) is preferred — one formula, in
  Python, unit-testable, identical to level 1:

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
- Recorder (`odom_node.py`): open the file once with truncation (not append per
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

## Per-scene run

For each row of `docs/scene_bags.csv`:

1. fetch the bag from subtdata (`rsync`, one at a time);
2. `ros2 bag play --rate R --clock` + body filter + FAST-LIO + recorder inside
   the sif; watch for the divergence that killed the original runs, and log
   where it starts if it happens;
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
