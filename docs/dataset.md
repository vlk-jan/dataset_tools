# Dataset layout

HELHEST is a small three-wheeled skid-steer field robot built at CTU in Prague
(VRAS group). It carries an **Odin1** lidar-camera-IMU unit and a
**Fixposition** GNSS/INS, plus wheel encoders. A HELHEST root holds one
directory per recording (scene), with every per-frame file index-aligned,
alongside root-level metadata files.

## Sensors

| Sensor | Topic | In a scene as |
| --- | --- | --- |
| Odin1 lidar, 256 x 192 raster at ~15 Hz | `/odin1/cloud_filtered/cloudini` | `clouds/`, in `base_link` |
| Odin1 camera, 1600 x 1296, fisheye | `/odin1/image/compressed` | `images/`, `camera_info.yaml`, `T_cam_to_base.txt` |
| Odin1 lidar-inertial odometry, 14.5 Hz, 6-DoF | `/odin1/odometry` | `odometry.txt`, `transforms.npy`, `gravity_align.npy` |
| Odin1 IMU, 400 Hz | `/odin1/imu` | used to deskew the sweeps |
| Fixposition GNSS/INS | `/fixposition/odometry_llh`, `/fixposition/odometry_enu` | `gps.npy`, and the check in `pose_check.txt` |
| Wheels | `/joint_states`, `/joint_setpoints`, `/cmd_vel` | `joint_states.npy`, `joint_setpoints.npy`, `cmd_vel.npy` |

Camera and lidar run 1:1, yielding approximately 15
frames per second.

## Root layout

```
<root>/
├── 0000/ … 0035/            one directory per recording (a scene)
├── manifest.json            what the release is: version, scenes, sessions, held-out scenes, defects
├── splits.json              train / validation split
├── scenes.csv               scene, bag, start time, frame count
├── keyframes.lst            usable frames of every scene
├── keyframes_train.lst      ... of the training scenes
├── keyframes_heldout.lst    ... of the held-out scenes
├── README.md                the release described in prose
├── validation.csv           helhest-validate's report
├── disk_usage.json          sizes per scene and modality
└── atlas/                   the viewer's curated index and thumbnails
```

The held-out scenes are listed in `manifest.json` under `held_out.scenes`.
A loader that trains on a root must keep them out of every training fold; the
provided keyframe lists (`keyframes_train.lst`) already exclude them.

## Scene files

`F` is the number of frames in the scene. Every per-frame file and every
`(F, ...)` array is index-aligned: `images/00042.jpg`, `clouds/00042.npy`,
`data_ts[42]`, `transforms[42]`, and `gravity_align[42]` all describe the exact
same instant.

### Per frame

| Path | Shape, dtype | Content |
| --- | --- | --- |
| `images/NNNNN.jpg` | 1600 x 1296 x 3 JPEG | the camera's raw compressed JPEG bytes, unmodified |
| `clouds/NNNNN.npy` | (N, 4) `float32` | x, y, z in metres and intensity, in `base_link`, deskewed onto the frame's instant. `N` varies; NaN and all-zero points are dropped. The robot's own body reflections are retained |
| `height_maps/NNNNN.npz` | 128 x 128 `float64`, key `height_map` | terrain height in metres, gravity-aligned base frame, 0.1 m cells, ±6.4 m. Rows correspond to x forward, columns to y left |
| `height_masks/NNNNN.npz` | 128 x 128 `uint8`, key `validity` | source of each cell's height: 0 extrapolated, 1 interpolated, 2 measured, 3 wheel contact |
| `data_ts.txt` | (F,) | the frame's instant, Unix seconds |
| `pcd_ts.txt` | (F,) | the lidar timestamp, on the same latency-corrected clock |
| `transforms.npy` | (F, 4, 4) `float64` | base pose per frame relative to frame 0, interpolated from odometry |
| `gravity_align.npy` | (F, 3, 3) `float64` | `Ry(pitch) @ Rx(roll)` of the base: applied to a cloud (`pts @ R.T`), it levels the ground. Yaw is omitted |

A frame with too few lidar returns produces no height map.

### Per scene

| Path | Content |
| --- | --- |
| `odometry.txt` | ground truth: base poses at the odometry's full rate; one row per pose containing timestamp followed by the row-major 4 x 4 matrix |
| `odometry_segments.txt` | segment table for `odometry.txt`; see [Ground truth](#ground-truth) |
| `camera_info.yaml` | full camera calibration: resolution, distortion model and coefficients, `K`, `R`, and `P` |
| `camera_intrinsics.txt` | the 3 x 3 intrinsic matrix `K` |
| `T_cam_to_base.txt` | 4 x 4 extrinsic transform: `p_base = T @ p_cam_optical` |
| `joint_states.npy` | (M, 1 + 3J): timestamp, followed by positions, velocities, and efforts of the J wheels at full rate (not frame-aligned) |
| `joint_names.txt` | wheel joint names matching the column order in `joint_states.npy` |
| `joint_setpoints.npy` | commanded wheel states with the same layout as `joint_states.npy`, where recorded |
| `cmd_vel.npy` | (M, 7): timestamp, linear and angular velocity in the base frame, where recorded |
| `gps.npy` | (M, 8): timestamp, latitude, longitude, altitude, fix status, and east, north, and up variances (not frame-aligned) |

### Extraction records

These describe how the scene was generated and provide provenance. They are the
first place to look when something in a scene seems wrong.

| Path | Content |
| --- | --- |
| `extract_meta.json` | the extraction run (git commit, all options), the source bag (topics and message counts), all streams with data gaps exceeding 1 s, and frame filtering records |
| `source_bag.txt` | the bag the scene originated from; also serves as the completion sentinel for the scene |
| `time_align.txt` | the estimated publish latency subtracted from the frame clock |
| `pose_check.txt` | odometry compared against the Fixposition at `base_link`: jumps, displacement disagreement over 1, 2, and 5 s, clock lag, and tilt |
| `gnss_guard.txt` | frames dropped for having no valid underlying pose |
| `tf_pins.txt` | fixed extrinsics used by the scene compared against what the bag itself published |
| `odom_frame.txt` | the rotation applied to odometry by `helhest-odom-frame` |
| `height_maps.json` | the configuration settings used to build height maps |

## Coordinate frames

- **`base_link`**: x forward, y left, z up, at the robot base. Clouds, poses,
  and height maps are all expressed in this frame.
- **Camera**: `T_cam_to_base.txt` maps points from the camera's optical frame
  (x right, y down, z forward) to `base_link`. The camera is a fisheye
  (equidistant model in `camera_info.yaml`), so undistort before projecting.
- **Gravity**: `gravity_align.npy` holds the base's roll and pitch per frame.
  It is the only source of absolute attitude in a scene.
- **Earth**: `gps.npy` is the only georeference. It is intended for mapping and
  comparison, not ground truth.

## Ground truth

`odometry.txt` is the Odin1's lidar-inertial odometry, expressed at
`base_link`. It is split into **segments**, listed in `odometry_segments.txt`,
which are numbered from 1. Each segment starts at the identity pose at its
first sample and runs from there. For each segment, the table gives its time
span, its pose count, and why it ended:

| Reason | Meaning |
| --- | --- |
| `gap` | the odometry was silent for more than 1 s |
| `teleport` | a jump faster than 6 m/s, such as the Odin1 relocalising |
| `fix_lost` | the pose source had no valid solution |
| `end_of_run` | the recording ended |

**Never bridge two segments.** Within a segment, compute relative motion with
`inv(T0) @ Ti`. A segment's frame is defined by the robot's attitude at its
first pose and is therefore not level: obtain absolute roll and pitch from
`gravity_align.npy`.

`transforms.npy` is the same odometry sampled at each frame. Where a frame has
no pose on either side of it, the extractor drops the frame rather than
extrapolating.

## Timing

`data_ts.txt` is the instant every per-frame file of that frame describes:

- **The publish latency is removed.** The Odin1 stamps some messages upon
  publish rather than capture, with a delay that varies under load. The
  extractor measures this delay per scene and subtracts it (`time_align.txt`).
- **Clouds are deskewed.** The Odin1 scans its raster from bottom to top over
  67 ms. Every return is moved to where the robot was at `data_ts`, using the
  odometry and the IMU, so a cloud is a snapshot at the exact instant of its
  camera image.

Full-rate streams (`odometry.txt`, `joint_states.npy`, `gps.npy`, ...) carry
their own timestamps on the same Unix clock.

## Recordings

The current release, HELHEST v2.5, holds 36 scenes from seven sessions in
September and October 2026:

| Scenes | Date | Site |
| --- | --- | --- |
| 0000-0001 | 2026-09-14 | Stromovka park, Prague: Robotour test runs on cobbles, gravel and asphalt |
| 0002 | 2026-09-19 | Stromovka park: the Robotour 2026 finals, driven autonomously |
| 0003-0006 | 2026-09-22 | Lipník nad Bečvou army UGV polygon: ploughed field, mud, banks, ballast |
| 0007-0015 | 2026-09-28 | Temešvár: tall-grass meadow, sprints, wheel-slip tests, two roll-overs, forest track |
| 0016-0021 | 2026-09-29 | Temešvár: the same meadow, late afternoon into dusk |
| 0022-0028 | 2026-10-02 | Temešvár: pine and mixed forest, the meadow |
| 0029-0035 | 2026-10-07 | Stromovka park, off the paths: lawn, woodland, tall weeds |

`manifest.json` lists each session, the held-out scenes, and the known defects
of every scene.
