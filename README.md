# dataset_tools

Tools for turning **HELHEST** robot recordings (ROS 2 bags) into a flat,
learning-friendly dataset: synchronised camera frames, point clouds in the
robot base frame, odometry, gravity alignment and proprioception.

Bags are read with the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags)
library, so **no ROS installation is required** for extraction — only the
optional `odom_recorder` node needs `rclpy`.

| Tool | Entry point | What it does |
| --- | --- | --- |
| `dataset_tools/extract_data.py` | `helhest-extract` | bags -> scene directories |
| `dataset_tools/validate_scene.py` | `helhest-validate` | check extracted scenes |
| `dataset_tools/sort_bag.py` | `helhest-sort-bag` | rewrite a bag sorted by header stamp |
| `dataset_tools/viewer/` | `helhest-viewer` | browse an extracted dataset root in a web viewer |
| `dataset_tools/disk_usage.py` | `helhest-disk-usage` | bytes per scene, modality and file, for the viewer's disk-usage panels |
| `dataset_tools/image_format.py` | `helhest-image-format` | declare (or read back) whether a root's images are the camera's JPEG or a PNG |
| `dataset_tools/odom_node.py` | `ros2 run dataset_tools odom_recorder` | record odometry to a text trajectory |

## Quickstart: from a recording to the viewer

For a bag recorded on the current robot (Odin1 + Fixposition, 2026-08 onwards):

```bash
python3 -m venv .venv && .venv/bin/pip install -e .                  # extraction: Python >= 3.10
.venv/bin/helhest-extract  /path/to/bags /path/to/dataset --jobs 8   # one scene per bag: 0000/, 0001/, ...
.venv/bin/helhest-validate /path/to/dataset --csv /path/to/dataset/validation.csv

python3.12 -m venv .venv-viewer && .venv-viewer/bin/pip install -e ".[viewer]"   # viewer: Python >= 3.12
.venv-viewer/bin/helhest-viewer /path/to/dataset                     # then open http://127.0.0.1:8080/
```

- The defaults are the current rig's topics; bags from the Livox + OAK-D robot
  need the flags under [Old bags](#old-bags-livox--luxonis-oak).
- The Odin1 clouds are cloudini-compressed. The decoder ships prebuilt
  (`dataset_tools/libcloudini_shim.so`, glibc >= 2.32 -- Ubuntu 22.04 and
  newer); on an older system run `cloudini_shim/build_shim.sh` once.
- Extraction resumes: a bag whose scene is complete is skipped, so an
  interrupted run can simply be started again. `--overwrite` redoes them.
- The viewer needs nothing beyond the scene directories; the curated index
  (`atlas/`) and the train/validation split are optional extras a root may
  carry. It is unauthenticated -- keep it on `127.0.0.1` or behind an SSH
  tunnel (see [docs/VIEWER.md](docs/VIEWER.md#who-can-reach-it)).

The HELHEST dataset itself is not in this repository; [docs/DATA.md](docs/DATA.md)
describes it and where it lives.

## The two HELHEST sensor generations

The robot was re-equipped in 2026, and the two generations need different
topics. Everything else about the extraction is identical.

| | **Old stack** (until 2026) | **New stack** (2026-08 onwards) |
| --- | --- | --- |
| Lidar | Livox front lidar, `/livox/lidar_front` (`sensor_msgs/PointCloud2`) | Odin1, `/odin1/cloud_filtered/cloudini` (`point_cloud_interfaces/CompressedPointCloud2`) |
| Camera | Luxonis OAK-D PoE, `/luxonis/oak/rgb/image_raw/compressed` | Odin1 camera, `/odin1/image/compressed` |
| Odometry | `/odom` (live robot odometry; the published GT was FAST-LIO run offline) | `/fixposition/odometry_enu` (Fixposition GNSS/INS) |
| Orientation (gravity) | `/ekf/imu/data` | `/fixposition/odometry_enu` |
| Proprioception | `/joint_states`, `/cmd_vel` | `/joint_states`, `/joint_setpoints`, `/cmd_vel` |
| GNSS position | Septentrio receiver (any `sensor_msgs/NavSatFix` topic) | `/fixposition/odometry_llh` (the ENU odometry, geodetically) |

The command-line defaults target the **new** stack; see
[Usage](#usage) for the old-stack invocation.

## Installation

### Plain Python (venv)

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[test]"
```

Python >= 3.10 (not declared in `pyproject.toml`: colcon's setup.py introspection cannot evaluate `requires-python`). Dependencies: `rosbags`, `numpy`, `opencv-python-headless`.

### Container (CTU cluster)

Build the venv with the container's own Python, then run everything through the
container. Note the **`--bind`**: the data shares are not mounted inside it by
default, so neither the bags nor the dataset are reachable without them.

```bash
SIF=/mnt/data/vras/data/helhest-jazzy.sif
singularity exec --bind /mnt/data $SIF python3 -m venv .venv
singularity exec --bind /mnt/data $SIF .venv/bin/pip install -e ".[test]"
singularity exec --bind /mnt/data $SIF \
    .venv/bin/python -m dataset_tools.extract_data <bags> <output> --jobs 8
```

The `.venv` is not in the repository (it is gitignored) -- `slurm/slurm_extract.batch`
and the Slurm example below both expect one built this way at `$REPO/.venv`.

Add further `--bind` options for any other filesystem you read or write
(e.g. `--bind /mnt/personal`).

### ROS 2 (only for `odom_recorder`)

The package still builds as an `ament_python` package:

```bash
colcon build --packages-select dataset_tools
ros2 run dataset_tools odom_recorder --ros-args -p topic:=/Odometry \
    -p file_path:=trajectory.txt
```

`setup.py`/`package.xml` are kept for that build; `pyproject.toml` is used by
plain `pip`. Both declare the same console scripts.

### Building the cloudini shim (new bags only)

The Odin1 clouds are cloudini-compressed. Decoding goes through a small C ABI
wrapper around [cloudini](https://github.com/facontidavide/cloudini):

```bash
cloudini_shim/build_shim.sh          # clones cloudini, builds dataset_tools/libcloudini_shim.so
```

Needs `cmake`, `g++` (C++20), `libzstd-dev`, `liblz4-dev` — all present in
`helhest-jazzy.sif`, so run it inside the container if in doubt. Old bags do
not need the shim.

## Usage

### New bags (2026-08 Odin1 + Fixposition)

```bash
python -m dataset_tools.extract_data <dir-with-bags> <output-dir> --jobs 8
```

which is equivalent to spelling out the defaults:

```
--cam /odin1/image/compressed --lidar /odin1/cloud_filtered/cloudini \
--odom /fixposition/odometry_enu --imu /fixposition/odometry_enu --gps auto
```

When `--odom` is a Fixposition topic, the extractor also writes the
ground-truth files (`--gt auto`): `odometry.txt` (base_link pose at full
rate, split into segments each re-anchored to the identity) and
`odometry_segments.txt` (never interpolate across segment boundaries).
Segments break at >1 s gaps, at GNSS teleports, and at the holes
`--gnss_guard` leaves where the receiver had no solution (`fix_lost`);
one-sample spikes are discarded. This matches the HELHEST v1.2 contract, so
segment-aware consumers need no changes, with one difference: segments in a
freshly written table are **numbered from 1**, so "segment 3 of 4" is the
third one. Roots written before this (v1.2 through v1.5) are numbered from 0
and are read as they stand -- a segment is addressed by the number in its own
file, never by its position.

The Odin1 mount extrinsic is not in the bags' TF tree; pass it once known:

```bash
python -m dataset_tools.extract_data <bags> <out> \
    --lidar_extrinsic X Y Z ROLL PITCH YAW
```

### Old bags (Livox + Luxonis OAK)

```bash
python -m dataset_tools.extract_data <dir-with-bags> <output-dir> \
    --cam /luxonis/oak/rgb/image_raw/compressed --lidar /livox/lidar_front \
    --odom /odom --imu /ekf/imu/data
```

Point cloud field layouts are parsed from the `PointCloud2` message, so any
lidar works.

### Options

| Flag | Meaning |
| --- | --- |
| `--jobs N` | worker processes, one bag each (default 4; `1` runs in-process) |
| `--overwrite` | re-extract scenes that are already complete (a scene's previous outputs are removed first) |
| `--gt auto\|on\|off` | write `odometry.txt` + `odometry_segments.txt` from the odometry stream; `auto` does so only for Fixposition topics |
| `--scenes 0 3 5-9 <bag-name>` | only these scene indices / ranges / bag directories |
| `--start_index N` | number the first scene `N` instead of `0000` |
| `--gps TOPIC` | GNSS fix topic written to `gps.npy`; `auto` (default) finds the `NavSatFix` topic carrying messages, `off` writes none |
| `--base_frame` | target frame for clouds, odometry and gravity (default `base_link`) |
| `--gnss_guard auto\|on\|off` | drop stretches where the GNSS/INS published a reset solution instead of a pose, and every frame left without a real one -- both the frames those stretches strand and the frames the pose stream never covered. `auto` turns it on for an absolute source (Fixposition) and off for wheel odometry. Recorded per scene in `gnss_guard.txt` |
| `--time_align auto\|off\|S` | how to put the frame clock on the pose clock. `auto` (default) measures the sensor unit's capture-to-publish latency against the clouds this run wrote, as a curve over the scene, and takes it off `data_ts.txt` / `pcd_ts.txt`; a number applies that constant; `off` leaves the stamps as recorded. Recorded per scene in `time_align.txt` |
| `--cam_time_offset S` | seconds added to the camera header stamp, for a camera that stamps differently from the lidar beside it (default `0.0`) |
| `--lidar_extrinsic` | manual lidar->base transform when TF has no path |
| `--tf_pins on\|off` | `on` (default): the two static transforms the Odin1 estimates online, `imu_link -> odin1_lidar` and `odin1_lidar -> odin1_camera_link`, are replaced by one constant for every bag (`dataset_tools.tf_pins.PINS`, the median of what 42 bags published), because the estimate wanders by 3 cm and half a degree between boots and even within a bag. What the bag said is recorded per scene in `tf_pins.txt`. `off` uses the bag's values, as every root before 2026-09-30 |
| `-v` | debug logging |

Behaviour worth knowing:

- **Input discovery.** `input_dir` may be a directory of bags or a single bag
  directory; only directories containing a `metadata.yaml` are treated as bags,
  anything else is skipped with a warning.
- **Resumable.** `source_bag.txt` is written as the last step of a scene, so a
  scene counts as complete only if it finished and its frame counts agree.
  Complete scenes are skipped unless `--overwrite` is given. Scenes extracted
  before this bookkeeping existed have no `source_bag.txt` and will be redone.
- **Numbering** follows the sorted bag order, so `--scenes` and `--start_index`
  keep indices stable rather than renumbering the selection.
- **Failures are collected**, not fatal: a bag that raises is logged with its
  traceback, the run continues, and the summary at the end lists every failed
  and every empty scene. The exit code is `1` if any bag failed.
- **GNSS is found by message type, not by name.** `--gps auto` picks the
  `sensor_msgs/NavSatFix` topic that actually carries messages, preferring a
  fused solution (`/fixposition/odometry_llh`) over one antenna's raw fix, so
  the same run works on the Fixposition and the Septentrio generation. The
  fixes are the receiver's own position — on the new rig the fusion point,
  a few tens of centimetres from `base_link` and well inside its own sigma —
  and they are **not** ground truth: `odometry.txt` is anchored per segment
  and says nothing about the Earth. A fix whose covariance the receiver called
  unknown carries `NaN` in the three covariance columns rather than zeros.
- **`scenes.csv`** at the dataset root maps `scene,bag,start_time,frames`. The parent
  rebuilds it from the finished scene directories when the run ends.

## Output format

One directory per bag, `0000/`, `0001/`, ... `F` is the number of synchronised
camera/lidar pairs; camera and lidar are matched by *mutual* nearest neighbour
on header stamps, so unmatched frames are dropped and all per-frame files are
index-aligned.

| Path | Shape | dtype | Units / frame | Notes |
| --- | --- | --- | --- | --- |
| `images/NNNNN.jpg` | H x W x 3 | `uint8` | the camera's own JPEG, `msg.data` unmodified | the compressed image topic, byte for byte. `.png` in a root made before 2026-09-29 or with `--image_format png`; `manifest.json` -> `images.extension` says which (no key: png). HELHEST v1.2-v1.5 (real files in v1.2, the others linked) and v2.1 were given their JPEGs back by `scripts/images_from_bag.py` (removed since, see git history) on 2026-09-29/30 and declare `jpg`; v2.0, v1.3_aux and v1_rear_archive still hold PNG |
| `clouds/NNNNN.npy` | (N, 4) | `float32` | x, y, z in m + intensity; `--base_frame` (`base_link`) | NaN rows and all-zero points dropped, so `N` varies |
| `data_ts.txt` | (F,) | text `float64` | seconds, Unix epoch | midpoint of the camera and lidar stamps of the pair, less the measured capture-to-publish latency (`--time_align`) |
| `pcd_ts.txt` | (F,) | text `float64` | seconds, Unix epoch | lidar header stamps, on that same corrected clock |
| `time_align.txt` | — | text | seconds | what the frame clock was corrected by and on what evidence: one row per window measured (`t`, `delta`, `gain`, `depth`) |
| `gnss_guard.txt` | — | text | seconds | the stretches of fabricated GNSS/INS pose this scene had removed, what they cost in poses and frames, and how many further frames went for having no pose under them at all; absent when the guard did not run |
| `tf_pins.txt` | — | text | m, degrees | the static transforms pinned to a constant (`--tf_pins`): for each, the pin the scene was built with and what the bag's own `/tf_static` said, so a bag whose estimate sat far from the constant can be found; absent when nothing was pinned (a Livox bag, or `--tf_pins off`) |
| `transforms.npy` | (F, 4, 4) | `float64` | m, SE(3) | base pose per frame, relative to frame 0 (identity); interpolated onto `data_ts` (lerp + SLERP) |
| `gravity_align.npy` | (F, 3, 3) | `float64` | rotation | `Ry(pitch) @ Rx(roll)` of the base at `data_ts`, yaw dropped |
| `camera_intrinsics.txt` | 3 x 3 | text `float64` | pixels | `K` from the matching `camera_info` topic |
| `camera_info.yaml` | — | YAML | — | the full `CameraInfo` (size, distortion model + coefficients, `K`, `R`, `P`); v1 scenes have only `camera_intrinsics.txt` |
| `T_cam_to_base.txt` | 4 x 4 | text `float64` | m, SE(3) | `p_base = T @ p_cam_optical`, from the static TF; v1 scenes instead carry `T_cam_to_lidar.txt` in the raw lidar frame |
| `joint_states.npy` | (M, 1 + 3J) | `float64` | `[t, positions..., velocities..., efforts...]` | full rate, sorted by `t`; not frame-aligned |
| `joint_setpoints.npy` | (M, 1 + 3J) | `float64` | same layout | only if `/joint_setpoints` exists |
| `cmd_vel.npy` | (M, 7) | `float64` | `[t, vx, vy, vz, wx, wy, wz]`, m/s and rad/s, base frame | only if `/cmd_vel` exists |
| `gps.npy` | (M, 8) | `float64` | `[t, lat, lon, alt, status, cov_ee, cov_nn, cov_uu]`, degrees, m and m² | full rate, sorted by `t`, not frame-aligned; only if the bag carries a `NavSatFix` |
| `source_bag.txt` | 1 line | text | — | source bag directory name; doubles as the completion marker |
| `../scenes.csv` | — | text | — | dataset-level index: `scene,bag,start_time,frames` |

Files whose source topic is missing from a bag are simply not written, and the
extractor logs a warning saying so. `helhest-validate` then reports the missing
`transforms.npy` / `gravity_align.npy` / `camera_intrinsics.txt` as errors: the
scene contract requires them, and a bag without odometry or orientation is a
recording to look into rather than a scene to ship.

## Validation

```bash
helhest-validate /path/to/dataset               # a whole dataset root
helhest-validate /path/to/dataset/0000 0001     # or individual scenes
helhest-validate /path/to/dataset --csv report.csv
helhest-validate /path/to/dataset --csv - --markdown -q
```

Checks file presence, equal frame counts across images / clouds / timestamps /
transforms / gravity alignment, contiguous frame numbering, monotonic
timestamps, cloud shape and dtype, the non-finite fraction, valid rotation
matrices, camera/lidar sync offset, frame-to-frame odometry jumps and tilt
statistics. When a scene carries ground truth, `odometry.txt` and
`odometry_segments.txt` are checked against each other: segment pose counts,
no overlaps, each segment anchored at the identity.

`gps.npy`, where a scene has one, is checked as an optional extra: column
count, a sorted clock, coordinates on the globe, and a time span that overlaps
`data_ts.txt`. Every finding there is a warning — nothing in the scene contract
is computed from it.

Structural problems are **errors** (exit code `1`); data-quality observations
(NaNs, jumps, extreme tilt) are **warnings** — `--strict` makes them fail too.
Written to `<root>/validation.csv`, the report is shown by the viewer: a chip
per scene on the index and a card on the scene page.
Thresholds: `--max-jump` (m), `--max-tilt` (deg), `--max-nan`, `--max-sync` (s);
`--sample-clouds N` sets how many clouds are opened per scene (default 8, keeps
the check fast on network storage).

The older dataset layout is supported: `odometry.txt`, `trajectory.txt`,
`T_cam_to_lidar.txt`, `height_maps/`, `height_masks/` and `<scene>_full.npz`
are reported as optional extras, and `float64` clouds are accepted.

## Viewer

```bash
pip install "dataset_tools[viewer]"     # needs Python 3.12+ (a SimView requirement)
helhest-viewer /path/to/dataset         # then open http://127.0.0.1:8080/
```

A web view of an extracted dataset root: an index of every scene and what each
one actually contains, a per-scene inventory, and a 3D view of each odometry
segment -- terrain from the height maps, the aggregated cloud, the robot's
pose, the camera frame and the proprioception plots alongside.

It is driven by the file structure rather than by a sensor generation, so a
scene missing any optional file still opens, with the absent files marked
rather than hidden. A view never spans an `odometry_segments.txt` boundary,
because segments are each re-anchored to their own first pose. The dataset root
is only read; built scenes are cached elsewhere.

See [docs/VIEWER.md](docs/VIEWER.md).

## Slurm

`slurm/slurm_extract.batch` is a working example (16 CPUs, 8 workers). The essential
part:

```bash
singularity exec --bind /mnt/personal --bind /mnt/data "$SIF" \
    "$REPO/.venv/bin/python" \
    -m dataset_tools.extract_data <bags> <output> --jobs 8
```

Because complete scenes are skipped, a re-queued job resumes where the previous
one stopped. The job files in `slurm/` are the ones the HELHEST releases were
built with on the CTU RCI cluster; [slurm/README.md](slurm/README.md) says what
each does and which paths are site-specific.

## Tests

No bags are needed; everything runs on synthetic data.

```bash
.venv/bin/pip install -e ".[test]"
.venv/bin/python -m pytest tests
```

Covered: nearest-neighbour synchronisation against a brute-force reference,
`PointCloud2` field parsing, static-TF graph composition, rotation helper
round-trips, bag discovery / scene selection / `scenes.csv` merging, a full
extraction of a synthetic bag written with `rosbags` (GNSS fixes included), the
frame-clock estimator against a synthetic drive whose latency is known by
construction (constant, and changing halfway), the scene validator, and the
viewer (probing, window enumeration, pose
interpolation, the map's geodesy, the segment page, rendering and every route,
against synthetic scene directories). `.github/workflows/ci.yml` runs
the same on Python 3.10 (extraction) and 3.12 (with the viewer), plus `black`
and `ruff`.

The viewer's tests need its extra too, and with it Python 3.12+; run them from
the viewer's own virtualenv:

```bash
.venv-viewer/bin/pip install -e ".[viewer,test]"
.venv-viewer/bin/python -m pytest tests
```

## Known caveats

- **Cloudini clouds.** The Odin1 lidar publishes
  `point_cloud_interfaces/CompressedPointCloud2`; decoding needs
  `dataset_tools/libcloudini_shim.so` (see above).
- **Missing Odin1 extrinsic.** The 2026-08 bags' TF tree does not connect
  `odin1_base_link` to `base_link` — the mount extrinsic was never published.
  Without `--lidar_extrinsic` the identity is used and a warning is logged, so
  those clouds are in the lidar frame.
- **The Odin1 re-estimates its own extrinsics.** From the 2026-09-14
  firmware the unit publishes `imu_link -> odin1_lidar` and
  `odin1_lidar -> odin1_camera_link` on `/tf_static`, but as a running
  estimate: across 42 bags the lidar-to-IMU translation spans 3 cm per axis
  between the 10th and 90th percentile, and one bag changes it seven times in
  its first minutes. The mount `base_link -> odin1_base_link` (the URDF's
  (0.10, 0, 0.10)) is not affected. Since 2026-09-30 the extractor pins both
  edges to the median (`--tf_pins`, `dataset_tools/tf_pins.py`, survey in
  `docs/odin_tf_static_2026-09-30.txt`); HELHEST v2.0 and v2.1 were built from
  the bags' own values, so their ground height differs by up to 3.5 cm from
  scene to scene for no physical reason.
- **Static-TF convention (fixed 2026-08-28).** `get_transform(G, src, dst)`
  now returns the point map `p_dst = T @ p_src`, matching tf2's
  `lookup_transform(dst, src)`, and a manual `--lidar_extrinsic X Y Z R P Y`
  is the pose of the lidar in the base frame (same convention). Earlier
  versions of this tool composed the static TF chain in the wrong direction.
  For HELHEST's front Livox (`base_link -> livox_front_frame`: x = +0.17 m,
  z = +0.145 m, rolled 180°) the rotation and z happened to survive
  (R_x(π) is its own inverse), but **all previously published v1 scenes have
  their clouds shifted by −0.34 m along x** (the lidar was placed at
  x = −0.17 m instead of +0.17 m). Re-extracting an old bag with the current
  tool therefore differs from the published scene by exactly +0.34 m in x;
  see `docs/DATA.md` for the recommended fix on existing data.
- **Fixposition topic choice.** Use `/fixposition/odometry_enu`, not
  `_smooth`: the smooth variant's child frame (`FP_POISH`) is not connected to
  `base_link` by static TF, so its poses cannot be re-expressed at the base.
- **Do not use `/imu/data` for gravity alignment** on the new stack: that IMU
  is mounted tilted (~5 deg roll, ~11 deg pitch) and its frame (`imu`) has no
  TF to `base_link`. The Fixposition orientation gives clean base roll/pitch.
- **Dataset defect, 2026-08-21 bags:** `/joint_states` carries only efforts —
  positions and velocities are all zero. Commanded wheel velocities are in
  `/joint_setpoints` (`joint_setpoints.npy`).
- **Cloud dtype.** The extractor writes `float32`; the previously published
  HELHEST dataset stores `float64` clouds. The validator accepts both.
- **One clock per frame (fixed 2026-09-09).** `transforms.npy` and
  `gravity_align.npy` are now sampled at `data_ts` -- the camera/lidar midpoint
  the frame index actually stands for. They used to be sampled at the camera and
  the lidar stamps respectively, so a consumer reading frame `k` from the three
  files got three slightly different instants (a median 3.9 ms apart on the
  2026-09 recordings, up to 67 ms).
- **Poses are interpolated, not snapped (fixed 2026-09-09).** Nearest-neighbour
  lookup onto a 10 Hz stream quantised every frame by up to 50 ms, which is 5 cm
  at 1 m/s. `interpolate_poses()` now lerps the translation and SLERPs the
  rotation, falling back to the nearest sample outside the sample range, across
  a gap over 0.5 s, or across a bracket implying more than 6 m/s (4 m/s until
  2026-09-29, which the sprints of the 2026-09-28 recordings exceed) -- a GNSS
  teleport, where blending the two sides would fabricate a pose the robot never
  held. The speed gate matches the one `write_ground_truth()` breaks segments
  on, so the two files agree about which steps are real motion. The count of
  interpolated versus fallen-back frames is logged per scene.
- **The sensor unit may stamp at publish, not at capture.** On the 2026-09
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

  A latency follows load, so it does not hold still -- scene 0000 of v2 beta
  spends 80 of its 290 s at ~0.05 s and the rest at ~0.69 s, and 0006-0008 have
  none at all. There is no per-rig or even per-bag number to configure, so
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
  inheriting its neighbour's offset. On HELHEST v1.3, whose ground truth comes
  off the same lidar, the profile comes back flat at zero, as it must.
- **A GNSS/INS with no fix keeps publishing poses.** When the Fixposition loses
  its solution -- and rolling the robot over is exactly how it loses it, with
  the antennas pointing at the ground -- the odometry topic does not stop. It
  fills with the reset state: a position hundreds of metres from where the
  robot is, held perfectly still, attitude snapped back to identity, and no
  flag on any of it. Scene 0010 of v2 beta reported a **1249 m step between two
  frames 64 ms apart** that way, and 0014 a pair of 1130 m round trips.

  `--gnss_guard auto` removes those stretches during extraction, before an
  image or a cloud is written, so the frames that survive are the frames the
  dataset can put a pose on and the indices stay contiguous. What marks the
  fabricated side is that the translation stops moving *at all* -- the reset
  state is one position repeated, held for 2.9 s and 46.5 s in the two v2 beta
  cases. That is never enough on its own, because a robot can be genuinely
  stopped and a Fixposition quantises to a micrometre, so a stopped robot
  publishes the same number for seconds too: scene 0009 does it three times
  while the driver leans on a robot that will not move, with a continuous fix,
  the scene's best covariance and a GNSS latitude and longitude every bit as
  still as the odometry. A freeze is therefore believed only where something
  independent agrees -- a jump implying four figures of metres per second into
  or out of it, or a hole in the stream of fixes that carried a position. A
  short island fenced in by a jump at both ends is an excursion rather than a
  visit, and stands on its own. So does an outage with a jump inside it: an
  outage alone is a solution coasting, a lone jump says one of its two sides is
  wrong without saying which, but together they are conclusive. That last rule
  has no catch of its own across v2 beta -- 0014's span is already convicted by
  its island -- but it is the only one that would reach a fabricated stretch
  which neither held still long enough nor came back.

  The outage is read as a *hole*, not as a flag, because the receiver reports
  losing its fix two different ways and 0014 uses both: 3 s of null-island
  `NavSatFix` around one jump, and 12.7 s of publishing nothing at all around
  the other. Only holes between fixes count -- a stream that starts late or
  stops early is the ordinary shape of a recording (0009 waits 11.7 s for its
  first fix, 0016 spends its last 57 s without one), and coasting is degraded,
  not fabricated.

  The guard fails towards keeping data: a fabricated stretch that arrives
  without a jump and that the receiver never admits to would not be found. A
  position cannot reach a reset origin hundreds of metres away without jumping
  there, so that combination does not arise in practice, and the alternative is
  deleting the frames of a robot that was merely stuck.

  What the guard deliberately does *not* read is the attitude. The 180 deg flip
  is the tempting signature and it is a trap: the robot really does roll over,
  and when it does, the flip is the measurement. Scene 0010 spends 56 s upside
  down **before** its solution resets, and that stretch is good data.

  Each scene records what it lost in `gnss_guard.txt`, and the hole reads as
  `fix_lost` in `odometry_segments.txt`.
- **Frames with no pose under them.** A fabricated pose does not need a
  fabricating receiver. `interpolate_poses` falls back to the nearest sample
  wherever it cannot interpolate -- before the pose stream starts, after it
  ends, or across a hole wider than its 0.5 s bracket -- and what comes out is a
  stationary pose under a frame the robot was driving through, written into
  `transforms.npy` looking exactly like a measurement. Scene 0014 of v2 beta
  exported a **3.29 m step between two frames 62 ms apart, 53 m/s**, with no
  fabricated pose anywhere near it: both frames sat in a 12 s hole in the
  odometry and borrowed the same distant sample.

  The same `--gnss_guard` stage drops them, on the same principle that governs
  a span -- a frame the dataset cannot put a real pose on has nothing to offer
  a consumer of a pose-annotated dataset. The margin is spent the other way
  round from a span, though: a span drops a frame that *might* fall inside it,
  support keeps a frame that *might* sit on a pose, so a hole has to exceed
  2.5 s before it costs anything. Across v2 beta this is 1,127 frames, 1.1%,
  two thirds of them 0016's last 57 s, where the receiver stops publishing while
  the camera runs on. With it, no scene in the root exports a step above
  3.2 m/s, and every one of those is driving.
- **Out-of-order bags.** Timestamps are sorted before synchronisation. If a bag
  is badly out of order for other reasons, `sort_bag.py` rewrites it sorted by
  header stamp.

## Dataset

See [docs/DATA.md](docs/DATA.md) for the dataset itself — recordings, contents
and statistics. [docs/ODOMETRY_REGEN.md](docs/ODOMETRY_REGEN.md) covers
regenerating the ground truth and
[docs/GT_HOLE_REPLAY.md](docs/GT_HOLE_REPLAY.md) recovering the stretches where
the estimator published nothing; [docs/README.md](docs/README.md) lists what
else is there. Code comments and the campaign records refer to **DRIFT**: that
is the downstream training code the roots are extracted for (a separate
repository), which also owns the height-map generator and the train/validation
split.

## License

BSD-3-Clause, see [LICENSE](LICENSE). Copyright (c) Jan Vlk,
Czech Technical University in Prague.
