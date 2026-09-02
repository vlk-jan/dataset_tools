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
| `dataset_tools/odom_node.py` | `ros2 run dataset_tools odom_recorder` | record odometry to a text trajectory |

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

The command-line defaults target the **new** stack; see
[Usage](#usage) for the old-stack invocation.

## Installation

### Plain Python (venv)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt      # or: pip install -e ".[test]"
```

Python >= 3.10 (not declared in `pyproject.toml`: colcon's setup.py introspection cannot evaluate `requires-python`). Dependencies: `rosbags`, `numpy`, `opencv-python-headless`,
`networkx`, `tqdm`.

### Container (CTU cluster)

The `.venv` in this repository was built with the container's Python 3.12, so
it has to be run through the container. Note the **`--bind /mnt/data`**: the
data share is not mounted inside the container by default, and neither the
bags nor the dataset are reachable without it.

```bash
singularity exec --bind /mnt/data /mnt/data/vras/data/helhest-jazzy.sif \
    /home/vlkjan6/projects/dataset_tools/.venv/bin/python \
    -m dataset_tools.extract_data <bags> <output> --jobs 8
```

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
--odom /fixposition/odometry_enu --imu /fixposition/odometry_enu
```

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
| `--overwrite` | re-extract scenes that are already complete |
| `--scenes 0 3 5-9 <bag-name>` | only these scene indices / ranges / bag directories |
| `--start_index N` | number the first scene `N` instead of `0000` |
| `--base_frame` | target frame for clouds, odometry and gravity (default `base_link`) |
| `--lidar_extrinsic` | manual lidar->base transform when TF has no path |
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
- **`scenes.csv`** at the dataset root maps `scene,bag,start_time,frames`. Each
  worker writes its own line into `.scenes_parts/`, and the parent merges them
  when the run ends (safe under multiprocessing).

## Output format

One directory per bag, `0000/`, `0001/`, ... `F` is the number of synchronised
camera/lidar pairs; camera and lidar are matched by *mutual* nearest neighbour
on header stamps, so unmatched frames are dropped and all per-frame files are
index-aligned.

| Path | Shape | dtype | Units / frame | Notes |
| --- | --- | --- | --- | --- |
| `images/NNNNN.png` | H x W x 3 | `uint8` | BGR pixels, camera frame | decoded from the compressed image topic |
| `clouds/NNNNN.npy` | (N, 4) | `float32` | x, y, z in m + intensity; `--base_frame` (`base_link`) | NaN rows and all-zero points dropped, so `N` varies |
| `data_ts.txt` | (F,) | text `float64` | seconds, Unix epoch | midpoint of the camera and lidar stamps of the pair |
| `pcd_ts.txt` | (F,) | text `float64` | seconds, Unix epoch | lidar header stamps |
| `transforms.npy` | (F, 4, 4) | `float64` | m, SE(3) | base pose per frame, relative to frame 0 (identity) |
| `gravity_align.npy` | (F, 3, 3) | `float64` | rotation | `Ry(pitch) @ Rx(roll)` of the base at each lidar stamp, yaw dropped |
| `camera_intrinsics.txt` | 3 x 3 | text `float64` | pixels | `K` from the matching `camera_info` topic |
| `camera_info.yaml` | — | YAML | — | the full `CameraInfo` (size, distortion model + coefficients, `K`, `R`, `P`); v1 scenes have only `camera_intrinsics.txt` |
| `T_cam_to_base.txt` | 4 x 4 | text `float64` | m, SE(3) | `p_base = T @ p_cam_optical`, from the static TF; v1 scenes instead carry `T_cam_to_lidar.txt` in the raw lidar frame |
| `joint_states.npy` | (M, 1 + 3J) | `float64` | `[t, positions..., velocities..., efforts...]` | full rate, sorted by `t`; not frame-aligned |
| `joint_setpoints.npy` | (M, 1 + 3J) | `float64` | same layout | only if `/joint_setpoints` exists |
| `cmd_vel.npy` | (M, 7) | `float64` | `[t, vx, vy, vz, wx, wy, wz]`, m/s and rad/s, base frame | only if `/cmd_vel` exists |
| `source_bag.txt` | 1 line | text | — | source bag directory name; doubles as the completion marker |
| `../scenes.csv` | — | text | — | dataset-level index: `scene,bag,start_time,frames` |

Files whose source topic is missing from a bag are simply not written, and the
extractor logs a warning saying so.

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
statistics.

Structural problems are **errors** (exit code `1`); data-quality observations
(NaNs, jumps, extreme tilt) are **warnings** — `--strict` makes them fail too.
Thresholds: `--max-jump` (m), `--max-tilt` (deg), `--max-nan`, `--max-sync` (s);
`--sample-clouds N` sets how many clouds are opened per scene (default 8, keeps
the check fast on network storage).

The older dataset layout is supported: `odometry.txt`, `trajectory.txt`,
`T_cam_to_lidar.txt`, `height_maps/`, `height_masks/` and `<scene>_full.npz`
are reported as optional extras, and `float64` clouds are accepted.

## Slurm

`slurm_extract.batch` is a working example (16 CPUs, 8 workers). The essential
part:

```bash
singularity exec --bind /mnt/personal --bind /mnt/data \
    /mnt/personal/vlkjan6/helhest_new/helhest-jazzy.sif \
    /home/vlkjan6/projects/dataset_tools/.venv/bin/python \
    -m dataset_tools.extract_data <bags> <output> --jobs 8
```

Because complete scenes are skipped, a re-queued job resumes where the previous
one stopped.

## Tests

No bags are needed; everything runs on synthetic data.

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest tests
```

Covered: nearest-neighbour synchronisation against a brute-force reference,
`PointCloud2` field parsing, static-TF graph composition, rotation helper
round-trips, bag discovery / scene selection / `scenes.csv` merging, and the
scene validator.

## Known caveats

- **Cloudini clouds.** The Odin1 lidar publishes
  `point_cloud_interfaces/CompressedPointCloud2`; decoding needs
  `dataset_tools/libcloudini_shim.so` (see above).
- **Missing Odin1 extrinsic.** The 2026-08 bags' TF tree does not connect
  `odin1_base_link` to `base_link` — the mount extrinsic was never published.
  Without `--lidar_extrinsic` the identity is used and a warning is logged, so
  those clouds are in the lidar frame.
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
- **Out-of-order bags.** Timestamps are sorted before synchronisation. If a bag
  is badly out of order for other reasons, `sort_bag.py` rewrites it sorted by
  header stamp.

## Dataset

See [docs/DATA.md](docs/DATA.md) for the dataset itself — recordings, contents
and statistics.

## License

BSD-3-Clause, see [LICENSE](LICENSE). Copyright (c) Jan Vlk,
Czech Technical University in Prague.
