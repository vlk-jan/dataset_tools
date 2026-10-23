# dataset_tools

Tools for turning **HELHEST** robot recordings (ROS 2 bags) into a flat,
learning-friendly dataset: synchronised camera frames, point clouds in the
robot base frame, odometry, gravity alignment and proprioception.

Bags are read with the pure-Python [`rosbags`](https://gitlab.com/ternaris/rosbags)
library, so **no ROS installation is required**.

| Tool | Entry point | What it does |
| --- | --- | --- |
| `dataset_tools/extract/` | `helhest-extract` | bags -> scene directories (`cli`, `scene` per bag, `scene_io`, `sync`, `pointcloud`, `deskew`, `gps`, `tf_graph`, `camera`, `ground_truth`, `pose_check`) |
| `dataset_tools/validate_scene.py` | `helhest-validate` | check extracted scenes |
| `dataset_tools/sort_bag.py` | `helhest-sort-bag` | rewrite a bag sorted by header stamp |
| `dataset_tools/viewer/` | `helhest-viewer` | browse an extracted dataset root in a web viewer |
| `dataset_tools/disk_usage.py` | `helhest-disk-usage` | bytes per scene, modality and file, for the viewer's disk-usage panels |
| `dataset_tools/image_format.py` | `helhest-image-format` | declare (or read back) whether a root's images are the camera's JPEG or a PNG |
| `dataset_tools/odom_frame.py` | `helhest-odom-frame` | measure and apply the rotation between the Odin1's pose frame and its clouds (needs the `build` extra) |
| `dataset_tools/heightmap.py` | `helhest-heightmap` | height maps and validity masks from a scene's clouds |
| `dataset_tools/keyframes.py` | `helhest-keyframes` | the root's keyframe lists (all / train / held out) |
| `dataset_tools/release.py` | `helhest-release` | a release's root-level files (manifest, splits, lists, README) from its spec in `docs/releases/` and its scenes |

A HELHEST root is built entirely from this repository: see
[Extracting a dataset, step by step](#extracting-a-dataset-step-by-step). Products other tools add
to a root afterwards (DRIFT's proprio labels and driven similarity) are theirs.

## Quickstart: from a recording to the viewer

For a bag recorded on the current robot (Odin1 + Fixposition, 2026-08 onwards):

```bash
python3 -m venv .venv && .venv/bin/pip install -e .                  # extraction only; Python >= 3.12
.venv/bin/helhest-extract  /path/to/bags /path/to/dataset --jobs 8   # one scene per bag: 0000/, 0001/, ...
.venv/bin/helhest-validate /path/to/dataset --csv /path/to/dataset/validation.csv

uv venv --python 3.12 .venv-viewer && uv pip install --python .venv-viewer -e ".[viewer]"   # plus the viewer
.venv-viewer/bin/helhest-viewer /path/to/dataset --open              # http://127.0.0.1:8080/
.venv-viewer/bin/helhest-viewer http://subtdata.felk.cvut.cz/HELHEST/   # or a root on a web server
```

- That is a bare root: scenes with clouds, images and poses. A release root
  also carries the odometry-frame rotation, height maps, keyframe lists and a
  manifest -- see [Extracting a dataset, step by step](#extracting-a-dataset-step-by-step).
- The viewer extra pulls in `torch`. `uv` takes it from the CPU-only index
  configured in `pyproject.toml`; plain `pip` ignores that and fetches the
  CUDA build (~3 GB), so with pip run
  `pip install torch --index-url https://download.pytorch.org/whl/cpu` first.
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
| Odometry | `/odom` (live robot odometry; the published GT was FAST-LIO run offline) | `/odin1/odometry` (Odin1 lidar-inertial odometry; v2 beta to v2.2 used `/fixposition/odometry_enu`, now the `--check_odom` reference) |
| Orientation (gravity) | `/ekf/imu/data` | `/odin1/odometry` (v2 beta to v2.2: `/fixposition/odometry_enu`) |
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

Python >= 3.12. Dependencies: `rosbags`, `numpy`, `opencv-python-headless`,
`scipy`. Extras: `test` (`pytest`, `httpx`), `build` (`open3d`, for
`helhest-odom-frame`; its Linux wheel wants glibc >= 2.29, so on the cluster
it runs inside the container) and `viewer` (see [Viewer](#viewer)). Building
a release root: `pip install -e ".[test,build]"`.

### Container (CTU cluster)

Build the venv with the container's own Python, then run everything through the
container. Note the **`--bind`**: the data shares are not mounted inside it by
default, so neither the bags nor the dataset are reachable without them.

```bash
SIF=/mnt/data/vras/data/helhest-jazzy.sif
singularity exec --bind /mnt/data $SIF python3 -m venv .venv
singularity exec --bind /mnt/data $SIF .venv/bin/pip install -e ".[test,build]"
singularity exec --bind /mnt/data $SIF \
    .venv/bin/python -m dataset_tools.extract <bags> <output> --jobs 8
```

The `.venv` is not in the repository (it is gitignored) -- `slurm/extract_v2.batch`
and the Slurm example below both expect one built this way at `$REPO/.venv`.

Add further `--bind` options for any other filesystem you read or write
(e.g. `--bind /mnt/personal`).

### Building the cloudini shim (new bags only)

The Odin1 clouds are cloudini-compressed. Decoding goes through a small C ABI
wrapper around [cloudini](https://github.com/facontidavide/cloudini):

```bash
cloudini_shim/build_shim.sh          # clones cloudini, builds dataset_tools/libcloudini_shim.so
```

Needs `cmake` and `g++` (C++20); `libzstd-dev` / `liblz4-dev` are used where
installed, otherwise cloudini's vendored copies are built. All of it is
present in `helhest-jazzy.sif`, so run it inside the container if in doubt.
The prebuilt `.so` needs GLIBC 2.32 and GLIBCXX 3.4.30. Old bags do not need
the shim.

## Extracting a dataset, step by step

A release root is built entirely from this repository, in two stages: per
bag (extract, rotate the odometry onto the clouds' frame, height maps,
validate) and per root (manifest, splits, keyframe lists, README, validation,
atlas). The scene contract is the same whether the steps run under Slurm or
by hand.

### 1. Describe the release

Two files in `docs/`, both committed:

- **The bag list**, `docs/v2_<n>_bags.csv`: one row per scene, in scene order,
  with header
  `scene,session,bag_dir_on_subtdata,bag_size_gb,bag_start_utc,duration_s,lidar_msgs,camera_msgs[,extract_args]`.
  The jobs read columns by position: `scene` must be the first and the bag
  directory the third. The optional `extract_args` column holds extractor
  options for that bag alone (`--trim_idle 60 10` in v2.4); it is cut on
  commas, so it may hold none and must not be quoted. Lists are cumulative:
  a release lists every scene of its root, earlier ones included.
- **The spec**, `docs/releases/v2_<n>.json`. Required: `dataset_version`,
  `bags` (the list above, relative to the repository root) and
  `held_out.scenes`. Optional: `inherit` (a spec to merge under this one,
  relative to the spec's directory; `helhest_v2.json` carries the rig's
  shared prose), `title`, `summary`, `held_out.decision`, `sessions`
  (`id`, `scenes`, `date`, `site`), `excluded`, `terrain_notes`, `defects`,
  `open_items`, `curation` (becomes `atlas/curation.json`). Copy
  `docs/releases/v2_4.json` as a template. Everything else in the manifest is
  measured from the scenes.

The v2.0 bags were chosen from a survey of every candidate bag
(`docs/v2_inventory.csv`, ~1 MB per bag, no extraction; see
[docs/V2_0_SELECTION.md](docs/V2_0_SELECTION.md)); its script was removed
after the selection, in af56814 (its last version: `git show 94955ed:scripts/survey_v2_bag.py`).

### 2. Extract every bag (CTU cluster)

From the repository root on a login node, with `.venv` built inside the
container as under [Container](#container-ctu-cluster):

```bash
mkdir -p slurm/slurm_out
CSV=docs/v2_4_bags.csv sbatch --array=0-35%4 slurm/extract_v2.batch /mnt/personal/$USER/HELHEST_v2_4
```

The array range is the bag list's rows (`0` to rows - 1; `%4` caps the jobs
running at once). **Always pass `CSV=` and the root**: their defaults are the
v2 beta list and root, and a forgotten one extracts the wrong bags. A root
that reuses an earlier release's scenes (linked in) is submitted with the
range of its new rows only.

Each array task leaves one finished scene:

1. fetches its bag by `rsync` from `$SUBTDATA` (default
   `$USER@subtdata.felk.cvut.cz`, key-based ssh) to node-local scratch;
2. `helhest-extract <bag> <scratch> --start_index <scene> --jobs 1` plus the
   row's `extract_args` and `$EXTRACT_ARGS`;
3. `helhest-odom-frame measure` then `apply`: the odometry and
   `gravity_align.npy` rotated onto the clouds' frame, recorded in
   `odom_frame.txt` (inside the container: `open3d` needs its glibc); a
   scene it cannot measure keeps the Odin1's own odometry, with a warning;
4. `helhest-heightmap --overwrite`: `height_maps/`, `height_masks/`,
   `height_maps.json`;
5. `helhest-validate` on the scene, then a copy swapped in as `<root>/<scene>/` and the
   report to `<root>/.validation/<scene>.csv`.

A scene already complete with the same bag is skipped, so resubmitting the
same range resumes. A scene that holds a *different* bag stops the task
(exit 13); `FORCE=1` re-extracts it. A scene that fails validation is not
published unless `FORCE_PUBLISH=1`. Exit codes name the failing step: 10
fetch, 11 extract, 12 publish, 14 validation, 15 the rotation, 16 the maps.
The log is
`slurm/slurm_out/`. `SIF` and `REPO` override the container image and the
checkout; [slurm/README.md](slurm/README.md) lists every site-specific value.

### 3. Finalise the root

On the login node, not through `sbatch`:

```bash
slurm/finalize_v2.sh /mnt/personal/$USER/HELHEST_v2_4 docs/v2_4_bags.csv docs/releases/v2_4.json
```

It runs `helhest-release` (`scenes.csv`, `keyframes.lst`,
`keyframes_train.lst`, `keyframes_heldout.lst`, `manifest.json` with the
image format, `splits.json`, `atlas/curation.json`, `README.md`), validates
the whole root into `<root>/validation.csv`, and draws the atlas
(`scripts/make_atlas.py <root> --bags <list>`). Its exit code is the
validation's. `helhest-release` refuses a root whose scenes do not match the
bag list, either way round, or a held-out scene that is not on disk.
DRIFT's loader keeps the `manifest.json` -> `held_out.scenes` out of every
training and validation fold whichever list it reads (`--keyframes`,
`--allow_held_out`); v2.1 and v2.2 did this with symlink roots (`*_train`,
`*_heldout`, `*_allframes_notrain`) instead.

### 4. Check it

```bash
helhest-validate <root> -v                      # INFO lines too
helhest-disk-usage <root>                       # refresh disk_usage.json for the viewer
helhest-viewer <root>                           # see Viewer below
```

### The same without Slurm

`helhest-extract` numbers scenes by the sorted bag directory names, and
`helhest-release` refuses a scene whose bag is not the list's, so the bag
directory must sort in list order. Rows with `extract_args` take a call of
their own on the single bag, with `--start_index <scene>`, as the batch job
does:

```bash
helhest-extract <bags> <root> --jobs 8 --scenes 0-28                        # rows without extract_args
helhest-extract <bags>/<bag_29> <root> --start_index 29 --trim_idle 60 10   # one per row with them
helhest-odom-frame measure <root> --fits fits.json      # every scene with clouds and no odom_frame.txt
helhest-odom-frame apply   <root> --fits fits.json
helhest-heightmap <root>                                # --workers N, --scenes 0,3
helhest-release   <root> --spec docs/releases/v2_4.json
helhest-validate  <root> --csv <root>/validation.csv
python scripts/make_atlas.py <root> --bags docs/v2_4_bags.csv
```

`helhest-odom-frame` needs the `build` extra. Re-extracting a scene removes
its `odom_frame.txt`, so run `measure` + `apply` again after it.
`helhest-release --out_dir DIR` writes the root-level files elsewhere (and
leaves the root's `scenes.csv` alone), to compare against a root that exists.

### Rebuilding the height maps only

After the clouds or `gravity_align.npy` of a root changed:

```bash
sbatch --array=0-35 slurm/heightmaps.batch /mnt/personal/$USER/HELHEST_v2_4   # array index = scene
helhest-heightmap <root> --scenes 3 --overwrite                                # or by hand
```

The batch job first checks a sample of the scene's existing maps against a
rebuild (`VERIFY=0` skips), passes `HM_ARGS` through (e.g.
`--set statistic=p90 --set tie_jitter=none`, the v2.0-v2.4 build; the defaults are v2.5's). `helhest-heightmap` refuses to mix maps built at
other settings than `height_maps.json` records, and to write through a linked
scene's `height_maps`, `height_masks` or `height_maps.json` symlink (build in
the root that owns them).

## Usage

### New bags (2026-08 Odin1 + Fixposition)

```bash
helhest-extract <dir-with-bags> <output-dir> --jobs 8
```

which is equivalent to spelling out the defaults:

```
--cam /odin1/image/compressed --lidar /odin1/cloud_filtered/cloudini \
--odom /odin1/odometry --imu /odin1/odometry --deskew_imu /odin1/imu \
--check_odom /fixposition/odometry_enu --gps auto
```

**Pose sources, from v2.3 on.** The ground truth is the Odin1's own
lidar-inertial odometry (`/odin1/odometry`: 14 Hz, 6-DoF, stamped on the same
clock as its clouds and images, locally within 2-5 cm of the Fixposition over
1-5 s windows). The Fixposition GNSS/INS stays the georeference (`gps.npy`)
and the check: every scene carries `pose_check.txt`, the odometry against it
at the base frame -- jumps, displacement over 1/2/5 s, clock lag, tilt -- and
a scene whose odometry relocalised in jumps is flagged there rather than
trusted. v2 beta to v2.2 were built the other way round, on the Fixposition;
`--odom /fixposition/odometry_enu --imu /fixposition/odometry_enu` reproduces
that.

When `--odom` is the Odin1 odometry or a Fixposition topic, the extractor also
writes the ground-truth files (`--gt auto`): `odometry.txt` (base_link pose at
full rate, split into segments each re-anchored to the identity) and
`odometry_segments.txt` (never interpolate across segment boundaries).
Segments break at >1 s gaps, at jumps (the Odin1's relocalisations, GNSS
teleports), and at the holes `--gnss_guard` leaves where a Fixposition had no
solution (`fix_lost`); a segment shorter than 20 poses is dropped. This matches the HELHEST v1.2 contract, so
segment-aware consumers need no changes, with one difference: segments in a
freshly written table are **numbered from 1**, so "segment 3 of 4" is the
third one. Roots written before this (v1.2 through v1.5) are numbered from 0
and are read as they stand -- a segment is addressed by the number in its own
file, never by its position.

The Odin1 mount extrinsic is not in the bags' TF tree; pass it once known:

```bash
helhest-extract <bags> <out> \
    --lidar_extrinsic X Y Z ROLL PITCH YAW
```

The full procedure for a release root is in
[Extracting a dataset, step by step](#extracting-a-dataset-step-by-step).

### Old bags (Livox + Luxonis OAK)

```bash
helhest-extract <dir-with-bags> <output-dir> \
    --cam /luxonis/oak/rgb/image_raw/compressed --lidar /livox/lidar_front \
    --odom /odom --imu /ekf/imu/data
```

Point cloud field layouts are parsed from the `PointCloud2` message, so any
lidar works.

### Options

| Flag | Meaning |
| --- | --- |
| `--jobs N` | worker processes, one bag each (default 4; `1` runs in-process). Within a bag, frames are decoded, deskewed and written on a thread pool of the CPUs the process may use |
| `--overwrite` | re-extract scenes that are already complete. Any scene that is extracted, complete or not, has its previous outputs removed first, `odom_frame.txt` included; `odometry.txt` and `odometry_segments.txt` only once the run is about to write new ones, so a bag without odometry, or a run that fails before that, leaves them in place |
| `--gt auto\|on\|off` | write `odometry.txt` + `odometry_segments.txt` from the odometry stream; `auto` does so for the Odin1 odometry and for Fixposition topics |
| `--scenes 0 3 5-9 <bag-name>` | only these scene indices / ranges / bag directories; comma-separated values work too |
| `--start_index N` | number the first scene `N` instead of `0000` |
| `--gps TOPIC` | GNSS fix topic written to `gps.npy`; `auto` (default) finds the `NavSatFix` topic carrying messages, `off` writes none |
| `--base_frame` | target frame for clouds, odometry and gravity (default `base_link`) |
| `--gnss_guard auto\|on\|off` | drop stretches where the GNSS/INS published a reset solution instead of a pose, and every frame left without a real one -- both the frames those stretches strand and the frames the pose stream never covered. `auto` turns the full guard on for a Fixposition source, runs only the coverage part for `/odin1/odometry` (it fabricates nothing, but frames it never covered still go), and nothing for wheel odometry; `on` forces the full guard on any source. Recorded per scene in `gnss_guard.txt` |
| `--time_align auto\|off\|S` | how to put the frame clock on the pose clock. `auto` (default) measures the sensor unit's capture-to-publish latency against the clouds this run wrote, as a curve over the scene, and takes it off `data_ts.txt` / `pcd_ts.txt`; a number applies that constant; `off` leaves the stamps as recorded. Recorded per scene in `time_align.txt`. A constant is known before the clouds are written, so `--deskew` and `--trim_idle` already run on the corrected clock; under `auto` they run on the stamped clock, which is exact for the Odin1 odometry (same unit, same latency) but with `--odom /fixposition/odometry_enu` leaves the sweep's pose bracket off by the latency -- pass the measured number there |
| `--cam_time_offset S` | seconds added to the camera header stamp, for a camera that stamps differently from the lidar beside it (default `0.0`) |
| `--deskew_imu TOPIC` | the gyro that gives the rotation inside each sweep when deskewing (default `/odin1/imu`, 400 Hz; needs a TF path to `--base_frame`). Used per sweep only where it agrees with the pose track's slerp to 1 deg, else the slerp; `off` or a missing topic means slerp throughout. Counted in `extract_meta.json` |
| `--check_odom TOPIC` | the pose stream the odometry is checked against at the base frame (default `/fixposition/odometry_enu`; `off` skips). Nothing in the scene is built from it; the result is `pose_check.txt` and `extract_meta.json` -> `frames.pose_check` |
| `--deskew auto\|off` | `auto` (default): where the cloud carries a per-point time (the Odin1's `offset_time`: its 256 x 192 raster is scanned bottom to top over 67 ms behind one stamp) and the pose stream covers the sweep, every return is moved to where the robot was at the frame's instant, so the cloud is a snapshot at `data_ts` like the image beside it. `off` writes the sweep as recorded, with its motion smear (4 deg across the raster at 60 deg/s), as every root before 2026-10-01. Counted per scene in `extract_meta.json` (`frames.deskew`) |
| `--image_format camera\|png` | `camera` (default): the camera's own compressed bytes, unmodified, as `.jpg` or `.png` by what they are. `png`: decode and store the pixels as PNG, four times the size; only for extending a root that already holds PNG |
| `--lidar_extrinsic X Y Z ROLL PITCH YAW` | manual lidar->base transform when TF has no path; metres and **radians** |
| `--allow_frame_fallback` | extract a bag whose TF does not reach `--base_frame` from the lidar (and no `--lidar_extrinsic`), the odometry child frame or the orientation frame, leaving that stream in its own frame. Without it such a scene **fails**: clouds, poses or gravity in the wrong frame look plausible and are found late. What fell back is listed in `extract_meta.json` |
| `--max_sync_dt S` | largest camera/lidar stamp difference a frame may pair (default `0.1`); further-apart mutual nearest neighbours, an image across a camera stall, are dropped and counted in `extract_meta.json` |
| `--tf_pins on\|off` | `on` (default): the two static transforms the Odin1 estimates online, `imu_link -> odin1_lidar` and `odin1_lidar -> odin1_camera_link`, are replaced by one constant for every bag (`dataset_tools.tf_pins.PINS`, the median of what 42 bags published), because the estimate wanders by 3 cm and half a degree between boots and even within a bag. What the bag said is recorded per scene in `tf_pins.txt`. `off` uses the bag's values, as every root before 2026-09-30 |
| `--trim_idle MAX_S KEEP_S` | off by default. Every stop longer than `MAX_S` seconds (odometry under 0.15 m/s and 5 deg/s over a 1 s window around the frame; split at frame-clock gaps over 1 s) keeps only its first `KEEP_S` seconds of frames; the rest are dropped after the guard and before any frame is written, so indices stay contiguous. The odometry (`odometry.txt`) is kept whole. `KEEP_S` may not exceed `MAX_S`. Recorded in `extract_meta.json` -> `frames.idle_trim` (`max_s`, `keep_s`, `frames_dropped`, `seconds_dropped`, `runs` as `[start, end, frames]` in seconds from the first frame). v2.4 uses `60 10` for its idle-heavy bags |
| `-v` | debug logging |

Behaviour worth knowing:

- **Input discovery.** `input_dir` may be a directory of bags or a single bag
  directory; only directories containing a `metadata.yaml` are treated as bags,
  anything else is skipped with a warning.
- **Resumable.** `source_bag.txt` is written as the last step of a scene, so a
  scene counts as complete only if it finished and its frame counts agree.
  Complete scenes are skipped unless `--overwrite` is given. Scenes extracted
  before this bookkeeping existed have no `source_bag.txt` and will be redone.
  A finished scene whose `source_bag.txt` names a different bag than the one
  the run would put there stops the run before anything is extracted, with or
  without `--overwrite`: the bag list moved, and skipping would leave the new
  bag out while overwriting would renumber the root.
- **Per-frame products follow the clock.** `height_maps/`, `height_masks/`,
  `proprio_labels/` and `driven_similarity/` are built after the extraction, one
  file per frame index. A re-extraction keeps them (and `height_maps.json`)
  when it writes the same `data_ts.txt`, and renames them
  `<name>.stale-<date>-<time>` when it does not, since they would describe
  frames that are now other moments.
- **Root bookkeeping after every run.** Besides `scenes.csv`, the run
  refreshes `disk_usage.json` for the scenes it touched and declares the image
  format in `manifest.json` -> `images` (other keys kept; an undeclared PNG
  root stays undeclared). Both are best effort and only warn on failure.
- **Stamps that were never set.** A header stamp before 2001 (zero, or time
  since boot from a driver that publishes before its clock syncs: the first
  51 `/joint_states` of HELHEST v2.1 scene 0000 say 0.12 s) is replaced by the
  message's receive time, and counted per topic in `extract_meta.json`.
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
  (`eye(4)` at each segment's first pose, not levelled: absolute roll/pitch
  is `gravity_align.npy`) and says nothing about the Earth. A fix whose
  covariance the receiver called
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
| `clouds/NNNNN.npy` | (N, 4) | `float32` | x, y, z in m + intensity; `--base_frame` (`base_link`) | NaN rows, all-zero points and returns without a finite per-point time dropped, so `N` varies. Since 2026-10-01 each sweep is deskewed onto `data_ts` (`--deskew`): every return stands where the robot saw it at the frame's instant. Roots made before that hold the raw 67 ms sweep |
| `data_ts.txt` | (F,) | text `float64` | seconds, Unix epoch | midpoint of the camera (plus `--cam_time_offset`) and lidar stamps of the pair, less the measured capture-to-publish latency (`--time_align`). With deskewed clouds this is the instant of the image and of every return alike; in roots made before 2026-10-01 the clouds' effective instant was the sweep's mean, and `time_align` had moved `data_ts` there, 27-52 ms after the image |
| `pcd_ts.txt` | (F,) | text `float64` | seconds, Unix epoch | lidar header stamps, on that same corrected clock |
| `time_align.txt` | — | text | seconds | what the frame clock was corrected by and on what evidence: one row per window measured (`t`, `delta`, `gain`, `depth`); no rows under a constant `--time_align`, and no file under `off` or without odometry |
| `odometry.txt`, `odometry_segments.txt` | — | text | `t` + row-major 4 x 4 SE(3), s and m | `--gt`: the base pose at the odometry's full rate, each segment re-anchored to the identity; the segment table (numbered from 1) gives each segment's span, pose count and why it ended (`gap`, `teleport`, `fix_lost`, `end_of_run`). See [Usage](#new-bags-2026-08-odin1--fixposition) |
| `pose_check.txt` | — | text | s, m, deg | the scene's odometry (`--odom`) against `--check_odom`: spans and overlap, path length and jumps (steps over the segment gate, and the metres they carry) in each, z range, the clock lag of the odometry against the reference, the median roll/pitch difference, and the |displacement| disagreement over 1/2/5 s windows (median, p90, scale). Warnings at the end where a figure is out of bounds |
| `gnss_guard.txt` | — | text | seconds | the stretches of fabricated GNSS/INS pose this scene had removed, what they cost in poses and frames, how many further frames went for having no pose under them at all (since 2026-10-01 split into before the pose stream, after it and inside holes), and how many more once the clock was corrected; absent when the guard did not run |
| `tf_pins.txt` | — | text | m, degrees | the static transforms pinned to a constant (`--tf_pins`): for each, the pin the scene was built with and what the bag's own `/tf_static` said, so a bag whose estimate sat far from the constant can be found; absent when nothing was pinned (a Livox bag, or `--tf_pins off`) |
| `transforms.npy` | (F, 4, 4) | `float64` | m, SE(3) | base pose per frame from `--odom`, relative to frame 0 (identity); interpolated onto `data_ts` (lerp + SLERP); the nearest sample outside the pose stream, across a gap over 0.5 s, or across a step over 6 m/s (a GNSS teleport, the gate `odometry.txt` breaks segments on) |
| `gravity_align.npy` | (F, 3, 3) | `float64` | rotation | `Ry(pitch) @ Rx(roll)` of the base at `data_ts`, yaw dropped |
| `camera_intrinsics.txt` | 3 x 3 | text `float64` | pixels | `K` from the matching `camera_info` topic |
| `camera_info.yaml` | — | YAML | — | the full `CameraInfo` (size, distortion model + coefficients, `K`, `R`, `P`); v1 scenes have only `camera_intrinsics.txt` |
| `T_cam_to_base.txt` | 4 x 4 | text `float64` | m, SE(3) | `p_base = T @ p_cam_optical`, from the static TF; v1 scenes instead carry `T_cam_to_lidar.txt` in the raw lidar frame |
| `joint_states.npy` | (M, 1 + 3J) | `float64` | `[t, positions..., velocities..., efforts...]` | full rate, sorted by `t`; not frame-aligned |
| `joint_setpoints.npy` | (M, 1 + 3J) | `float64` | same layout | only if `/joint_setpoints` exists |
| `cmd_vel.npy` | (M, 7) | `float64` | `[t, vx, vy, vz, wx, wy, wz]`, m/s and rad/s, base frame | only if `/cmd_vel` exists |
| `gps.npy` | (M, 8) | `float64` | `[t, lat, lon, alt, status, cov_ee, cov_nn, cov_uu]`, degrees, m and m² | full rate, sorted by `t`, not frame-aligned; fixes at non-finite or (0, 0) positions skipped; absent without a usable `NavSatFix` or with `--gps off` |
| `joint_names.txt` | J lines | text | — | the joint order of the `joint_states.npy` / `joint_setpoints.npy` columns: the first message's. Later messages are reordered to it; one naming other joints is skipped (counted in `extract_meta.json`); an empty column (no effort) is NaN |
| `extract_meta.json` | — | JSON | seconds since the bag began | the run (git commit and dirty flag, argv, every resolved option, library versions), the bag (path, duration, every topic's type and message count), the decisions (GPS topic, ground truth, guard), **each stream read: messages, first/last stamp and every gap over 1 s** -- the first place to look when a scene has a hole, to see which sensor went quiet -- and the frame bookkeeping (`frames`): messages, pairs, pairs over `--max_sync_dt`, `deskew`, `pose_check`, `idle_trim`, what the guard removed and where, frames trimmed after the clock correction, frames written. Also `unset_stamps`, `malformed_joint_messages` and `frame_fallbacks`. Since 2026-10-01 |
| `source_bag.txt` | 1 line | text | — | source bag directory name; doubles as the completion marker |
| `odom_frame.txt` | — | text + JSON last line | rotation, degrees | written by `helhest-odom-frame apply` (and removed by a re-extraction): the rotation `C` (cloud frame -> Odin1 pose frame) that `odometry.txt`, `transforms.npy` and `gravity_align.npy` were re-expressed by, and how it was measured (ICP between clouds 1.4-3 s apart). Absent where it was not applied |
| `height_maps/NNNNN.npz` | 128 x 128 | `float64` (`height_map`) | m, gravity-aligned base frame, `[row = x forward, col = y left]`, 0.1 m cells, +-6.4 m | `helhest-heightmap`: median of the returns per cell (0.65 m < r, z < 1.0 m), Delaunay-interpolated (fixed tie-break jitter since v2.5; v2.0-v2.4: p90, no jitter), filled, smoothed (`dataset_tools/heightmap.py` says why each setting). A frame with fewer than 250 such returns, or with fewer than 4 cells they fall in, has none |
| `height_masks/NNNNN.npz` | 128 x 128 | `uint8` (`validity`) | — | where each cell's value came from: 0 extrapolated, 1 interpolated, 2 measured, 3 a wheel-contact anchor |
| `height_maps.json` | — | JSON | — | the settings the maps were built with and their counts; `helhest-heightmap` refuses to add maps at other settings to them |
| `../keyframes.lst`, `../keyframes_train.lst`, `../keyframes_heldout.lst` | — | text | — | `helhest-keyframes`: `<scene>/clouds/NNNNN.npy <scene>/images/NNNNN.<ext>` per frame with a cloud, image, map and mask, minus frames within 4 s before a tilt past 45 deg; all scenes / all but `manifest.json` -> `held_out.scenes` / only those |
| `../manifest.json`, `../splits.json`, `../README.md`, `../atlas/curation.json` | — | JSON, markdown | — | `helhest-release`: the spec's prose plus what was measured per scene. The extractor itself only writes `manifest.json` -> `images` |
| `../disk_usage.json` | — | JSON | bytes | bytes and files per scene and modality, for the viewer; refreshed by every extraction run and by `helhest-disk-usage <root>` |
| `../validation.csv` | — | text | — | `helhest-validate <root> --csv <root>/validation.csv`, shown by the viewer |
| `../atlas/` | — | JSON, JPEG | — | `scripts/make_atlas.py`: the curated index and thumbnails the viewer's index shows |
| `../scenes.csv` | — | text | — | dataset-level index: `scene,bag,start_time,frames` |

Files whose source topic is missing from a bag are simply not written, and the
extractor logs it (a warning for odometry, orientation and `camera_info`;
INFO for the proprioception and GNSS topics). `helhest-validate` then reports the missing
`transforms.npy` / `gravity_align.npy` / `camera_intrinsics.txt` as errors: the
scene contract requires them, and a bag without odometry or orientation is a
recording to look into rather than a scene to ship.

## Validation

```bash
helhest-validate /path/to/dataset               # a whole dataset root
helhest-validate /path/to/dataset/0000 0001     # or individual scenes
helhest-validate /path/to/dataset --csv report.csv
helhest-validate /path/to/dataset --csv - --markdown -q
helhest-validate /path/to/dataset --json /path/to/dataset/validation.json --strict
```

Checks file presence, equal frame counts across images / clouds / timestamps /
transforms / gravity alignment, contiguous frame numbering, monotonic
timestamps, cloud shape and dtype, the non-finite fraction, valid rotation
matrices, camera/lidar sync offset, frame-to-frame odometry jumps and tilt
statistics, duplicate timestamps (warning), `camera_intrinsics.txt` (3 x 3,
positive focal lengths) and `T_cam_to_base.txt` / `T_cam_to_lidar.txt` (a
finite rigid 4 x 4). Images must all be in the one format the root declares
(error otherwise). When a scene carries ground truth, `odometry.txt` and
`odometry_segments.txt` are checked against each other: segment pose counts,
unique segment numbers, no overlaps (errors); poses outside every segment and
a segment not anchored at the identity (warnings); a segment table without
`odometry.txt` is an error. A ground-truth scene without `time_align.txt` is
INFO.

`gps.npy`, where a scene has one, is checked as an optional extra: column
count, a sorted clock, coordinates on the globe, and a time span that overlaps
`data_ts.txt`. Every finding there is a warning — nothing in the scene contract
is computed from it.

Also checked:

- **Proprioception** (`joint_states.npy`, `joint_setpoints.npy`, `cmd_vel.npy`):
  rows stamped before 2001 (never-set or boot-clock stamps), unsorted stamps, a
  stream that starts or ends more than 2 s inside the frames, and gaps over 1 s
  with frames under them. `cmd_vel` is published only while the autonomy
  drives, so its coverage and gaps are INFO.
- **Holes in the frame clock**: every gap over 1 s in `data_ts.txt`, with where
  it is (INFO; `max_frame_gap_s` and `hole_time_s` in the stats).
- **Every cloud's point count**, from its `.npy` header: empty clouds warn,
  ones under 100 points are INFO. The full loads stay a sample.
- **Per-frame products** (`height_maps/`, `height_masks/`, `proprio_labels/`,
  `driven_similarity/`): every file names a frame of the scene, one loads, and
  directories a re-extraction set aside (`*.stale-*`) are listed.
- **The root**, when a root is given on the command line (over the scenes
  directly below it): two scenes from one bag (error); a `manifest.json` that
  is unreadable or whose scenes, frame counts or bags no longer match the disk,
  and a `T_cam_to_base.txt` more than 5 mm from the median across scenes
  (warnings); the keyframe lists (`ROOT_KEYFRAMES`): a held-out scene in
  `keyframes_train.lst` is an error, a listed scene not on disk or a scene in
  `keyframes_heldout.lst` that the manifest does not hold out a warning.

Structural problems are **errors** (exit code `1`; a missing path or an
unreadable expect file exits `2`); data-quality observations (NaNs, implausible
speed, extreme tilt, proprioception not covering the frames) are **warnings** —
`--strict` makes them fail too. What is worth knowing but no defect is **INFO**
(`-v` prints it): a hole in the frame clock, an odometry step over
`--max-jump` (a step faster than 6 m/s warns, `ODO_SPEED`). Every message
carries a stable code, printed in brackets.

`--strict` is a usable gate once a root lists what it is known to contain in
`<root>/validation_expect.json` (or `--expect FILE`):

```json
{"expected": [["0008", "TILT"], ["0009", "TILT"], ["*", "CLOUD_EMPTY"]]}
```

Those findings are still reported, marked expected, and no longer count; an
entry that matches nothing is reported as `EXPECT_UNUSED` (INFO), so the list
cannot silently outlive its reason. Only scene-level warnings and errors can be
listed; root findings cannot. `--json PATH` writes every message (`scene`,
`code`, `severity`, `msg`, `t0`/`t1` in seconds from the first frame where
the finding has a place, `expected` where listed) under `scenes.<name>` with
its `status` and `stats`, and the root's under `root`.
Written to `<root>/validation.csv`, the report is shown by the viewer: a chip
per scene on the index and a card on the scene page.
Thresholds: `--max-jump` (m), `--max-tilt` (deg), `--max-nan`, `--max-sync` (s);
`--sample-clouds N` sets how many clouds are opened per scene (default 8; `0`
turns the cloud checks off, the header scan of every cloud included, which is most of a run's time on slow
storage -- 16 s warm, a minute cold, for HELHEST v2.1's 133k frames on a USB
drive).

The older dataset layout is supported: `trajectory.txt` and
`<scene>_full.npz` are reported as optional extras, `T_cam_to_lidar.txt` is
checked as the extrinsic, and `float64` clouds are accepted.

## Viewer

A web view of an extracted dataset root: an index of every scene and what each
one actually contains, a per-scene inventory with the validation findings, a
map and the signals over time, a per-frame page (camera, height map, cloud on
the image), and a 3D view of each odometry segment -- terrain from the height
maps, the aggregated cloud, the robot's pose and the camera frame, with the
proprioception plots alongside. [docs/VIEWER.md](docs/VIEWER.md) describes
every page and panel.

It is driven by the file structure rather than by a sensor generation, so a
scene missing any optional file still opens, with the absent files marked
rather than hidden. A view never spans an `odometry_segments.txt` boundary,
because segments are each re-anchored to their own first pose. The scenes are
only read: built 3D windows are cached elsewhere, and the one thing the viewer
writes into the root is reviewer notes (`<root>/notes/`). A root may also be
the URL of one published as a directory index; it is then mirrored into the
cache as it is browsed, and its notes are kept there
([docs/VIEWER.md](docs/VIEWER.md#a-root-on-a-web-server)).

### Install

```bash
uv venv --python 3.12 .venv-viewer
uv pip install --python .venv-viewer -e ".[viewer]"     # fastapi, uvicorn, jinja2, simview >= 6.1.1, CPU torch
```

With plain pip, install the CPU torch first (see [Quickstart](#quickstart-from-a-recording-to-the-viewer)).
The viewer refuses to start with SimView older than 6.1.1.

### On your machine

```bash
.venv-viewer/bin/helhest-viewer /path/to/root --open     # serves http://127.0.0.1:8080/
.venv-viewer/bin/helhest-viewer http://subtdata.felk.cvut.cz/HELHEST/ --open   # a root on a web server
scripts/viewer_public.sh                                 # the same, on the public release ($VIEWER_URL)
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--host`, `--port` | `127.0.0.1`, `8080` | bind address. There is no authentication and notes are writes: keep it on loopback and tunnel, or pass `--no-notes` before binding `0.0.0.0` |
| `--max-seconds`, `--max-frames` | `600`, `12000` | a segment longer than this is split into equal viewing windows |
| `--cache-dir` | `${XDG_CACHE_HOME:-~/.cache}/dataset_tools_viewer/<root name>-<hash>` | probes and built 3D windows |
| `--cache-max-gb` | `5.0` | built windows kept on disk, least recently used evicted first |
| `--clear-cache` | | drop this root's cached probes and windows (and a URL root's mirror) before starting |
| `--no-prefetch` | | do not build windows in the background (the next one, a scene page's first; for slow storage) |
| `--notes-dir`, `--no-notes`, `--author` | `<root>/notes` (`<cache-dir>/notes` for a URL root), on, `$USER` | reviewer notes |
| `--open`, `-v` | | open a browser; debug logging |

### On the CTU cluster

The viewer runs as a Slurm job next to the data; a tunnel brings it to the
laptop.

1. On the login node, from the repository root, with `.venv-viewer` built as
   above:
   ```bash
   mkdir -p slurm/slurm_out
   sbatch slurm/viewer.batch /mnt/personal/$USER/HELHEST_v2_4      # 8 h; -p cpulong -t 2-00:00:00 for longer
   ```
   The job picks a free port, caches under
   `$VIEWER_CACHE/<root name>` (default `/mnt/personal/$USER/viewer_cache`)
   and writes `slurm/slurm_out/viewer_<job>.conn` (node and port); its log
   prints the manual `ssh -N -J ... -L ...` line.
2. On the laptop:
   ```bash
   scripts/viewer_connect.sh            # [local_port=8080] [job_id]; newest running viewer job by default
   ```
   It finds the node and port and holds the tunnel open (`VIEWER_USER`,
   `VIEWER_LOGIN`, `VIEWER_REPO` for a different account, login node or
   checkout). Ctrl-C closes the tunnel; the viewer keeps running until the job
   ends.
3. Open http://127.0.0.1:8080/.

### What it reads from a root

Only the scene directories are needed. Each of these optional root files adds
to the pages, and is re-read when it changes:

| File | Shows as | Made by |
| --- | --- | --- |
| `validation.csv`, `validation.json` | status chips on the index (`/?status=fail`), the scene page's Validation card with each finding | `helhest-validate <root> --csv <root>/validation.csv --json <root>/validation.json` |
| `disk_usage.json` | sizes and disk bars | every `helhest-extract` run; `helhest-disk-usage <root> [--all]` |
| `manifest.json`, `splits.json`, `keyframes*.lst` | the dataset version, the VAL/TRAIN band and `?split=val\|train\|excluded` filters | `helhest-release` |
| `atlas/` | the index's curated masthead and thumbnails | `scripts/make_atlas.py`, `atlas/curation.json` from the spec |
| `notes/` | notes on scenes and frame ranges, with a verdict (`keep`, `drop`, `fix`) | the viewer itself |

### Using it

- **`/`**: one card per scene, filterable by validation status, note verdict
  and split. **`/segments`** lists every ground-truth window in one table;
  **`/notes`** every note.
- **`/scene/<id>`**: the ground-truth timeline, the file inventory (each chip
  opens that file's details), the map, the extraction, disk and validation
  cards, the signals, the windows table with "Open 3D", and the notes.
- **`/scene/<id>/segment/<n>`**: one ground-truth segment, by its number in
  `odometry_segments.txt`: what it is, what the robot did in it, its map,
  notes, windows and frames.
- **`/scene/<id>/frame/<n>`**: the image, the height map, validity and cloud
  tiles, and the cloud projected onto the image (a check on the extrinsic).
  Click a tile to zoom; in the zoom, hover or arrow keys read single cells.
- **`/view/<scene>/<window>`**: the 3D view. Space plays and pauses, `[` and
  `]` step to the previous and next window, and the notes panel takes a
  frame range from the current frame ("now"). The first open of a window
  builds it; later ones come from the cache, and the next window is
  prefetched.

After re-extracting scenes the viewer notices the changed files by itself;
`--clear-cache` is only for a cache that went wrong.
[docs/VIEWER.md](docs/VIEWER.md#troubleshooting) has the troubleshooting.

## Slurm

The job files in `slurm/` are the ones the HELHEST releases were built with
on the CTU RCI cluster: `extract_v2.batch` and `finalize_v2.sh` (see
[Extracting a dataset, step by step](#extracting-a-dataset-step-by-step)),
`heightmaps.batch`, `viewer.batch` and `probe_clock_v2.batch`. The simplest
shape -- one job extracting a whole directory of bags (16 CPUs, 8 workers;
take the `#SBATCH` header and container setup from `extract_v2.batch`) -- is:

```bash
singularity exec --bind /mnt/personal --bind /mnt/data "$SIF" \
    "$REPO/.venv/bin/python" \
    -m dataset_tools.extract <bags> <output> --jobs 8
```

Because complete scenes are skipped, a re-queued job resumes where the previous
one stopped. Submit from the repository root (Slurm runs a spooled copy, so
the jobs find the checkout through the submit directory).
[slurm/README.md](slurm/README.md) says what each job does and which paths
are site-specific.

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
construction (constant, and changing halfway), sweep deskew, idle trimming,
the GNSS guard, ground-truth segmentation and I/O, pose interpolation and the
pose check, TF pins, camera info and extrinsics, image formats, the
odometry-frame rotation, height maps (against a stored parity sample of the
DRIFT builder), keyframe lists, release assembly from a spec, the scene
validator, disk usage, the atlas, the segment filter and the map-mismatch
test behind the segment scripts, `helhest-sort-bag`, and the viewer (probing,
window enumeration, the map's geodesy, splits, notes, the segment page,
rendering, the URL-root mirror and every route, against synthetic scene
directories).

`.github/workflows/ci.yml` runs three jobs: `lint` (`black` and `ruff`, pinned),
`extraction` (`.[test]`, every non-viewer console script's `--help`, pytest with the
viewer tests skipped) and `viewer` (`.[viewer,test]`, `helhest-viewer --help`,
pytest). [CONTRIBUTING.md](CONTRIBUTING.md) covers the two virtualenvs
(`.venv`, `.venv-viewer`) and the lint.

## Known caveats

- **A lidar frame is a sweep, not a snapshot.** The Odin1 scans its raster
  bottom to top over 67 ms and stamps the frame at the start; each point's
  own time is in the `offset_time` field. Until 2026-10-01 the extractor
  dropped that field and wrote the sweep as if simultaneous, `--time_align`
  then measured the clouds' effective instant 27-52 ms after the stamp (the
  mean of the offsets) and moved `data_ts` there -- so against `data_ts` the
  image, exposed at the stamp, read 30-50 ms early, and the cloud projected
  onto the image slid sideways in every fast turn. `--deskew auto` undoes the
  motion within the sweep; on such a root `time_align.txt` should read as
  already aligned. For a root made before, the image's instant is
  `data_ts - delta` with `delta` the correction the scene's `time_align.txt`
  gives at that time (line 2 summarises it, a range where it varied; 0 where
  none was applied), and the smear within the sweep
  cannot be undone without the bag.
- **The 2026-08 bags are not dataset material.** Their TF tree has no
  `base_link -> odin1_base_link` edge, so a scene fails without
  `--lidar_extrinsic`; with `--allow_frame_fallback` the identity is used and
  listed in `extract_meta.json` -> `frame_fallbacks`, leaving those clouds in
  the lidar frame. The 2026-08-21 `/joint_states` carry efforts only
  (commanded wheel velocities are in `joint_setpoints.npy`). The rig has had
  both since 2026-09-02 (`docs/NEW_RIG_READINESS.md`).
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
- **Published v1 clouds sit 0.34 m back (static-TF direction, fixed
  2026-08-28).** Earlier versions of this tool composed the static TF chain
  backwards; for the front Livox (x = +0.17 m, rolled 180°) only x came out
  wrong, so every previously published v1 cloud is shifted −0.34 m along x and
  re-extracting an old bag differs from it by exactly +0.34 m. Now
  `get_transform(G, src, dst)` returns `p_dst = T @ p_src` (tf2's
  `lookup_transform(dst, src)`) and `--lidar_extrinsic` is the lidar's pose in
  the base frame. Fix for existing data: `docs/DATA.md` §5.
- **`/imu/data` is mounted tilted** (~5 deg roll, ~11 deg pitch) and its frame
  has no TF to `base_link`: do not pass it as `--imu`. The default
  `/odin1/odometry` (or the Fixposition orientation) gives clean base
  roll/pitch.
- **The sensor unit may stamp at publish, not at capture.** On the 2026-09
  Odin1 recordings the poses run ahead of the imagery and the clouds by up to
  **~0.69 s**, varying with load within one bag; no clock on the robot is
  wrong. `--time_align auto` measures it per scene from the sharpness of the
  aggregated cloud and takes it off; `time_align.txt` records the curve. How
  and why: `docs/EXTRACTION_GUARDS.md`.
- **A GNSS/INS with no fix keeps publishing poses.** After a roll-over the
  Fixposition fills its odometry with a reset state hundreds of metres away
  and no flag (a 1249 m step in 64 ms in v2 beta 0010). `--gnss_guard auto`
  drops a frozen stretch only where a >6 m/s jump or a hole in the fixes
  confirms it, never on attitude alone, and fails towards keeping data
  (Fixposition topics only, so v2 beta to v2.2):
  `gnss_guard.txt`, `fix_lost` in `odometry_segments.txt`. Details:
  `docs/EXTRACTION_GUARDS.md`.
- **Frames with no pose under them.** Where `interpolate_poses` cannot
  interpolate (before or after the pose stream, or across a hole over 0.5 s)
  it copies the nearest sample, a stationary pose that looks like a
  measurement. `--gnss_guard` drops frames in holes over 2 s, and once the
  clock is final every frame with no pose on either side; before 2026-10-01
  those were exported (184 frames, all at scene ends, in HELHEST v2.2).
  Details: `docs/EXTRACTION_GUARDS.md`.
- **Out-of-order bags.** Timestamps are sorted before synchronisation. If a bag
  is badly out of order for other reasons, `helhest-sort-bag` rewrites it sorted
  by header stamp.

## Dataset

See [docs/DATA.md](docs/DATA.md) for the dataset itself — recordings, contents
and statistics. [docs/ODOMETRY_REGEN.md](docs/ODOMETRY_REGEN.md) covers
regenerating the ground truth and
[docs/GT_HOLE_REPLAY.md](docs/GT_HOLE_REPLAY.md) recovering the stretches where
the estimator published nothing; [docs/README.md](docs/README.md) lists what
else is there. Code comments and the campaign records refer to **DRIFT**: that
is the downstream training code the roots are extracted for (a separate
repository). It adds its own per-frame products to a root (proprio labels,
driven similarity); the height maps, the keyframe lists and the split are this
repository's (until 2026-10-12 the maps and lists were built by DRIFT's code,
which `dataset_tools.heightmap` and `dataset_tools.keyframes` reproduce).

## License

BSD-3-Clause, see [LICENSE](LICENSE). Copyright (c) VRAS (Vision for Robotics
and Autonomous Systems), Czech Technical University in Prague.
