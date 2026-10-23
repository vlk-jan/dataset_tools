# Tools

Every tool is a command installed with the package. Run any of them with
`--help` for the full list of options.

| Command | What it does |
| --- | --- |
| [`helhest-extract`](#helhest-extract) | Extracts ROS 2 bags into scene directories |
| [`helhest-odom-frame`](#helhest-odom-frame) | Aligns odometry with the lidar cloud coordinate frame |
| [`helhest-heightmap`](#helhest-heightmap) | Generates height maps and validity masks from a scene's clouds |
| [`helhest-keyframes`](#helhest-keyframes) | Generates keyframe lists for a root |
| [`helhest-release`](#helhest-release) | Generates root-level release files from a specification |
| [`helhest-validate`](#helhest-validate) | Validates scenes and complete roots |
| [`helhest-viewer`](viewer.md) | Browses a root in a web browser |
| [`helhest-disk-usage`](#helhest-disk-usage) | Summarises disk usage per scene and modality |
| [`helhest-image-format`](#helhest-image-format) | Declares or checks the image format stored in a root |
| [`helhest-sort-bag`](#helhest-sort-bag) | Rewrites a bag sorted by header timestamp |

## helhest-extract

Extracts ROS 2 bags into scene directories (`0000/`, `0001/`, etc.), producing
one scene per bag. See the [dataset layout](dataset.md) for details on all
generated files.

```bash
helhest-extract <dir-with-bags> <root> --jobs 8
```

`<dir-with-bags>` can be a directory containing multiple bags or a single bag
directory. Only directories containing `metadata.yaml` are recognised as bags.
Scenes are numbered sequentially according to the sorted names of the bag
directories.

By default, the extractor reads the robot's primary sensors: the Odin1
lidar-camera unit and the Fixposition GNSS/INS. Spelled out in full, the default
arguments are:

```
--cam /odin1/image/compressed --lidar /odin1/cloud_filtered/cloudini \
--odom /odin1/odometry --imu /odin1/odometry --deskew_imu /odin1/imu \
--check_odom /fixposition/odometry_enu --gps auto
```

**Processing pipeline.** Camera and lidar frames are matched by mutual nearest
neighbour on their header timestamps, and unpaired frames are dropped. Each
lidar sweep that carries per-point times and is bracketed by odometry is
deskewed to its frame's instant; other sweeps are written as recorded and
counted in `extract_meta.json`. Odometry is interpolated onto each frame. With
`--time_align auto`, the publish latency is then measured from the written
clouds and subtracted from the frame clock. See [Timing in the dataset
layout](dataset.md#timing) for clock details.

**Resuming.** A scene counts as complete when `source_bag.txt`, written last,
is present and its image, cloud and timestamp counts agree. Completed scenes
are skipped by default, allowing an interrupted run to be resumed cleanly;
`--overwrite` re-extracts them. If an existing scene corresponds to a different
bag than expected, the run stops before extracting anything (exit code `2`) to
prevent indexing conflicts from an altered bag list.

**Failures.** If a bag fails, its traceback is logged and the run continues
with remaining bags. The summary at the end lists every failed and empty scene,
and the command exits with code `1` if any bag failed. It exits with `2` when
no bag is found, `--scenes` selects none, or an existing scene holds a
different bag.

### Main options

| Option | Meaning |
| --- | --- |
| `--jobs N` | Number of worker processes, processing one bag each (default: 4; `1` runs in-process) |
| `--scenes 0 3 5-9 <bag-name>` | Extract only specified scene indices, ranges, or bag directory names |
| `--start_index N` | Number the first scene `N` instead of `0000` |
| `--overwrite` | Re-extract scenes that are already complete |
| `--base_frame` | Target coordinate frame for clouds, odometry, and gravity alignment (default: `base_link`) |
| `--gt auto/on/off` | Write ground-truth `odometry.txt` and `odometry_segments.txt` (`auto`: enables this for Odin1 and Fixposition odometry) |
| `--time_align auto/off/S` | Correct frame timestamps for publish latency: measured per scene (`auto`, default), fixed duration in seconds, or disabled (`off`) |
| `--deskew auto/off` | Motion-compensate each lidar return to the robot's pose at the frame timestamp, where the sweep has per-point times and odometry covers it (default: `auto`) |
| `--deskew_imu TOPIC` | IMU topic providing angular velocity for intra-sweep deskewing (default: `/odin1/imu`; `off` interpolates poses instead) |
| `--imu TOPIC` | Orientation source for `gravity_align.npy` (default: `/odin1/odometry`) |
| `--check_odom TOPIC` | Reference odometry stream compared against primary odometry in `pose_check.txt` (default: `/fixposition/odometry_enu`; `off` skips) |
| `--gnss_guard auto/on/off` | Drop frames the pose stream does not cover and stretches of transient GNSS/INS reset solutions. `auto` (default) runs the full guard for Fixposition odometry, the coverage-only part for `/odin1/odometry`; recorded in `gnss_guard.txt` |
| `--trim_idle MAX_S KEEP_S` | Cut every stop (odometry under 0.15 m/s and 5 deg/s) longer than `MAX_S` seconds to its first `KEEP_S` seconds of frames; `odometry.txt` is kept whole (off by default) |
| `--gps TOPIC` | GNSS fix topic for `gps.npy`; `auto` (default) selects the active `NavSatFix` topic, `off` records none |
| `--tf_pins on/off` | Replace online Odin1 lidar and camera extrinsic estimates (which drift across and within bags) with one constant: the median of what 42 bags published (default: `on`) |
| `--lidar_extrinsic X Y Z ROLL PITCH YAW` | Fixed lidar-to-base transform when unavailable from the TF tree, in metres and radians |
| `--allow_frame_fallback` | Extract even when TF does not reach `--base_frame`, leaving that stream in its native frame (otherwise the scene fails) |
| `--max_sync_dt S` | Maximum allowable timestamp difference between synchronised camera and lidar frames (default: `0.1`) |
| `--cam_time_offset S` | Seconds added to the camera stamps before pairing; unlike `--time_align`, this changes which camera and lidar frames are paired (default: `0`) |
| `--image_format camera/png` | Store raw compressed JPEG bytes from the camera (default) or decode to PNG |
| `-v` | Enable verbose/debug logging |

All options used during extraction, along with automatic decisions, are
recorded in each scene's `extract_meta.json`.

**Preserving derived files on re-extraction.** Height maps and other derived
products are built after extraction. If a re-extraction produces the same
`data_ts.txt`, they are kept. If the timestamps differ, each derived directory
(`height_maps/`, `height_masks/`, `proprio_labels/`, `driven_similarity/`) and
`height_maps.json` is renamed whole to `<name>.stale-YYYYmmdd-HHMMSS`, so no
product is left misaligned with the new frames.

## helhest-odom-frame

The Odin1 computes its odometry with an online estimate of the lidar-to-IMU
extrinsic. The clouds use the fixed value `--tf_pins` sets. Consequently, a
scene's odometry frame and cloud frame can differ by up to 0.8 degrees. This
tool measures that angular offset by registering clouds against each other
(ICP) and rotates `odometry.txt`, `transforms.npy`, and `gravity_align.npy`
into the cloud frame. Only `measure` needs the `build` extra. Both subcommands
take `--scenes`. `apply` refuses a scene that already has `odom_frame.txt` or a
correction over 3 degrees. After `apply`, rebuild the scene's height maps and
the keyframe lists.

```bash
helhest-odom-frame measure <root> --fits fits.json   # every scene with clouds and no odom_frame.txt
helhest-odom-frame apply   <root> --fits fits.json
```

The result is recorded in each scene's `odom_frame.txt`. Re-extracting a scene
removes this file, so rerun `measure` and `apply` afterwards.

## helhest-heightmap

Builds a height map and validity mask for each frame. The cloud is levelled
using `gravity_align.npy` and binned onto a 128 x 128 grid of 0.1 m cells around
`base_link`, then interpolated, filled, and smoothed.

```bash
helhest-heightmap <root>                           # every scene
helhest-heightmap <root> --scenes 0003 --overwrite    # rebuild one scene
helhest-heightmap <root> --workers 16 --set statistic=p90 --set tie_jitter=none
```

The parameters used are recorded in each scene's `height_maps.json`. The tool's
defaults are not the settings HELHEST v2.4 was built with: v2.4's maps use
`--set statistic=p90 --set tie_jitter=none`. Frames that already have a map are
kept unless `--overwrite` is passed; adding maps with different settings to a scene
is refused, so rebuild the whole scene with `--overwrite`. Frames with too few
lidar returns receive no map.

## helhest-keyframes

Writes the root's three keyframe lists: `keyframes.lst` (all scenes),
`keyframes_train.lst` (training scenes), and `keyframes_heldout.lst` (held-out
scenes). Each line lists one frame that contains a cloud, an image, a height map,
and a mask:

```
0003/clouds/00042.npy 0003/images/00042.jpg
```

Frames tilted more than 45 degrees, and the 4 s before such a tilt, are
excluded (`--max_tilt_deg`, `--tilt_horizon`; a negative `--max_tilt_deg`
disables the gate). `--out_dir` writes the lists elsewhere. Held-out scenes are
determined from `manifest.json`. The image extension comes from `manifest.json`
(`images.extension`). `helhest-release` runs this step automatically.

## helhest-release

Generates a release's root-level files from its specification and scenes:
`scenes.csv`, the keyframe lists, `manifest.json`, `splits.json`,
`atlas/curation.json`, and `README.md`.

```bash
helhest-release <root> --spec docs/releases/v2_4.json
helhest-release <root> --spec docs/releases/v2_4.json --out_dir /tmp/compare
```

With `--out_dir`, the files go to that directory, except `scenes.csv`, which is
only updated in the root itself.

The specification holds curated metadata that cannot be derived automatically:
the bag list, held-out scene selections and rationales, session groupings, and
known defects. The spec's descriptive blocks and keys such as `bag_host`, `images`
and `corrections` are copied or merged into the manifest; everything else is
measured from the scenes. The tool rejects a root whose scene directories do
not match the bag list (a missing, extra or mismatched scene), or where a
designated held-out scene is missing from disk. See
[Building a release root](building.md#1-describe-the-release).

## helhest-validate

Validates extracted scenes and prints a diagnostic report.

```bash
helhest-validate <root>                                # a whole root
helhest-validate <root>/0000 <root>/0001               # single scenes
helhest-validate <root> --csv <root>/validation.csv    # the report the viewer shows
helhest-validate <root> --json <root>/validation.json --strict
```

The validator verifies that required files are present, that frame counts agree
across images, clouds, timestamps, poses, and gravity alignment, that
timestamps are monotonic, that a sample of clouds (8 per scene,
`--sample-clouds N`, `0` skips them) has the expected `(N, 4)` shape and a
consistent float dtype, that rotation matrices are valid, and that camera
calibration parameters and extrinsics are reasonable. It also checks
ground-truth segments, proprioception and GNSS streams, per-frame products,
and, for a whole root, the manifest and keyframe lists.

Findings are reported at three severity levels:

- **`ERROR`**: Structural defects. Exit code `1`.
- **`WARN`**: Data-quality issues, such as NaNs, implausible speeds, or
  extreme tilt. Passing `--strict` causes warnings to fail with exit code `1`
  as well.
- **`INFO`**: Informational notices that do not indicate defects, such as a gap
  in the frame clock. Displayed with `-v`.

Exit code `2` means a path was not found or the expect file is unreadable.

Every finding has a stable identifier printed in brackets. To use `--strict` on
a root with known acceptable anomalies, list them in
`<root>/validation_expect.json`:

```json
{"expected": [["0008", "TILT"], ["*", "CLOUD_EMPTY"]]}
```

Expected findings no longer fail the run; they stay in `--json` output and are
printed with `-v`. An entry that matches nothing is reported as an `INFO` finding
`EXPECT_UNUSED` (shown with `-v`), so stale exceptions are noticed. `--expect PATH`
reads another file.

## helhest-disk-usage

Generates `<root>/disk_usage.json`, recording bytes and file counts per scene
and modality for the viewer's disk-usage panels. It scans only scenes not yet
present in the file, unless scenes are specified explicitly or `--all` is
passed. Use `--all` to refresh sizes after a scene changed. Extraction runs
update this file automatically for the scenes they process.

```bash
helhest-disk-usage <root> [scene ...] [--all]
```

## helhest-image-format

Declares in `manifest.json` whether a root stores raw camera JPEG bytes or PNG
images. Any scene containing images in a format different from the declared one
is reported as an error. `helhest-extract` runs the declaration at the end of
every run, so `manifest.json` can exist before the release step. A JPEG root is
declared as `images.extension: jpg`; a root without the declaration is read as
PNG. Both modes exit `1` when the scenes do not hold one format or disagree
with the declaration.

```bash
helhest-image-format <root>            # declare what the scenes hold
helhest-image-format <root> --check    # report only
```

## helhest-sort-bag

Rewrites a bag with its messages sorted by header timestamp; messages without a
usable header stamp keep their bag receive time. The extractor already sorts
timestamps before pairing frames, so this tool is rarely needed. The entire bag
is loaded into memory during sorting. An existing output is refused unless
`--overwrite` is passed.

```bash
helhest-sort-bag <input-bag> <output-bag> [--overwrite]
```
