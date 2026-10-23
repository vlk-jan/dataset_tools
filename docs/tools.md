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
matched pair's timestamp is corrected for sensor publish latency, each lidar
sweep is deskewed to that instant, and odometry is interpolated onto it. See
[Timing in the dataset layout](dataset.md#timing) for clock details.

**Resuming.** A scene is marked complete once `source_bag.txt` is written at
the end of extraction. Completed scenes are skipped by default, allowing an
interrupted run to be resumed cleanly; `--overwrite` re-extracts them. If an
existing scene corresponds to a different bag than expected, the run stops
immediately before extracting anything to prevent indexing conflicts from an
altered bag list.

**Failures.** If a bag fails, its traceback is logged and the run continues
with remaining bags. The summary at the end lists every failed and empty scene,
and the command exits with code `1` if any bag failed.

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
| `--deskew auto/off` | Motion-compensate each lidar return to the robot's pose at the frame timestamp (default: `auto`) |
| `--deskew_imu TOPIC` | IMU topic providing angular velocity for intra-sweep deskewing (default: `/odin1/imu`; `off` interpolates poses instead) |
| `--check_odom TOPIC` | Reference odometry stream compared against primary odometry in `pose_check.txt` (default: `/fixposition/odometry_enu`; `off` skips) |
| `--gnss_guard auto/on/off` | Drop frames lacking valid pose estimates or containing transient GNSS/INS reset solutions after losing fix |
| `--trim_idle MAX_S KEEP_S` | In stationary periods longer than `MAX_S` seconds, retain only the first `KEEP_S` seconds of frames (disabled by default) |
| `--gps TOPIC` | GNSS fix topic for `gps.npy`; `auto` (default) selects the active `NavSatFix` topic, `off` records none |
| `--tf_pins on/off` | Replace online Odin1 lidar and camera extrinsic estimates (which drift across and within bags) with one constant: the median of what 42 bags published (default: `on`) |
| `--lidar_extrinsic X Y Z ROLL PITCH YAW` | Fixed lidar-to-base transform when unavailable from the TF tree, in metres and radians |
| `--allow_frame_fallback` | Extract even when TF does not reach `--base_frame`, leaving that stream in its native frame (otherwise the scene fails) |
| `--max_sync_dt S` | Maximum allowable timestamp difference between synchronised camera and lidar frames (default: `0.1`) |
| `--image_format camera/png` | Store raw compressed JPEG bytes from the camera (default) or decode to PNG |
| `-v` | Enable verbose/debug logging |

All options used during extraction, along with automatic decisions, are
recorded in each scene's `extract_meta.json`.

**Preserving derived files on re-extraction.** Height maps and other per-frame
products are generated after extraction, one file per frame. If a re-extraction
produces the same `data_ts.txt`, these files are kept; if timestamps differ,
outdated files are renamed to `<name>.stale-<date>-<time>` to avoid timestamp
misalignment.

## helhest-odom-frame

The Odin1 computes its odometry with an online estimate of the lidar-to-IMU
extrinsic, while the clouds go through the fixed value `--tf_pins` sets. Consequently, a scene's
odometry frame and cloud frame can differ by up to 0.8 degrees. This tool
measures that angular offset by registering clouds against each other (ICP) and
rotates `odometry.txt`, `transforms.npy`, and `gravity_align.npy` into the cloud
frame. It requires the `build` extra.

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
helhest-heightmap <root> --scenes 3 --overwrite    # rebuild one scene
helhest-heightmap <root> --workers 16 --set statistic=p90
```

The parameters used are recorded in each scene's `height_maps.json`. The tool
refuses to add maps built with different settings to an existing scene. Frames
with too few lidar returns receive no map.

## helhest-keyframes

Writes the root's three keyframe lists: `keyframes.lst` (all scenes),
`keyframes_train.lst` (training scenes), and `keyframes_heldout.lst` (held-out
scenes). Each line lists one frame that contains a cloud, an image, a height map,
and a mask:

```
0003/clouds/00042.npy 0003/images/00042.jpg
```

Frames within 4 s before the robot tilts more than 45 degrees are excluded.
Held-out scenes are determined from `manifest.json`. `helhest-release` runs this
step automatically.

## helhest-release

Generates a release's root-level files from its specification and scenes:
`scenes.csv`, the keyframe lists, `manifest.json`, `splits.json`,
`atlas/curation.json`, and `README.md`.

```bash
helhest-release <root> --spec docs/releases/v2_5.json
helhest-release <root> --spec docs/releases/v2_5.json --out_dir /tmp/compare
```

The specification holds curated metadata that cannot be derived automatically:
the bag list, held-out scene selections and rationales, session groupings, and
known defects. All other manifest fields are measured directly from the scenes.
The tool rejects any root whose scenes do not match the bag list, or where a
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
across images, clouds, timestamps, poses, and gravity alignment, that timestamps
are monotonic, that clouds have the expected shape and dtype, that
rotation matrices are valid, and that camera calibration parameters and
extrinsics are reasonable. It also checks ground-truth segments, proprioception
and GNSS streams, per-frame products, and, for a whole root, the manifest and
keyframe lists.

Findings are reported at three severity levels:

- **Errors**: Structural defects. Exit code `1`.
- **Warnings**: Data-quality issues, such as NaNs, implausible speeds, or
  extreme tilt. Passing `--strict` causes warnings to fail with exit code `1`
  as well.
- **INFO**: Informational notices that do not indicate defects, such as a gap
  in the frame clock. Displayed with `-v`.

Every finding has a stable identifier printed in brackets. To use `--strict` on
a root with known acceptable anomalies, list them in
`<root>/validation_expect.json`:

```json
{"expected": [["0008", "TILT"], ["*", "CLOUD_EMPTY"]]}
```

Expected findings are still reported in the log but no longer trigger failure.
Any entry that matches nothing is reported as `EXPECT_UNUSED`, preventing
outdated exceptions from persisting.

## helhest-disk-usage

Generates `<root>/disk_usage.json`, recording bytes and file counts per scene
and modality for the viewer's disk-usage panels. It scans only scenes not yet
present in the file, unless scenes are specified explicitly or `--all` is
passed. Extraction runs update this file automatically for the scenes they
process.

```bash
helhest-disk-usage <root> [scene ...] [--all]
```

## helhest-image-format

Declares in `manifest.json` whether a root stores raw camera JPEG bytes or PNG
images. Any scene containing images in a format different from the declared one
is reported as an error.

```bash
helhest-image-format <root>            # declare what the scenes hold
helhest-image-format <root> --check    # report only
```

## helhest-sort-bag

Rewrites a bag with its messages sorted chronologically by header timestamp,
for bags with out-of-order records. The extractor already sorts timestamps
before pairing frames, so this tool is rarely needed. The entire bag is loaded
into memory during sorting.

```bash
helhest-sort-bag <input-bag> <output-bag>
```
