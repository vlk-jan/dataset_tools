# Building a release root

A release root is built entirely using tools in this repository, in two stages.
The first stage operates per bag: extracting the data, rotating odometry into
the cloud coordinate frame, generating height maps, and validating each scene.
The second stage operates on the whole root: producing the manifest, dataset
splits, keyframe lists, README, full validation report, and atlas.

The steps below use the Slurm jobs in `slurm/` on the CTU RCI cluster. The same
steps [without Slurm](#without-slurm) follow at the end.

## 1. Describe the release

A release is described by two tracked files in `docs/`. These files live in
the repository, not on this site:

**The bag list**,
[`docs/v2_<n>_bags.csv`](https://github.com/vlk-jan/dataset_tools/tree/main/docs)
(e.g.
[`docs/v2_4_bags.csv`](https://github.com/vlk-jan/dataset_tools/blob/main/docs/v2_4_bags.csv)):
one row per scene, in scene order, with the header

```
scene,session,bag_dir_on_subtdata,bag_size_gb,bag_start_utc,duration_s,lidar_msgs,camera_msgs[,extract_args]
```

The jobs parse columns by position, so `scene` must be the first column and the
bag directory the third. The optional `extract_args` column holds extractor
options for that bag alone, such as `--trim_idle 60 10`. Bag lists are
cumulative: each release lists every scene in its root, including those from
earlier releases. The jobs split rows on raw commas (not quoted CSV), so no
field, `extract_args` included, may contain a comma.

**The spec**,
[`docs/releases/v2_<n>.json`](https://github.com/vlk-jan/dataset_tools/tree/main/docs/releases):

- Required: `dataset_version`, `bags` (the list above, relative to the
  repository root) and `held_out.scenes`.
- Optional: `inherit` (a spec to merge under this one; [`helhest_v2.json`](https://github.com/vlk-jan/dataset_tools/blob/main/docs/releases/helhest_v2.json)
  holds text shared across releases for this rig), `title`, `summary`,
  `held_out.decision` and extra `held_out` keys (e.g. `held_out.support`),
  `bag_host`, `images`, `corrections`, `sessions`, `excluded`, `terrain_notes`,
  `defects`, `open_items` and `curation` (becomes `atlas/curation.json`).

Use
[`docs/releases/v2_4.json`](https://github.com/vlk-jan/dataset_tools/blob/main/docs/releases/v2_4.json)
as a template. The spec's descriptive blocks (`rig`, `camera`, `gps`,
`odometry`, `sessions`, ...) and `bag_host`, `images`, `corrections` and
`held_out` are copied or merged into the manifest; everything else is measured
from the scenes.

## 2. Extract every bag

From the repository root on a login node, with `.venv` built inside the
container (see [Installation](installation.md#container-ctu-cluster)), and
key-based ssh from the cluster to subtdata: each task rsyncs its bag from
`$SUBTDATA` (default `$USER@subtdata.felk.cvut.cz`). `EXTRACT_ARGS` adds
extractor options to every task of a submission, after the row's own
`extract_args`.

```bash
mkdir -p slurm/slurm_out
CSV=docs/v2_4_bags.csv HM_ARGS="--set statistic=p90 --set tie_jitter=none" \
    sbatch --array=0-35%4 slurm/extract_v2.batch /mnt/personal/$USER/HELHEST_v2_4
```

The array range covers the rows of the bag list (`0` to rows - 1); `%4` caps
how many jobs run at once. `CSV=` and the root are required. `HM_ARGS` sets
the height-map settings; the tool's defaults are not v2.4's, so a v2.4 build
passes the p90 settings shown. A root that links in an
earlier release's scenes is submitted with the range of its new rows only.

Each array task produces one finished scene:

1. copies its bag with `rsync` from `subtdata` to node-local scratch;
2. runs `helhest-extract` on it, with the row's `extract_args`;
3. runs `helhest-odom-frame measure`, then `apply` if the measurement succeeded
   (a scene it cannot measure keeps the Odin1's own odometry and gravity alignment,
   with a warning);
4. runs `helhest-heightmap` with `HM_ARGS`;
5. validates the scene, moves it into `<root>/<scene>/`, and writes the report
   to `<root>/.validation/<scene>.csv`.

A task skips a scene whose `source_bag.txt` already names the same bag, so
resubmitting the same range resumes. The job checks only that file, not the
frame counts. A scene that holds a *different* bag stops the task. `FORCE=1`
re-extracts the scene in both cases. A scene that fails validation is not
published unless
`FORCE_PUBLISH=1` is set. The exit code names the failing step:

| Exit code | Step |
| --- | --- |
| 10 | fetching the bag |
| 11 | extraction |
| 12 | publishing the scene |
| 13 | the scene holds a different bag |
| 14 | validation |
| 15 | the odometry-frame rotation (`apply`) |
| 16 | the height maps |

Logs go to `slurm/slurm_out/`. `SIF` and `REPO` override the container image
and the checkout; `slurm/README.md` lists every site-specific value.

## 3. Finalise the root

On the login node, not through `sbatch`:

```bash
slurm/finalize_v2.sh /mnt/personal/$USER/HELHEST_v2_4 docs/v2_4_bags.csv docs/releases/v2_4.json
```

This runs `helhest-release` (root-level files), validates the whole root into
`<root>/validation.csv`, and draws the viewer's atlas with
`scripts/make_atlas.py`. It exits 2 if `helhest-release` fails; otherwise with
the validation step's exit code. Atlas problems are logged but do not change
the exit code. The spec (third argument) is optional: without it the script
only rewrites `scenes.csv` and declares the image format, with no manifest or
keyframe lists.

## 4. Check it

```bash
helhest-validate <root> -v           # include INFO findings
helhest-disk-usage <root> --all     # rescan every scene for the viewer
helhest-viewer <root> --open         # look at it
```

## Without Slurm

Extract one bag per bag-list row, each with `--start_index` set to its scene
number and that row's `extract_args` (bags fetched into one local directory
`<bags>`, keeping their directory names):

```bash
tail -n +2 docs/v2_4_bags.csv | while IFS=, read -r scene _ bag _ _ _ _ _ args; do
    helhest-extract "<bags>/$(basename "$bag")" <root> --start_index "$((10#$scene))" $args
done
helhest-odom-frame measure <root> --fits fits.json
helhest-odom-frame apply   <root> --fits fits.json
helhest-heightmap <root> --set statistic=p90 --set tie_jitter=none
helhest-release   <root> --spec docs/releases/v2_4.json
helhest-validate  <root> --csv <root>/validation.csv
.venv/bin/python scripts/make_atlas.py <root> --bags docs/v2_4_bags.csv
```

Run `helhest-odom-frame` before `helhest-heightmap`, because `apply` rotates
`gravity_align.npy`, which the height maps are levelled with; if odom-frame is
re-run on a scene that already has maps, rebuild them with `--overwrite`.

## Rebuilding only the height maps

After a root's clouds or `gravity_align.npy` change:

```bash
HM_ARGS="--set statistic=p90 --set tie_jitter=none" \
    sbatch --array=0-35 slurm/heightmaps.batch /mnt/personal/$USER/HELHEST_v2_4   # array index = scene
helhest-heightmap <root> --scenes 0003 --overwrite                            # or by hand
```

The batch job first checks a sample of the existing maps against a rebuild
(`VERIFY=0` skips this) and passes `HM_ARGS` through to `helhest-heightmap`.
The tool's defaults are not the settings v2.4 was built with; v2.4's maps use
`statistic=p90` and no tie jitter, so pass
`HM_ARGS="--set statistic=p90 --set tie_jitter=none"` to the batch job and
`--set statistic=p90 --set tie_jitter=none` by hand. Each scene's
`height_maps.json` records the settings it was built with. `--scenes` takes
comma-separated scene directory names, e.g. `0003,0004`. The tool will not
write through a scene whose height maps are linked from another root; build them
in the root that owns them.
