# Building a release root

A release root is built entirely using tools in this repository, in two stages.
The first stage operates per bag: extracting the data, rotating odometry into
the cloud coordinate frame, generating height maps, and validating each scene.
The second stage operates on the whole root: producing the manifest, dataset
splits, keyframe lists, README, full validation report, and atlas.

The steps below use the Slurm jobs in `slurm/` on the CTU cluster. The same
steps [without Slurm](#without-slurm) follow at the end.

## 1. Describe the release

A release is described by two tracked files in `docs/`:

**The bag list**, `docs/v2_<n>_bags.csv`: one row per scene, in scene order,
with the header

```
scene,session,bag_dir_on_subtdata,bag_size_gb,bag_start_utc,duration_s,lidar_msgs,camera_msgs[,extract_args]
```

The jobs parse columns by position, so `scene` must be the first column and the
bag directory the third. The optional `extract_args` column holds extractor
options for that bag alone, such as `--trim_idle 60 10`. Bag lists are
cumulative: each release lists every scene in its root, including those from
earlier releases.

**The spec**, `docs/releases/v2_<n>.json`:

- Required: `dataset_version`, `bags` (the list above, relative to the
  repository root) and `held_out.scenes`.
- Optional: `inherit` (a spec to merge under this one; `helhest_v2.json`
  holds text shared across releases for this rig), `title`, `summary`,
  `held_out.decision`, `sessions`, `excluded`, `terrain_notes`, `defects`,
  `open_items` and `curation` (becomes `atlas/curation.json`).

Use `docs/releases/v2_4.json` as a template. All other metadata in the manifest
is measured directly from the scenes.

## 2. Extract every bag

From the repository root on a login node, with `.venv` built inside the
container (see [Installation](installation.md#container-ctu-cluster)):

```bash
mkdir -p slurm/slurm_out
CSV=docs/v2_4_bags.csv sbatch --array=0-35%4 slurm/extract_v2.batch /mnt/personal/$USER/HELHEST_v2_4
```

The array range covers the rows of the bag list (`0` to rows - 1); `%4` caps
how many jobs run at once. **Always pass `CSV=` and the root**: omitting either
falls back to a default that extracts the wrong bags. A root that links in an
earlier release's scenes is submitted with the range of its new rows only.

Each array task produces one finished scene:

1. copies its bag with `rsync` from `subtdata` to node-local scratch;
2. runs `helhest-extract` on it, with the row's `extract_args`;
3. runs `helhest-odom-frame measure` and `apply`;
4. runs `helhest-heightmap`;
5. validates the scene, moves it into `<root>/<scene>/`, and writes the report
   to `<root>/.validation/<scene>.csv`.

A scene that is already complete with the same bag is skipped, so
resubmitting the same range resumes cleanly. A scene that holds a *different*
bag stops the task; set `FORCE=1` to re-extract it. A scene that fails
validation is not published unless `FORCE_PUBLISH=1` is set. The exit code
names the failing step:

| Exit code | Step |
| --- | --- |
| 10 | fetching the bag |
| 11 | extraction |
| 12 | publishing the scene |
| 13 | the scene holds a different bag |
| 14 | validation |
| 15 | the odometry-frame rotation |
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
`scripts/make_atlas.py`. Its exit code is that of the validation step.

## 4. Check it

```bash
helhest-validate <root> -v       # include INFO findings
helhest-disk-usage <root>        # refresh disk_usage.json for the viewer
helhest-viewer <root> --open     # look at it
```

## Without Slurm

`helhest-extract` numbers scenes by the sorted bag directory names, and
`helhest-release` refuses a scene whose bag does not match the list, so the
bag directories must sort in list order. Rows with `extract_args` need an
individual call on that single bag, with `--start_index`:

```bash
helhest-extract <bags> <root> --jobs 8 --scenes 0-28                        # rows without extract_args
helhest-extract <bags>/<bag_29> <root> --start_index 29 --trim_idle 60 10   # one call per row with them
helhest-odom-frame measure <root> --fits fits.json
helhest-odom-frame apply   <root> --fits fits.json
helhest-heightmap <root>
helhest-release   <root> --spec docs/releases/v2_4.json
helhest-validate  <root> --csv <root>/validation.csv
python scripts/make_atlas.py <root> --bags docs/v2_4_bags.csv
```

## Rebuilding only the height maps

After a root's clouds or `gravity_align.npy` change:

```bash
sbatch --array=0-35 slurm/heightmaps.batch /mnt/personal/$USER/HELHEST_v2_4   # array index = scene
helhest-heightmap <root> --scenes 3 --overwrite                                # or by hand
```

The batch job first checks a sample of the existing maps against a rebuild
(`VERIFY=0` skips this) and passes `HM_ARGS` through to `helhest-heightmap`.
The tool will not write through a scene whose height maps are linked from
another root; build them in the root that owns them.
