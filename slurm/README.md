# Slurm job files

The jobs the HELHEST releases were built with, on the CTU **RCI** cluster. They
are kept as working examples rather than as portable tooling: every one of them
assumes the RCI layout (a Singularity image with ROS 2 Jazzy, `/mnt/personal`
and `/mnt/data` mounts, raw bags reachable by key-based ssh from
`subtdata.felk.cvut.cz`, node-local NVMe under `/data`). Anywhere else, read
them for the *shape* of a run and change the handful of paths at the top.

## Submit from the repository root

```bash
cd dataset_tools && mkdir -p slurm/slurm_out
CSV=docs/v2_4_bags.csv sbatch --array=0-35%4 slurm/extract_v2.batch /mnt/personal/$USER/HELHEST_v2_4
```

Slurm copies a batch file into its spool and runs it from there, so `$0` does
not point at the checkout. Each job finds the repository through the submit
directory (`$SLURM_SUBMIT_DIR`, or `REPO=...` to override) and refuses to run
from anywhere that is not a checkout (`viewer.batch` looks for
`pyproject.toml`, the others for `slurm/common.sh`). The `#SBATCH --output` paths are
relative to the same directory, and `sbatch` fails outright when the directory
they name is missing -- hence the `mkdir`.

## The jobs

| File | What it does | Array index |
| --- | --- | --- |
| `extract_v2.batch` | v2 rig (Odin1 + Fixposition): fetch one bag to node-local scratch, extract, rotate the odometry onto the clouds' frame (`odom_frame`), build the height maps (`dataset_tools.heightmap`), validate, copy the finished scene out and swap it in; per-bag extractor options come from the bag list's `extract_args` column, height-map settings from `HM_ARGS` (the defaults are not v2.4's: `HM_ARGS="--set statistic=p90 --set tie_jitter=none"`) | row of the bag list (`CSV=`, required; each release has its own cumulative `docs/v2_<n>_bags.csv`, so a root that reuses an earlier release's scenes is submitted with the array range of its new rows only -- v2.1, linked into v2.0, as `--array=12-17`) = scene number |
| `finalize_v2.sh` | after a v2 array run: with a release spec (third argument, `docs/releases/<n>.json`) write the root-level files with `dataset_tools.release` (scenes.csv, keyframe lists, manifest, splits, curation, README); without one only `scenes.csv` and the image format. Then validate the whole root and regenerate the atlas (second argument: the bag list). Run directly on a login node, not through `sbatch` | -- |
| `heightmaps.batch` | rebuild the height maps of scenes already in a root (after clouds or `gravity_align.npy` changed): build on node-local scratch, check a sample of the scene's own maps first (`VERIFY=0` skips), replace both directories | scene number |
| `probe_clock_v2.batch` | `scripts/probe_clock_skew.py` over one v2 bag: every publisher's clock against the recorder's | row of the bag list (`CSV=`, required) |
| `viewer.batch` | run `helhest-viewer` on a compute node, on a Unix socket in a private `/tmp` directory (other users share the node); `scripts/viewer_connect.sh` opens the tunnel from a laptop | -- |

## Site-specific values

Every path that names a machine, a share or an account is an environment
variable with the RCI value as its default, so a different site changes the
environment rather than the file:

| Variable | Default | Used by |
| --- | --- | --- |
| `REPO` | the submit directory | all |
| `SIF` | `/mnt/data/vras/data/helhest-jazzy.sif` | all |
| `SUBTDATA` | `$USER@subtdata.felk.cvut.cz` | the fetching jobs |
| `VIEWER_CACHE` | `/mnt/personal/$USER/viewer_cache` | `viewer.batch` |

The root is the first positional argument of every job and is required (for
`viewer.batch` it is the root to serve, for `probe_clock_v2.batch` the output
directory); so is the bag list, `CSV=` for the array jobs and the second
argument of `finalize_v2.sh`.

Nothing here calls another repository: extraction, the odom_frame correction,
the height maps and the root files are all this one's.

The Python that runs inside the container is the repository's own `.venv`
(`.venv-viewer` for the viewer), built with the container's interpreter as the
top-level README describes; the jobs do not install anything.

The fetching jobs share their preamble (bag-list row, node-local scratch,
fetch) through `common.sh`, which they source from `$REPO`. The v1
regeneration job (`regen_scene.batch`) and the PNG-to-JPEG conversion
(`images_from_bag.batch`) were removed once used; see git history.
