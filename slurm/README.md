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
sbatch --array=0-25%4 slurm/extract_v2.batch /mnt/personal/$USER/HELHEST_v2_beta
```

Slurm copies a batch file into its spool and runs it from there, so `$0` does
not point at the checkout. Each job finds the repository through the submit
directory (`$SLURM_SUBMIT_DIR`, or `REPO=...` to override) and refuses to run
from anywhere that has no `pyproject.toml`. The `#SBATCH --output` paths are
relative to the same directory, and `sbatch` fails outright when the directory
they name is missing -- hence the `mkdir`.

## The jobs

| File | What it does | Array index |
| --- | --- | --- |
| `extract_v2.batch` | v2 rig (Odin1 + Fixposition): fetch one bag to node-local scratch, extract, validate, build the height maps with DRIFT's batch, rsync the finished scene out | row of the bag list (`CSV=`, default `docs/v2_bags.csv`; v2.0 uses `docs/v2_0_bags.csv`, v2.1 `docs/v2_1_bags.csv` from row 12: rows 0-11 are v2.0's scenes, linked, and must not be extracted into v2.1) = scene number |
| `finalize_v2.sh` | after a v2 array run: rebuild `scenes.csv`, validate the whole root, regenerate the atlas (second argument: the bag list). Run directly on a login node, not through `sbatch` | -- |
| `survey_v2.batch` | `scripts/survey_v2_bag.py` over one bag of `docs/v2_inventory.csv`: what it carries, how the Odin behaved, where the robot went, thumbnails -- ~1 MB per bag, no extraction. Basis of `docs/V2_0_SELECTION.md` | row of `docs/v2_inventory.csv` |
| `probe_clock_v2.batch` | `scripts/probe_clock_skew.py` over one v2 bag: every publisher's clock against the recorder's | row of `docs/v2_bags.csv` |
| `slurm_extract.batch` | the simplest shape: extract a whole directory of bags in one job with `--jobs 8` | -- |
| `viewer.batch` | run `helhest-viewer` on a compute node, bound to `127.0.0.1`; `scripts/viewer_connect.sh` opens the tunnel from a laptop | -- |

## Site-specific values

Every path that names a machine, a share or an account is an environment
variable with the RCI value as its default, so a different site changes the
environment rather than the file:

| Variable | Default | Used by |
| --- | --- | --- |
| `REPO` | the submit directory | all |
| `SIF` | `/mnt/data/vras/data/helhest-jazzy.sif` | all |
| `SUBTDATA` | `$USER@subtdata.felk.cvut.cz` | the fetching jobs |
| `DRIFT` | `$HOME/projects/DRIFT` | `extract_v2.batch` (height maps) |
| `VIEWER_CACHE` | `/mnt/personal/$USER/viewer_cache` | `viewer.batch` |

The output root is the first positional argument of every job, defaulting to
`/mnt/personal/$USER/<release>`.

The Python that runs inside the container is the repository's own `.venv`
(`.venv-viewer` for the viewer), built with the container's interpreter as the
top-level README describes; the jobs do not install anything.

The fetching jobs share their preamble (bag-list row, node-local scratch,
fetch) through `common.sh`, which they source from `$REPO`. The v1
regeneration job (`regen_scene.batch`) and the PNG-to-JPEG conversion
(`images_from_bag.batch`) were removed once used; see git history.
