#!/bin/bash
# Dataset-root bookkeeping after slurm/extract_v2.batch: rebuild scenes.csv
# from the published scenes (extract_v2 runs one bag per task, so no single
# run ever sees them all) and validate the whole root.
#
#   slurm/finalize_v2.sh [out_root]
set -uo pipefail
OUT=${1:-/mnt/personal/vlkjan6/HELHEST_v2_beta}
REPO=/home/vlkjan6/projects/dataset_tools
SIF=/mnt/data/vras/data/helhest-jazzy.sif
PY="$REPO/.venv/bin/python"
BINDS="--bind /mnt/personal --bind /mnt/data"

singularity exec $BINDS "$SIF" "$PY" -c "
from pathlib import Path
from dataset_tools.extract_data import merge_scenes_csv
print(merge_scenes_csv(Path('$OUT')))
"
column -s, -t "$OUT/scenes.csv"

singularity exec $BINDS "$SIF" "$PY" -m dataset_tools.validate_scene \
    "$OUT" --csv "$OUT/validation.csv"
