#!/bin/bash
# Root bookkeeping after slurm/extract_v2.batch: rebuild scenes.csv, declare the
# image format, validate the root, and regenerate the atlas (last, so its numbers
# match the current scenes).
#
#   slurm/finalize_v2.sh [out_root] [bag_list.csv]
set -uo pipefail
OUT=${1:-/mnt/personal/$USER/HELHEST_v2_beta}
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/slurm/common.sh"
BAGS=${2:-$REPO/docs/v2_bags.csv}
BINDS="--bind /mnt/personal --bind /mnt/data"

singularity exec $BINDS "$SIF" "$PY" -c "
from pathlib import Path
from dataset_tools.extract.scene_io import merge_scenes_csv
print(merge_scenes_csv(Path('$OUT')))
"
column -s, -t "$OUT/scenes.csv"

# Writes manifest.json images.extension; fails if scenes disagree (partial re-extraction).
singularity exec $BINDS "$SIF" "$PY" -m dataset_tools.image_format "$OUT" \
    || echo "IMAGE FORMAT: the scenes of $OUT do not hold one format -- see above"

singularity exec $BINDS "$SIF" "$PY" -m dataset_tools.validate_scene \
    "$OUT" --csv "$OUT/validation.csv"
vrc=$?

# --bags must be this release's list: v1's numbers its scenes the same way.
singularity exec $BINDS "$SIF" "$PY" "$REPO/scripts/make_atlas.py" "$OUT" --bags "$BAGS" \
    || echo "atlas generation reported problems"

exit $vrc
