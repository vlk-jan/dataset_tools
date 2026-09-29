#!/bin/bash
# Dataset-root bookkeeping after slurm/extract_v2.batch: rebuild scenes.csv
# from the published scenes (extract_v2 runs one bag per task, so no single run
# ever sees them all), validate the whole root, and regenerate the atlas.
#
# The atlas has to be last and it has to be here rather than by hand: every
# number on the index page is rendered from the root, and the pose guard can
# change a scene's frame count, so an atlas from before a re-extraction shows
# the previous run's figures under the current title. v1.3 already shipped once
# with a stale atlas.
#
#   slurm/finalize_v2.sh [out_root] [bag_list.csv]
set -uo pipefail
OUT=${1:-/mnt/personal/$USER/HELHEST_v2_beta}
REPO=$(cd "$(dirname "$0")/.." && pwd)
BAGS=${2:-$REPO/docs/v2_bags.csv}
SIF=${SIF:-/mnt/data/vras/data/helhest-jazzy.sif}
PY="$REPO/.venv/bin/python"
BINDS="--bind /mnt/personal --bind /mnt/data"

singularity exec $BINDS "$SIF" "$PY" -c "
from pathlib import Path
from dataset_tools.extract_data import merge_scenes_csv
print(merge_scenes_csv(Path('$OUT')))
"
column -s, -t "$OUT/scenes.csv"

# What the images are, declared where a consumer looks for it (manifest.json ->
# images.extension) before anything validates against it. Fails on a root whose
# scenes do not agree, which is the state a partial re-extraction leaves.
singularity exec $BINDS "$SIF" "$PY" -m dataset_tools.image_format "$OUT" \
    || echo "IMAGE FORMAT: the scenes of $OUT do not hold one format -- see above"

singularity exec $BINDS "$SIF" "$PY" -m dataset_tools.validate_scene \
    "$OUT" --csv "$OUT/validation.csv"
vrc=$?

# --bags: make_atlas defaults to the v1 bag list, which numbers its scenes the
# same way and so silently lends every v2 card a v1 bag size and session.
singularity exec $BINDS "$SIF" "$PY" "$REPO/scripts/make_atlas.py" "$OUT" --bags "$BAGS" \
    || echo "atlas generation reported problems"

exit $vrc
