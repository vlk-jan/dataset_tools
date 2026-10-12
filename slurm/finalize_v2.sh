#!/bin/bash
# Root bookkeeping after slurm/extract_v2.batch. With a release spec: the
# root-level files from it and the scenes (dataset_tools.release: scenes.csv,
# keyframe lists, manifest.json, splits.json, atlas/curation.json, README.md);
# without one: scenes.csv and the image-format declaration only. Then validate
# the root and regenerate the atlas (last, so its numbers match the scenes).
#
#   slurm/finalize_v2.sh [out_root] [bag_list.csv] [release_spec.json]
#   slurm/finalize_v2.sh /mnt/personal/$USER/HELHEST_v2_4 docs/v2_4_bags.csv docs/releases/v2_4.json
set -uo pipefail
OUT=${1:-/mnt/personal/$USER/HELHEST_v2_beta}
REPO=$(cd "$(dirname "$0")/.." && pwd)
. "$REPO/slurm/common.sh"
BAGS=${2:-$REPO/docs/v2_bags.csv}
SPEC=${3:-}
BINDS="--bind /mnt/personal --bind /mnt/data"
run() { singularity exec $BINDS "$SIF" env PYTHONPATH="$REPO" "$PY" "$@"; }

if [ -n "$SPEC" ]; then
    (cd "$REPO" && run -m dataset_tools.release "$OUT" --spec "$SPEC") || { echo "RELEASE: failed -- see above"; exit 2; }
else
    run -c "
from pathlib import Path
from dataset_tools.extract.scene_io import merge_scenes_csv
print(merge_scenes_csv(Path('$OUT')))
"
    # Writes manifest.json images.extension; fails if scenes disagree (partial re-extraction).
    run -m dataset_tools.image_format "$OUT" \
        || echo "IMAGE FORMAT: the scenes of $OUT do not hold one format -- see above"
fi
column -s, -t "$OUT/scenes.csv"

run -m dataset_tools.validate_scene "$OUT" --csv "$OUT/validation.csv"
vrc=$?

# --bags must be this release's list: v1's numbers its scenes the same way.
run "$REPO/scripts/make_atlas.py" "$OUT" --bags "$BAGS" \
    || echo "atlas generation reported problems"

exit $vrc
