#!/bin/bash
# Fan-out driver for the v1 odometry regeneration on dgx07 (or any host with
# singularity, ssh access to subtdata and the RCI mounts).
#
#   scripts/dgx07_regen.sh <scene_bags.csv> <out_root> [jobs] [rate] [scenes...]
#
# Per scene: rsync the bag from subtdata -> scripts/run_scene.sh (FAST-LIO +
# recorder with segment restarts) -> extract_data.py (clouds, images,
# camera_info.yaml, transforms.npy) -> validate_scene.py -> delete the bag.
# Every slot gets its own ROS_DOMAIN_ID; a scene that already has
# run_stats.txt with rc=0 is skipped, so re-running resumes.
#
# Environment (override as needed):
#   SIF        singularity image with ROS Jazzy + PCL  (the 2026-02-18 laptop image)
#   FLIO_WS    colcon workspace with fast_lio built inside $SIF
#   REPO       this dataset_tools checkout (branch odometry-regen-recorder)
#   CONFIG     fast_lio params file name in share/fast_lio/config (default mid360_v2.yaml)
#   SUBTDATA   ssh host holding the bags
#   STAGE      local staging dir for bags (needs ~70 GB per concurrent job)
set -uo pipefail

CSV=${1:?scene_bags.csv}; OUT=${2:?out_root}; JOBS=${3:-8}; RATE=${4:-1.0}; shift 4 2>/dev/null || shift $#
ONLY=("$@")

SIF=${SIF:-/mnt/personal/vlkjan6/sif/helhest-jazzy-2026-02-18.sif}
FLIO_WS=${FLIO_WS:-$HOME/flio_ws}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
CONFIG=${CONFIG:-mid360_v2.yaml}   # a file NAME under the installed share/fast_lio/config
SUBTDATA=${SUBTDATA:-vlkjan6@subtdata.felk.cvut.cz}
STAGE=${STAGE:-/mnt/personal/vlkjan6/regen_stage}
BINDS="--bind /mnt/personal --bind /mnt/data --bind /tmp"

mkdir -p "$OUT" "$STAGE" "$OUT/logs"
for tool in parallel rsync singularity; do command -v $tool >/dev/null || { echo "missing $tool" >&2; exit 1; }; done
[ -f "$SIF" ] || { echo "no sif at $SIF" >&2; exit 1; }
[ -d "$FLIO_WS/install" ] || { echo "no built workspace at $FLIO_WS" >&2; exit 1; }

run_one() {
    local scene=$1 bagdir=$2 slot=$3
    local out="$OUT/$scene" bag="$STAGE/$(basename "$bagdir")" log="$OUT/logs/$scene.log"
    if grep -qs '^rc=0' "$out/run_stats.txt" 2>/dev/null; then echo "$scene: done, skipping"; return 0; fi
    mkdir -p "$out"
    {
    echo "[$(date '+%F %T')] $scene fetch $bagdir"
    rsync -a --partial "$SUBTDATA:$bagdir/" "$bag/" || { echo "$scene: fetch failed"; return 10; }
    echo "[$(date '+%F %T')] $scene odometry (slot $slot, domain $((100+slot)))"
    ROS_DOMAIN_ID=$((100+slot)) ROS_LOCALHOST_ONLY=1 \
    singularity exec $BINDS "$SIF" bash -c "
        source /opt/ros/jazzy/setup.bash && source '$FLIO_WS/install/setup.bash' &&
        PYTHONPATH='$REPO' '$REPO/scripts/run_scene.sh' '$bag' '$out' '$CONFIG' '$RATE' --restart"
    local rc=$?
    echo "[$(date '+%F %T')] $scene run_scene rc=$rc"
    if [ $rc -eq 0 ] || [ $rc -eq 4 ]; then
        echo "[$(date '+%F %T')] $scene extract"
        singularity exec $BINDS "$SIF" "$REPO/.venv/bin/python" "$REPO/dataset_tools/extract_data.py" \
            "$(dirname "$bag")" "$OUT" --scenes "$(basename "$bag")" --start_index "$((10#$scene))" --jobs 4 \
            || echo "$scene: extraction failed"
        singularity exec $BINDS "$SIF" "$REPO/.venv/bin/python" "$REPO/dataset_tools/validate_scene.py" "$out" \
            --csv "$out/validation.csv" || echo "$scene: validation reported problems"
    fi
    rm -rf "$bag"
    echo "[$(date '+%F %T')] $scene finished rc=$rc"
    return $rc
    } >> "$log" 2>&1
}
export -f run_one
export OUT STAGE SIF FLIO_WS REPO CONFIG SUBTDATA RATE BINDS

# scene,bag_dir_on_subtdata,... ; longest scenes first so the tail of the run is short
tail -n +2 "$CSV" | awk -F, '{print $1","$2","$6}' | sort -t, -k3 -nr | cut -d, -f1,2 |
while IFS=, read -r scene bagdir; do
    if [ ${#ONLY[@]} -gt 0 ]; then printf '%s\n' "${ONLY[@]}" | grep -qx "$scene" || continue; fi
    echo "$scene $bagdir"
done | parallel -j "$JOBS" --colsep ' ' --joblog "$OUT/logs/joblog.txt" run_one {1} {2} {%}
echo "all done; joblog at $OUT/logs/joblog.txt"
