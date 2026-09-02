#!/bin/bash
# Fan-out driver for the v1 odometry regeneration (HELHEST v1.2) on dgx07, or
# any host with singularity, ssh access to subtdata and the RCI mounts.
#
#   scripts/dgx07_regen.sh <scene_bags.csv> <out_root> [jobs] [rate] [scenes...]
#
# Per scene, in its own staging dir and ROS domain (100 + scene number):
#   Scene layout: consumer files at the scene top level, provenance in
#   <scene>/regen/ (see the dataset root README.md).
#   1. rsync the bag from subtdata
#   2. run_scene.sh          FAST-LIO + recorder with segment restarts  -> <scene>/regen/
#   3. extract_odom_ref.py   /odom reference,  extract_wheels.py wheel speeds -> <scene>/regen/
#   4. filter_segments.py    post-hoc segment filter                  -> <scene>/odometry.txt
#   5. extract_data.py       clouds, images, camera_info.yaml, transforms.npy, ...
#   6. validate_scene.py
#   7. delete the staged bag
# A scene whose <scene>/done marker exists is skipped, so re-running resumes.
#
# Environment (override as needed):
#   SIF          image with ROS Jazzy + PCL (the 2026-02-18 laptop image)  - FAST-LIO
#   EXTRACT_SIF  image the dataset_tools .venv was built in               - extraction, audits
#   FLIO_WS      colcon workspace with fast_lio built inside $SIF
#   REGEN_REPO   dataset_tools checkout of branch odometry-regen-recorder (run_scene.sh, audits)
#   REPO         this dataset_tools checkout (extract_data.py, filter_segments.py, .venv)
#   CONFIG       fast_lio params file name in share/fast_lio/config (default mid360_v2.yaml)
#   SUBTDATA     ssh host holding the bags
#   STAGE        local staging root for bags (~70 GB per concurrent job)
set -uo pipefail

CSV=${1:?scene_bags.csv}; OUT=${2:?out_root}; JOBS=${3:-8}; RATE=${4:-1.0}
shift 4 2>/dev/null || shift $#
ONLY=("$@")

SIF=${SIF:-/mnt/personal/vlkjan6/sif/helhest-jazzy-2026-02-18.sif}
EXTRACT_SIF=${EXTRACT_SIF:-/mnt/data/vras/data/helhest-jazzy.sif}
FLIO_WS=${FLIO_WS:-$HOME/flio_ws}
REGEN_REPO=${REGEN_REPO:-$HOME/flio_ws/src/dataset_tools_regen}
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
CONFIG=${CONFIG:-mid360_v2.yaml}
SUBTDATA=${SUBTDATA:-vlkjan6@subtdata.felk.cvut.cz}
STAGE=${STAGE:-/mnt/personal/vlkjan6/regen_stage}
BINDS="--bind /mnt/personal --bind /mnt/data --bind /tmp"
# The FAST-LIO container is started with --pwd <scene>/regen: run_scene.sh runs
# the recorder as `python3 -m dataset_tools.odom_node`, and the working directory
# is first on sys.path, so a checkout of dataset_tools as cwd would shadow the
# branch's package with whatever is checked out there.
PY="$REPO/.venv/bin/python"

mkdir -p "$OUT" "$STAGE" "$OUT/logs"
for tool in xargs rsync singularity; do command -v $tool >/dev/null || { echo "missing $tool" >&2; exit 1; }; done
[ -f "$SIF" ] || { echo "no sif at $SIF" >&2; exit 1; }
[ -x "$FLIO_WS/install/fast_lio/lib/fast_lio/fastlio_mapping" ] || { echo "fast_lio not built in $FLIO_WS" >&2; exit 1; }
[ -x "$REGEN_REPO/scripts/run_scene.sh" ] || { echo "no run_scene.sh in $REGEN_REPO" >&2; exit 1; }

run_one() {
    local scene=$1 bagdir=$2
    local out="$OUT/$scene" regen="$OUT/$scene/regen" stage="$STAGE/$scene"
    local bag="$stage/$(basename "$bagdir")" log="$OUT/logs/$scene.log"
    if [ -e "$out/source_bag.txt" ] && [ -e "$out/regen/run_stats.txt" ]; then echo "$scene: complete, skipping"; return 0; fi
    mkdir -p "$out" "$regen" "$stage"
    {
    echo "[$(date '+%F %T')] $scene fetch $bagdir"
    rsync -a --partial "$SUBTDATA:$bagdir/" "$bag/" || { echo "$scene: fetch failed"; return 10; }

    echo "[$(date '+%F %T')] $scene odometry (ROS domain $((100 + 10#$scene)))"
    ROS_DOMAIN_ID=$((100 + 10#$scene)) ROS_LOCALHOST_ONLY=1 \
    singularity exec --pwd "$regen" $BINDS "$SIF" bash -c "
        source /opt/ros/jazzy/setup.bash && source '$FLIO_WS/install/setup.bash' &&
        '$REGEN_REPO/scripts/run_scene.sh' '$bag' '$regen' '$CONFIG' '$RATE' --restart"
    local rc=$?
    echo "[$(date '+%F %T')] $scene run_scene rc=$rc"

    echo "[$(date '+%F %T')] $scene references"
    singularity exec $BINDS "$EXTRACT_SIF" "$PY" "$REGEN_REPO/scripts/extract_odom_ref.py" "$bag" "$regen/odom_ref.txt" || echo "$scene: odom_ref failed"
    singularity exec $BINDS "$EXTRACT_SIF" "$PY" "$REGEN_REPO/scripts/extract_wheels.py" "$bag" "$regen/wheels.txt" || echo "$scene: wheels failed"

    if [ $rc -eq 0 ] && [ -s "$regen/odometry.txt" ] && [ -s "$regen/odom_ref.txt" ]; then
        echo "[$(date '+%F %T')] $scene filter"
        singularity exec $BINDS "$EXTRACT_SIF" "$PY" "$REPO/scripts/filter_segments.py" "$regen" "$regen/odom_ref.txt" "$out" || echo "$scene: filter failed"
    else
        echo "$scene: no usable odometry (rc=$rc); scene will have no odometry.txt"
    fi

    echo "[$(date '+%F %T')] $scene extract"
    singularity exec $BINDS "$EXTRACT_SIF" "$PY" "$REPO/dataset_tools/extract_data.py" "$stage" "$OUT" \
        --start_index "$((10#$scene))" --jobs 1 --overwrite \
        --cam /luxonis/oak/rgb/image_raw/compressed --lidar /livox/lidar_front --odom /odom --imu /ekf/imu/data \
        || echo "$scene: extraction failed"
    singularity exec $BINDS "$EXTRACT_SIF" "$PY" "$REPO/dataset_tools/validate_scene.py" "$out" --csv "$regen/validation.csv" \
        || echo "$scene: validation reported problems"

    mv "$regen/odometry.txt" "$regen/odometry_prefilter.txt" 2>/dev/null
    rm -rf "$stage"
    echo "[$(date '+%F %T')] $scene finished rc=$rc"
    echo "$(date '+%F %T') $scene rc=$rc" >> "$OUT/logs/joblog.txt"
    return $rc
    } >> "$log" 2>&1
}
export -f run_one
export OUT STAGE SIF EXTRACT_SIF FLIO_WS REGEN_REPO REPO CONFIG SUBTDATA RATE BINDS PY

# scene,bag_dir_on_subtdata,...,duration_s ; longest scenes first
tail -n +2 "$CSV" | awk -F, '{print $1","$2","$6}' | sort -t, -k3 -nr | cut -d, -f1,2 |
while IFS=, read -r scene bagdir; do
    if [ ${#ONLY[@]} -gt 0 ]; then printf '%s\n' "${ONLY[@]}" | grep -qx "$scene" || continue; fi
    echo "$scene $bagdir"
done | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"'
echo "all done; joblog at $OUT/logs/joblog.txt"
