#!/bin/bash
# Replay each hole through FAST-LIO once its bag has landed; delete the bag when its holes are done.
#
#   replay_holes.sh <campaign.csv> [stage_dir] [out_dir]
#
# CSV columns: scene,bag_path,first_frame_ts,hole_t0_rel,hole_dur (docs/gt_hole_campaign.csv).
# Each hole is replayed with 15 s of lead-in so FAST-LIO has initialised before
# the hole starts, and 20 s of run-out; the output lands in
# <out_dir>/<scene>_hole_t<t0>/ and is scored with score_hole_replay.py before
# any of it reaches a root.
#
# Needs run_scene.sh INSIDE the container to honour START_OFFSET / PLAY_DURATION
# -- that is scripts/run_scene_offset.sh here. A container carrying the older
# run_scene.sh ignores both and replays the whole bag.
set -u
CSV=${1:?usage: replay_holes.sh <campaign.csv> [stage_dir] [out_dir]}
ST=${2:-/mnt/personal/vlkjan6/regen_stage}
PO=${3:-/mnt/personal/vlkjan6/regen_pilot}
SIF=${SIF:-/mnt/personal/vlkjan6/sif/helhest-jazzy-2026-02-18.sif}
RUN_SCENE=${RUN_SCENE:-~/flio_ws/src/dataset_tools_regen/scripts/run_scene.sh}
CONFIG=${CONFIG:-mid360_v2.yaml}
RATE=${RATE:-0.5}
LEAD_IN=${LEAD_IN:-15.0}
RUN_OUT=${RUN_OUT:-20.0}
DOM=${DOM:-80}
mkdir -p "$PO"

prev_name=""
while IFS=, read -r scene bag T0 t0 dur; do
  name=$(basename "$bag")
  if [ -n "$prev_name" ] && [ "$prev_name" != "$name" ]; then
    while pgrep -f "bag play $ST/$prev_name" >/dev/null; do sleep 15; done
    echo "[$(date '+%T')] holes of $prev_name done, removing bag"; rm -rf "${ST:?}/$prev_name"
  fi
  prev_name=$name
  out=$PO/${scene}_hole_t$(LC_ALL=C printf '%.0f' "$t0")
  [ -e "$out/run_stats.txt" ] && { echo "[$(date '+%T')] $out exists, skip"; continue; }
  while [ ! -e "$ST/$name.fetched" ]; do sleep 30; done
  # the CSV holds the scene's first FRAME timestamp; the bag starts earlier
  bstart=$(grep -m1 nanoseconds_since_epoch "$ST/$name/metadata.yaml" | awk '{printf "%.6f", $2/1e9}')
  off=$(python3 -c "print(max(0.0, round($t0 + ($T0 - $bstart) - $LEAD_IN, 1)))")
  pdur=$(python3 -c "print(round($dur + $LEAD_IN + $RUN_OUT, 1))")
  DOM=$((DOM + 1)); rm -rf "$out"; mkdir -p "$out"
  echo "[$(date '+%T')] replay $scene t+$t0 (+$dur s): offset $off, duration $pdur, domain $DOM"
  singularity exec --bind /mnt/personal --bind /tmp "$SIF" bash -c "
    source /opt/ros/jazzy/setup.bash; source ~/flio_ws/install/setup.bash
    export ROS_DOMAIN_ID=$DOM ROS_LOCALHOST_ONLY=1 START_OFFSET=$off PLAY_DURATION=$pdur
    $RUN_SCENE $ST/$name $out $CONFIG $RATE --restart" 2>&1 \
    | grep -E "^RESULT|ERROR" | sed "s/^/[$(date '+%T')] $scene t+$t0: /"
done < "$CSV"
[ -n "$prev_name" ] && {
  while pgrep -f "bag play $ST/$prev_name" >/dev/null; do sleep 15; done
  echo "[$(date '+%T')] holes of $prev_name done, removing bag"; rm -rf "${ST:?}/$prev_name"
}
echo "[$(date '+%T')] replay loop done"
