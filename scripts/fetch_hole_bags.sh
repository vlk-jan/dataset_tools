#!/bin/bash
# Fetch every bag named in a hole campaign CSV, one at a time, marking <bag>.fetched.
#
#   fetch_hole_bags.sh <campaign.csv> [stage_dir]
#
# CSV columns: scene,bag_path_on_remote,first_frame_ts,hole_t0_rel,hole_dur
# (see docs/gt_hole_campaign.csv). One bag is on disk at a time by design -- the
# v1.3 campaign moved 571 GB through a 14 GB working set. Run this alongside
# replay_holes.sh, which waits for the .fetched marks and deletes each bag when
# its holes are done.
set -u
CSV=${1:?usage: fetch_hole_bags.sh <campaign.csv> [stage_dir]}
ST=${2:-/mnt/personal/vlkjan6/regen_stage}
REMOTE=${REMOTE:-vlkjan6@subtdata.felk.cvut.cz}
mkdir -p "$ST"

cut -d, -f2 "$CSV" | awk '!seen[$0]++' | while read -r bag; do
  name=$(basename "$bag")
  [ -e "$ST/$name.fetched" ] && { echo "[$(date '+%T')] $name already fetched"; continue; }
  while pgrep -f "rsync.*$name" >/dev/null; do sleep 15; done   # a fetch started by hand
  echo "[$(date '+%T')] fetch $name"
  rsync -a --info=progress2 "$REMOTE:$bag" "$ST/" > "$ST/fetch_$name.log" 2>&1 \
    && touch "$ST/$name.fetched" && echo "[$(date '+%T')] fetched $name $(du -sh "$ST/$name" | cut -f1)" \
    || echo "[$(date '+%T')] FETCH FAILED $name"
done
echo "[$(date '+%T')] fetch loop done"
