#!/bin/bash
# Replay one v1 bag through FAST-LIO and record a base_link trajectory.
#
# This is the recorder repo's scripts/run_scene.sh (branch odometry-regen-recorder)
# with the optional replay window added -- START_OFFSET / PLAY_DURATION, which is
# what makes a single hole replayable instead of a whole bag. Kept here because
# replay_holes.sh calls it inside the container and the campaign needs it to
# survive; the recorder repo remains its home.
#
# Relaunches fastlio_mapping when the recorder flags divergence, so a scene
# that used to be lost at the first failure comes back as a set of re-anchored
# segments. Run inside the helhest sif with the workspace sourced.
#
# CONTRACT (this is wrapped by the parallel driver on dgx07)
#
#   usage: run_scene.sh <bag_dir> <out_dir> [config] [rate] [--restart]
#     config     fast_lio config file name, default mid360_v2.yaml
#     rate       ros2 bag play --rate, default 0.5
#     --restart  segment and relaunch on divergence; omit to log flags only
#
#   inputs   <bag_dir> must be a readable rosbag2 directory (metadata.yaml)
#   outputs  written into <out_dir>:
#     odometry.txt            base_link poses, segments concatenated,
#                             each re-anchored to the identity
#     odometry_imu_raw.txt    raw IMU-frame poses, one line per scan
#     odometry_flags.txt      "timestamp test detail" per divergence flag
#     odometry_segments.txt   "segment start_ts end_ts n_poses reason"
#     run_stats.txt           key=value: rate, config, wall_s, peak_rss_mb,
#                             poses, segments, flags
#     run.log                 combined stdout/stderr of the ROS processes
#
#   exit codes
#     0  success, at least one pose recorded
#     1  usage error
#     2  bag directory missing or not a rosbag2
#     3  launch failure (fast_lio or the recorder died before any pose)
#     4  ran to completion but recorded no poses at all
#
#   stdout ends with a single machine-readable line:
#     RESULT scene=<out_dir> poses=<n> segments=<n> flags=<n> \
#            wall_s=<s> peak_rss_mb=<mb> rc=<code>

set -u

BAG="${1:-}"
OUT="${2:-}"
CONFIG="${3:-mid360_v2.yaml}"
RATE="${4:-0.5}"
RESTART="${5:-}"

if [ -z "$BAG" ] || [ -z "$OUT" ]; then
  echo "usage: run_scene.sh <bag_dir> <out_dir> [config] [rate] [--restart]" >&2
  exit 1
fi
if [ ! -d "$BAG" ] || [ ! -e "$BAG/metadata.yaml" ]; then
  echo "ERROR: '$BAG' is not a rosbag2 directory (no metadata.yaml)" >&2
  exit 2
fi

mkdir -p "$OUT"
LOG="$OUT/run.log"
REQ="$OUT/restart_request"
rm -f "$REQ"

SEGMENT_ON_FLAG=false
[ "$RESTART" = "--restart" ] && SEGMENT_ON_FLAG=true

PEAK_RSS_KB=0
FASTLIO_PID=""
RECORDER_PID=""
PLAY_PID=""

# Run the estimator binary directly. `ros2 launch` (and `ros2 run`) spawn it as
# a child, so killing the wrapper orphans the node: it keeps publishing
# /Odometry and the next run's recorder receives two estimators at once. That
# silently corrupted a whole set of runs before it was noticed.
FASTLIO_PREFIX="$(ros2 pkg prefix fast_lio 2>/dev/null)"
FASTLIO_BIN="$FASTLIO_PREFIX/lib/fast_lio/fastlio_mapping"
CONFIG_PATH="$FASTLIO_PREFIX/share/fast_lio/config/$CONFIG"
if [ ! -x "$FASTLIO_BIN" ] || [ ! -e "$CONFIG_PATH" ]; then
  echo "ERROR: fast_lio not built or config '$CONFIG' missing ($CONFIG_PATH)" >&2
  echo "RESULT scene=$OUT poses=0 segments=0 flags=0 wall_s=0 peak_rss_mb=0 rc=3"
  exit 3
fi

start_fastlio() {
  "$FASTLIO_BIN" --ros-args --params-file "$CONFIG_PATH" >>"$LOG" 2>&1 &
  FASTLIO_PID=$!
}

stop_fastlio() {
  [ -n "$FASTLIO_PID" ] || return 0
  kill "$FASTLIO_PID" 2>/dev/null || true
  # Give it a second to flush, then make sure it is really gone.
  for _ in 1 2 3 4 5; do
    kill -0 "$FASTLIO_PID" 2>/dev/null || return 0
    sleep 0.2
  done
  kill -9 "$FASTLIO_PID" 2>/dev/null || true
}

# Sample RSS across the whole process group. The driver needs a real number to
# size the 8-wide fan-out, and the estimator's map growth is what varies.
sample_rss() {
  # pgrep -g is unreliable inside the container, so read the process group
  # out of /proc/PID/stat (field 5) directly.
  # Our own pgrp, not $$: under `apptainer exec bash -c` the shell is not the
  # process group leader, so $$ never matches.
  local total=0 rss pid pgrp mygrp
  mygrp=$(sed 's/.*) //' "/proc/$$/stat" 2>/dev/null | awk '{print $3}')
  [ -n "$mygrp" ] || return 0
  for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do
    # /proc/PID/stat field 2 (comm) can contain spaces and parens, so index
    # from after the closing paren: state, ppid, pgrp.
    pgrp=$(sed 's/.*) //' "/proc/$pid/stat" 2>/dev/null | awk '{print $3}') || continue
    [ "$pgrp" = "$mygrp" ] || continue
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    total=$((total + ${rss:-0}))
  done
  [ "$total" -gt "$PEAK_RSS_KB" ] && PEAK_RSS_KB=$total
  return 0
}

cleanup() {
  kill "$RECORDER_PID" "$PLAY_PID" 2>/dev/null || true
  stop_fastlio
}
trap cleanup EXIT

START=$(date +%s)
echo "== fast_lio ($CONFIG), recorder (segmenting=$SEGMENT_ON_FLAG), bag at rate $RATE =="
start_fastlio

# Run the recorder straight from the source tree rather than through colcon.
# It is a pure rclpy node, and dataset_tools cannot currently be colcon-built:
# pyproject.toml's requires-python puts a SpecifierSet repr into the dist attrs
# that colcon_python_setup_py tries to ast.literal_eval, which raises.
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." &> /dev/null && pwd)"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m dataset_tools.odom_node --ros-args \
  -p file_path:="$OUT/odometry.txt" \
  -p reference_topic:=/odom \
  -p detect:=true \
  -p segment_on_flag:="$SEGMENT_ON_FLAG" \
  -p restart_request_path:="$REQ" >>"$LOG" 2>&1 &
RECORDER_PID=$!

# Give both a moment to come up; if either is already gone the run is doomed,
# and the recorder refuses to start when its outputs already exist.
sleep 3
if ! kill -0 "$RECORDER_PID" 2>/dev/null || ! kill -0 "$FASTLIO_PID" 2>/dev/null; then
  echo "ERROR: fast_lio or the recorder failed to start; see $LOG" >&2
  echo "RESULT scene=$OUT poses=0 segments=0 flags=0 wall_s=0 peak_rss_mb=0 rc=3"
  exit 3
fi

# Only the three topics the pipeline consumes; everything else in these bags is
# irrelevant here and some of it is expensive to deserialize.
# Optional window: START_OFFSET (s from the bag start) and PLAY_DURATION (s)
# replay only part of a bag, for re-running a stretch the full pass lost
# (FAST-LIO silent stalls, 2026-09-09). Both are plain seconds; unset = whole bag.
PLAY_WINDOW=()
[ -n "${START_OFFSET:-}" ] && PLAY_WINDOW+=(--start-offset "$START_OFFSET")
[ -n "${PLAY_DURATION:-}" ] && PLAY_WINDOW+=(--playback-duration "$PLAY_DURATION")
ros2 bag play "$BAG" --rate "$RATE" --clock "${PLAY_WINDOW[@]}" \
  --topics /livox/lidar_front /livox/imu_front /odom >>"$LOG" 2>&1 &
PLAY_PID=$!

LAST_WD=$(date +%s)
while kill -0 "$PLAY_PID" 2>/dev/null; do
  sample_rss
  if [ -e "$REQ" ]; then
    echo "== restart requested: $(cat "$REQ") =="
    rm -f "$REQ"
    stop_fastlio
    wait "$FASTLIO_PID" 2>/dev/null || true
    start_fastlio
    LAST_WD=$(date +%s)
  fi
  # Watchdog. A relaunched fastlio can come up, pass IMU init, and then never
  # receive a single scan (DDS re-discovery failing under load) - it stays
  # alive and silent, the detector sees no /Odometry, and the rest of the bag
  # is lost (12 of 43 scenes in the first v1.2 pass). The recorder flushes
  # odometry_imu_raw.txt per pose, so its mtime is the liveness signal.
  if [ "$SEGMENT_ON_FLAG" = "true" ]; then
    now=$(date +%s)
    if ! kill -0 "$FASTLIO_PID" 2>/dev/null; then
      echo "== watchdog: fastlio died, relaunching =="
      start_fastlio
      LAST_WD=$now
    elif [ $((now - LAST_WD)) -gt 20 ]; then
      hb=$(stat -c %Y "$OUT/odometry_imu_raw.txt" 2>/dev/null || echo "$now")
      if [ $((now - hb)) -gt 20 ]; then
        echo "== watchdog: no pose for $((now - hb)) s, restarting fastlio =="
        stop_fastlio
        wait "$FASTLIO_PID" 2>/dev/null || true
        start_fastlio
        LAST_WD=$now
      fi
    fi
  fi
  sleep 0.2
done

wait "$PLAY_PID" 2>/dev/null || true
# Let the last scans drain before tearing the recorder down.
sleep 2
sample_rss
kill "$RECORDER_PID" 2>/dev/null || true
wait "$RECORDER_PID" 2>/dev/null || true
stop_fastlio
wait "$FASTLIO_PID" 2>/dev/null || true

WALL=$(( $(date +%s) - START ))
PEAK_RSS_MB=$((PEAK_RSS_KB / 1024))
count() { if [ -e "$1" ]; then grep -vc '^#' "$1" || true; else echo 0; fi; }
POSES=$(count "$OUT/odometry.txt")
SEGMENTS=$(count "$OUT/odometry_segments.txt")
FLAGS=$(count "$OUT/odometry_flags.txt")

{
  echo "rate=$RATE"
  echo "config=$CONFIG"
  echo "segment_on_flag=$SEGMENT_ON_FLAG"
  echo "wall_s=$WALL"
  echo "peak_rss_mb=$PEAK_RSS_MB"
  echo "poses=$POSES"
  echo "segments=$SEGMENTS"
  echo "flags=$FLAGS"
} > "$OUT/run_stats.txt"

echo "== segments =="
cat "$OUT/odometry_segments.txt" 2>/dev/null || true

RC=0
[ "$POSES" -eq 0 ] && RC=4
echo "RESULT scene=$OUT poses=$POSES segments=$SEGMENTS flags=$FLAGS wall_s=$WALL peak_rss_mb=$PEAK_RSS_MB rc=$RC"
exit "$RC"
