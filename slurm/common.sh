# Sourced by the slurm jobs, never run: the preamble they all share. A job finds
# the checkout first (slurm runs a copy of the job from its spool, so not
# through $0) and sources this from it:
#
#   REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
#   . "$REPO/slurm/common.sh" || { echo "not a dataset_tools checkout: $REPO -- submit from the repository root"; exit 1; }

SIF=${SIF:-/mnt/data/vras/data/helhest-jazzy.sif}
SUBTDATA=${SUBTDATA:-$USER@subtdata.felk.cvut.cz}
PY="$REPO/.venv/bin/python"

# Row $SLURM_ARRAY_TASK_ID of $CSV (0-based, header skipped) into $row, its
# third column -- the bag directory on subtdata -- into $bagdir, and its
# extract_args column (if the header has one) into $row_args.
read_row() {
    row=$(tail -n +2 "$CSV" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
    [ -n "$row" ] || { echo "no row $SLURM_ARRAY_TASK_ID in $CSV"; exit 1; }
    bagdir=$(echo "$row" | cut -d, -f3)
    row_args=""
    local col
    col=$(head -1 "$CSV" | tr ',' '\n' | grep -nx 'extract_args' | cut -d: -f1)
    [ -n "$col" ] && row_args=$(echo "$row" | cut -d, -f"$col")
}

# Node-local scratch for this job into $SCRATCH, removed on exit: /data (NVMe)
# preferred, /tmp fallback. $1 prefixes the directory name. Sets $BINDS to what
# the container needs to see.
make_scratch() {
    SCRATCH=""
    for c in /data/temporary/$USER /data/$USER /tmp/$USER; do
        mkdir -p "$c/$1_${SLURM_JOB_ID}" 2>/dev/null && SCRATCH="$c/$1_${SLURM_JOB_ID}" && break
    done
    [ -n "$SCRATCH" ] || { echo "no writable scratch"; exit 1; }
    trap 'rm -rf "$SCRATCH"' EXIT
    BINDS="--bind /mnt/personal --bind /mnt/data --bind $SCRATCH"
}

# rsync $bagdir from subtdata into $2; $1 names it in the log. Exits 10 on failure.
fetch_bag() {
    echo "[$(date '+%F %T')] $1 fetch $bagdir -> $2"
    rsync -a --partial "$SUBTDATA:$bagdir/" "$2/" || { echo "$1: fetch failed"; exit 10; }
    echo "[$(date '+%F %T')] $1 fetched $(du -sh "$2" | cut -f1)"
}
