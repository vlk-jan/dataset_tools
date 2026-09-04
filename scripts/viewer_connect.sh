#!/bin/bash
# Open an SSH tunnel from this machine to a viewer running under
# slurm/viewer.batch on an RCI compute node, then leave it up.
#
#   scripts/viewer_connect.sh [local_port] [job_id]
#
# With no job id it takes your newest running helhest_viewer job. The hop is
# `ssh -J`, because node sshd runs pam_slurm_adopt: the login node lets you
# through to a node only while you hold a job there.
set -uo pipefail

LOGIN=${VIEWER_LOGIN:-login3.rci.cvut.cz}
REPO=${VIEWER_REPO:-/home/vlkjan6/projects/dataset_tools}
LOCAL_PORT=${1:-8080}
JOB=${2:-}

if [ -z "$JOB" ]; then
    JOB=$(ssh -o BatchMode=yes "$LOGIN" \
        "squeue -h -u \$USER -n helhest_viewer -t RUNNING -O JobID --sort=-V | head -1") || exit 1
    JOB=${JOB// /}
fi
if [ -z "$JOB" ]; then
    pend=$(ssh -o BatchMode=yes "$LOGIN" "squeue -h -u \$USER -n helhest_viewer -o '%i %T %r'" 2>/dev/null)
    if [ -n "$pend" ]; then
        echo "no viewer running yet; queued:"; echo "$pend"
    else
        echo "no viewer job. Start one with:"
        echo "  ssh $LOGIN 'cd $REPO && sbatch slurm/viewer.batch'"
    fi
    exit 1
fi

conn=$(ssh -o BatchMode=yes "$LOGIN" "cat $REPO/slurm_out/viewer_$JOB.conn" 2>/dev/null)
[ -n "$conn" ] || { echo "job $JOB has not published a node/port yet -- still starting?"; exit 1; }
node=${conn%% *}
port=${conn##* }

echo "job $JOB on $node:$port -> http://127.0.0.1:$LOCAL_PORT/"
echo "ctrl-c closes the tunnel; the viewer keeps running until the job ends."
exec ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -J "$LOGIN" -L "$LOCAL_PORT:127.0.0.1:$port" "$node"
