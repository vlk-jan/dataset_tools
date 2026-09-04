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
NODE_HOSTS=${VIEWER_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_rci_nodes}
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
        # mkdir: slurm/slurm_out is untracked, and sbatch fails outright if the
        # directory its --output names is missing.
        echo "  ssh $LOGIN 'cd $REPO && mkdir -p slurm/slurm_out && sbatch slurm/viewer.batch'"
    fi
    exit 1
fi

# ssh options for the second hop. Compute node keys live in their own file so
# that learning one never writes to (or weakens the checking of) the login
# hosts in ~/.ssh/known_hosts; both files are searched, new keys go to the
# first. A reimaged node changes key and ssh then refuses to connect:
#   ssh-keygen -R <node> -f ~/.ssh/known_hosts_rci_nodes
NODE_SSH=(-o BatchMode=yes
          -o UserKnownHostsFile="$NODE_HOSTS $HOME/.ssh/known_hosts"
          -o StrictHostKeyChecking=accept-new)

conn=$(ssh -o BatchMode=yes "$LOGIN" "cat $REPO/slurm/slurm_out/viewer_$JOB.conn" 2>/dev/null)
if [ -z "$conn" ]; then
    # slurm/slurm_out is untracked, so a `git clean` or a purge can take the .conn
    # file out from under a job that is still happily serving. Ask slurm which
    # node, and the node which port the viewer is listening on.
    node=$(ssh -o BatchMode=yes "$LOGIN" \
        "squeue -h -j $JOB -t RUNNING -O NodeList" 2>/dev/null)
    node=${node// /}
    [ -n "$node" ] || { echo "job $JOB has not published a node/port yet -- still starting?"; exit 1; }
    # One viewer per job, but a node can host several jobs; the first listener
    # is the right guess only when it is yours, hence the port in the banner.
    port=$(ssh "${NODE_SSH[@]}" -J "$LOGIN" "$node" \
        'pid=$(pgrep -u "$USER" -f helhest-viewer | head -1)
         [ -n "$pid" ] && ss -ltnpH 2>/dev/null | grep -o "127.0.0.1:[0-9]*.*pid=$pid," |
             head -1 | cut -d: -f2 | cut -d" " -f1' 2>/dev/null)
    [ -n "$port" ] || { echo "job $JOB runs on $node but publishes no port -- viewer not up?"; exit 1; }
    echo "no slurm/slurm_out/viewer_$JOB.conn; found the viewer on $node:$port instead."
else
    node=${conn%% *}
    port=${conn##* }
fi

echo "job $JOB on $node:$port -> http://127.0.0.1:$LOCAL_PORT/"
echo "ctrl-c closes the tunnel; the viewer keeps running until the job ends."
exec ssh -N "${NODE_SSH[@]}" -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -J "$LOGIN" -L "$LOCAL_PORT:127.0.0.1:$port" "$node"
