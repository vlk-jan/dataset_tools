#!/bin/bash
# Open an SSH tunnel from this machine to a viewer running under
# slurm/viewer.batch on an RCI compute node, then leave it up.
#
#   scripts/viewer_connect.sh [local_port] [job_id]
#
# With no job id it takes your newest running helhest_viewer job. Nodes are
# reached via `ssh -J` through the login node (pam_slurm_adopt).
set -uo pipefail

# RCI account name; may differ from the local one.
RCI_USER=${VIEWER_USER:-$USER}
LOGIN=$RCI_USER@${VIEWER_LOGIN:-login3.rci.cvut.cz}
REPO=${VIEWER_REPO:-/home/$RCI_USER/projects/dataset_tools}
NODE_HOSTS=${VIEWER_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_rci_nodes}
# A 3D window is ~26 MB raw, ~10 MB compressed: a win below ~100 Mbit/s,
# a loss on wired gigabit (zlib runs at ~37 MB/s). 0 turns it off.
[ "${VIEWER_COMPRESS:-1}" = 0 ] && COMPRESSION=no || COMPRESSION=yes
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
        # sbatch fails if its --output directory is missing.
        echo "  ssh $LOGIN 'cd $REPO && mkdir -p slurm/slurm_out && sbatch slurm/viewer.batch'"
    fi
    exit 1
fi

# Node keys go to their own known_hosts file. A reimaged node needs:
#   ssh-keygen -R <node> -f ~/.ssh/known_hosts_rci_nodes
NODE_SSH=(-o BatchMode=yes
          -o UserKnownHostsFile="$NODE_HOSTS $HOME/.ssh/known_hosts"
          -o StrictHostKeyChecking=accept-new)

# The viewer listens on a Unix socket (see slurm/viewer.batch), at a path the
# user and job id fix; the .conn file says so too, plus the node.
conn=$(ssh -o BatchMode=yes "$LOGIN" "cat $REPO/slurm/slurm_out/viewer_$JOB.conn" 2>/dev/null)
if [ -z "$conn" ]; then
    # .conn file gone (untracked dir): ask slurm for the node.
    node=$(ssh -o BatchMode=yes "$LOGIN" \
        "squeue -h -j $JOB -t RUNNING -O NodeList" 2>/dev/null)
    node=${node// /}
    [ -n "$node" ] || { echo "job $JOB has not published a node yet -- still starting?"; exit 1; }
    sock=/tmp/helhest_viewer_${RCI_USER}_$JOB/viewer.sock
    echo "no slurm/slurm_out/viewer_$JOB.conn; trying $node:$sock."
else
    node=${conn%% *}
    sock=${conn##* }
fi

echo "job $JOB on $node:$sock -> http://127.0.0.1:$LOCAL_PORT/"
echo "ctrl-c closes the tunnel; the viewer keeps running until the job ends."
exec ssh -N "${NODE_SSH[@]}" -o ExitOnForwardFailure=yes -o Compression=$COMPRESSION \
    -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
    -J "$LOGIN" -L "127.0.0.1:$LOCAL_PORT:$sock" "$RCI_USER@$node"
