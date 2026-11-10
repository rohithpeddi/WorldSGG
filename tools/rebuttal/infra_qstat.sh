#!/bin/bash
# INFRA (rebuttal): queue + GPU status of a remote host, from CS93371.
#   bash /data3/rohith/ag/runs/rebuttal/infra/infra_qstat.sh <host> [<job name> [<n log lines>]]
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/infra_hosts.sh" "$1" || exit 2
NAME=${2:-}; N=${3:-20}
$SSH -n "$RHOST" "for s in pending running done failed; do printf '%-8s %s\n' \$s \"\$(ls $RQ/\$s 2>/dev/null | tr '\n' ' ')\"; done; \
  echo; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader; \
  echo; tmux ls 2>/dev/null | grep gpuq_; df -h $RROOT | tail -1; \
  if [ -n '$NAME' ]; then echo; cat $RQ/logs/$NAME.status.json 2>/dev/null; tail -n $N $RQ/logs/$NAME.log; fi" 2>/dev/null
