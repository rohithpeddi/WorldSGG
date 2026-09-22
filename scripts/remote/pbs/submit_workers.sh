#!/bin/sh
# Submit N MLLM queue workers to PBS (pragya / IITD).
#
# Usage:  sh scripts/remote/pbs/submit_workers.sh [N] [jobs-file] [walltime]
# Default: 4 workers, $HOME/DATA/ag/logs/mllm.jobs, 24 h each.
#
# The `standard` queue allows 10 running and 10 queued jobs per user, so N > 10
# is refused here rather than by the server.  Workers are independent: each one
# claims whatever is unlocked, so submitting more simply widens the fan-out, and
# a worker that finds nothing runnable exits and releases its allocation.
set -u
N="${1:-4}"
JOBS="${2:-$HOME/DATA/ag/logs/mllm.jobs}"
WALL="${3:-24:00:00}"
ROOT="${MLLM_ROOT:-$HOME/CODE/WorldSGG}"
LOGDIR="${MLLM_LOGDIR:-$HOME/DATA/ag/logs}"
PBSBIN=/opt/pbs/2026.0.0/bin

[ -f "$JOBS" ] || { echo "no jobs file: $JOBS" >&2; exit 1; }
[ "$N" -le 10 ] || { echo "max_run for this user is 10; asked for $N" >&2; exit 1; }
mkdir -p "$LOGDIR"

i=1
while [ "$i" -le "$N" ]; do
  "$PBSBIN/qsub" -N "mllm_w$i" -l "walltime=$WALL" \
    -o "$LOGDIR/pbs_w$i.log" -j oe \
    -v "JOBS=$JOBS,MLLM_ROOT=$ROOT,MLLM_LOGDIR=$LOGDIR" \
    "$ROOT/scripts/remote/pbs/mllm_worker.pbs"
  i=$((i + 1))
done

echo
"$PBSBIN/qstat" -u "$USER"
