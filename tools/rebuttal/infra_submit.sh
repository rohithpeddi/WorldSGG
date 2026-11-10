#!/bin/bash
# INFRA (rebuttal): submit a job script from CS93371 to a remote host's GPU queue (gpuq_worker.sh). Run on CS93371:
#
#   bash /data3/rohith/ag/runs/rebuttal/infra/infra_submit.sh <host> <job.sh> <PP> [<name>]
#
#   host  cs94085 | cs94092;  PP = 2-digit priority (BRIEF §4: INFER 10-29, TRAIN 30-59; lower runs first)
#   name  defaults to the basename of <job.sh> without .sh; must start with trn_ / inf_ / infra_ and be unique on
#         that host (pending, running, done, failed). The script lands atomically as <queue>/pending/<PP>_<name>.sh.
# The job runs on the remote host: use its paths (source <RW>/host.env, see docs/rebuttal_runs/INFRA.md).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/infra_hosts.sh" "$1" || exit 2
JOB=${2:?job script}; PP=${3:?priority}; NAME=${4:-$(basename "$JOB" .sh)}
NAME=${NAME#[0-9][0-9]_}
[ -f "$JOB" ] || { echo "no such file: $JOB" >&2; exit 2; }
[[ "$PP" =~ ^[0-9][0-9]$ ]] || { echo "PP must be two digits" >&2; exit 2; }
[[ "$NAME" =~ ^(trn|inf|infra)_[A-Za-z0-9_.-]+$ ]] || { echo "name must start with trn_/inf_/infra_: $NAME" >&2; exit 2; }
grep -q '^# GPU_MEM_GB=' "$JOB" || echo "warning: no '# GPU_MEM_GB=' header (worker assumes 4 GB)" >&2
clash=$($SSH -n "$RHOST" "ls $RQ/pending $RQ/running $RQ/done $RQ/failed 2>/dev/null | grep -E '(^|_)${NAME}\.sh\$'" 2>/dev/null)
[ -z "$clash" ] || { echo "name already used on $RHOST: $clash" >&2; exit 3; }
$SSH "$RHOST" "cat > $RQ/pending/$NAME.tmp && chmod +x $RQ/pending/$NAME.tmp && mv $RQ/pending/$NAME.tmp $RQ/pending/${PP}_$NAME.sh" \
  < "$JOB" || { echo "submit failed" >&2; exit 1; }
echo "submitted $RHOST:$RQ/pending/${PP}_$NAME.sh  (log: $RQ/logs/$NAME.log)"
