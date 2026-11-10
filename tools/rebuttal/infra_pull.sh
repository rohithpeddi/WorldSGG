#!/bin/bash
# INFRA (rebuttal): copy the small host-local files of the remote GPU hosts (queue logs, status json, finished job
# scripts, $WSGG_LOCAL scratch) into CS93371 every 30 min. Job outputs themselves are written straight to CS93371
# through the rw mount (<RW>/mnt/reb = /data3/rohith/ag/runs/rebuttal), so nothing large is pulled.
# Run in tmux reb_infra_pull:
#
#   tmux new -d -s reb_infra_pull 'bash /data3/rohith/ag/runs/rebuttal/infra/infra_pull.sh cs94085'
#
# Per host H, into /data3/rohith/ag/runs/rebuttal/remote/H/:  queue/ <- H:$RQ/ (minus pending/),  local/ <- H:$RRUNS/
# Never deletes anything on either side. Files over 200 MB are skipped.
# Stop: touch /data3/rohith/ag/runs/rebuttal/infra/STOP_pull.   One pass now: bash infra_pull.sh --once cs94085
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ONCE=0; [ "${1:-}" = "--once" ] && { ONCE=1; shift; }
HOSTS="$*"; [ -n "$HOSTS" ] || { echo "usage: $0 [--once] <host>..." >&2; exit 2; }
BASE=/data3/rohith/ag/runs/rebuttal/remote
STOPF=/data3/rohith/ag/runs/rebuttal/infra/STOP_pull
PERIOD=${PERIOD:-1800}
while :; do
  for h in $HOSTS; do
    (
      . "$HERE/infra_hosts.sh" "$h" || exit 0
      D=$BASE/$h; mkdir -p "$D/queue" "$D/local"
      rsync -a --max-size=200m --exclude=/pending/ --exclude=.claim.lock -e "$SSH" "$RHOST:$RQ/" "$D/queue/" < /dev/null; r1=$?
      rsync -a --max-size=200m -e "$SSH" "$RHOST:$RRUNS/" "$D/local/" < /dev/null; r2=$?
      echo "[$(date '+%F %T')] $h: pulled queue rc=$r1 local rc=$r2"
    )
  done
  [ $ONCE -eq 1 ] && break
  for _ in $(seq $((PERIOD / 30))); do [ -f "$STOPF" ] && break 2; sleep 30; done
done
echo "[$(date '+%F %T')] pull loop exit"
