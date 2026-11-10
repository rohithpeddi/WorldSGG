#!/bin/bash
# INFRA (rebuttal): watchdog for the read-write `reb` sshfs mount on cs94085 (/data1/rohith/wsgg/mnt/reb).
# On Nov 7 2026 at 11:25 that mount deadlocked: CS93371's sftp-server and ssh for it went idle (no CPU), the remote
# sshfs never exited, and every cs94085 process touching the mount blocked in D state until it was restarted by hand.
# Every cycle this starts a background read of a small file through the mount on cs94085 and checks it 120 s later.
# After two misses in a row (the read never finished), it restarts the mount: kills CS93371's sftp-server/ssh pair
# for reb and the remote sshfs, by PID; infra_mount.sh then re-serves it about 10 s later. Jobs blocked on the mount
# fail at that point: requeue them (trainings can resume from their last checkpoint).
# An ssh failure (network, timeout) is not counted as a miss.
#
#   tmux new -d -s reb_infra_wd_cs94085 'bash /data3/rohith/ag/runs/rebuttal/infra/infra_reb_watchdog.sh'
# Stop: touch /data3/rohith/ag/runs/rebuttal/infra/mnt/STOP_wd_cs94085
set -u
INFRA=/data3/rohith/ag/runs/rebuttal/infra
LOG=$INFRA/logs/reb_watchdog_cs94085.log
STOP=$INFRA/mnt/STOP_wd_cs94085
SRC=/data3/rohith/ag/runs/rebuttal
PROBE=/data1/rohith/wsgg/mnt/reb/splits/val750.txt
RT=/data1/rohith/wsgg/tmp/rebwd
rssh() { timeout 40 ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 \
           -o HostName=10.176.140.100 -o HostKeyAlias=cs94085.utdallas.edu cs94085 "$@" 2>/dev/null; }
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

restart() {
  local spid sub sftp rpid
  spid=$(pgrep -u "$(id -un)" -f "^ssh .* cs94085 exec sshfs .*fsname=cs93371:$SRC x:$SRC " | head -1)
  sub=${spid:+$(ps -o ppid= -p "$spid" | tr -d ' ')}
  sftp=${sub:+$(pgrep -P "$sub" -x sftp-server | head -1)}
  log "RESTART reb mount: CS93371 ssh=${spid:-none} sftp-server=${sftp:-none}"
  [ -n "${sftp:-}" ] && kill "$sftp"
  [ -n "${spid:-}" ] && kill "$spid"
  sleep 2
  rpid=$(rssh "pgrep -u \$(id -un) -f '[s]shfs -o slave.*fsname=cs93371:$SRC x:$SRC '")   # [s]: skip this shell
  log "remote sshfs pid(s): ${rpid:-none}"
  for p in $rpid; do rssh "kill -9 $p"; done
  sleep 60
  tail -2 "$INFRA/logs/mount_cs94085.log" | sed 's/^/    mount log: /' | tee -a "$LOG"
}

log "start (probe $PROBE every ~2.5 min)"
miss=0
while [ ! -f "$STOP" ]; do
  id=$(date +%s)
  rssh "mkdir -p $RT; find $RT -maxdepth 1 -name 'ok.*' -mmin +30 -delete; ( cat $PROBE >/dev/null 2>&1 && touch $RT/ok.$id ) </dev/null >/dev/null 2>&1 &"
  sleep 120
  rssh "test -f $RT/ok.$id && rm -f $RT/ok.$id"; rc=$?
  if [ "$rc" -eq 0 ]; then
    [ "$miss" -gt 0 ] && log "probe ok again after $miss miss(es)"
    miss=0
  elif [ "$rc" -eq 1 ]; then
    miss=$((miss + 1)); log "probe $id did not finish within 120 s (miss $miss)"
    if [ "$miss" -ge 2 ]; then restart; miss=0; fi
  else
    log "ssh to cs94085 failed (rc=$rc); not counted"
  fi
  sleep 30
done
log "stopped"
