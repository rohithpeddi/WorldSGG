#!/bin/bash
# INFRA (rebuttal): serve CS93371 directories to a remote GPU host as sshfs mounts in slave mode. No copy, no new SSH
# key: CS93371's local sftp-server talks to `sshfs -o slave` on the remote host through the existing ssh login.
# Run on CS93371 inside tmux:
#
#   tmux new -d -s reb_infra_mnt_cs94085 'bash /data3/rohith/ag/runs/rebuttal/infra/infra_mount.sh cs94085'
#
# Mounts under <RW>/mnt/ on the remote host (one ssh connection each; read-only ones also run sftp-server -R, so the
# server refuses every write):
#   data   <- /data/rohith/ag                           ro  data_path (features, annotations, frames_annotated, checkpoints)
#   grids  <- /code/rohith/ag/cache/pp_grids            ro  grid_cache_root (WorldWise++)
#   env    <- ~/anaconda3/envs/scene4cast               ro  python env (bin/python)
#   runs   <- /data3/rohith/ag/runs                     ro  existing runs (WorldWise+/++ checkpoints, saved dumps)
#   code   <- ~/CODE/Scene4Cast_rebuttal_train          rw  TRAIN worktree (trainers append results/<exp>_metrics.jsonl)
#   reb    <- /data3/rohith/ag/runs/rebuttal            rw  job outputs land directly on CS93371
# Symlinks are resolved on CS93371 (follow_symlinks), so the derived-stream links into /data3 work.
# A mount whose pipe dies is lazily unmounted and re-served after 10 s (jobs reading it at that moment fail).
# Stop: touch /data3/rohith/ag/runs/rebuttal/infra/mnt/STOP_<host>, then kill this script's sftp-server PIDs
#       (logs/mount_<host>.log lists them); the remote mounts then end and are lazily unmounted.
# Only some mounts:  MOUNTS="data grids" bash infra_mount.sh cs94085
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/infra_hosts.sh" "$1" || exit 2
MD=$INFRA/mnt; mkdir -p "$MD" "$INFRA/logs"
RM=$RW/mnt
RO_OPTS="ro,follow_symlinks,kernel_cache,dir_cache=yes,dcache_timeout=3600,attr_timeout=3600,entry_timeout=3600,negative_timeout=30"
RW_OPTS="follow_symlinks,dir_cache=yes,dcache_timeout=5,attr_timeout=1,entry_timeout=1"
declare -A SRCP=([data]=/data/rohith/ag [grids]=/code/rohith/ag/cache/pp_grids [env]=$HOME/anaconda3/envs/scene4cast
                 [runs]=/data3/rohith/ag/runs [code]=$HOME/CODE/Scene4Cast_rebuttal_train
                 [reb]=/data3/rohith/ag/runs/rebuttal)
declare -A MODE=([data]=ro [grids]=ro [env]=ro [runs]=ro [code]=rw [reb]=rw)
MOUNTS=${MOUNTS:-data grids env runs code reb}
log() { echo "[$(date '+%F %T')] [mnt $RHOST] $*"; }

serve() {  # <name>
  local n=$1 src=${SRCP[$1]} fifo=$MD/$RHOST.$1.fifo opts sflag rc
  if [ "${MODE[$n]}" = ro ]; then opts=$RO_OPTS; sflag=-R; else opts=$RW_OPTS; sflag=; fi
  while [ ! -f "$MD/STOP_$RHOST" ]; do
    $SSH -n "$RHOST" "mkdir -p $RM/$n; fusermount3 -u -z $RM/$n 2>/dev/null; true" 2>/dev/null
    rm -f "$fifo"; mkfifo "$fifo"
    log "serve $n: $src -> $RHOST:$RM/$n (${MODE[$n]})"
    /usr/lib/openssh/sftp-server $sflag -l ERROR < "$fifo" \
      | $SSH "$RHOST" "exec sshfs -o slave,$opts,fsname=cs93371:$src x:$src $RM/$n" > "$fifo" &
    log "serve $n: pipeline pid $! (sftp-server: $(pgrep -n -f "^/usr/lib/openssh/sftp-server $sflag" || echo ?))"
    wait $!; rc=$?
    log "serve $n: ended rc=$rc"
    [ -f "$MD/STOP_$RHOST" ] && break
    sleep 10
  done
  $SSH -n "$RHOST" "fusermount3 -u -z $RM/$n 2>/dev/null; true" 2>/dev/null
  rm -f "$fifo"
}

log "start mounts: $MOUNTS"
for n in $MOUNTS; do serve "$n" & sleep 2; done
wait
log "all mounts stopped"
