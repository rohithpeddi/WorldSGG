#!/bin/bash
# INFRA (rebuttal): gpuq_worker.sh (tools/rebuttal/gpuq/, same queue layout, CLI and job headers) plus three guards
# for the remote hosts, which read everything from CS93371 through sshfs mounts. Run inside tmux:
#
#   bash infra_gpuq_worker.sh <queue_root> <gpu> [slots=2] [headroom_gb=4]
#   tmux new -d -s gpuq_g0 'PRECHECK=/data1/rohith/wsgg/infra/infra_mounts_ok.sh bash infra_gpuq_worker.sh /data1/rohith/wsgg/queue 0 2 4'
#
# Same as gpuq_worker.sh: a job is <queue_root>/pending/<PP>_<name>.sh (write <name>.tmp, then mv); headers
#   # GPU_MEM_GB=<n>   memory the job needs (default 4); started only if the GPU has GPU_MEM_GB + headroom_gb free
#   # REQUIRES=<job_name>[,...]   start only after <queue_root>/done/<job_name>.sh exists
# Added:
#   # LOCK=<name>      at most one running job per lock name on this queue root (all GPUs). WorldWise++ jobs use
#                      LOCK=pp_grids: one of them saturates the 1 GbE link to CS93371 (see docs/rebuttal_runs/INFRA.md).
#   PRECHECK=<cmd>     env: claim nothing while <cmd> fails or hangs > 30 s (mounts down -> jobs stay pending).
#   MAX_UTIL=<pct>     env: claim only if the GPU's utilisation is <= pct in 3 samples 2 s apart (cs94092: GPUs are
#                      shared with other users' jobs; default 100 = off).
#   EXCLUSIVE=1        env: claim only if no other user's process holds memory on the GPU (processes of this user,
#                      e.g. our own running jobs, are allowed). A PID nvidia-smi lists but ps cannot see (a container)
#                      counts as foreign. Default 0 = off.
# Claims are serialised with flock on <queue_root>/.claim.lock (LOCK checks are race-free across GPUs).
# Stop: touch <queue_root>/STOP (workers exit after their running jobs end); touch <queue_root>/PAUSE_gpu<G>.
Q=$1; G=$2; SLOTS=${3:-2}; HEAD=${4:-4}
[ -n "$Q" ] && [ -n "$G" ] || { echo "usage: $0 <queue_root> <gpu> [slots] [headroom_gb]"; exit 2; }
MAX_UTIL=${MAX_UTIL:-100}; PRECHECK=${PRECHECK:-}; EXCLUSIVE=${EXCLUSIVE:-0}; ME=$(id -un)
mkdir -p "$Q"/pending "$Q"/running "$Q"/done "$Q"/failed "$Q"/logs
HOST=$(hostname -s)
log() { echo "[$(date '+%F %T')] [$HOST gpu$G] $*" >> "$Q/logs/worker_${HOST}_gpu$G.log"; }
log "worker start slots=$SLOTS headroom=${HEAD}GB max_util=$MAX_UTIL exclusive=$EXCLUSIVE precheck=${PRECHECK:-none} pid=$$"
deps_ok() {  # $1 = job file
  local req; req=$(grep -m1 '^# REQUIRES=' "$1" | cut -d= -f2 | tr -d ' ')
  [ -z "$req" ] && return 0
  local IFS=,; for r in $req; do [ -f "$Q/done/$r.sh" ] || ls "$Q"/done/*_"$r".sh >/dev/null 2>&1 || return 1; done
  return 0
}
lock_ok() {  # $1 = job file
  local lk; lk=$(grep -m1 '^# LOCK=' "$1" | cut -d= -f2 | tr -d ' ')
  [ -z "$lk" ] && return 0
  ! grep -qE "^# LOCK= *${lk} *$" "$Q"/running/*.sh 2>/dev/null
}
util_ok() {
  [ "$MAX_UTIL" -ge 100 ] && return 0
  local i u
  for i in 1 2 3; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$G" 2>/dev/null | head -1)
    [ "${u:-100}" -le "$MAX_UTIL" ] || return 1
    [ $i -lt 3 ] && sleep 2
  done
  return 0
}
excl_ok() {
  [ "$EXCLUSIVE" = 1 ] || return 0
  local uuid pid owner
  uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$G" 2>/dev/null | head -1)
  [ -n "$uuid" ] || return 1
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader 2>/dev/null                | awk -F', *' -v u="$uuid" '$1==u {print $2}'); do
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$owner" = "$ME" ] || return 1
  done
  return 0
}
pre_ok=1
while [ ! -f "$Q/STOP" ]; do
  if [ ! -f "$Q/PAUSE_gpu$G" ] && ls "$Q"/pending/*.sh >/dev/null 2>&1; then
    if [ -n "$PRECHECK" ] && ! timeout 30 bash -c "$PRECHECK" >/dev/null 2>&1; then
      [ $pre_ok -eq 1 ] && log "precheck failed ($PRECHECK): not claiming"; pre_ok=0
    else
      [ $pre_ok -eq 0 ] && log "precheck ok again"; pre_ok=1
      nrun=$(ls "$Q"/running/gpu${G}__*.sh 2>/dev/null | wc -l)
      if [ "$nrun" -lt "$SLOTS" ] && excl_ok && util_ok; then
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" 2>/dev/null | head -1)
        free=$(( ${free:-0} / 1024 ))
        exec 9>"$Q/.claim.lock"; flock 9
        claimed=
        for job in $(ls "$Q"/pending/*.sh 2>/dev/null | sort); do
          need=$(grep -m1 '^# GPU_MEM_GB=' "$job" | cut -d= -f2 | tr -d ' '); need=${need:-4}
          [ "$free" -ge $(( need + HEAD )) ] || continue
          deps_ok "$job" || continue
          lock_ok "$job" || continue
          f=$(basename "$job"); name=${f%.sh}; name=${name#[0-9][0-9]_}
          run="$Q/running/gpu${G}__$f"
          mv "$job" "$run" 2>/dev/null || continue
          claimed=1; break
        done
        flock -u 9; exec 9>&-
        if [ -n "$claimed" ]; then
          log "start $name (need ${need}GB, free ${free}GB)"
          (
            start=$(date -Is)
            CUDA_VISIBLE_DEVICES=$G REB_GPU=$G REB_HOST=$HOST bash "$run" > "$Q/logs/$name.log" 2>&1
            rc=$?
            printf '{"job":"%s","host":"%s","gpu":%s,"rc":%s,"start":"%s","end":"%s"}\n' \
              "$name" "$HOST" "$G" "$rc" "$start" "$(date -Is)" > "$Q/logs/$name.status.json"
            if [ "$rc" -eq 0 ]; then mv "$run" "$Q/done/$f"; else mv "$run" "$Q/failed/$f"; fi
            log "end $name rc=$rc"
          ) &
          sleep 45   # let the job allocate before the next free-memory reading
        fi
      fi
    fi
  fi
  sleep 20
done
log "STOP seen; waiting for running jobs"
wait
log "worker exit"
