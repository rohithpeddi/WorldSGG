#!/bin/bash
# File-based GPU job queue: one worker per GPU, run inside tmux.
#
#   bash gpuq_worker.sh <queue_root> <gpu> [slots=2] [headroom_gb=4]
#   tmux new -d -s gpuq_g0 'bash gpuq_worker.sh /data3/rohith/ag/runs/rebuttal/queue 0 2'
#
# A job is an executable bash script in <queue_root>/pending/ named <PP>_<name>.sh (PP = 2-digit priority, lower
# runs first). Write it as <name>.tmp and mv it to .sh so a worker never claims a half-written file. Optional header
# lines:   # GPU_MEM_GB=<n>   memory the job needs (default 4)
#          # REQUIRES=<job_name>[,<job_name>...]   start only after <queue_root>/done/<job_name>.sh exists
# The worker claims a job with an atomic mv to running/gpu<G>__<file>, runs it with CUDA_VISIBLE_DEVICES=<gpu>, writes
# logs/<name>.log and logs/<name>.status.json, then moves the script to done/ (rc 0) or failed/ (rc != 0).
# It starts a job only if the GPU has GPU_MEM_GB + headroom_gb free and fewer than <slots> of its jobs are running.
# Stop: touch <queue_root>/STOP (workers exit after their running jobs end); touch <queue_root>/PAUSE_gpu<G> pauses
# claiming on one GPU.
Q=$1; G=$2; SLOTS=${3:-2}; HEAD=${4:-4}
[ -n "$Q" ] && [ -n "$G" ] || { echo "usage: $0 <queue_root> <gpu> [slots] [headroom_gb]"; exit 2; }
mkdir -p "$Q"/pending "$Q"/running "$Q"/done "$Q"/failed "$Q"/logs
HOST=$(hostname -s)
log() { echo "[$(date '+%F %T')] [$HOST gpu$G] $*" >> "$Q/logs/worker_${HOST}_gpu$G.log"; }
log "worker start slots=$SLOTS headroom=${HEAD}GB pid=$$"
deps_ok() {  # $1 = job file
  local req; req=$(grep -m1 '^# REQUIRES=' "$1" | cut -d= -f2 | tr -d ' ')
  [ -z "$req" ] && return 0
  local IFS=,; for r in $req; do [ -f "$Q/done/$r.sh" ] || ls "$Q"/done/*_"$r".sh >/dev/null 2>&1 || return 1; done
  return 0
}
while [ ! -f "$Q/STOP" ]; do
  if [ ! -f "$Q/PAUSE_gpu$G" ]; then
    nrun=$(ls "$Q"/running/gpu${G}__*.sh 2>/dev/null | wc -l)
    if [ "$nrun" -lt "$SLOTS" ]; then
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$G" 2>/dev/null | head -1)
      free=$(( ${free:-0} / 1024 ))
      for job in $(ls "$Q"/pending/*.sh 2>/dev/null | sort); do
        need=$(grep -m1 '^# GPU_MEM_GB=' "$job" | cut -d= -f2 | tr -d ' '); need=${need:-4}
        [ "$free" -ge $(( need + HEAD )) ] || continue
        deps_ok "$job" || continue
        f=$(basename "$job"); name=${f%.sh}; name=${name#[0-9][0-9]_}
        run="$Q/running/gpu${G}__$f"
        mv "$job" "$run" 2>/dev/null || continue   # another worker won the claim
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
        break
      done
    fi
  fi
  sleep 20
done
log "STOP seen; waiting for running jobs"
wait
log "worker exit"
