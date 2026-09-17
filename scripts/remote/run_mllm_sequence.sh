#!/bin/sh
# MLLM job queue worker (B3 baselines, Stage-1 graph fill, Track A/B, Thinking subset).
# Usage:  MLLM_GPU=<gpu> sh scripts/remote/run_mllm_sequence.sh <worker_name> <jobs file>
# Jobs file: one job per line  "<job_name> <python module> <args...>"; '#' comments.
# The file is RE-READ before every job, so jobs can be added / reordered live.
# Several workers (one per GPU, different MLLM_GPU) may serve the same file: a job
# is claimed with an atomic  mkdir /data3/rohith/ag/logs/<job>.lock ; a claimed
# job whose status pid is dead (worker killed) is unlocked and re-run (all runners
# are resumable per video); "done" jobs are skipped; a worker exits when no job is
# runnable and none is running.
# Per job: CUDA_VISIBLE_DEVICES=$MLLM_GPU (default 0), log /data3/rohith/ag/logs/<job>.log,
# status /data3/rohith/ag/logs/<job>.status.json ({state: running|done|failed, pid, gpu, ...}).
# Launch detached:
#   setsid nohup sh scripts/remote/run_mllm_sequence.sh b3 /data3/rohith/ag/logs/b3.jobs \
#       > /data3/rohith/ag/logs/b3.queue.log 2>&1 < /dev/null &
set -u
WORKER="$1"; JOBS="$2"
ROOT="$HOME/CODE/Scene4Cast_mllm"
LOGDIR=/data3/rohith/ag/logs
PY="$HOME/anaconda3/envs/wsg/bin/python"
GPU="${MLLM_GPU:-0}"
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
echo "[$WORKER] start $(date -Is) gpu=$GPU commit=$(git rev-parse --short HEAD)"

state_of() { sed -n 's/.*"state": *"\([a-z]*\)".*/\1/p' "$1" 2>/dev/null; }
pid_of()   { sed -n 's/.*"pid": *\([0-9]*\).*/\1/p' "$1" 2>/dev/null; }

while :; do
  NEXT=""; BUSY=0
  while read -r JOB MOD ARGS; do
    case "$JOB" in ""|\#*) continue;; esac
    STATUS="$LOGDIR/$JOB.status.json"; LOCK="$LOGDIR/$JOB.lock"
    ST=$(state_of "$STATUS")
    [ "$ST" = "done" ] && continue
    if [ -d "$LOCK" ]; then
      P=$(pid_of "$STATUS")
      if [ "$ST" = "running" ] && [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then BUSY=1; continue; fi
      # claimed but its worker died (or it never started): release the stale lock
      echo "[$WORKER] stale lock for $JOB (state=$ST pid=$P) -> unlocking"; rmdir "$LOCK" 2>/dev/null
    fi
    if mkdir "$LOCK" 2>/dev/null; then NEXT="$JOB $MOD $ARGS"; break; fi
    BUSY=1
  done < "$JOBS"
  if [ -z "$NEXT" ]; then
    [ "$BUSY" = 1 ] && { sleep 60; continue; }
    break
  fi
  set -- $NEXT; JOB="$1"; MOD="$2"; shift 2; ARGS="$*"
  STATUS="$LOGDIR/$JOB.status.json"; LOG="$LOGDIR/$JOB.log"; LOCK="$LOGDIR/$JOB.lock"
  echo "[$WORKER] run  $JOB  $(date -Is)   ($MOD $ARGS)"
  START=$(date -Is)
  $PY -m $MOD $ARGS >> "$LOG" 2>&1 &
  P=$!
  printf '{"state": "running", "pid": %d, "started": "%s", "gpu": "%s", "worker": "%s", "cmd": "%s"}\n' \
    "$P" "$START" "$GPU" "$WORKER" "$MOD $ARGS" > "$STATUS"
  wait $P; RC=$?
  STATE=done; [ $RC -eq 0 ] || STATE=failed
  printf '{"state": "%s", "pid": %d, "started": "%s", "ended": "%s", "exit_code": %d, "gpu": "%s", "worker": "%s", "cmd": "%s"}\n' \
    "$STATE" "$P" "$START" "$(date -Is)" $RC "$GPU" "$WORKER" "$MOD $ARGS" > "$STATUS"
  rmdir "$LOCK" 2>/dev/null
  echo "[$WORKER] $STATE $JOB rc=$RC $(date -Is)"
  # vLLM's EngineCore child survives its parent (e.g. after a kill) and keeps
  # the whole GPU; kill any that belongs to THIS GPU's job (children of $P are
  # gone with it; orphans are found by their env CUDA_VISIBLE_DEVICES)
  sleep 3
  for EP in $(pgrep -f "^VLLM::EngineCor[e]"); do
    if tr '\0' '\n' < /proc/$EP/environ 2>/dev/null | grep -q "^CUDA_VISIBLE_DEVICES=$GPU$"; then
      kill $EP 2>/dev/null && echo "[$WORKER] killed leftover VLLM::EngineCore $EP (gpu $GPU)"; sleep 5
    fi
  done
  [ "$STATE" = failed ] && sleep 120
done
echo "[$WORKER] end $(date -Is) (no runnable jobs left)"
