#!/bin/sh
# Sequential MLLM job queue for GPU 0 (B3 baselines, Stage-1 graph fill, tracks).
# Usage:  sh scripts/remote/run_mllm_sequence.sh <queue_name> <jobs file>
# Jobs file: one job per line  "<job_name> <python module> <args...>"; '#' comments.
# The file is RE-READ before every job, so jobs can be added / reordered live.
# Per job: CUDA_VISIBLE_DEVICES=$MLLM_GPU (default 0), log /data3/rohith/ag/logs/<job>.log,
# status /data3/rohith/ag/logs/<job>.status.json ({state: running|done|failed, pid, ...}).
# A job whose status is "done" is skipped; "running" with a live pid is waited for
# (another queue instance owns it); "running" with a dead pid or "failed" is re-run
# (all runners are resumable per video).  Launch detached:
#   setsid nohup sh scripts/remote/run_mllm_sequence.sh b3 /data3/rohith/ag/logs/b3.jobs \
#       > /data3/rohith/ag/logs/b3.queue.log 2>&1 < /dev/null &
set -u
QUEUE="$1"; JOBS="$2"
ROOT="$HOME/CODE/Scene4Cast_mllm"
LOGDIR=/data3/rohith/ag/logs
PY="$HOME/anaconda3/envs/wsg/bin/python"
GPU="${MLLM_GPU:-0}"
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING
echo "[$QUEUE] start $(date -Is) gpu=$GPU commit=$(git rev-parse --short HEAD)"

state_of() { sed -n 's/.*"state": *"\([a-z]*\)".*/\1/p' "$1" 2>/dev/null; }
pid_of()   { sed -n 's/.*"pid": *\([0-9]*\).*/\1/p' "$1" 2>/dev/null; }

while :; do
  NEXT=""
  # pick the first job that is not done and not owned by a live process
  while read -r JOB MOD ARGS; do
    case "$JOB" in ""|\#*) continue;; esac
    STATUS="$LOGDIR/$JOB.status.json"
    ST=$(state_of "$STATUS")
    if [ "$ST" = "done" ]; then continue; fi
    if [ "$ST" = "running" ]; then
      P=$(pid_of "$STATUS")
      if [ -n "$P" ] && kill -0 "$P" 2>/dev/null; then NEXT="__WAIT__ $JOB"; break; fi
      echo "[$QUEUE] stale running status for $JOB (pid $P dead) -> re-run"
    fi
    NEXT="$JOB $MOD $ARGS"; break
  done < "$JOBS"
  [ -z "$NEXT" ] && break
  case "$NEXT" in
    "__WAIT__ "*) sleep 60; continue;;
  esac
  set -- $NEXT; JOB="$1"; MOD="$2"; shift 2; ARGS="$*"
  STATUS="$LOGDIR/$JOB.status.json"; LOG="$LOGDIR/$JOB.log"
  echo "[$QUEUE] run  $JOB  $(date -Is)   ($MOD $ARGS)"
  START=$(date -Is)
  # run in background so the recorded pid is the python process
  $PY -m $MOD $ARGS >> "$LOG" 2>&1 &
  P=$!
  printf '{"state": "running", "pid": %d, "started": "%s", "gpu": "%s", "queue": "%s", "cmd": "%s"}\n' \
    "$P" "$START" "$GPU" "$QUEUE" "$MOD $ARGS" > "$STATUS"
  wait $P; RC=$?
  STATE=done; [ $RC -eq 0 ] || STATE=failed
  printf '{"state": "%s", "pid": %d, "started": "%s", "ended": "%s", "exit_code": %d, "gpu": "%s", "queue": "%s", "cmd": "%s"}\n' \
    "$STATE" "$P" "$START" "$(date -Is)" $RC "$GPU" "$QUEUE" "$MOD $ARGS" > "$STATUS"
  echo "[$QUEUE] $STATE $JOB rc=$RC $(date -Is)"
done
echo "[$QUEUE] end $(date -Is) (no runnable jobs left)"
