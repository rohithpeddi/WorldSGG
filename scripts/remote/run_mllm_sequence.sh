#!/bin/sh
# Sequential MLLM job queue for GPU 0 (B3 baselines + Stage-1 graph fill).
# Usage:  sh scripts/remote/run_mllm_sequence.sh <queue_name> <jobs file>
# Jobs file: one job per line  "<job_name> <python module> <args...>"; '#' comments.
# Each job: CUDA_VISIBLE_DEVICES=0, log -> /data3/rohith/ag/logs/<job>.log,
# status -> /data3/rohith/ag/logs/<job>.status.json; the queue itself writes
# /data3/rohith/ag/logs/<queue>.queue.log. Jobs whose status is "done" are skipped
# (resumable). Launch detached:
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
grep -v '^\s*#' "$JOBS" | grep -v '^\s*$' | while read -r JOB MOD ARGS; do
  STATUS="$LOGDIR/$JOB.status.json"; LOG="$LOGDIR/$JOB.log"
  if [ -f "$STATUS" ] && grep -q '"state": "done"' "$STATUS"; then
    echo "[$QUEUE] skip $JOB (done)"; continue
  fi
  echo "[$QUEUE] run  $JOB  $(date -Is)"
  printf '{"state": "running", "pid": %d, "started": "%s", "gpu": "%s", "queue": "%s", "cmd": "%s"}\n' \
    $$ "$(date -Is)" "$GPU" "$QUEUE" "$MOD $ARGS" > "$STATUS"
  $PY -m $MOD $ARGS >> "$LOG" 2>&1
  RC=$?
  STATE=done; [ $RC -eq 0 ] || STATE=failed
  printf '{"state": "%s", "pid": %d, "started": "%s", "ended": "%s", "exit_code": %d, "gpu": "%s", "queue": "%s", "cmd": "%s"}\n' \
    "$STATE" $$ "$(sed -n 's/.*"started": *"\([^"]*\)".*/\1/p' "$STATUS")" "$(date -Is)" $RC "$GPU" "$QUEUE" "$MOD $ARGS" > "$STATUS"
  echo "[$QUEUE] $STATE $JOB rc=$RC $(date -Is)"
done
echo "[$QUEUE] end $(date -Is)"
