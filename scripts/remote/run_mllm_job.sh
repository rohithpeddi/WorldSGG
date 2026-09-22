#!/bin/sh
# Detached launcher for MLLM jobs on CS93371 (GPU 0, wsg env).
#   sh scripts/remote/run_mllm_job.sh <job_name> <python module> [args...]
# Writes /data3/rohith/ag/logs/<job_name>.log and
#        /data3/rohith/ag/logs/<job_name>.status.json  ({state, pid, started, ended, exit_code, cmd})
# Refuses to start if another job with the same name is still running.
set -u
JOB="$1"; shift
MOD="$1"; shift
ROOT="$HOME/CODE/Scene4Cast_mllm"
LOGDIR=/data3/rohith/ag/logs
PY="$HOME/anaconda3/envs/wsg/bin/python"
GPU="${MLLM_GPU:-0}"
mkdir -p "$LOGDIR"
STATUS="$LOGDIR/$JOB.status.json"
LOG="$LOGDIR/$JOB.log"
if [ -f "$STATUS" ]; then
  OLDPID=$(sed -n 's/.*"pid": *\([0-9]*\).*/\1/p' "$STATUS")
  if [ -n "$OLDPID" ] && kill -0 "$OLDPID" 2>/dev/null && grep -q '"state": "running"' "$STATUS"; then
    echo "job $JOB already running (pid $OLDPID)"; exit 2
  fi
fi
CMD="$PY -m $MOD $*"
cat > "$LOGDIR/$JOB.run.sh" <<EOS
#!/bin/sh
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_LOGGING_LEVEL=WARNING
printf '{"state": "running", "pid": %d, "started": "%s", "gpu": "%s", "cmd": "%s"}\n' \$\$ "\$(date -Is)" "$GPU" "$(echo "$CMD" | sed 's/"/\\\\"/g')" > "$STATUS"
$CMD
RC=\$?
printf '{"state": "%s", "pid": %d, "started": "%s", "ended": "%s", "exit_code": %d, "gpu": "%s", "cmd": "%s"}\n' \
  "\$( [ \$RC -eq 0 ] && echo done || echo failed )" \$\$ "\$(sed -n 's/.*"started": *"\([^"]*\)".*/\1/p' "$STATUS")" "\$(date -Is)" \$RC "$GPU" "$(echo "$CMD" | sed 's/"/\\\\"/g')" > "$STATUS"
exit \$RC
EOS
chmod +x "$LOGDIR/$JOB.run.sh"
setsid nohup sh "$LOGDIR/$JOB.run.sh" > "$LOG" 2>&1 < /dev/null &
sleep 1
echo "launched $JOB -> $LOG"; cat "$STATUS"
