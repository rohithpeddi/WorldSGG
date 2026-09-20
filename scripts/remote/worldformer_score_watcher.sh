#!/bin/sh
# C4 automation: wait until every WorldFormer C1 cell has logged its final epoch, then score.
#   setsid nohup sh scripts/remote/worldformer_score_watcher.sh > /data3/rohith/ag/logs/score_watcher.out 2>&1 < /dev/null &
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
LOG=/data3/rohith/ag/logs/score_watcher.log
NEPOCH=${NEPOCH:-20}
cd "$REPO" || exit 1
log() { echo "[score-watcher $(date +%FT%T)] $*" >> "$LOG"; }
all_done() {
  for m in predcls sgdet; do for st in dinov3tok pi3tok fused; do
    f=results/worldformer_c1_${st}_${m}_metrics.jsonl
    [ -f "$f" ] || return 1
    last=$(tail -1 "$f" | grep -o '"epoch": [0-9]*' | awk '{print $2}')
    [ -n "$last" ] && [ "$last" -ge "$NEPOCH" ] || return 1
  done; done
  return 0
}
log "waiting for 6 cells x $NEPOCH epochs"
until all_done; do sleep 600; done
log "all cells finished -> scoring"
sh scripts/remote/score_worldformer_c1.sh >> "$LOG" 2>&1
log "scoring finished: $(tail -1 /data3/rohith/ag/runs/worldformer/score/status.txt)"
cat /data3/rohith/ag/runs/worldformer/score/table_nc.md >> "$LOG"
