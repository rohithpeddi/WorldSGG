#!/bin/sh
# Wait until every WorldWise++ cell has logged its final epoch, then run the variant
# scoring chain (WorldWise+ cells are picked up in the same pass).
#   setsid nohup sh scripts/remote/worldwise_pp_score_watcher.sh > /data3/rohith/ag/logs/pp_score_watcher.out 2>&1 < /dev/null &
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
LOG=/data3/rohith/ag/logs/pp_score_watcher.log
NEPOCH=${NEPOCH:-20}
cd "$REPO" || exit 1
log() { echo "[pp-score-watcher $(date +%FT%T)] $*" >> "$LOG"; }
all_done() {
  for st in worldwise_pp_dinov3_predcls worldwise_pp_dinov3_sgdet \
            worldwise_pp_dinov3_nodet_predcls worldwise_pp_dinov3_nodet_sgdet; do
    f=results/${st}_metrics.jsonl
    [ -f "$f" ] || return 1
    last=$(tail -1 "$f" | grep -o '"epoch": [0-9]*' | awk '{print $2}')
    [ -n "$last" ] && [ "$last" -ge "$NEPOCH" ] || return 1
  done
  return 0
}
log "waiting for 4 WorldWise++ cells x $NEPOCH epochs"
until all_done; do sleep 600; done
log "all cells finished -> scoring"
sh scripts/remote/score_worldwise_variants.sh >> "$LOG" 2>&1
log "scoring finished: $(tail -1 /data3/rohith/ag/runs/worldwise_pp/score/status.txt)"
cat /data3/rohith/ag/runs/worldwise_pp/score/worldwise_variants_table.md >> "$LOG"
