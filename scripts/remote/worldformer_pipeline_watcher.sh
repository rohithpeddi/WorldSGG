#!/bin/sh
# Overnight driver for the WorldFormer C1 pipeline (detached; polls every 5 min; idempotent).
#   setsid nohup sh scripts/remote/worldformer_pipeline_watcher.sh > /data3/rohith/ag/logs/pipeline_watcher.out 2>&1 < /dev/null &
# Phase A: both DINOv3 train shards done -> derive train dinov3_tok -> launch the 2 dinov3_tok cells on GPU 2
#          (memory guard; Pi3 cacher rate before / 10 min after; auto-pause trainers on a >30% drop).
# Phase B: both Pi3 test shards done  -> derive test pi3_tok + fused.
# Phase C: both Pi3 train shards done -> derive train pi3_tok + fused -> launch the 4 remaining cells on GPUs 1+2.
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
LOGD=/data3/rohith/ag/logs
LOG=$LOGD/pipeline_watcher.log
TOK=/data3/rohith/ag/cache/tokens
FLAGS=/data3/rohith/ag/runs/worldformer/flags
mkdir -p "$FLAGS"
cd "$REPO" || exit 1

log() { echo "[watcher $(date +%FT%T)] $*" >> "$LOG"; }
both_ended() { # $1 = "dinov3l train" etc.
  grep -q "END $1" $LOGD/tokens_chain_shard0of2.log 2>/dev/null && grep -q "END $1" $LOGD/tokens_chain_shard1of2.log 2>/dev/null
}
derive_sync() { # $1 split, $2.. streams — runs run_derive's payload in the foreground
  split=$1; shift
  for st in "$@"; do
    log "derive $split $st START"
    $PY tools/register_token_cache.py --stream dinov3l --split $split >> "$LOG" 2>&1
    case $st in pi3_tok|fused) $PY tools/register_token_cache.py --stream pi3 --split $split >> "$LOG" 2>&1 ;; esac
    $PY datasets/preprocess/tokens/derive_roi_features.py --split $split --mode all --stream $st --workers 16 --register \
        >> $LOGD/derive_${split}_${st}.log 2>&1
    log "derive $split $st END rc=$?"
  done
}
avail_gb() { free -g | awk '/^Mem:/{print $7}'; }
pi3_clip_frames() { # sum of clip_frames over both shards of the running pi3 stage
  for s in 0 1; do for sp in train test; do f=$TOK/pi3/$sp/status_shard${s}of2.json; [ -f $f ] && grep -o '"clip_frames": [0-9]*' $f | awk '{print $2}'; done; done | awk '{s+=$1} END{print s+0}'
}
pi3_rate() { for s in 0 1; do for sp in train test; do f=$TOK/pi3/$sp/status_shard${s}of2.json; [ -f $f ] && grep '"state": "running"' $f >/dev/null && grep -o '"clip_frames_per_s": [0-9.]*' $f | awk '{print $2}'; done; done | awk '{s+=$1} END{print s+0}'; }

# ---------------- Phase A ----------------
if [ ! -f $FLAGS/phaseA.done ]; then
  until both_ended "dinov3l train"; do sleep 300; done
  log "Phase A: dinov3l train caches complete"
  derive_sync train dinov3_tok
  while [ "$(avail_gb)" -lt 20 ]; do log "Phase A: waiting for host memory (avail $(avail_gb) GB)"; sleep 300; done
  r0=$(pi3_rate); c0=$(pi3_clip_frames)
  log "Phase A: launching dinov3_tok cells on GPU 2 (pi3 cacher cumulative rate $r0 clip-fr/s, clip_frames $c0, mem avail $(avail_gb) GB)"
  mkdir -p logs/grid results
  setsid nohup $PY tools/run_configs_multigpu.py --gpus 2 --per-gpu 2 --python $PY \
      --configs configs/methods/predcls/worldformer_c1_dinov3tok_predcls.yaml configs/methods/sgdet/worldformer_c1_dinov3tok_sgdet.yaml \
      >> $LOGD/train_worldformer_c1_dinov3tok.log 2>&1 < /dev/null &
  echo $! > $FLAGS/trainers_dinov3tok.pid
  log "Phase A: trainer queue pid $(cat $FLAGS/trainers_dinov3tok.pid)"
  sleep 600
  c1=$(pi3_clip_frames); r1=$(awk "BEGIN{print ($c1-$c0)/600}")
  log "Phase A: pi3 cacher rate 10 min after launch: $r1 clip-fr/s (before: $r0)"
  if [ "$(awk "BEGIN{print ($r0>0 && $r1 < 0.7*$r0) ? 1 : 0}")" = 1 ]; then
    log "Phase A: WARNING rate dropped >30% -> pausing trainers (resume with --ckpt)"
    pkill -P $(cat $FLAGS/trainers_dinov3tok.pid); kill $(cat $FLAGS/trainers_dinov3tok.pid) 2>/dev/null
    pkill -f "[w]orldformer_c1_dinov3tok_"
    touch $FLAGS/trainers_paused_rate_drop
  fi
  touch $FLAGS/phaseA.done
fi

# ---------------- Phase B ----------------
if [ ! -f $FLAGS/phaseB.done ]; then
  until both_ended "pi3 test"; do sleep 300; done
  log "Phase B: pi3 test caches complete"
  derive_sync test pi3_tok fused
  touch $FLAGS/phaseB.done
fi

# ---------------- Phase C ----------------
if [ ! -f $FLAGS/phaseC.done ]; then
  until both_ended "pi3 train"; do sleep 300; done
  log "Phase C: pi3 train caches complete"
  derive_sync train pi3_tok fused
  while [ "$(avail_gb)" -lt 20 ]; do log "Phase C: waiting for host memory"; sleep 300; done
  log "Phase C: launching pi3_tok + fused cells on GPUs 1+2"
  setsid nohup $PY tools/run_configs_multigpu.py --gpus 1 2 --per-gpu 2 --python $PY \
      --configs configs/methods/predcls/worldformer_c1_pi3tok_predcls.yaml configs/methods/sgdet/worldformer_c1_pi3tok_sgdet.yaml \
                configs/methods/predcls/worldformer_c1_fused_predcls.yaml configs/methods/sgdet/worldformer_c1_fused_sgdet.yaml \
      >> $LOGD/train_worldformer_c1_pi3fused.log 2>&1 < /dev/null &
  echo $! > $FLAGS/trainers_pi3fused.pid
  touch $FLAGS/phaseC.done
fi
$PY tools/cache_manifest.py list >> "$LOG" 2>&1
log "watcher finished"
