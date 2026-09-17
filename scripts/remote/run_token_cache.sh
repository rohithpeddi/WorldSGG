#!/bin/sh
# Launch one token-cache shard, detached, on one GPU (WorldFormer C1a / C1b).
#   sh scripts/remote/run_token_cache.sh <stream: dinov3l|pi3> <split> <shard> <n_shards> <gpu> [extra args]
# Logs: /data3/rohith/ag/logs/tokens_<stream>_<split>_shard<k>of<n>.log
set -e
STREAM=$1; SPLIT=$2; SHARD=$3; NSHARDS=$4; GPU=$5; shift 5
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
LOGDIR=/data3/rohith/ag/logs
mkdir -p "$LOGDIR"
case "$STREAM" in
  dinov3l) PY=$HOME/anaconda3/envs/scene4cast/bin/python; SCRIPT=datasets/preprocess/tokens/dinov3_tokens.py ;;
  pi3)     PY=$HOME/anaconda3/envs/pi3/bin/python;        SCRIPT=datasets/preprocess/tokens/pi3_tokens.py ;;
  *) echo "unknown stream $STREAM"; exit 1 ;;
esac
LOG=$LOGDIR/tokens_${STREAM}_${SPLIT}_shard${SHARD}of${NSHARDS}.log
cd "$REPO"
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  setsid nohup "$PY" "$SCRIPT" --split "$SPLIT" --shard "$SHARD" --n_shards "$NSHARDS" "$@" \
  >> "$LOG" 2>&1 < /dev/null &
echo "launched $STREAM $SPLIT shard $SHARD/$NSHARDS on GPU $GPU pid $! log $LOG"
