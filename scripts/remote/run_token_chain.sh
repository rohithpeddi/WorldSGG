#!/bin/sh
# One detached process per GPU that runs the whole token-cache sequence for one shard:
#   dinov3l test -> dinov3l train -> pi3 test -> pi3 train   (each stage is resumable)
#   sh scripts/remote/run_token_chain.sh <shard> <n_shards> <gpu>
# Log: /data3/rohith/ag/logs/tokens_chain_shard<k>of<n>.log (+ per-stage logs from run_token_cache.sh)
SHARD=$1; NSHARDS=$2; GPU=$3
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
LOGDIR=/data3/rohith/ag/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/tokens_chain_shard${SHARD}of${NSHARDS}.log
cd "$REPO"
setsid nohup sh -c "
for stage in 'dinov3l test' 'dinov3l train' 'pi3 test' 'pi3 train'; do
  set -- \$stage; STREAM=\$1; SPLIT=\$2
  case \$STREAM in
    dinov3l) PY=\$HOME/anaconda3/envs/scene4cast/bin/python; SCRIPT=datasets/preprocess/tokens/dinov3_tokens.py ;;
    pi3)     PY=\$HOME/anaconda3/envs/pi3/bin/python;        SCRIPT=datasets/preprocess/tokens/pi3_tokens.py ;;
  esac
  echo \"[chain \$(date +%FT%T)] START \$STREAM \$SPLIT shard $SHARD/$NSHARDS gpu $GPU\"
  CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 \$PY \$SCRIPT --split \$SPLIT --shard $SHARD --n_shards $NSHARDS \
    >> $LOGDIR/tokens_\${STREAM}_\${SPLIT}_shard${SHARD}of${NSHARDS}.log 2>&1
  echo \"[chain \$(date +%FT%T)] END \$STREAM \$SPLIT rc=\$?\"
  df -h /data3 | tail -1
done
echo \"[chain \$(date +%FT%T)] ALL DONE\"
" >> "$LOG" 2>&1 < /dev/null &
echo "launched chain shard $SHARD/$NSHARDS on GPU $GPU pid $! log $LOG"
