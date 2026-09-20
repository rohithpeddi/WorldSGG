#!/bin/sh
# Re-score the existing checkpoints (best epoch) on the worldbbox test set:
# all-frame prediction dumps + re-evaluation + visibility-bucket breakdown.
# CPU only (the GPUs belong to the token-caching / MLLM jobs); a full
# 1,511-video dump takes ~1 min per checkpoint.
#   setsid nohup sh scripts/remote/rescore_worldbbox.sh > /data3/rohith/ag/logs/rescore.out 2>&1 < /dev/null &
REPO=/home/rxp190007/CODE/Scene4Cast_wb
PY=/home/rxp190007/anaconda3/envs/scene4cast/bin/python
OUT=/data3/rohith/ag/runs/rescore
LOG=/data3/rohith/ag/logs
cd $REPO || exit 1
mkdir -p $OUT/dumps $OUT/reeval
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
: > $OUT/status.txt

run_mode() {
  mode=$1
  for stem in w_sttran_${mode}_resnet50 w_sttran_pp_${mode}_resnet50 w_dsgdetr_${mode}_resnet50 \
              w_dsgdetr_pp_${mode}_resnet50 w_usg_${mode}_resnet50 worldwise_${mode}_resnet50 \
              worldwise_${mode}_dinov2b worldwise_${mode}_dinov2l worldwise_${mode}_dinov3l; do
    cfg=configs/methods/$mode/$stem.yaml
    $PY tools/dump_predictions.py --config $cfg --ckpt best --frames all \
        --out $OUT/dumps/${stem}__all.pkl > $LOG/rescore_dump_$stem.log 2>&1
    echo "dump $stem exit=$?" >> $OUT/status.txt
    $PY tools/reeval_test.py --config $cfg --ckpt best --frames both \
        --out $OUT/reeval/reeval_$stem.json > $LOG/rescore_reeval_$stem.log 2>&1
    echo "reeval $stem exit=$?" >> $OUT/status.txt
  done
}
run_mode predcls &
run_mode sgdet &
wait
$PY tools/bucketed_breakdown.py --dumps $OUT/dumps --out $OUT/bucketed_breakdown_worldbbox.json \
    > $LOG/rescore_bucketed.log 2>&1
echo "bucketed exit=$?" >> $OUT/status.txt
echo DONE >> $OUT/status.txt
