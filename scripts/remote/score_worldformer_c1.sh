#!/bin/sh
# C4 — score the six WorldFormer C1 cells on the worldbbox test set (CPU):
# all-frame prediction dumps -> re-evaluation (all + last frames) -> visibility-bucket breakdown.
#   setsid nohup sh scripts/remote/score_worldformer_c1.sh > /data3/rohith/ag/logs/score_c1.out 2>&1 < /dev/null &
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
OUT=/data3/rohith/ag/runs/worldformer/score
LOG=/data3/rohith/ag/logs
CKPT=${CKPT:-best}
cd $REPO || exit 1
mkdir -p $OUT/dumps $OUT/reeval
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8
: > $OUT/status.txt

run_mode() {
  mode=$1
  for stream in dinov3tok pi3tok fused; do
    stem=worldformer_c1_${stream}_${mode}
    cfg=configs/methods/$mode/$stem.yaml
    $PY tools/dump_predictions.py --config $cfg --ckpt $CKPT --frames all \
        --out $OUT/dumps/${stem}__all.pkl > $LOG/score_c1_dump_$stem.log 2>&1
    echo "dump $stem exit=$?" >> $OUT/status.txt
    $PY tools/reeval_test.py --config $cfg --ckpt $CKPT --frames both \
        --out $OUT/reeval/reeval_$stem.json > $LOG/score_c1_reeval_$stem.log 2>&1
    echo "reeval $stem exit=$?" >> $OUT/status.txt
  done
}
run_mode predcls &
run_mode sgdet &
wait
# the WorldWise@dinov3l reference dumps (rescore run) sit next to ours for the bucket table
for m in predcls sgdet; do
  ref=/data3/rohith/ag/runs/rescore/dumps/worldwise_${m}_dinov3l__all.pkl
  [ -f $ref ] && ln -sfn $ref $OUT/dumps/worldwise_${m}_dinov3l__all.pkl
done
$PY tools/bucketed_breakdown.py --dumps $OUT/dumps --out $OUT/bucketed_breakdown_c1.json \
    > $LOG/score_c1_bucketed.log 2>&1
echo "bucketed exit=$?" >> $OUT/status.txt
$PY tools/render_worldformer_c1_results.py --score_dir $OUT --constraint nc --k 20 > $OUT/table_nc.md
$PY tools/render_worldformer_c1_results.py --score_dir $OUT --constraint wc --k 20 > $OUT/table_wc.md
echo DONE >> $OUT/status.txt
