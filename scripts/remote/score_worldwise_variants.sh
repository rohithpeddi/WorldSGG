#!/bin/sh
# Score the WorldWise+ and WorldWise++ cells on the worldbbox test set (CPU, --ckpt best):
# all-frame prediction dumps -> re-evaluation (all + last frames) -> visibility-bucket
# breakdown -> aggregate tables.  Stems whose reeval JSON already exists are skipped, and
# a WorldWise+ stem re-uses the dump/reeval of its identical worldformer_c1 checkpoint
# (same weights, only the name changed) when that already exists.
#   setsid nohup sh scripts/remote/score_worldwise_variants.sh > /data3/rohith/ag/logs/score_variants.out 2>&1 < /dev/null &
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
PLUS_OUT=/data3/rohith/ag/runs/worldwise_plus/score
PP_OUT=/data3/rohith/ag/runs/worldwise_pp/score
C1_OUT=/data3/rohith/ag/runs/worldformer/score
RESCORE=/data3/rohith/ag/runs/rescore
LOG=/data3/rohith/ag/logs
CKPT=${CKPT:-best}
cd $REPO || exit 1
mkdir -p $PLUS_OUT/dumps $PLUS_OUT/reeval $PP_OUT/dumps $PP_OUT/reeval
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8

score_stem() {
  # $1 = stem, $2 = out dir, $3 = mode
  stem=$1; out=$2; mode=$3
  cfg=configs/methods/$mode/$stem.yaml
  if [ -s $out/reeval/reeval_$stem.json ] && [ -s $out/dumps/${stem}__all.pkl ]; then
    echo "skip $stem (already scored)" >> $out/status.txt; return
  fi
  [ -s $out/dumps/${stem}__all.pkl ] || {
    $PY tools/dump_predictions.py --config $cfg --ckpt $CKPT --frames all \
        --out $out/dumps/${stem}__all.pkl > $LOG/score_dump_$stem.log 2>&1
    echo "dump $stem exit=$?" >> $out/status.txt; }
  [ -s $out/reeval/reeval_$stem.json ] || {
    $PY tools/reeval_test.py --config $cfg --ckpt $CKPT --frames both \
        --out $out/reeval/reeval_$stem.json > $LOG/score_reeval_$stem.log 2>&1
    echo "reeval $stem exit=$?" >> $out/status.txt; }
}

link_c1() {
  # WorldWise+ cell == worldformer_c1 cell: reuse its dump / reeval if already produced
  stream=$1; mode=$2
  plus=worldwise_plus_${stream}_${mode}; c1=worldformer_c1_${stream}_${mode}
  [ -s $C1_OUT/dumps/${c1}__all.pkl ] && [ ! -e $PLUS_OUT/dumps/${plus}__all.pkl ] && \
    ln -sfn $C1_OUT/dumps/${c1}__all.pkl $PLUS_OUT/dumps/${plus}__all.pkl
  [ -s $C1_OUT/reeval/reeval_$c1.json ] && [ ! -e $PLUS_OUT/reeval/reeval_$plus.json ] && \
    ln -sfn $C1_OUT/reeval/reeval_$c1.json $PLUS_OUT/reeval/reeval_$plus.json
  return 0
}

run_mode() {
  mode=$1
  for stream in dinov3tok pi3tok fused; do
    link_c1 $stream $mode
    score_stem worldwise_plus_${stream}_${mode} $PLUS_OUT $mode
  done
  score_stem worldwise_pp_dinov3_${mode} $PP_OUT $mode
  score_stem worldwise_pp_dinov3_nodet_${mode} $PP_OUT $mode
}
run_mode predcls &
run_mode sgdet &
wait

# WorldWise@dinov3l reference dumps next to ours for the bucket tables
for m in predcls sgdet; do
  ref=$RESCORE/dumps/worldwise_${m}_dinov3l__all.pkl
  [ -f $ref ] && ln -sfn $ref $PLUS_OUT/dumps/worldwise_${m}_dinov3l__all.pkl
  [ -f $ref ] && ln -sfn $ref $PP_OUT/dumps/worldwise_${m}_dinov3l__all.pkl
done
$PY tools/bucketed_breakdown.py --dumps $PLUS_OUT/dumps --out $PLUS_OUT/bucketed_breakdown.json \
    > $LOG/score_bucketed_plus.log 2>&1
echo "bucketed exit=$?" >> $PLUS_OUT/status.txt
$PY tools/bucketed_breakdown.py --dumps $PP_OUT/dumps --out $PP_OUT/bucketed_breakdown.json \
    > $LOG/score_bucketed_pp.log 2>&1
echo "bucketed exit=$?" >> $PP_OUT/status.txt
# aggregate table across the reference rescore root and both variant roots
$PY tools/aggregate_rescore.py --root $RESCORE $PLUS_OUT $PP_OUT > $PP_OUT/worldwise_variants_table.md 2> $LOG/score_aggregate.log
echo "aggregate exit=$?" >> $PP_OUT/status.txt
echo DONE >> $PLUS_OUT/status.txt
echo DONE >> $PP_OUT/status.txt
