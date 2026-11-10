#!/bin/bash
# TRAIN: ground-truth references + full-GT caches for scoring the rebuttal-round dumps (CPU only, ~1 h).
#   bash tools/rebuttal/train_prep_gt.sh <feature_model>     (resnet50 | dinov3l | dinov3_tok)
# Per feature model F (the PredCls box space the SGDet dumps of that family are matched against):
#   1. GT reference with the historical layout   -> gate: equals the published PredCls dump's GT fields exactly
#   2. GT reference with per-video slot alignment -> full-GT cache (rescore_rebuttal.build_gt, parametrised)
#   3. gate: per-frame class-level GT triplets and the full-inventory totals equal the published ones
set -e -o pipefail
F=$1
cd ~/CODE/Scene4Cast_rebuttal_train
PY=~/anaconda3/envs/scene4cast/bin/python
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
O=/data3/rohith/ag/runs/rebuttal/train/gt
mkdir -p $O
case $F in
  resnet50)   PUB=/data3/rohith/ag/runs/rescore/dumps/w_sttran_predcls_resnet50__all.pkl ;;
  dinov3l)    PUB=/data3/rohith/ag/runs/rescore/dumps/worldwise_predcls_dinov3l__all.pkl ;;
  dinov3_tok) PUB=/data3/rohith/ag/runs/worldwise_pp/score/dumps/worldwise_pp_dinov3_predcls__all.pkl ;;
  *) echo "unknown feature model $F"; exit 2 ;;
esac
echo "[prep $F] $(date -Is) reference (historical layout)"
$PY tools/rebuttal/train_rescore.py gt-ref --feature_model $F --slot_alignment none --out $O/predcls_${F}_none.pkl
$PY tools/rebuttal/train_rescore.py gate-ref --ref $O/predcls_${F}_none.pkl --published $PUB --exact \
    --out $O/gate_${F}_none.json
echo "[prep $F] $(date -Is) reference (aligned)"
$PY tools/rebuttal/train_rescore.py gt-ref --feature_model $F --slot_alignment per_video_class \
    --out $O/predcls_${F}_aligned.pkl
$PY tools/rebuttal/train_rescore.py build-gt --ref $O/predcls_${F}_aligned.pkl --cache $O/full_gt_${F}_aligned.tmp.pkl
$PY tools/rebuttal/train_rescore.py gate-ref --ref $O/predcls_${F}_aligned.pkl --published $PUB \
    --cache $O/full_gt_${F}_aligned.tmp.pkl --out $O/gate_${F}_aligned.json
mv $O/full_gt_${F}_aligned.tmp.pkl $O/full_gt_${F}_aligned.pkl     # eval jobs wait for this file
echo "[prep $F] $(date -Is) PASS"
