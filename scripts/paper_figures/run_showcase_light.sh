#!/bin/bash
# Dump every training-based method (WorldWise family + the five adapted baselines, PredCls
# checkpoints, light theme) for the showcase video of the processing-unit figures.
# TM0BV: the WorldBBox test video on which WorldWise++ gets every person-object pair right
# at the three key frames (t = 13 visible, t = 21 and 23 unobserved); picked by
# find_showcase.py over the all-frame prediction dumps.  Runs on CS93371 from ~/CODE/Scene4Cast:
#   setsid nohup bash scripts/paper_figures/run_showcase_light.sh > /data3/rohith/ag/runs/intermediates_light/logs/run_showcase.log 2>&1 < /dev/null &
cd ~/CODE/Scene4Cast
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OUT=/data3/rohith/ag/runs/intermediates_light
PY=~/anaconda3/envs/scene4cast/bin/python
VIDEO=${VIDEO:-TM0BV}
KEYS=${KEYS:-"13 21 23"}
echo "=== $VIDEO keys $KEYS $(date)"
$PY scripts/paper_figures/dump_intermediates.py --video $VIDEO --out-dir $OUT --keyframes $KEYS \
  --methods worldwise worldwise_plus worldwise_pp w_sttran w_sttran_pp w_dsgdetr w_dsgdetr_pp w_usg \
  --theme light --caps title
echo "SHOWCASE-DONE $(date)"
