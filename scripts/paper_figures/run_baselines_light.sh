#!/bin/bash
# Dump the five adapted-baseline cells (light theme) for 00T1E, the video the
# processing-unit figures are drawn for.  Runs on CS93371 from ~/CODE/Scene4Cast:
#   setsid nohup bash scripts/paper_figures/run_baselines_light.sh > /data3/rohith/ag/runs/intermediates_light/logs/run_baselines.log 2>&1 < /dev/null &
# 00T1E is outside the WorldBBox split: legacy annotation folder (see dark/README.md).
cd ~/CODE/Scene4Cast
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
OUT=/data3/rohith/ag/runs/intermediates_light
PY=~/anaconda3/envs/scene4cast/bin/python
echo "=== 00T1E $(date)"
$PY scripts/paper_figures/dump_intermediates.py --video 00T1E --out-dir $OUT \
  --methods w_sttran w_sttran_pp w_dsgdetr w_dsgdetr_pp w_usg --theme light --caps title \
  --annot-dir world4d_rel_annotations
echo "BASELINES-DONE $(date)"
