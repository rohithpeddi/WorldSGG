#!/bin/bash
# Re-render the inference panels of every dumped video in the light (white
# background, Title Case) theme.  Runs on CS93371 from ~/CODE/Scene4Cast:
#   setsid nohup bash scripts/paper_figures/run_light.sh > /data3/rohith/ag/runs/intermediates_light/logs/run_light.log 2>&1 < /dev/null &
# The dark dumps stay in /data3/rohith/ag/runs/intermediates/.
cd ~/CODE/Scene4Cast
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}
OUT=/data3/rohith/ag/runs/intermediates_light
PY=~/anaconda3/envs/scene4cast/bin/python
for v in 12XD3 AQQQ5 GFK4S IDXZK QI0EL R4SJJ HI75B UG4M2; do
  echo "=== $v $(date)"
  $PY scripts/paper_figures/dump_intermediates.py --video $v --out-dir $OUT --theme light --caps title
done
# 00T1E is outside the WorldBBox split: legacy annotation, recovered canonical frame (see dark/README.md)
echo "=== 00T1E $(date)"
$PY scripts/paper_figures/dump_intermediates.py --video 00T1E --out-dir $OUT --theme light --caps title --annot-dir world4d_rel_annotations
echo "LIGHT-DONE $(date)"
