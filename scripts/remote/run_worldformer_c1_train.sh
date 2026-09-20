#!/bin/sh
# C2 — train the six WorldFormer C1 cells on GPUs 1+2, two per GPU (detached).
# Requires the derived feature PKLs (derive_roi_features.py, train + test) and the
# symlinks under /data/rohith/ag/features/roi_features/<mode>/<stream>/.
#   sh scripts/remote/run_worldformer_c1_train.sh [extra run_configs_multigpu args]
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
LOG=/data3/rohith/ag/logs/train_worldformer_c1.log
cd $REPO || exit 1
mkdir -p /data3/rohith/ag/runs/worldformer logs/grid results
setsid nohup $PY tools/run_configs_multigpu.py --gpus 1 2 --per-gpu 2 --python $PY \
    --configs "configs/methods/predcls/worldformer_c1_*_predcls.yaml" \
              "configs/methods/sgdet/worldformer_c1_*_sgdet.yaml" "$@" \
    >> $LOG 2>&1 < /dev/null &
echo "launched C1 training queue pid $! log $LOG"
