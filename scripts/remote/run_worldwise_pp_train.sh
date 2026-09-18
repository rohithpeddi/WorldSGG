#!/bin/sh
# Train the four WorldWise++ cells (detached), one launcher per GPU:
#   GPU 0: worldwise_pp_dinov3_predcls        GPU 1: worldwise_pp_dinov3_sgdet
#   GPU 2: worldwise_pp_dinov3_nodet_{predcls,sgdet} (two concurrent slots)
# Requires the pp grid cache (/data3/rohith/ag/cache/pp_grids/{train,test}) to be complete.
#   sh scripts/remote/run_worldwise_pp_train.sh [extra run_configs_multigpu args]
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
LOGDIR=/data3/rohith/ag/logs
cd $REPO || exit 1
mkdir -p /data3/rohith/ag/runs/worldwise_pp logs/grid results $LOGDIR
setsid nohup $PY tools/run_configs_multigpu.py --gpus 0 --per-gpu 1 --python $PY \
    --configs configs/methods/predcls/worldwise_pp_dinov3_predcls.yaml "$@" \
    >> $LOGDIR/train_worldwise_pp_gpu0.log 2>&1 < /dev/null &
echo "launched worldwise_pp predcls on GPU 0: launcher pid $!"
sleep 2
setsid nohup $PY tools/run_configs_multigpu.py --gpus 1 --per-gpu 1 --python $PY \
    --configs configs/methods/sgdet/worldwise_pp_dinov3_sgdet.yaml "$@" \
    >> $LOGDIR/train_worldwise_pp_gpu1.log 2>&1 < /dev/null &
echo "launched worldwise_pp sgdet on GPU 1: launcher pid $!"
sleep 2
setsid nohup $PY tools/run_configs_multigpu.py --gpus 2 --per-gpu 2 --python $PY \
    --configs configs/methods/predcls/worldwise_pp_dinov3_nodet_predcls.yaml \
              configs/methods/sgdet/worldwise_pp_dinov3_nodet_sgdet.yaml "$@" \
    >> $LOGDIR/train_worldwise_pp_gpu2.log 2>&1 < /dev/null &
echo "launched worldwise_pp nodet predcls+sgdet on GPU 2: launcher pid $!"
echo "per-run logs: logs/grid/worldwise_pp_*.log ; launcher logs: $LOGDIR/train_worldwise_pp_gpu*.log"
