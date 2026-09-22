#!/bin/sh
# Parallel CPU scoring of the pending MLLM cells on CS93371.
# One score_all process per cell so the 58 idle cores are actually used;
# each is pinned to 4 BLAS threads and denied a GPU.
W="$1"
cd /home/rxp190007/CODE/Scene4Cast_mllm || exit 1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
LOG=/data3/rohith/ag/logs/score22/$W.log
echo "[$W] start $(date -Is)" > "$LOG"
/home/rxp190007/anaconda3/envs/wsg/bin/python -m lib.mllm.eval.score_all \
    --runs /data3/rohith/ag/logs/score22/$W.json \
    --out_prefix /data3/rohith/ag/runs/mllm/score22/$W \
    --no_legacy >> "$LOG" 2>&1
echo "[$W] exit=$? $(date -Is)" >> "$LOG"
