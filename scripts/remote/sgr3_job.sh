#!/bin/sh
# One rag_sgr3 run on GPU 1 (wsg env), 150-video list, Qwen3-VL-8B predcls.
#   sh scripts/remote/sgr3_job.sh <name> <runner args...>
NAME=$1; shift
cd $HOME/CODE/Scene4Cast_sgr3
export CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false VLLM_LOGGING_LEVEL=WARNING OMP_NUM_THREADS=8
echo "START $(date -Is) $NAME $*"
$HOME/anaconda3/envs/wsg/bin/python -m lib.mllm.methods.rag_sgr3.runner --model_name qwen3vl_8b --mode predcls \
    --temperature 0.2 --video_list /data3/rohith/ag/splits/test_worldbbox_thinking150.txt "$@"
echo "END $(date -Is) $NAME rc=$?"
