#!/bin/sh
# Noise floor: a second R0 draw (temperature 0.2), queued behind sgr3_arms_extra.sh (waits for EXTRA_DONE).
set -u
ROOT=$HOME/CODE/Scene4Cast_sgr3
PY=$HOME/anaconda3/envs/wsg/bin/python
LOGS=/data3/rohith/ag/logs/ext
RUNS=/data3/rohith/ag/runs/mllm/rag_sgr3
LIST=/data3/rohith/ag/splits/test_worldbbox_thinking150.txt
cd $ROOT
until grep -q "EXTRA_DONE" $LOGS/sgr3_arms_extra.log; do sleep 60; done
TAG=R0b
echo "ARM_START $(date -Is) $TAG"
sh $ROOT/scripts/remote/sgr3_job.sh $TAG --tag _$TAG --arm none > $LOGS/arm_$TAG.log 2>&1
echo "ARM_END $(date -Is) $TAG rc=$?"
for p in $(nvidia-smi --query-compute-apps=pid -i 1 --format=csv,noheader); do
  ps -o cmd= -p $p | grep -q "VLLM::EngineCore" && kill $p
done
$PY -m lib.mllm.tools.sgr3_check_run $RUNS/predcls/qwen3vl_8b_$TAG --video_list $LIST >> $RUNS/scores/checks.jsonl
OMP_NUM_THREADS=8 $PY -m lib.mllm.eval.score_run --method rag_all --model qwen3vl_8b_$TAG --mode predcls \
    --pred_dir $RUNS --video_list $LIST --subset_tag s150 --no_legacy \
    --out $RUNS/scores/qwen3vl_8b_$TAG.json > $LOGS/score_qwen3vl_8b_$TAG.log 2>&1
echo "EXTRA2_DONE $(date -Is)"
