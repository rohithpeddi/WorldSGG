#!/bin/sh
# Phase-4 SGR3-style retrieval arms, sequential on GPU 1 (CS93371), then check + score.
#   setsid nohup sh scripts/remote/sgr3_arms.sh > /data3/rohith/ag/logs/ext/sgr3_arms.log 2>&1 < /dev/null &
# Every arm: Qwen3-VL-8B, predcls, 150-video list, temperature 0.2, default max_new_tokens (128),
# skip-verification, retrieved-context budget B tokens.  Resumable (per-video pkls).
set -u
B=${SGR3_BUDGET:-90}
ROOT=$HOME/CODE/Scene4Cast_sgr3
PY=$HOME/anaconda3/envs/wsg/bin/python
LOGS=/data3/rohith/ag/logs/ext
RUNS=/data3/rohith/ag/runs/mllm/rag_sgr3
LIST=/data3/rohith/ag/splits/test_worldbbox_thinking150.txt
mkdir -p $RUNS/scores
cd $ROOT

score() {  # <model dir name> <pred root>
  $PY -m lib.mllm.tools.sgr3_check_run $2/predcls/$1 --video_list $LIST >> $RUNS/scores/checks.jsonl
  OMP_NUM_THREADS=8 $PY -m lib.mllm.eval.score_run --method rag_all --model $1 --mode predcls \
      --pred_dir $2 --video_list $LIST --subset_tag s150 --no_legacy \
      --out $RUNS/scores/$1.json > $LOGS/score_$1.log 2>&1
}

arm() {  # <tag> <runner args...>
  TAG=$1; shift
  echo "ARM_START $(date -Is) $TAG $*"
  sh $ROOT/scripts/remote/sgr3_job.sh "$TAG" --tag "_$TAG" "$@" > $LOGS/arm_$TAG.log 2>&1
  echo "ARM_END $(date -Is) $TAG rc=$?"
  # kill a leftover engine of this GPU (runner exited, EngineCore may linger)
  for p in $(nvidia-smi --query-compute-apps=pid -i 1 --format=csv,noheader); do
    ps -o cmd= -p $p | grep -q "VLLM::EngineCore" && kill $p
  done
  score qwen3vl_8b_$TAG $RUNS &
}

# R1 as it exists (uncapped, 2026-09-20) rescored with this checkout's scorer
score qwen3vl_8b_150 /data3/rohith/ag/runs/mllm/rag_all &

arm R2k1 --arm sgr3 --k 1 --budget $B
arm R0   --arm none
arm R3   --arm hybrid --k 1 --budget $B
arm R2k3 --arm sgr3 --k 3 --budget $B
arm R2k5 --arm sgr3 --k 5 --budget $B
arm R1B  --arm bge --budget $B
wait
echo "ALL_DONE $(date -Is)"
