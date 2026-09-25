#!/bin/sh
# MLLM figure intermediates for the sandbox split: the headline commands of scripts/remote/b3_baselines.jobs,
# on one GPU, with every output / cache / log redirected into the sandbox by $WSGG_MLLM_CONFIG.
#   SB=<sandbox> GPU=<free gpu> setsid nohup sh run_chain.sh > <sandbox>/logs/chain.out 2>&1 < /dev/null &
SB=${SB:-/data3/rohith/ag/runs/intermediates/mllm}
TOOLS=${TOOLS:-$SB/tools}
PY=/home/rxp190007/anaconda3/envs/wsg/bin/python
SPLIT=$SB/split.txt
export WSGG_MLLM_CONFIG=$SB/config.yaml CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd /home/rxp190007/CODE/Scene4Cast_mllm || exit 1
LOG=$SB/logs/chain.log

step() {    # step <name> <timeout s> <command ...>
    name=$1; to=$2; shift 2
    echo "$(date '+%F %T') START $name" >> "$LOG"
    # own session => own process group; a VLLM::EngineCore left behind is reaped with the group
    setsid timeout "$to" "$@" > "$SB/logs/$name.log" 2>&1 &
    pid=$!
    echo "$(date '+%F %T')       $name pid=$pid" >> "$LOG"
    wait "$pid"; rc=$?
    kill -9 -- "-$pid" 2>/dev/null
    echo "$(date '+%F %T') END   $name rc=$rc" >> "$LOG"
}

step graphs          2400 $PY -m lib.mllm.methods.graphs.runner --model_name qwen25vl_7b --tensor_parallel_size 1 --fast --video_list $SPLIT
step rag_predcls     2400 $PY $TOOLS/capture_rag.py $SB/capture --model_name qwen25vl_7b --mode predcls --tensor_parallel_size 1 --skip-verification --video_list $SPLIT
step rag_sgdet       2400 $PY $TOOLS/capture_rag.py $SB/capture --model_name qwen25vl_7b --mode sgdet --tensor_parallel_size 1 --skip-verification --video_list $SPLIT
step track_a_predcls 2400 $PY -m lib.mllm.methods.track_a_prompt.runner --model_name qwen3vl_8b --mode predcls --max_new_tokens 1024 --temperature 0.2 --video_list $SPLIT --status $SB/logs/track_a_predcls.status.json
step track_a_sgdet   2400 $PY -m lib.mllm.methods.track_a_prompt.runner --model_name qwen3vl_8b --mode sgdet --max_new_tokens 1024 --temperature 0.2 --video_list $SPLIT --status $SB/logs/track_a_sgdet.status.json
step track_b_predcls 2400 $PY -m lib.mllm.methods.track_b_agent.runner --model_name qwen3vl_8b --mode predcls --max_new_tokens 1024 --temperature 0.2 --video_list $SPLIT --status $SB/logs/track_b_predcls.status.json
step track_b_sgdet   2400 $PY -m lib.mllm.methods.track_b_agent.runner --model_name qwen3vl_8b --mode sgdet --max_new_tokens 1024 --temperature 0.2 --video_list $SPLIT --status $SB/logs/track_b_sgdet.status.json
echo "$(date '+%F %T') CHAIN DONE" >> "$LOG"
