#!/bin/sh
# The two context-free / caption-prefixed unlocalized runners on the sandbox split, for the
# zero_shot and caption_all figures: the headline commands of scripts/remote/b3_baselines.jobs
# (qwen25vl_7b, --skip-verification), wrapped by capture_ctx.py, one GPU, every output / cache / log
# redirected into the sandbox by $WSGG_MLLM_CONFIG.  Needs the Stage-1 graphs of run_chain.sh.
#   SB=<sandbox> GPU=<free gpu> setsid nohup sh run_ctx_chain.sh > <sandbox>/logs/ctx_chain.out 2>&1 < /dev/null &
SB=${SB:-/data3/rohith/ag/runs/intermediates/mllm}
TOOLS=${TOOLS:-$SB/tools}
PY=/home/rxp190007/anaconda3/envs/wsg/bin/python
SPLIT=$SB/split.txt
export WSGG_MLLM_CONFIG=$SB/config.yaml CUDA_VISIBLE_DEVICES=${GPU:-2} OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
cd /home/rxp190007/CODE/Scene4Cast_mllm || exit 1
LOG=$SB/logs/ctx_chain.log

step() {    # step <name> <timeout s> <command ...>
    name=$1; to=$2; shift 2
    echo "$(date '+%F %T') START $name" >> "$LOG"
    setsid timeout "$to" "$@" > "$SB/logs/$name.log" 2>&1 &
    pid=$!
    echo "$(date '+%F %T')       $name pid=$pid" >> "$LOG"
    wait "$pid"; rc=$?
    kill -9 -- "-$pid" 2>/dev/null
    echo "$(date '+%F %T') END   $name rc=$rc" >> "$LOG"
}

step zero_shot_predcls   2400 $PY $TOOLS/capture_ctx.py $SB/capture zero_shot   --model_name qwen25vl_7b --mode predcls --tensor_parallel_size 1 --skip-verification --video_list $SPLIT
step zero_shot_sgdet     2400 $PY $TOOLS/capture_ctx.py $SB/capture zero_shot   --model_name qwen25vl_7b --mode sgdet   --tensor_parallel_size 1 --skip-verification --video_list $SPLIT
step caption_all_predcls 2400 $PY $TOOLS/capture_ctx.py $SB/capture caption_all --model_name qwen25vl_7b --mode predcls --tensor_parallel_size 1 --skip-verification --video_list $SPLIT
step caption_all_sgdet   2400 $PY $TOOLS/capture_ctx.py $SB/capture caption_all --model_name qwen25vl_7b --mode sgdet   --tensor_parallel_size 1 --skip-verification --video_list $SPLIT
echo "$(date '+%F %T') CTX CHAIN DONE" >> "$LOG"
