#!/bin/bash
# INFRA gate (b): WorldWise++ PredCls training smoke through the sshfs mounts. N training videos (every 150th of the
# sorted dinov3_tok train list), 1 epoch, end-of-epoch evaluation on 10 other training videos (val_video_list, so
# the test split is not loaded), with a 10-s monitor of the trainer's process states, GPU and NIC receive rate.
#   CUDA_VISIBLE_DEVICES=<g> bash infra_smoke_pp.sh [N=50] [tag=a] [stride=150] [offset=7] [config]
# (train videos: line numbers = offset mod stride; use a new offset to measure uncached reads)
# Outputs: checkpoint -> CS93371 /data3/rohith/ag/runs/rebuttal/infra/smoke/; log + monitor -> $WSGG_LOCAL/infra_smoke_<tag>/
set -e
case ${REB_HOST:-$(hostname -s)} in
  cs94085) . /data1/rohith/wsgg/host.env ;;
  cs94092) . /data3/rohith/wsgg/host.env ;;
  *) . /data3/rohith/ag/runs/rebuttal/infra/host_cs93371.env ;;
esac
N=${1:-50}; TAG=${2:-a}; STRIDE=${3:-150}; OFF=${4:-7}
CONF=${5:-configs/methods/predcls/worldwise_pp_dinov3_predcls.yaml}
L=$WSGG_LOCAL/infra_smoke_$TAG; mkdir -p "$L"
cd "$WSGG_CODE"
ls "$WSGG_DATA/features/roi_features/predcls/dinov3_tok/train" | sed 's/\.pkl$//' | LC_ALL=C sort > "$L/all_train.txt"
awk -v s="$STRIDE" -v o="$OFF" 'NR%s==o' "$L/all_train.txt" | head -n "$N" > "$L/train.txt"
awk -v o="$OFF" 'NR%150==(o+77)%150' "$L/all_train.txt" | grep -vxFf "$L/train.txt" | head -n 10 > "$L/val.txt"
CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" "$CONF" "infra_smoke_pp_$TAG")
printf 'train_video_list: %s\nval_video_list: %s\n' "$L/train.txt" "$L/val.txt" >> "$CFG"
[ -n "${EXTRA_YAML:-}" ] && printf '%b\n' "$EXTRA_YAML" >> "$CFG"   # e.g. EXTRA_YAML='num_workers: 4\nuse_wandb: false'
SAVE=$(hp /data3/rohith/ag/runs/rebuttal/infra/smoke)
NIC=$(ip route get 10.176.140.104 2>/dev/null | grep -o 'dev [^ ]*' | cut -d' ' -f2); NIC=${NIC:-eno1}
echo "[$(date '+%F %T')] smoke $TAG on $REB_SITE gpu ${CUDA_VISIBLE_DEVICES:-?}: $(wc -l < "$L/train.txt") train / $(wc -l < "$L/val.txt") val videos, nic $NIC"
t0=$(date +%s)
"$PY" train_wsgg_methods.py --config "$CFG" --nepoch 1 --save_path "$SAVE" \
  --experiment_name "infra_smoke_pp_predcls_${REB_SITE}_$TAG" > "$L/train.log" 2>&1 &
TP=$!
(
  prev=$(cat /sys/class/net/$NIC/statistics/rx_bytes)
  while kill -0 $TP 2>/dev/null; do
    sleep 10
    cur=$(cat /sys/class/net/$NIC/statistics/rx_bytes)
    st=$(ps -o stat= -p $TP --ppid $TP 2>/dev/null | cut -c1 | sort | uniq -c | tr -s ' ' | tr '\n' ' ')
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES%%,*}" 2>/dev/null)
    echo "$(date +%T) +$(( $(date +%s) - t0 ))s rx=$(( (cur - prev) / 10 / 1048576 ))MB/s states=[${st}] gpu_util,mem=[$g] load=$(cut -d' ' -f1 /proc/loadavg)"
    prev=$cur
  done
) > "$L/monitor.log" 2>&1 &
wait $TP; rc=$?
echo "[$(date '+%F %T')] smoke $TAG rc=$rc wall=$(( $(date +%s) - t0 ))s"
tr '\r' '\n' < "$L/train.log" | grep -E "Epoch 1/1|Evaluating|video_ids filter|grids from|Train:|DataLoader" | tail -8
exit $rc
