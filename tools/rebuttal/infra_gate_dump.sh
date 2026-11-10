#!/bin/bash
# INFRA gate (a): dump 3 test videos on this host with the saved best checkpoints (read through the mounts) and
# write the dumps to CS93371 (/data3/rohith/ag/runs/rebuttal/infra/gate/), to be compared there with
# infra_gate_compare.py against the saved full-split dumps. Usage (GPU chosen by CUDA_VISIBLE_DEVICES):
#   bash infra_gate_dump.sh [w_dsgdetr_pp] [worldwise_pp]
set -e
case ${REB_HOST:-$(hostname -s)} in
  cs94085) . /data1/rohith/wsgg/host.env ;;
  cs94092) . /data3/rohith/wsgg/host.env ;;
  *) . /data3/rohith/ag/runs/rebuttal/infra/host_cs93371.env ;;
esac
cd "$WSGG_CODE"
OUTD=$(hp /data3/rohith/ag/runs/rebuttal/infra/gate); mkdir -p "$OUTD"
VIDS="015XE 02DPI 03PRW"
for m in ${@:-w_dsgdetr_pp worldwise_pp}; do
  case $m in
    w_dsgdetr_pp) cfg=configs/methods/predcls/w_dsgdetr_pp_predcls_resnet50.yaml ;;
    worldwise_pp) cfg=configs/methods/predcls/worldwise_pp_dinov3_predcls.yaml ;;
    *) echo "unknown gate model $m"; exit 2 ;;
  esac
  CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" "$cfg" "gate_${m}_predcls")
  echo "[$(date '+%F %T')] gate $m on $REB_SITE gpu ${CUDA_VISIBLE_DEVICES:-?} cfg $CFG"
  grep -E '^(data_path|save_path|grid_cache_root):' "$CFG"
  t0=$(date +%s)
  "$PY" tools/dump_predictions.py --config "$CFG" --ckpt best --frames all --videos $VIDS \
    --out "$OUTD/${m}_predcls_3vid_${REB_SITE}.pkl"
  echo "[$(date '+%F %T')] gate $m done in $(( $(date +%s) - t0 )) s"
done
