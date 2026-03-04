#!/bin/bash
# ===========================================================================
# run_dump_all.sh — Dump predictions for all 24 method configs
# ===========================================================================
# Usage:
#   bash run_dump_all.sh
#   bash run_dump_all.sh --min_epoch 5
# ===========================================================================

MIN_EPOCH="${1:-4}"
OUTPUT_ROOT="/data/rohith/ag/wsgg_logits"

echo "============================================================"
echo "Dumping predictions for all configs (min_epoch=${MIN_EPOCH})"
echo "Output root: ${OUTPUT_ROOT}"
echo "============================================================"

for config in configs/methods/predcls/*.yaml configs/methods/sgdet/*.yaml; do
    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "Config: ${config}"
    echo "────────────────────────────────────────────────────────"
    python dump_predictions.py \
        --config "${config}" \
        --min_epoch "${MIN_EPOCH}" \
        --output_root "${OUTPUT_ROOT}"
done

echo ""
echo "============================================================"
echo "All dumps complete."
echo "============================================================"
