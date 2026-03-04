#!/bin/bash
# ===========================================================================
# run_eval_all.sh — Evaluate predictions for all 24 method configs
# ===========================================================================
# Prerequisite: run run_dump_all.sh first to generate prediction PKLs.
#
# Usage:
#   bash run_eval_all.sh
#   bash run_eval_all.sh --use_wandb
# ===========================================================================

LOGIT_ROOT="/data/rohith/ag/wsgg_logits"
ANNOT_DIR="/data/rohith/ag/world4d_rel_annotations/test"
USE_WANDB=""

# Parse optional --use_wandb flag
if [[ "$1" == "--use_wandb" ]]; then
    USE_WANDB="--use_wandb"
fi

echo "============================================================"
echo "Evaluating all method predictions"
echo "Logit root:  ${LOGIT_ROOT}"
echo "Annot dir:   ${ANNOT_DIR}"
echo "WandB:       ${USE_WANDB:-disabled}"
echo "============================================================"

for mode in predcls sgdet; do
    MODE_DIR="${LOGIT_ROOT}/${mode}"
    if [ ! -d "${MODE_DIR}" ]; then
        echo "Skipping ${mode}: directory not found"
        continue
    fi

    for method_dir in "${MODE_DIR}"/*/; do
        experiment_name=$(basename "${method_dir}")
        echo ""
        echo "────────────────────────────────────────────────────────"
        echo "Mode: ${mode} | Experiment: ${experiment_name}"
        echo "────────────────────────────────────────────────────────"
        python evaluate_predictions.py \
            --logit_root "${method_dir}" \
            --annot_dir "${ANNOT_DIR}" \
            --mode "${mode}" \
            --experiment_name "${experiment_name}" \
            ${USE_WANDB}
    done
done

echo ""
echo "============================================================"
echo "All evaluations complete."
echo "============================================================"
