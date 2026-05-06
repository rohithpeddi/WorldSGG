#!/bin/bash
# ============================================================================
# WorldWise Ablation Experiments — Batch Runner
# ============================================================================
# Runs all WorldWise ablation experiments (removing one component at a time)
# across all 3 Dino backbones and both evaluation modes.
#
# Usage:
#   bash scripts/run_ablations.sh
#   bash scripts/run_ablations.sh predcls    # Run only PredCls ablations
#   bash scripts/run_ablations.sh sgdet      # Run only SGDet ablations
# ============================================================================

set -e

MODE_FILTER="${1:-}"  # Optional: "predcls" or "sgdet" to filter

echo "========================================="
echo "WorldWise Ablation Experiments"
echo "========================================="

# PredCls ablations
if [ -z "$MODE_FILTER" ] || [ "$MODE_FILTER" = "predcls" ]; then
    echo ""
    echo "--- PredCls Ablations ---"
    for cfg in configs/methods/predcls/worldwise_no_*.yaml; do
        echo ""
        echo "Running: $cfg"
        python train_wsgg_methods.py --config "$cfg"
    done
fi

# SGDet ablations
if [ -z "$MODE_FILTER" ] || [ "$MODE_FILTER" = "sgdet" ]; then
    echo ""
    echo "--- SGDet Ablations ---"
    for cfg in configs/methods/sgdet/worldwise_no_*.yaml; do
        echo ""
        echo "Running: $cfg"
        python train_wsgg_methods.py --config "$cfg"
    done
fi

echo ""
echo "========================================="
echo "All ablation experiments complete!"
echo "========================================="
