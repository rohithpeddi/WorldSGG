#!/bin/sh
# C1c — derive the ROI-feature PKLs of one split from the token caches (CPU, detached),
# for both modes and the given streams, with /data symlinks + manifest registration.
#   sh scripts/remote/run_derive.sh <split: train|test> [streams: dinov3_tok pi3_tok fused | all]
# Log: /data3/rohith/ag/logs/derive_<split>.log
SPLIT=$1; shift
STREAMS=${*:-all}
REPO=${REPO:-$HOME/CODE/Scene4Cast_worldformer}
PY=$HOME/anaconda3/envs/scene4cast/bin/python
LOG=/data3/rohith/ag/logs/derive_${SPLIT}.log
cd "$REPO" || exit 1
setsid nohup sh -c "
for st in $STREAMS; do
  echo \"[derive \$(date +%FT%T)] START $SPLIT \$st\"
  $PY tools/register_token_cache.py --stream dinov3l --split $SPLIT
  [ \"\$st\" = pi3_tok ] || [ \"\$st\" = fused ] || [ \"\$st\" = all ] && $PY tools/register_token_cache.py --stream pi3 --split $SPLIT
  $PY datasets/preprocess/tokens/derive_roi_features.py --split $SPLIT --mode all --stream \$st --workers 16 --register
  echo \"[derive \$(date +%FT%T)] END $SPLIT \$st rc=\$?\"
done
$PY tools/cache_manifest.py list
" >> "$LOG" 2>&1 < /dev/null &
echo "launched derive $SPLIT [$STREAMS] pid $! log $LOG"
