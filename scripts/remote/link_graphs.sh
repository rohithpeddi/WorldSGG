#!/bin/sh
# Build the merged Stage-1 graph directory used by configs/mllm/server.yaml:
#   /data3/rohith/ag/cache/mllm/graphs/qwen25vl_7b/<video>.mp4.pkl
# = symlinks to the reusable /data/rohith/ag/graphs/qwen25vl_7b/*.pkl (never
# regenerated) + graphs generated later for missing test videos (real files).
set -eu
SRC=/data/rohith/ag/graphs/qwen25vl_7b
DST=/data3/rohith/ag/cache/mllm/graphs/qwen25vl_7b
mkdir -p "$DST"
n=0
for f in "$SRC"/*.pkl; do
  b=$(basename "$f")
  [ -e "$DST/$b" ] || { ln -s "$f" "$DST/$b"; n=$((n+1)); }
done
echo "linked $n new; total $(ls "$DST" | wc -l) in $DST"
