#!/bin/bash
# INFRA (rebuttal): rewrite a WSGG config's CS93371 paths for the host a job runs on. Run on that host after sourcing
# its host.env (which defines hp); prints the path of the rewritten copy (the input is never modified):
#
#   CFG=$(bash $WSGG_INFRA/infra_hostcfg.sh configs/methods/predcls/worldwise_pp_dinov3_predcls.yaml [<tag>])
#   $PY train_wsgg_methods.py --config "$CFG"
#
# Every "key: <absolute path>" value is mapped with hp (host.env): /data/rohith/ag -> $WSGG_DATA,
# /data3/rohith/ag/runs/rebuttal -> $WSGG_OUT (rw), other /data3/rohith/ag/runs -> $WSGG_RUNS_RO (read-only: fine
# for evaluation, a training save_path must be under /data3/rohith/ag/runs/rebuttal/), pp_grids -> $WSGG_GRIDS.
# A /data, /data3 or /code value that no mount covers (left unchanged) is reported on stderr.
# Output: $WSGG_INFRA/cfg/<tag or config basename>.yaml (temp file + mv).
set -eu
IN=${1:?config}; TAG=${2:-$(basename "$IN" .yaml)}
: "${WSGG_INFRA:?source host.env first}"
type hp >/dev/null 2>&1 || { echo "infra_hostcfg: hp() missing, source host.env in this shell" >&2; exit 2; }
mkdir -p "$WSGG_INFRA/cfg"
OUT=$WSGG_INFRA/cfg/$TAG.yaml
: > "$OUT.tmp.$$"
while IFS= read -r line || [ -n "$line" ]; do
  if [[ "$line" =~ ^([A-Za-z0-9_]+):[[:space:]]+[\"\']?(/[^\"\' ]*)[\"\']?[[:space:]]*$ ]]; then
    k=${BASH_REMATCH[1]}; v=${BASH_REMATCH[2]}; nv=$(hp "$v")
    [ "$nv" = "$v" ] && [ "$REB_SITE" != cs93371 ] && case "$nv" in /data/*|/data3/*|/code/*) echo "infra_hostcfg: $k: $nv was not mapped (no mount covers it on $REB_SITE)" >&2 ;; esac
    line="$k: $nv"
  fi
  printf '%s\n' "$line" >> "$OUT.tmp.$$"
done < "$IN"
mv "$OUT.tmp.$$" "$OUT"
echo "$OUT"
