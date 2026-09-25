#!/bin/sh
# PUF adapted to monocular per-timestamp WSGG (setup/EXT_PUF.md): run every arm over the
# W-DSGDetr++ front-end dumps and score them (all frames, predcls + sgdet).  CPU only.
#
#   sh scripts/remote/ext_puf.sh smoke|s150|full
#   setsid nohup sh scripts/remote/ext_puf.sh full > /data3/rohith/ag/logs/ext/puf_full.log 2>&1 < /dev/null &
#
# Prerequisites (once): tools/ext_puf.py geom (Pi3 geometry cache) and tools/ext_puf.py prior.
# VARIANTS lines are "<name> <arm> [extra run args]"; override with the VARIANTS env var.
STAGE=${1:-s150}
REPO=/home/rxp190007/CODE/Scene4Cast_puf
PY=/home/rxp190007/anaconda3/envs/scene4cast/bin/python
ROOT=/data3/rohith/ag/runs/ext/puf
FE_DIR=/data3/rohith/ag/runs/rescore/dumps
WORKERS=${WORKERS:-12}
MODES=${MODES:-"predcls sgdet"}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd $REPO || exit 1
case $STAGE in
  smoke) VF=$ROOT/smoke5.txt ;;
  s150)  VF=/data3/rohith/ag/splits/test_worldbbox_thinking150.txt ;;
  full)  VF="" ;;
  *) echo "unknown stage $STAGE"; exit 1 ;;
esac
OUT=$ROOT/$STAGE
mkdir -p $OUT/dumps $OUT/score
DEFAULT_VARIANTS="frontend frontend
lks lks
lks_bi lks_bi
fross fross
puf puf
puf_prior puf_prior
puf_prior_vis puf_prior_vis"
VARIANTS=${VARIANTS:-$DEFAULT_VARIANTS}

for mode in $MODES; do
  FE=$FE_DIR/w_dsgdetr_pp_${mode}_resnet50__all.pkl
  REF=$OUT/dumps/ref_wdsgdetrpp_${mode}.pkl
  if [ -n "$VF" ]; then
    [ -f $REF ] || $PY tools/ext_puf.py subset --frontend $FE --videos-file $VF \
        --name ref_wdsgdetrpp_${mode} --out $REF
    VARG="--videos-file $VF"
  else
    [ -e $REF ] || ln -s $FE $REF
    VARG=""
  fi
  echo "$VARIANTS" | while read name arm extra; do
    [ -z "$name" ] && continue
    D=$OUT/dumps/puf_${name}_${mode}.pkl
    if [ -f $D ]; then echo "[ext_puf] have $D"; continue; fi
    echo "[ext_puf] $(date +%T) run $name ($arm $extra) $mode"
    $PY tools/ext_puf.py run --mode $mode --arm $arm --frontend $FE $VARG --workers $WORKERS \
        --name puf_${name}_${mode} --out $D $extra > $OUT/dumps/puf_${name}_${mode}.log 2>&1
    echo "[ext_puf] run $name $mode exit=$?"
  done
done

# score everything not yet scored (4 at a time)
n=0
for D in $OUT/dumps/*.pkl; do
  stem=$(basename $D .pkl)
  [ -f $OUT/score/bucketed_breakdown_${stem}.json ] && continue
  echo "[ext_puf] $(date +%T) score $stem"
  $PY tools/ext_puf.py score --dump $D --out-dir $OUT/score --name $stem > $OUT/score/$stem.log 2>&1 &
  n=$((n + 1))
  if [ $((n % 4)) -eq 0 ]; then wait; fi
done
wait
grep -h "\[score\]" $OUT/score/*.log
echo "[ext_puf] DONE $STAGE"
