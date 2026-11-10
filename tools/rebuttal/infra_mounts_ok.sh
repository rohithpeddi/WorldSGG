#!/bin/bash
# INFRA (rebuttal): exit 0 iff every sshfs mount from CS93371 on this host is mounted and answers (each probe <= 10 s).
#   bash infra_mounts_ok.sh [mount_root=/data1/rohith/wsgg/mnt]      (used as PRECHECK by infra_gpuq_worker.sh)
M=${1:-/data1/rohith/wsgg/mnt}
for p in data/features/roi_features grids/test env/bin/python code/wsgg_base.py runs reb; do
  timeout 10 stat -t "$M/$p" >/dev/null 2>&1 || { echo "mount check failed: $M/$p"; exit 1; }
done
for n in data grids env runs code reb; do
  mountpoint -q "$M/$n" || { echo "not a mountpoint: $M/$n"; exit 1; }
done
echo "mounts ok"
