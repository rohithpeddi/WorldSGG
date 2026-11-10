#!/bin/bash
# GPU_MEM_GB=2
# LOCK=infra_selftest
# INFRA (rebuttal): template of a host-portable queue job, also the queue self-test. The same file runs on CS93371,
# cs94085 and cs94092: it picks the host's paths from host.env by $REB_HOST (set by the worker). Submit from CS93371:
#   bash /data3/rohith/ag/runs/rebuttal/infra/infra_submit.sh cs94085 infra_job_example.sh 60 infra_selftest_1
# Headers: GPU_MEM_GB (worker starts the job only with that + headroom free), optional REQUIRES=<job>, and LOCK=<name>
# (infra_gpuq_worker.sh: one running job per lock; WorldWise++ training/eval must use  # LOCK=pp_grids).
set -e
case ${REB_HOST:-$(hostname -s)} in
  cs94085) . /data1/rohith/wsgg/host.env ;;
  cs94092) . /data3/rohith/wsgg/host.env ;;
  *) . /data3/rohith/ag/runs/rebuttal/infra/host_cs93371.env ;;
esac
cd "$WSGG_CODE"
OUT=$(hp /data3/rohith/ag/runs/rebuttal/infra/selftest); mkdir -p "$OUT"     # = CS93371 path, on every host
# A training job would continue like this (configs keep their CS93371 paths; infra_hostcfg.sh maps them):
#   CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" configs/methods/predcls/rebuttal/<cfg>.yaml <unique tag>)
#   "$PY" train_wsgg_methods.py --config "$CFG"      # save_path must be under /data3/rohith/ag/runs/rebuttal/
"$PY" -c "import torch, socket, os; print(socket.gethostname(), os.environ.get('CUDA_VISIBLE_DEVICES'), torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.mem_get_info()[0]/2**30, 1), 'GB free')" \
  | tee "$OUT/${REB_SITE}_gpu${REB_GPU:-x}_$(date +%H%M%S).txt"
sleep "${SELFTEST_SLEEP:-40}"
echo "selftest ok"
