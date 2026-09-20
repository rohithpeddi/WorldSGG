#!/usr/bin/env python3
"""
Queue an explicit list of training configs over a set of GPUs
(``--per-gpu`` concurrent slots per GPU), e.g. the six WorldFormer C1 cells:

    python tools/run_configs_multigpu.py --gpus 1 2 --per-gpu 2 \
        --configs configs/methods/predcls/worldformer_c1_*_predcls.yaml \
                  configs/methods/sgdet/worldformer_c1_*_sgdet.yaml

Each run: ``python train_wsgg_methods.py --config <cfg>`` with CUDA_VISIBLE_DEVICES
pinned, stdout/stderr -> logs/grid/<experiment>.log; completed runs (metrics jsonl
holding the final epoch) are skipped. Detach the launcher itself with setsid nohup.
"""
import argparse
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _completed(cfg: dict) -> bool:
    p = os.path.join(REPO, cfg.get("results_path", "results"), f"{cfg['experiment_name']}_metrics.jsonl")
    if not os.path.exists(p):
        return False
    last = None
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return bool(last) and int(last.get("epoch", -1)) >= int(cfg.get("nepoch", 20)) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--gpus", nargs="+", type=int, required=True)
    ap.add_argument("--per-gpu", type=int, default=2)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--no-skip-completed", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfgs = []
    for pat in args.configs:
        cfgs.extend(sorted(glob.glob(pat)) or [pat])
    q: "queue.Queue[str]" = queue.Queue()
    for c in cfgs:
        with open(c) as f:
            cfg = yaml.safe_load(f)
        if not args.no_skip_completed and _completed(cfg):
            print(f"[skip] {cfg['experiment_name']} already complete")
            continue
        q.put(c)
    print(f"{q.qsize()} runs queued on GPUs {args.gpus} x {args.per_gpu}")
    if args.dry_run:
        return
    os.makedirs(os.path.join(REPO, "logs", "grid"), exist_ok=True)
    status_path = os.path.join(REPO, "results", "grid_run_status.csv")

    def worker(gpu: int, slot: int):
        while True:
            try:
                c = q.get_nowait()
            except queue.Empty:
                return
            with open(c) as f:
                name = yaml.safe_load(f)["experiment_name"]
            log = os.path.join(REPO, "logs", "grid", f"{name}.log")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            t0 = time.time()
            print(f"[gpu{gpu}/slot{slot}] START {name}", flush=True)
            with open(log, "a") as lf:
                rc = subprocess.call([args.python, "train_wsgg_methods.py", "--config", c],
                                     cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
            with open(status_path, "a") as sf:
                sf.write(f"{time.strftime('%FT%T')},{name},gpu{gpu},rc={rc},{(time.time() - t0) / 60:.1f}min\n")
            print(f"[gpu{gpu}/slot{slot}] END {name} rc={rc} ({(time.time() - t0) / 60:.1f} min)", flush=True)

    threads = [threading.Thread(target=worker, args=(g, s), daemon=True)
               for g in args.gpus for s in range(args.per_gpu)]
    for t in threads:
        t.start()
        time.sleep(5)
    for t in threads:
        t.join()
    print("all runs finished")


if __name__ == "__main__":
    main()
