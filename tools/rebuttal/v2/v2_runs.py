"""V2 (round 3): configs, smoke, queue submission and scoring daemon for all 31 V2 runs (every V1 row: baselines,
WorldWise, WorldWise+, WorldWise++, seeds, logit adjustment, completion variants, ResNet-50 detector).

V2 runs from its own code folder, /data3/rohith/ag/runs/rebuttal/code_v2 (symlinked as ~/CODE/Scene4Cast_rebuttal_v2;
inside the rw output mount so cs94085 can run it), and writes to /data3/rohith/ag/runs/rebuttal/train_v2. V1 runs
from ~/CODE/Scene4Cast_rebuttal_train into .../train and is not touched. Each V2 run = the matching V1 round-2
config (WorldWise @DINOv3-L, WorldWise++ @DINOv3 tokens; full inventory, slot alignment, validation selection, SGDet
visibility off) with ``method_name: worldwise_3d`` / ``worldwise_pp_3d`` and the V2 keys below.

Scoring (CS93371, tmux ``r3_scorer``): train_rescore.py (2-D rules, V1's full-inventory GT references),
rescore_3d.py (3-D rules; "refined" = V2 boxes) in both settings, and v2_localize.py (hidden-object boxes).

  python tools/rebuttal/v2/v2_runs.py configs | list | smoke | submit [--dry] | score_daemon
"""
import argparse
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import inv_runs as iv  # noqa: E402
import train_runs as tr  # noqa: E402

REPO = tr.REPO
V2_CODE = "/data3/rohith/ag/runs/rebuttal/code_v2"
OUT = "/data3/rohith/ag/runs/rebuttal/train_v2"
GT_DIR = "/data3/rohith/ag/runs/rebuttal/train/gt"
QUEUE = tr.QUEUE
INFRA = tr.INFRA
PY = tr.PY
MANIFEST = f"{OUT}/manifest_r3.json"
V2_KEYS = {"lambda_geo": 1.0, "p_span_mask": 0.3, "max_span": 10}
# Every V1 row gets a V2 counterpart (user, Nov 11: "we should be running all the experiments completely"): the same
# 31 runs as round 2 (R1 Table 2 x 2 modes, R2 seeds, R3 logit adjustment + completion variants, R4 ResNet-50
# detector), each under the V2 rules (no box input for hidden / masked slots, inferred geometry token + decoded box,
# 3-D loss on masked items only). Host placement follows round 2 (WorldWise+ / ++ on CS93371, the rest on cs94085).
METHOD = {"worldwise": "worldwise_3d", "worldwise_pp": "worldwise_pp_3d", "worldwise_plus": "worldwise_plus_3d",
          "w_sttran": "w_sttran_3d", "w_sttran_pp": "w_sttran_pp_3d", "w_dsgdetr": "w_dsgdetr_3d",
          "w_dsgdetr_pp": "w_dsgdetr_pp_3d", "w_usg": "w_usg_3d"}
V2_TRAIN_PRIO_OFFSET = 2   # V2 trainings sort after the V1 evals (31_) in every queue, longest first


def v2_name(exp1, row):
    """Round-2 experiment name -> V2 name: r2_<row><rest> -> r3_<METHOD[row]><rest>."""
    assert exp1.startswith(f"r2_{row}"), (exp1, row)
    return f"r3_{METHOD[row]}" + exp1[len(f"r2_{row}"):]


def runs():
    """[(exp, row, mode, seed, overrides, gt space, host)] — one V2 run per round-2 run."""
    return [(v2_name(exp, row), row, mode, seed, ov, space, host)
            for rnd, exp, row, mode, seed, ov, space, host in iv.runs() if row in METHOD]


def config_path(exp, mode):
    return os.path.join(REPO, "configs", "methods", mode, "rebuttal3", f"{exp}.yaml")


def cfg_rel(exp, mode):
    return os.path.relpath(config_path(exp, mode), REPO).replace(os.sep, "/")


def build(exp, row, mode, seed, ov, save_path=None):
    src, c = iv.build(exp, row, mode, seed, ov, save_path=save_path or f"{OUT}/runs")
    c["method_name"] = METHOD[row]
    c.update(V2_KEYS)
    return src, c


def env_lines(mode):
    return iv.env_lines(mode) + "\nexport PYTHONDONTWRITEBYTECODE=1"


def cmd_configs(a):
    for exp, row, mode, seed, ov, space, host in runs():
        src, c = build(exp, row, mode, seed, ov)
        tr.write_config(config_path(exp, mode), src, c, note="; V2 (round 3, explicit geometry token)")
        print("wrote", cfg_rel(exp, mode))


def cmd_list(a):
    for r in runs():
        print(*r[:3], r[5], r[6])


def local_train_job(exp, mode, mem):
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# V2 (round 3, explicit geometry token): {exp}
set -e -o pipefail
cd {V2_CODE}
PY={PY}
{env_lines(mode)}
echo "[trn3] start $(date -Is) host=$REB_HOST gpu=$REB_GPU"
$PY train_wsgg_methods.py --config {cfg_rel(exp, mode)}
test -f {OUT}/runs/{exp}/best_model.pth
test -f {OUT}/runs/{exp}/checkpoint_19/checkpoint_state.pth
echo "[trn3] done $(date -Is)"
"""


def local_eval_job(exp, mode, mem):
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# REQUIRES=trn3_{exp}
# V2 (round 3): test dumps with V2 boxes (val-selected + final checkpoint) of {exp}; scored on CS93371
set -e -o pipefail
cd {V2_CODE}
PY={PY}
{env_lines(mode)}
$PY tools/rebuttal/v2/v2_dump.py --config {cfg_rel(exp, mode)} --out_dir {OUT}/dumps3d
date -Is > {OUT}/dumps3d/{exp}.dumped
echo "[trn3-eval] done $(date -Is)"
"""


def remote_preamble(mode):
    return f"""set -e -o pipefail
case $REB_HOST in cs94085) . /data1/rohith/wsgg/host.env ;; cs94092) . /data3/rohith/wsgg/host.env ;;
                  *) echo "V2 remote job started on $REB_HOST" >&2; exit 2 ;; esac
cd "$(hp {V2_CODE})"
{env_lines(mode)}"""


def remote_train_job(exp, mode, mem):
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# V2 (round 3, remote): {exp}
{remote_preamble(mode)}
echo "[trn3] start $(date -Is) host=$REB_HOST gpu=$REB_GPU"
CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" {cfg_rel(exp, mode)} trn_{exp})
test -s "$CFG"
"$PY" train_wsgg_methods.py --config "$CFG"
test -f "$(hp {OUT}/runs/{exp}/best_model.pth)"
test -f "$(hp {OUT}/runs/{exp}/checkpoint_19/checkpoint_state.pth)"
echo "[trn3] done $(date -Is)"
"""


def remote_eval_job(exp, mode, mem):
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# REQUIRES=trn_{exp}
# V2 (round 3, remote): test dumps with V2 boxes of {exp}; scored on CS93371
{remote_preamble(mode)}
CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" {cfg_rel(exp, mode)} trn_eval_{exp})
test -s "$CFG"
sed -i "s#^save_path: $WSGG_OUT/train_v2/runs\\$#save_path: $WSGG_RUNS_RO/rebuttal/train_v2/runs#" "$CFG"
grep -q "^save_path: $WSGG_RUNS_RO/rebuttal/train_v2/runs\\$" "$CFG"
"$PY" tools/rebuttal/v2/v2_dump.py --config "$CFG" --out_dir "$(hp {OUT}/dumps3d)"
date -Is > "$(hp {OUT}/dumps3d/{exp}.dumped)"
echo "[trn3-eval] done $(date -Is)"
"""


def write_manifest():
    man = {exp: {"row": row, "mode": mode, "seed": seed, "gt_space": space, "host": host, "config": cfg_rel(exp, mode)}
           for exp, row, mode, seed, ov, space, host in runs()}
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST + ".tmp", "w") as f:
        json.dump(man, f, indent=1)
    os.replace(MANIFEST + ".tmp", MANIFEST)


def cmd_submit(a):
    write_manifest()
    for exp, row, mode, seed, ov, space, host in runs():
        if a.only and exp not in a.only:
            continue
        host = a.force_host or host
        mem = tr.GPU_MEM.get(row, 4)
        prio = tr.priority(row, mode, "train") + V2_TRAIN_PRIO_OFFSET   # after the V1 evals (31_), longest first
        if host == "cs93371":
            tr.submit_file(f"trn3_{exp}", prio, local_train_job(exp, mode, mem), a.dry)
            tr.submit_file(f"trn3_eval_{exp}", 31, local_eval_job(exp, mode, mem), a.dry)
        else:
            iv.OUT = OUT
            iv.submit_remote(host, f"trn_{exp}", prio, remote_train_job(exp, mode, mem), a.dry)
            iv.submit_remote(host, f"trn_eval_{exp}", 31, remote_eval_job(exp, mode, mem), a.dry)


def cmd_smoke(a):
    """CS93371 GPU job: 1 epoch on 50 / 20 videos for each V2 run, then a 3-video dump and all three scorers."""
    sdir = f"{OUT}/smoke3"
    os.makedirs(f"{sdir}/configs", exist_ok=True)
    tr_ids = [v.strip() for v in open(f"{tr.SPLITS}/train_minus_val.txt") if v.strip()][:50]
    va_ids = [v.strip() for v in open(f"{tr.SPLITS}/val750.txt") if v.strip()][:20]
    open(f"{sdir}/train.txt", "w").write("\n".join(tr_ids) + "\n")
    open(f"{sdir}/val.txt", "w").write("\n".join(va_ids) + "\n")
    te = sorted(f[:-len(".mp4.pkl")] for f in os.listdir(iv.ANNOT) if f.endswith(".mp4.pkl"))[:3]
    open(f"{sdir}/test3.txt", "w").write("\n".join(te) + "\n")
    cmds = []
    for exp, row, mode, seed, ov, space, host in runs():
        if a.only and exp not in a.only:
            continue
        sexp = "smoke3_" + exp
        src, c = build(sexp, row, mode, seed, ov, save_path=f"{sdir}/runs")
        c.update({"nepoch": 1, "train_video_list": f"{sdir}/train.txt", "val_video_list": f"{sdir}/val.txt",
                  "log_every": 10})
        p = f"{sdir}/configs/{sexp}.yaml"
        tr.write_config(p, src, c, note="; V2 SMOKE")
        ref, cache = f"{GT_DIR}/predcls_{space}_full.pkl", f"{GT_DIR}/full_gt_{space}_full.pkl"
        d = f"{sdir}/dumps/{sexp}__best__all3d.pkl"
        cmds.append(
            f"echo \"=== {sexp} $(date -Is)\"; {env_lines(mode).replace(chr(10), '; ')}; "
            f"$PY train_wsgg_methods.py --config {p} > {sdir}/{sexp}.log 2>&1 && "
            f"$PY tools/rebuttal/v2/v2_dump.py --config {p} --ckpts best --out_dir {sdir}/dumps --video_list {sdir}/test3.txt >> {sdir}/{sexp}.log 2>&1 && "
            f"CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/train_rescore.py score --mode {mode} --dump {d} --ref {ref} --cache {cache} --out {sdir}/scores/{sexp}.json --allow_subset >> {sdir}/{sexp}.log 2>&1 && "
            f"CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/rescore_3d.py --dump {d} --name {sexp} --ref {ref} --cache {cache} --out {sdir}/score3d >> {sdir}/{sexp}.log 2>&1 && "
            f"CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/v2/v2_localize.py --dump {d} --out {sdir}/loc/{sexp}.json >> {sdir}/{sexp}.log 2>&1 "
            f"&& echo OK {sexp} || {{ echo FAIL {sexp}; FAILED=1; }}")
    job = f"""#!/bin/bash
# GPU_MEM_GB=8
# V2 smoke: 1 epoch / 50 videos per V2 run + 3-video dump + scorers
cd {V2_CODE}
PY={PY}
mkdir -p {sdir}/scores {sdir}/score3d {sdir}/loc
FAILED=0
{chr(10).join(cmds)}
exit $FAILED
"""
    if a.out_script:            # run outside the queue (e.g. a GPU with spare memory while every slot is taken)
        with open(a.out_script, "w", newline="\n") as f:
            f.write(job)
        os.chmod(a.out_script, 0o755)
        print("wrote", a.out_script)
        return
    tr.submit_file("trn3_smoke", 29, job, a.dry)


def score_cmds(exp, info, ck):
    py = PY.replace("~", os.path.expanduser("~"))
    ref, cache = f"{GT_DIR}/predcls_{info['gt_space']}_full.pkl", f"{GT_DIR}/full_gt_{info['gt_space']}_full.pkl"
    d = f"{OUT}/dumps3d/{exp}__{ck}__all3d.pkl"
    cmds = []
    if not os.path.exists(f"{OUT}/scores/{exp}__{ck}.json"):
        cmds.append([py, "tools/rebuttal/train_rescore.py", "score", "--mode", info["mode"], "--dump", d, "--ref", ref,
                     "--cache", cache, "--out", f"{OUT}/scores/{exp}__{ck}.json"])
    if not os.path.exists(f"{OUT}/score3d/{exp}__{ck}.json"):
        cmds.append([py, "tools/rebuttal/rescore_3d.py", "--dump", d, "--name", f"{exp}__{ck}", "--ref", ref,
                     "--cache", cache, "--out", f"{OUT}/score3d"])
    if not os.path.exists(f"{OUT}/loc/{exp}__{ck}.json"):
        cmds.append([py, "tools/rebuttal/v2/v2_localize.py", "--dump", d, "--out", f"{OUT}/loc/{exp}__{ck}.json"])
    return cmds


def cmd_score_daemon(a):
    import time
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    running, failed = {}, set()
    os.makedirs(f"{OUT}/logs", exist_ok=True)
    while True:
        man = json.load(open(MANIFEST))
        todo = []
        for m in sorted(glob.glob(f"{OUT}/dumps3d/*.dumped")):
            exp = os.path.basename(m)[:-len(".dumped")]
            if exp not in man:
                continue
            for ck in ("best", "checkpoint_19"):
                for c in score_cmds(exp, man[exp], ck):
                    key = (exp, ck, c[1])
                    if key not in running and key not in failed:
                        todo.append((key, c))
        for key, p in list(running.items()):
            rc = p.poll()
            if rc is not None:
                del running[key]
                print(f"[score3-daemon] {time.strftime('%H:%M:%S')} {key} rc={rc}", flush=True)
                if rc != 0:
                    failed.add(key)
        for key, c in todo:
            if len(running) >= a.workers:
                break
            log = open(f"{OUT}/logs/score_{key[0]}__{key[1]}__{os.path.basename(key[2])[:-3]}.log", "a")
            running[key] = subprocess.Popen(c, cwd=V2_CODE, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"[score3-daemon] {time.strftime('%H:%M:%S')} start {key}", flush=True)
        if a.once and not running and not todo:
            break
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    sub.add_parser("list")
    p = sub.add_parser("smoke")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--out_script", default=None)
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("submit")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--force_host", default=None)
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("score_daemon")
    p.add_argument("--once", action="store_true")
    p.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"configs": cmd_configs, "list": cmd_list, "smoke": cmd_smoke, "submit": cmd_submit,
     "score_daemon": cmd_score_daemon}[a.cmd](a)
