"""Round 2 (full object inventory, Nov 10 2026): configs, smoke, queue submission and scoring daemon.

Same 31 runs as round 1 (tools/rebuttal/train_runs.py: registry, base configs, validation selection, slot alignment,
SGDet visibility off), renamed ``r2_*``, with the loader's ``inventory: full`` (every object gets a slot; missing
objects are natural masks; see dataloader/world_ag_dataset.py INVENTORIES) and, in SGDet, ``held_min_score: 0.2``.
The loss code is unchanged. Round-1 outputs were moved to /data3/rohith/ag/runs/rebuttal/train_bak.

Placement: WorldWise+ / WorldWise++ on CS93371 (token streams on /data3, grid cache over the 1 GbE link); the
ROI-feature-only rows (baselines, WorldWise and its controls) on cs94085 through the sshfs mounts (nothing is copied;
checkpoints, dumps and scores are written to CS93371). Pending remote jobs can be moved to cs94092 when its GPUs free.

Each run = a training job ``trn2_<exp>`` and an evaluation job ``trn2_eval_<exp>`` (REQUIRES the training) that
writes one test dump per checkpoint (val-selected ``best`` and final ``checkpoint_19``) with the 3-D boxes kept,
then a marker ``<exp>.dumped``. Scoring is CPU-only and runs on CS93371 (``score_daemon``): 2-D rules
(train_rescore.py, full-inventory GT reference of the run's feature box space) for every run, plus 3-D rules
(rescore_3d.py) for SGDet.

  python tools/rebuttal/inv_runs.py configs
  python tools/rebuttal/inv_runs.py list
  python tools/rebuttal/inv_runs.py smoke --mode predcls|sgdet            # server: CS93371 GPU smoke job
  python tools/rebuttal/inv_runs.py submit [--hosts cs93371 cs94085] [--only exp ...] [--dry]
  python tools/rebuttal/inv_runs.py score_daemon [--once] [--workers 3]   # server, CS93371
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_runs as tr  # noqa: E402

REPO = tr.REPO
OUT = tr.OUT                       # /data3/rohith/ag/runs/rebuttal/train (fresh for round 2)
QUEUE = tr.QUEUE
INFRA = tr.INFRA
SERVER_REPO = tr.SERVER_REPO
PY = tr.PY
ANNOT = "/data/rohith/ag/world4d_rel_annotations_worldbbox/test"
HELD_MIN_SCORE = 0.2
LOCAL_ROWS = ("worldwise_pp", "worldwise_plus")
MANIFEST = f"{OUT}/manifest_r2.json"


def runs():
    """[(round, exp, row, mode, seed, overrides, gt space, host)]"""
    out = []
    for rnd, exp, row, mode, seed, ov in tr.run_list():
        exp2 = "r2_" + exp[len("reb_"):]
        host = "cs93371" if row in LOCAL_ROWS else "cs94085"
        out.append((rnd, exp2, row, mode, seed, ov, tr.gt_space(row, ov), host))
    return out


def config_path(exp, mode):
    return os.path.join(REPO, "configs", "methods", mode, "rebuttal2", f"{exp}.yaml")


def cfg_rel(exp, mode):
    return os.path.relpath(config_path(exp, mode), REPO).replace(os.sep, "/")


def build(exp, row, mode, seed, ov, save_path=None):
    src, c = tr.build_config(exp, row, mode, seed, ov, save_path=save_path or f"{OUT}/runs")
    c["inventory"] = "full"
    if mode == "sgdet":
        c["held_min_score"] = HELD_MIN_SCORE
    return src, c


def env_lines(mode):
    return tr.env_lines(mode) + "\nexport WSGG_INVENTORY=full"


def ref_paths(space):
    return f"{OUT}/gt/predcls_{space}_full.pkl", f"{OUT}/gt/full_gt_{space}_full.pkl"


# ----------------------------------------------------------------------------------------------------------------
def cmd_configs(a):
    for rnd, exp, row, mode, seed, ov, space, host in runs():
        src, c = build(exp, row, mode, seed, ov)
        tr.write_config(config_path(exp, mode), src, c, note=f"; round 2 (full inventory), {rnd}")
        print("wrote", cfg_rel(exp, mode))


def cmd_list(a):
    for rnd, exp, row, mode, seed, ov, space, host in runs():
        print(f"{rnd} {exp:38s} {mode:7s} seed={seed} gt={space:10s} host={host} "
              f"{ {k: v for k, v in ov.items() if not k.startswith('_')} }")


# ----------------------------------------------------------------------------------------------------------------
SUFFIX = ""          # job-name suffix for resubmissions (e.g. "_hf" for the held-frame fix reruns)


def local_train_job(exp, mode, mem, requires):
    req = f"# REQUIRES={requires}\n" if requires else ""
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
{req}# ROUND 2 (full inventory): {exp}
set -e -o pipefail
cd {SERVER_REPO}
PY={PY}
{env_lines(mode)}
echo "[trn2] start $(date -Is) host=$REB_HOST gpu=$REB_GPU"
$PY train_wsgg_methods.py --config {cfg_rel(exp, mode)}
test -f {OUT}/runs/{exp}/best_model.pth
test -f {OUT}/runs/{exp}/checkpoint_19/checkpoint_state.pth
echo "[trn2] done $(date -Is)"
"""


def local_eval_job(exp, mode, mem):
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# REQUIRES=trn2_{exp}{SUFFIX}
# ROUND 2 (full inventory): test dumps with 3-D boxes (val-selected + final checkpoint) of {exp}; scored on CS93371
set -e -o pipefail
cd {SERVER_REPO}
PY={PY}
{env_lines(mode)}
$PY tools/rebuttal/train_dump.py --keep_3d --config {cfg_rel(exp, mode)} --ckpts best checkpoint_19 --out_dir {OUT}/dumps3d
test -s {OUT}/dumps3d/{exp}__best__all3d.pkl
test -s {OUT}/dumps3d/{exp}__checkpoint_19__all3d.pkl
date -Is > {OUT}/dumps3d/{exp}.dumped
echo "[trn2-eval] done $(date -Is)"
"""


def remote_preamble(mode):
    return f"""set -e -o pipefail
case $REB_HOST in cs94085) . /data1/rohith/wsgg/host.env ;; cs94092) . /data3/rohith/wsgg/host.env ;;
                  *) echo "ROUND 2 remote job started on $REB_HOST" >&2; exit 2 ;; esac
cd "$WSGG_CODE"
{env_lines(mode)}"""


def remote_train_job(exp, mode, mem):
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# ROUND 2 (full inventory, remote): {exp}
{remote_preamble(mode)}
echo "[trn2] start $(date -Is) host=$REB_HOST gpu=$REB_GPU"
CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" {cfg_rel(exp, mode)} trn2_{exp}{SUFFIX})
test -s "$CFG"
"$PY" train_wsgg_methods.py --config "$CFG"
test -f "$(hp {OUT}/runs/{exp}/best_model.pth)"
test -f "$(hp {OUT}/runs/{exp}/checkpoint_19/checkpoint_state.pth)"
echo "[trn2] done $(date -Is)"
"""


def remote_eval_job(exp, mode, mem):
    # checkpoints are read through the read-only runs mount (big reads off the rw output mount, INFRA.md 9)
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# REQUIRES=trn_{exp}{SUFFIX}
# ROUND 2 (full inventory, remote): test dumps with 3-D boxes of {exp}; scored on CS93371
{remote_preamble(mode)}
CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" {cfg_rel(exp, mode)} trn2_eval_{exp}{SUFFIX})
test -s "$CFG"
sed -i "s#^save_path: $WSGG_OUT/train/runs\\$#save_path: $WSGG_RUNS_RO/rebuttal/train/runs#" "$CFG"
grep -q "^save_path: $WSGG_RUNS_RO/rebuttal/train/runs\\$" "$CFG"
"$PY" tools/rebuttal/train_dump.py --keep_3d --config "$CFG" --ckpts best checkpoint_19 --out_dir "$(hp {OUT}/dumps3d)"
test -s "$(hp {OUT}/dumps3d/{exp}__best__all3d.pkl)"
test -s "$(hp {OUT}/dumps3d/{exp}__checkpoint_19__all3d.pkl)"
date -Is > "$(hp {OUT}/dumps3d/{exp}.dumped)"
echo "[trn2-eval] done $(date -Is)"
"""


def submit_local(name, prio, text, dry):
    return tr.submit_file(name, prio, text, dry)


def submit_remote(host, name, prio, text, dry):
    jd = f"{OUT}/jobs_remote"
    os.makedirs(jd, exist_ok=True)
    path = f"{jd}/{name}.sh"
    with open(path, "w", newline="\n") as f:
        f.write(text)
    if dry:
        print(f"[dry] {host}: {path}")
        return
    r = subprocess.run(["bash", f"{INFRA}/infra_submit.sh", host, path, f"{prio:02d}", name],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())


def write_manifest():
    man = {exp: {"round": rnd, "row": row, "mode": mode, "seed": seed, "gt_space": space, "host": host,
                 "config": cfg_rel(exp, mode)}
           for rnd, exp, row, mode, seed, ov, space, host in runs()}
    os.makedirs(OUT, exist_ok=True)
    with open(MANIFEST + ".tmp", "w") as f:
        json.dump(man, f, indent=1)
    os.replace(MANIFEST + ".tmp", MANIFEST)
    return man


def cmd_submit(a):
    global SUFFIX
    SUFFIX = a.suffix
    write_manifest()
    for rnd, exp, row, mode, seed, ov, space, host in runs():
        if a.only and exp not in a.only:
            continue
        if a.mode and mode != a.mode:
            continue
        host = a.force_host or host
        if host not in a.hosts:
            continue
        if not os.path.exists(config_path(exp, mode)):
            sys.exit(f"missing config {cfg_rel(exp, mode)}: run `configs` first")
        mem = tr.GPU_MEM.get(row, 4)
        prio = tr.priority(row, mode, "train", rnd)
        if host == "cs93371":
            req = None if a.no_smoke_dep else f"trn2_smoke_{mode}"
            submit_local(f"trn2_{exp}{SUFFIX}", prio, local_train_job(exp, mode, mem, req), a.dry)
            submit_local(f"trn2_eval_{exp}{SUFFIX}", 31, local_eval_job(exp, mode, mem), a.dry)
        else:
            # remote queues accept trn_/inf_/infra_ names only: trn_r2_<..> / trn_eval_r2_<..>
            submit_remote(host, f"trn_{exp}{SUFFIX}", prio, remote_train_job(exp, mode, mem), a.dry)
            submit_remote(host, f"trn_eval_{exp}{SUFFIX}", 31, remote_eval_job(exp, mode, mem), a.dry)


# ----------------------------------------------------------------------------------------------------------------
def cmd_smoke(a):
    """CS93371 GPU job: 1 epoch on 50 train-minus-val videos (select on 20 val videos) for every run variant of one
    mode, then a 3-video test dump (3-D boxes kept) + 2-D score (+ 3-D score for SGDet) of two variants."""
    sdir = f"{OUT}/smoke2_{a.mode}"
    os.makedirs(f"{sdir}/configs", exist_ok=True)
    tr_ids = [v.strip() for v in open(f"{tr.SPLITS}/train_minus_val.txt") if v.strip()][:a.n_train]
    va_ids = [v.strip() for v in open(f"{tr.SPLITS}/val750.txt") if v.strip()][:a.n_val]
    open(f"{sdir}/train.txt", "w").write("\n".join(tr_ids) + "\n")
    open(f"{sdir}/val.txt", "w").write("\n".join(va_ids) + "\n")
    te = sorted(f[:-len(".mp4.pkl")] for f in os.listdir(ANNOT) if f.endswith(".mp4.pkl"))[:3]
    open(f"{sdir}/test3.txt", "w").write("\n".join(te) + "\n")
    cmds, dump_cmds, seen = [], [], set()
    for rnd, exp, row, mode, seed, ov, space, host in runs():
        if mode != a.mode or seed != 0:
            continue
        key = (row, tuple(sorted((k, str(v)) for k, v in ov.items())))
        if key in seen:
            continue
        seen.add(key)
        sexp = "smoke2_" + exp
        src, c = build(sexp, row, mode, seed, ov, save_path=f"{sdir}/runs")
        c.update({"nepoch": 1, "train_video_list": f"{sdir}/train.txt", "val_video_list": f"{sdir}/val.txt",
                  "log_every": 10})
        p = f"{sdir}/configs/{sexp}.yaml"
        tr.write_config(p, src, c, note="; ROUND 2 SMOKE")
        cmds.append(f"echo \"=== {sexp} $(date -Is)\"; $PY train_wsgg_methods.py --config {p} "
                    f"> {sdir}/{sexp}.log 2>&1 && echo OK {sexp} || {{ echo FAIL {sexp}; FAILED=1; }}")
        if row in ("w_dsgdetr_pp", "worldwise_pp") and not ov:
            ref, cache = ref_paths(space)
            d = f"{sdir}/dumps/{sexp}__best__all3d.pkl"
            line = (f"$PY tools/rebuttal/train_dump.py --keep_3d --config {p} --ckpts best --out_dir {sdir}/dumps "
                    f"--video_list {sdir}/test3.txt > {sdir}/{sexp}.dump.log 2>&1 && "
                    f"CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/train_rescore.py score --mode {mode} --dump {d} "
                    f"--ref {ref} --cache {cache} --out {sdir}/scores/{sexp}.json --allow_subset "
                    f">> {sdir}/{sexp}.dump.log 2>&1")
            if mode == "sgdet":
                line += (f" && CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/rescore_3d.py --dump {d} --name {sexp} "
                         f"--ref {ref} --cache {cache} --out {sdir}/score3d "
                         f">> {sdir}/{sexp}.dump.log 2>&1")
            dump_cmds.append(line + f" && echo OK dump+score {sexp} || {{ echo FAIL dump/score {sexp}; FAILED=1; }}")
    job = f"""#!/bin/bash
# GPU_MEM_GB=8
# ROUND 2 smoke ({a.mode}): 1 epoch / {a.n_train} videos per run variant, then dump + score of two variants
cd {SERVER_REPO}
PY={PY}
{env_lines(a.mode)}
FAILED=0
{chr(10).join(cmds)}
for i in $(seq 1 240); do ls {OUT}/gt/full_gt_*_full.pkl 2>/dev/null | wc -l | grep -q '^3$' && break; sleep 30; done
mkdir -p {sdir}/scores {sdir}/score3d
{chr(10).join(dump_cmds)}
exit $FAILED
"""
    submit_local(f"trn2_smoke_{a.mode}", 30, job, a.dry)


# ----------------------------------------------------------------------------------------------------------------
def score_cmds(exp, info, ck):
    ref, cache = ref_paths(info["gt_space"])
    d = f"{OUT}/dumps3d/{exp}__{ck}__all3d.pkl"
    cmds = []
    s2 = f"{OUT}/scores/{exp}__{ck}.json"
    if not os.path.exists(s2):
        cmds.append([PY.replace("~", os.path.expanduser("~")), "tools/rebuttal/train_rescore.py", "score",
                     "--mode", info["mode"], "--dump", d, "--ref", ref, "--cache", cache, "--out", s2])
    s3 = f"{OUT}/score3d/{exp}__{ck}.json"
    if info["mode"] == "sgdet" and not os.path.exists(s3):
        cmds.append([PY.replace("~", os.path.expanduser("~")), "tools/rebuttal/rescore_3d.py", "--dump", d,
                     "--name", f"{exp}__{ck}", "--ref", ref, "--cache", cache, "--out", f"{OUT}/score3d",
                     "--strict-gate"])
    return cmds


def cmd_score_daemon(a):
    """Score every dumped run (marker <exp>.dumped) that has no score JSON yet. CPU only; run on CS93371 in tmux."""
    repo = os.path.expanduser(SERVER_REPO)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    running = {}                                   # key -> Popen
    failed = set()
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
                print(f"[score-daemon] {time.strftime('%H:%M:%S')} {key} rc={rc}", flush=True)
                if rc != 0:
                    failed.add(key)
        for key, c in todo:
            if len(running) >= a.workers:
                break
            log = open(f"{OUT}/logs/score_{key[0]}__{key[1]}__{os.path.basename(key[2])[:-3]}.log", "a")
            running[key] = subprocess.Popen(c, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"[score-daemon] {time.strftime('%H:%M:%S')} start {key}", flush=True)
        if a.once and not running and not todo:
            break
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    sub.add_parser("list")
    p = sub.add_parser("smoke")
    p.add_argument("--mode", choices=tr.MODES, required=True)
    p.add_argument("--n_train", type=int, default=50)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("submit")
    p.add_argument("--hosts", nargs="+", default=["cs93371", "cs94085"])
    p.add_argument("--force_host", default=None)
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--no_smoke_dep", action="store_true")
    p.add_argument("--mode", default=None, choices=["predcls", "sgdet"])
    p.add_argument("--suffix", default="")
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("score_daemon")
    p.add_argument("--once", action="store_true")
    p.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    {"configs": cmd_configs, "list": cmd_list, "smoke": cmd_smoke, "submit": cmd_submit,
     "score_daemon": cmd_score_daemon}[a.cmd](a)
