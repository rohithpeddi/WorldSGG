"""TRAIN: run registry, config generator and queue submitter for the rebuttal retraining round.

Every run = a published Table 2 config + the round's corrections:
  * validation selection: train on ``train_minus_val.txt`` (6,766), select ``best_model.pth`` on ``val750.txt``
    (WC R@20, last frame, the existing criterion); the test split is never loaded during training;
  * ``slot_alignment: per_video_class`` (C1);
  * SGDet: ``visibility_from_annotation: false`` (C2), also exported as WSGG_VISIBILITY_FROM_ANNOTATION=0;
  * ``num_workers: 4``, wandb off, explicit seed, own save_path.
plus the run's variant (seed, B2 logit adjustment, B3 completion, C4 detector).

  python tools/rebuttal/train_runs.py configs                 # (re)write configs/methods/<mode>/rebuttal/*.yaml
  python tools/rebuttal/train_runs.py smoke --mode predcls    # server: 1-epoch / 50-video smoke configs + job
  python tools/rebuttal/train_runs.py submit --rounds R1 ...  # server: train + chained eval jobs -> queue
  python tools/rebuttal/train_runs.py list
"""
import argparse
import copy
import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS = "/data3/rohith/ag/runs/rebuttal/splits"
OUT = "/data3/rohith/ag/runs/rebuttal/train"
QUEUE = "/data3/rohith/ag/runs/rebuttal/queue"
SERVER_REPO = "~/CODE/Scene4Cast_rebuttal_train"
PY = "~/anaconda3/envs/scene4cast/bin/python"
MODES = ("predcls", "sgdet")

# Table 2 rows: key -> (published config stem, feature box space of its PredCls GT reference)
ROWS = {
    "w_sttran": ("w_sttran_{m}_resnet50", "resnet50"),
    "w_sttran_pp": ("w_sttran_pp_{m}_resnet50", "resnet50"),
    "w_dsgdetr": ("w_dsgdetr_{m}_resnet50", "resnet50"),
    "w_dsgdetr_pp": ("w_dsgdetr_pp_{m}_resnet50", "resnet50"),
    "w_usg": ("w_usg_{m}_resnet50", "resnet50"),
    "worldwise": ("worldwise_{m}_dinov3l", "dinov3l"),          # = worldwise_v2e_<m>_dinov3l_v2 (Table 2 dump)
    "worldwise_plus": ("worldwise_plus_dinov3tok_{m}", "dinov3_tok"),
    "worldwise_pp": ("worldwise_pp_dinov3_{m}", "dinov3_tok"),
}
# hours per run (t10_compute, PredCls / SGDet) -> priority and expectations only
HOURS = {"w_sttran": (3.1, 4.0), "w_sttran_pp": (3.2, 4.1), "w_dsgdetr": (3.4, 4.3), "w_dsgdetr_pp": (3.6, 4.4),
         "w_usg": (3.5, 4.4), "worldwise": (4.0, 9.2), "worldwise_plus": (4.5, 9.5), "worldwise_pp": (11.6, 22.3)}
GPU_MEM = {"worldwise_pp": 8, "worldwise_plus": 6, "worldwise": 6}


def run_list():
    """[(round, exp, row, mode, seed, overrides)]"""
    runs = []
    for m in MODES:
        for row in ROWS:
            runs.append(("R1", f"reb_{row}_{m}_s0", row, m, 0, {}))
    for m in MODES:
        for row in ("w_dsgdetr_pp", "worldwise_pp"):
            for s in (1, 2):
                runs.append(("R2", f"reb_{row}_{m}_s{s}", row, m, s, {}))
    for m in MODES:
        runs.append(("R3", f"reb_w_dsgdetr_pp_la_{m}_s0", "w_dsgdetr_pp", m, 0, {
            "use_logit_adjustment": True, "logit_adjustment_tau": 0.5,
            "predicate_priors_path": f"features/predicate_priors_{m}.json"}))
        runs.append(("R3", f"reb_worldwise_hold_{m}_s0", "worldwise", m, 0, {"completion": "hold_last_visible"}))
        runs.append(("R3", f"reb_worldwise_masktok_{m}_s0", "worldwise", m, 0, {"completion": "mask_token"}))
    # C4: WorldWise++ SGDet on the baselines' ResNet-50 detector (boxes, labels, scores, 1024-d ROI features);
    # the DINOv3/Pi3 grids are frame-level and detector-independent; boxes are mapped into the grid image space.
    runs.append(("R4", "reb_worldwise_pp_r50det_sgdet_s0", "worldwise_pp", "sgdet", 0, {
        "feature_model": "resnet50", "d_detector_roi": 1024, "d_union_roi": 1024, "token_streams": [1024],
        "rescale_boxes_to_grid": True, "_gt_space": "dinov3_tok"}))
    return runs


def gt_space(row, overrides):
    return overrides.get("_gt_space", ROWS[row][1])


def build_config(exp, row, mode, seed, overrides, save_path=None):
    src = os.path.join(REPO, "configs", "methods", mode, ROWS[row][0].format(m=mode) + ".yaml")
    with open(src) as f:
        c = yaml.safe_load(f)
    c = copy.deepcopy(c)
    c.update({
        "experiment_name": exp,
        "save_path": save_path or f"{OUT}/runs",
        "seed": int(seed),
        "ckpt": None,
        "use_wandb": False,
        "num_workers": 4,
        "prefetch_factor": 4,
        "slot_alignment": "per_video_class",
        "train_video_list": f"{SPLITS}/train_minus_val.txt",
        "val_video_list": f"{SPLITS}/val750.txt",
        "skip_test": False,
    })
    if mode == "sgdet":
        c["visibility_from_annotation"] = False
    c.update({k: v for k, v in overrides.items() if not k.startswith("_")})
    return src, c


def config_path(exp, mode):
    return os.path.join(REPO, "configs", "methods", mode, "rebuttal", f"{exp}.yaml")


def write_config(path, src, c, note=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    head = (f"# Rebuttal retraining round (TRAIN agent, docs/rebuttal_runs/TRAIN.md). Generated by\n"
            f"# tools/rebuttal/train_runs.py from {os.path.relpath(src, REPO).replace(os.sep, '/')}{note}.\n")
    with open(path, "w", newline="\n") as f:
        f.write(head)
        yaml.safe_dump(c, f, sort_keys=False, default_flow_style=False)


def cmd_configs(a):
    for rnd, exp, row, mode, seed, ov in run_list():
        src, c = build_config(exp, row, mode, seed, ov)
        write_config(config_path(exp, mode), src, c, note=f"; round {rnd}")
        print("wrote", os.path.relpath(config_path(exp, mode), REPO))


def cmd_list(a):
    for rnd, exp, row, mode, seed, ov in run_list():
        h = HOURS[row][MODES.index(mode)]
        print(f"{rnd} {exp:40s} {row:15s} {mode:7s} seed={seed} ~{h}h gt={gt_space(row, ov)} "
              f"{ {k: v for k, v in ov.items() if not k.startswith('_')} }")


# ----------------------------------------------------------------------------------------------------------------
def env_lines(mode):
    lines = ["export OMP_NUM_THREADS=4", "export WSGG_SLOT_ALIGNMENT=per_video_class"]
    if mode == "sgdet":
        lines.append("export WSGG_VISIBILITY_FROM_ANNOTATION=0")
    return "\n".join(lines)


def train_job(exp, mode, cfg_rel, mem, requires=None):
    req = f"# REQUIRES={requires}\n" if requires else ""
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
{req}# TRAIN rebuttal round: {exp}
set -e -o pipefail
cd {SERVER_REPO}
PY={PY}
{env_lines(mode)}
echo "[trn] start $(date -Is) host=$REB_HOST gpu=$REB_GPU commit=$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)"
$PY train_wsgg_methods.py --config {cfg_rel}
test -f {OUT}/runs/{exp}/best_model.pth
test -f {OUT}/runs/{exp}/checkpoint_19/checkpoint_state.pth
echo "[trn] done $(date -Is)"
"""


def eval_job(exp, mode, cfg_rel, space, mem):
    ref = f"{OUT}/gt/predcls_{space}_aligned.pkl"
    cache = f"{OUT}/gt/full_gt_{space}_aligned.pkl"
    dumps, scores = f"{OUT}/dumps", f"{OUT}/scores"
    lines = [f"$PY tools/rebuttal/train_dump.py --config {cfg_rel} --ckpts best checkpoint_19 --out_dir {dumps}"]
    for ck in ("best", "checkpoint_19"):
        lines.append(f"CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/train_rescore.py score --mode {mode} "
                     f"--dump {dumps}/{exp}__{ck}__all.pkl --ref {ref} --cache {cache} "
                     f"--out {scores}/{exp}__{ck}.json")
    body = "\n".join(lines)
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# REQUIRES=trn_{exp}
# TRAIN rebuttal round: test-split dump (val-selected + final checkpoint) and re-score of {exp}
set -e
cd {SERVER_REPO}
PY={PY}
{env_lines(mode)}
test -f {cache}
{body}
echo "[trn-eval] done $(date -Is)"
"""


def priority(row, mode, kind, rnd="R1"):
    """Longest first within the TRAIN band (30-59); evals right after their training."""
    if kind == "eval":
        return 31
    h = HOURS[row][MODES.index(mode)]
    return 30 if h > 20 else 33 if h > 10 else 36 if h > 8 else 40 if mode == "sgdet" else 44


def queued_names(queue=QUEUE):
    """{job name: queue dir} for every job file (pending PP_name.sh, running gpuG__PP_name.sh, done/failed)."""
    out = {}
    for d in ("pending", "running", "done", "failed"):
        for f in os.listdir(f"{queue}/{d}"):
            b = f.split("__", 1)[-1]
            if b.endswith(".sh") and len(b) > 3 and b[:2].isdigit() and b[2] == "_":
                out[b[3:-3]] = d
    return out


def submit_file(name, prio, text, dry):
    tmp = f"{QUEUE}/pending/{name}.tmp"
    dst = f"{QUEUE}/pending/{prio:02d}_{name}.sh"
    where = queued_names().get(name)
    if where is not None:
        print(f"skip {name}: already in {where}/")
        return
    if dry:
        print(f"[dry] {dst}\n{text}")
        return
    with open(tmp, "w", newline="\n") as f:
        f.write(text)
    os.chmod(tmp, 0o755)
    os.rename(tmp, dst)
    print("submitted", dst)


def cmd_submit(a):
    sel = [r for r in run_list() if r[0] in a.rounds and (not a.only or r[1] in a.only)]
    for rnd, exp, row, mode, seed, ov in sel:
        cfg_rel = os.path.relpath(config_path(exp, mode), REPO).replace(os.sep, "/")
        if not os.path.exists(config_path(exp, mode)):
            sys.exit(f"missing config {cfg_rel}: run `configs` first")
        mem = GPU_MEM.get(row, 4)
        req = None if a.no_smoke_dep else f"trn_smoke_{mode}"
        submit_file(f"trn_{exp}", priority(row, mode, "train", rnd), train_job(exp, mode, cfg_rel, mem, req), a.dry)
        if not a.no_eval:
            submit_file(f"trn_eval_{exp}", priority(row, mode, "eval"),
                        eval_job(exp, mode, cfg_rel, gt_space(row, ov), mem), a.dry)


# ----------------------------------------------------------------------------------------------------------------
# Remote hosts (INFRA: docs/rebuttal_runs/INFRA.md). Code, env, data and outputs are CS93371 directories mounted on
# the remote host; configs keep their CS93371 paths and are rewritten per host by infra_hostcfg.sh, every other
# CS93371 path is mapped with hp(). INFRA placement: ROI-feature-only families (baselines, WorldWise @DINOv3-L) on
# cs94085; WorldWise+ (slot streams on /data3) and WorldWise++ (grids, 1 GbE uplink) stay on CS93371.
INFRA = "/data3/rohith/ag/runs/rebuttal/infra"
REMOTE_ROWS = ("w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg", "worldwise")
REMOTE_HOST = "cs94085"
JOBS_DIR = f"{OUT}/jobs_remote"


def remote_preamble(mode):
    return f"""set -e -o pipefail
case $REB_HOST in cs94085) . /data1/rohith/wsgg/host.env ;; cs94092) . /data3/rohith/wsgg/host.env ;;
                  *) echo "TRAIN remote job started on $REB_HOST" >&2; exit 2 ;; esac
cd "$WSGG_CODE"
{env_lines(mode)}"""


def remote_train_job(exp, mode, cfg_rel, mem, requires):
    req = f"# REQUIRES={requires}\n" if requires else ""
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
{req}# TRAIN rebuttal round ({REMOTE_HOST}): {exp}
{remote_preamble(mode)}
echo "[trn] start $(date -Is) host=$REB_HOST gpu=$REB_GPU"
CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" {cfg_rel} trn_{exp})
test -s "$CFG"
"$PY" train_wsgg_methods.py --config "$CFG"
test -f "$(hp {OUT}/runs/{exp}/best_model.pth)"
test -f "$(hp {OUT}/runs/{exp}/checkpoint_19/checkpoint_state.pth)"
echo "[trn] done $(date -Is)"
"""


def remote_eval_job(exp, mode, cfg_rel, space, mem):
    ref, cache = f"{OUT}/gt/predcls_{space}_aligned.pkl", f"{OUT}/gt/full_gt_{space}_aligned.pkl"
    lines = [f'CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh" {cfg_rel} trn_eval_{exp})', 'test -s "$CFG"',
             f'"$PY" tools/rebuttal/train_dump.py --config "$CFG" --ckpts best checkpoint_19 --out_dir "$(hp {OUT}/dumps)"']
    for ck in ("best", "checkpoint_19"):
        lines.append(f'CUDA_VISIBLE_DEVICES= "$PY" tools/rebuttal/train_rescore.py score --mode {mode} '
                     f'--dump "$(hp {OUT}/dumps/{exp}__{ck}__all.pkl)" --ref "$(hp {ref})" --cache "$(hp {cache})" '
                     f'--out "$(hp {OUT}/scores/{exp}__{ck}.json)"')
    body = "\n".join(lines)
    return f"""#!/bin/bash
# GPU_MEM_GB={mem}
# REQUIRES=trn_{exp}
# TRAIN rebuttal round ({REMOTE_HOST}): test-split dump (val-selected + final checkpoint) and re-score of {exp}
{remote_preamble(mode)}
test -f "$(hp {cache})"
{body}
echo "[trn-eval] done $(date -Is)"
"""


def remote_submit(name, prio, text, dry):
    os.makedirs(JOBS_DIR, exist_ok=True)
    path = f"{JOBS_DIR}/{name}.sh"
    with open(path, "w", newline="\n") as f:
        f.write(text)
    if dry:
        print(f"[dry] {path}\n{text}")
        return
    import subprocess
    r = subprocess.run(["bash", f"{INFRA}/infra_submit.sh", REMOTE_HOST, path, f"{prio:02d}", name],
                       capture_output=True, text=True)
    print((r.stdout + r.stderr).strip())


def withdraw_local(name, dry):
    """Remove a TRAIN job from the CS93371 queue if (and only if) it is still pending."""
    hit = [f for f in os.listdir(f"{QUEUE}/pending") if f.endswith(f"_{name}.sh") and f[3:] == f"{name}.sh"]
    where = queued_names().get(name)
    if where not in (None, "pending"):
        print(f"keep {name}: already {where} on CS93371")
        return False
    for f in hit:
        if dry:
            print(f"[dry] withdraw {QUEUE}/pending/{f}")
        else:
            os.rename(f"{QUEUE}/pending/{f}", f"{JOBS_DIR}/withdrawn_cs93371__{f}")
            print(f"withdrew {f} from CS93371 pending")
    return True


def cmd_move(a):
    """Move the REMOTE_ROWS runs that are still pending on CS93371 to cs94085 (training + eval), behind a
    cs94085 GPU smoke (trn_smoke85_<mode>)."""
    os.makedirs(JOBS_DIR, exist_ok=True)
    sel = [r for r in run_list() if r[2] in REMOTE_ROWS and (not a.only or r[1] in a.only)]
    for mode in MODES:
        if any(r[3] == mode for r in sel) and not a.no_smoke:
            remote_submit(f"trn_smoke85_{mode}", 30, remote_smoke_job(mode), a.dry)
    for rnd, exp, row, mode, seed, ov in sel:
        if not withdraw_local(f"trn_{exp}", a.dry):
            continue
        withdraw_local(f"trn_eval_{exp}", a.dry)
        cfg_rel = os.path.relpath(config_path(exp, mode), REPO).replace(os.sep, "/")
        mem = GPU_MEM.get(row, 4)
        remote_submit(f"trn_{exp}", priority(row, mode, "train", rnd),
                      remote_train_job(exp, mode, cfg_rel, mem, None if a.no_smoke else f"trn_smoke85_{mode}"),
                      a.dry)
        remote_submit(f"trn_eval_{exp}", priority(row, mode, "eval"),
                      remote_eval_job(exp, mode, cfg_rel, gt_space(row, ov), mem), a.dry)


def remote_smoke_job(mode, n_train=50, n_val=20):
    """cs94085 GPU smoke of every moved run variant of one mode (1 epoch / 50 videos) + a 3-video dump+score."""
    sdir = f"{OUT}/smoke_c85"
    os.makedirs(f"{sdir}/configs", exist_ok=True)
    for name, src in ((f"train{n_train}.txt", "train_minus_val.txt"), (f"val{n_val}.txt", "val750.txt")):
        n = n_train if name.startswith("train") else n_val
        ids = [v.strip() for v in open(f"{SPLITS}/{src}") if v.strip()][:n]
        open(f"{sdir}/{name}", "w").write("\n".join(ids) + "\n")
    te = sorted(f[:-len(".mp4.pkl")] for f in os.listdir("/data/rohith/ag/world4d_rel_annotations_worldbbox/test")
                if f.endswith(".mp4.pkl"))[:3]
    open(f"{sdir}/test3.txt", "w").write("\n".join(te) + "\n")
    cmds, seen = [], set()
    for rnd, exp, row, m, seed, ov in run_list():
        if m != mode or seed != 0 or row not in REMOTE_ROWS:
            continue
        key = (row, tuple(sorted((k, str(v)) for k, v in ov.items())))
        if key in seen:
            continue
        seen.add(key)
        sexp = "smoke85_" + exp
        src, c = build_config(sexp, row, mode, seed, ov, save_path=f"{sdir}/runs")
        c.update({"nepoch": 1, "train_video_list": f"{sdir}/train{n_train}.txt",
                  "val_video_list": f"{sdir}/val{n_val}.txt", "log_every": 10})
        p = f"{sdir}/configs/{sexp}.yaml"
        write_config(p, src, c, note="; SMOKE cs94085")
        log = f'"$(hp {sdir}/{sexp}.log)"'
        cmds.append(f'C=$(bash "$WSGG_INFRA/infra_hostcfg.sh" "$(hp {p})" {sexp}) && test -s "$C" && '
                    f'"$PY" train_wsgg_methods.py --config "$C" '
                    f'> {log} 2>&1 && echo OK {sexp} || {{ echo FAIL {sexp}; FAILED=1; }}')
        if row == "w_dsgdetr_pp" and not ov:
            space = gt_space(row, ov)
            cmds.append(f'"$PY" tools/rebuttal/train_dump.py --config "$C" --ckpts best checkpoint_0 '
                        f'--out_dir "$(hp {sdir}/dumps)" --video_list "$(hp {sdir}/test3.txt)" >> {log} 2>&1 && '
                        f'CUDA_VISIBLE_DEVICES= "$PY" tools/rebuttal/train_rescore.py score --mode {mode} '
                        f'--dump "$(hp {sdir}/dumps/{sexp}__best__all.pkl)" '
                        f'--ref "$(hp {OUT}/gt/predcls_{space}_aligned.pkl)" '
                        f'--cache "$(hp {OUT}/gt/full_gt_{space}_aligned.pkl)" '
                        f'--out "$(hp {sdir}/scores/{sexp}.json)" --allow_subset >> {log} 2>&1 '
                        f'&& echo OK dump+score {sexp} || {{ echo FAIL dump/score {sexp}; FAILED=1; }}')
    return f"""#!/bin/bash
# GPU_MEM_GB=8
# TRAIN rebuttal round: cs94085 smoke ({mode}) of the runs placed there
set +e
case $REB_HOST in cs94085) . /data1/rohith/wsgg/host.env ;; cs94092) . /data3/rohith/wsgg/host.env ;;
                  *) echo "remote smoke on $REB_HOST" >&2; exit 2 ;; esac
cd "$WSGG_CODE"
{env_lines(mode)}
mkdir -p "$(hp {sdir}/scores)"
FAILED=0
{chr(10).join(cmds)}
exit $FAILED
"""


def cmd_remote_smoke(a):
    """(Re)submit only the cs94085 smoke of one mode."""
    remote_submit(f"trn_smoke85_{a.mode}", 30, remote_smoke_job(a.mode), a.dry)


def cmd_reprioritize(a):
    """Rename pending CS93371 TRAIN jobs to the current priority() (e.g. after a placement change)."""
    for rnd, exp, row, mode, seed, ov in run_list():
        name = f"trn_{exp}"
        want = priority(row, mode, "train", rnd)
        for f in os.listdir(f"{QUEUE}/pending"):
            if f[3:] == f"{name}.sh" and int(f[:2]) != want:
                dst = f"{want:02d}_{name}.sh"
                print(f"{f} -> {dst}")
                if not a.dry:
                    os.rename(f"{QUEUE}/pending/{f}", f"{QUEUE}/pending/{dst}")


# ----------------------------------------------------------------------------------------------------------------
def cmd_smoke(a):
    """1 epoch on 50 train-minus-val videos, select on 20 val videos, for every run variant of one mode; then a
    3-video test dump of two of them. One queue job per mode."""
    sdir = f"{OUT}/smoke{a.tag}"
    os.makedirs(f"{sdir}/configs", exist_ok=True)
    tr = [v.strip() for v in open(f"{SPLITS}/train_minus_val.txt") if v.strip()][:a.n_train]
    va = [v.strip() for v in open(f"{SPLITS}/val750.txt") if v.strip()][:a.n_val]
    open(f"{sdir}/train{a.n_train}.txt", "w").write("\n".join(tr) + "\n")
    open(f"{sdir}/val{a.n_val}.txt", "w").write("\n".join(va) + "\n")
    te = sorted(f[:-len(".mp4.pkl")] for f in os.listdir("/data/rohith/ag/world4d_rel_annotations_worldbbox/test")
                if f.endswith(".mp4.pkl"))[:3]
    open(f"{sdir}/test3.txt", "w").write("\n".join(te) + "\n")
    seen, cmds, dump_cmds = set(), [], []
    for rnd, exp, row, mode, seed, ov in run_list():
        if mode != a.mode or seed != 0:
            continue
        key = (row, tuple(sorted((k, str(v)) for k, v in ov.items())))
        if key in seen:
            continue
        seen.add(key)
        sexp = "smoke_" + exp
        src, c = build_config(sexp, row, mode, seed, ov, save_path=f"{sdir}/runs")
        c.update({"nepoch": 1, "train_video_list": f"{sdir}/train{a.n_train}.txt",
                  "val_video_list": f"{sdir}/val{a.n_val}.txt", "log_every": 10})
        p = f"{sdir}/configs/{sexp}.yaml"
        write_config(p, src, c, note="; SMOKE")
        cmds.append(f"echo \"=== {sexp} $(date -Is)\"; $PY train_wsgg_methods.py --config {p} "
                    f"> {sdir}/{sexp}.log 2>&1 && echo OK {sexp} || {{ echo FAIL {sexp}; FAILED=1; }}")
        if row in a.dump_rows:
            space = gt_space(row, ov)
            d = f"{sdir}/dumps/{sexp}__best__all.pkl"
            dump_cmds.append(f"$PY tools/rebuttal/train_dump.py --config {p} --ckpts best checkpoint_0 "
                        f"--out_dir {sdir}/dumps --video_list {sdir}/test3.txt > {sdir}/{sexp}.dump.log 2>&1 "
                        f"&& CUDA_VISIBLE_DEVICES= $PY tools/rebuttal/train_rescore.py score --mode {mode} --dump {d} "
                        f"--ref {OUT}/gt/predcls_{space}_aligned.pkl --cache {OUT}/gt/full_gt_{space}_aligned.pkl "
                        f"--out {sdir}/scores/{sexp}.json --allow_subset >> {sdir}/{sexp}.dump.log 2>&1 "
                        f"&& echo OK dump+score {sexp} || {{ echo FAIL dump/score {sexp}; FAILED=1; }}")
    job = f"""#!/bin/bash
# GPU_MEM_GB=8
# TRAIN rebuttal round: smoke ({a.mode}) - 1 epoch / {a.n_train} videos per run variant
cd {SERVER_REPO}
PY={PY}
{env_lines(a.mode)}
FAILED=0
{chr(10).join(cmds)}
# wait (up to 2 h) for the GT caches of tools/rebuttal/train_prep_gt.sh before the dump + score smoke
for i in $(seq 1 240); do ls {OUT}/gt/full_gt_*_aligned.pkl 2>/dev/null | wc -l | grep -q '^3$' && break; sleep 30; done
mkdir -p {sdir}/scores
{chr(10).join(dump_cmds)}
exit $FAILED
"""
    if a.write:
        with open(a.write, "w", newline="\n") as f:
            f.write(job)
        print("wrote", a.write)
        return
    submit_file(f"trn_smoke_{a.mode}", 30, job, a.dry)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("configs")
    sub.add_parser("list")
    p = sub.add_parser("submit")
    p.add_argument("--rounds", nargs="+", default=["R1"])
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--no_eval", action="store_true")
    p.add_argument("--no_smoke_dep", action="store_true", help="do not wait for trn_smoke_<mode> in done/")
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("move")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--no_smoke", action="store_true")
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("reprioritize")
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("remote_smoke")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--dry", action="store_true")
    p = sub.add_parser("smoke")
    p.add_argument("--mode", choices=MODES, required=True)
    p.add_argument("--n_train", type=int, default=50)
    p.add_argument("--n_val", type=int, default=20)
    p.add_argument("--dump_rows", nargs="*", default=["w_dsgdetr_pp", "worldwise_pp"])
    p.add_argument("--dry", action="store_true")
    p.add_argument("--write", default=None, help="write the job script here instead of submitting it")
    p.add_argument("--tag", default="", help="suffix for the smoke dir (e.g. _cpu)")
    a = ap.parse_args()
    {"configs": cmd_configs, "list": cmd_list, "submit": cmd_submit, "smoke": cmd_smoke, "move": cmd_move,
     "reprioritize": cmd_reprioritize, "remote_smoke": cmd_remote_smoke}[a.cmd](a)
