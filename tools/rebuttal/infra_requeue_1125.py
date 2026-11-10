"""Requeue cs94085 jobs after the reb sshfs mount hang of Nov 7 2026 (11:25-11:52 CST). Run ON cs94085, queue paused:

    $PY /data1/rohith/wsgg/mnt/reb/infra/requeue_1125.py [--dry]

1. Each 31_trn_eval_*.sh in pending/ and failed/ reads its checkpoints (save_path) and its scoring inputs
   (--dump/--ref/--cache) through the read-only `runs` mount, so the rw `reb` pipe carries little except writes.
   failed/ evals move back to pending/.
2. The six trainings that crashed (failed/44_trn_<exp>.sh) become pending/29_trn_<exp>.sh, resuming from their last
   complete checkpoint, which is also read through the `runs` mount (ckpt is relative to the experiment dir).
Originals and their logs are kept in failed/reb_hang_1125/ and logs/failed_reb_hang_1125/.
"""
import os
import re
import shutil
import sys

Q = "/data1/rohith/wsgg/queue"
DRY = "--dry" in sys.argv
RESUME = {  # last checkpoint that loads with a matching epoch (checked on CS93371); worldwise checkpoint_10 truncated
    "reb_w_sttran_predcls_s0": 12,
    "reb_w_sttran_pp_predcls_s0": 11,
    "reb_w_usg_predcls_s0": 10,
    "reb_worldwise_predcls_s0": 9,
    "reb_worldwise_hold_predcls_s0": 11,
    "reb_worldwise_masktok_predcls_s0": 11,
}
CFG_PREFIX = 'CFG=$(bash "$WSGG_INFRA/infra_hostcfg.sh"'
ARC_JOBS = f"{Q}/failed/reb_hang_1125"
ARC_LOGS = f"{Q}/logs/failed_reb_hang_1125"


def write_atomic(path, text):
    print(f"  write {path}")
    if DRY:
        return
    tmp = path + ".tmp"          # the worker globs pending/*.sh, so the .tmp name is never claimed
    with open(tmp, "w") as f:
        f.write(text)
    os.chmod(tmp, 0o755)
    os.replace(tmp, path)


def move(src, dst_dir):
    if os.path.exists(src):
        print(f"  move {src} -> {dst_dir}/")
        if not DRY:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, os.path.join(dst_dir, os.path.basename(src)))


def cfg_index(lines, path):
    idx = [k for k, l in enumerate(lines) if l.startswith(CFG_PREFIX)]
    assert len(idx) == 1, f"{path}: expected one CFG= line, found {len(idx)}"
    return idx[0]


# 1. evaluation jobs
EVAL_TAG = "# read checkpoints through the read-only runs mount"
for d in ("pending", "failed"):
    for name in sorted(os.listdir(f"{Q}/{d}")):
        if not (name.startswith("31_trn_eval_") and name.endswith(".sh")):
            continue
        path = f"{Q}/{d}/{name}"
        src = open(path).read()
        if EVAL_TAG in src:
            print(f"skip (already fixed) {path}")
            continue
        lines = src.split("\n")
        i = cfg_index(lines, path)
        assert lines[i + 1].startswith('"$PY" tools/rebuttal/train_dump.py'), f"{path}: unexpected line after CFG"
        add = [
            f"{EVAL_TAG} (the rw reb pipe hung on Nov 7 under mixed big reads and writes)",
            'sed -i "s#^save_path: $WSGG_OUT/train/runs\\$#save_path: $WSGG_RUNS_RO/rebuttal/train/runs#" "$CFG"',
            'grep -q "^save_path: $WSGG_RUNS_RO/rebuttal/train/runs\\$" "$CFG"',
        ]
        lines = lines[:i + 1] + add + lines[i + 1:]
        text, n = re.subn(r'(--(?:dump|ref|cache) )"\$\(hp /data3/rohith/ag/runs/rebuttal/([^)]*)\)"',
                          r'\1"$WSGG_RUNS_RO/rebuttal/\2"', "\n".join(lines))
        assert n == 6, f"{path}: expected 6 scoring inputs to reroute, got {n}"
        print(f"eval {name} ({d}): save_path + {n} scoring inputs -> runs mount")
        if d == "failed":
            job = name[3:-3]
            for ext in (".log", ".status.json"):
                move(f"{Q}/logs/{job}{ext}", ARC_LOGS)
            write_atomic(f"{Q}/pending/{name}", text)
            if not DRY:
                os.remove(path)
        else:
            write_atomic(path, text)

# 2. training jobs
for exp, n in RESUME.items():
    old = f"{Q}/failed/44_trn_{exp}.sh"
    assert os.path.exists(old), f"missing {old}"
    lines = open(old).read().split("\n")
    i = cfg_index(lines, old)
    assert lines[i + 1].startswith('"$PY" train_wsgg_methods.py --config "$CFG"'), f"{old}: unexpected line after CFG"
    res = f"../../../../runs/rebuttal/train/runs/{exp}/checkpoint_{n}"
    add = [
        "# Resume after the reb mount hang of Nov 7 11:25 (first attempt: failed/reb_hang_1125/). The checkpoint is",
        "# read through the read-only runs mount: RES is relative to the experiment dir on the reb mount.",
        f"RES={res}",
        f'test -f "$WSGG_OUT/train/runs/{exp}/$RES/checkpoint_state.pth"',
        "grep -q '^ckpt: null$' \"$CFG\"",
        'sed -i "s#^ckpt: null\\$#ckpt: $RES#" "$CFG"',
        'grep -q "^ckpt: $RES\\$" "$CFG"',
        f'echo "[trn] resume from checkpoint_{n} via $RES"',
    ]
    lines = lines[:i + 1] + add + lines[i + 1:]
    print(f"train {exp}: resume checkpoint_{n}")
    for ext in (".log", ".status.json"):
        move(f"{Q}/logs/trn_{exp}{ext}", ARC_LOGS)
    write_atomic(f"{Q}/pending/29_trn_{exp}.sh", "\n".join(lines))
    move(old, ARC_JOBS)
print("dry run" if DRY else "done")
