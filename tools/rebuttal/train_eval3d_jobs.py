"""TRAIN: queue the 3-D IoU re-score of rebuttal-round SGDet runs (user request, Nov 7 2026).

For each experiment, the job is derived from that run's own 2-D evaluation job (``31_trn_eval_<exp>.sh`` in the
host queue's pending/, running/ or done/), so config, environment and ground truth stay identical:
  * ``train_dump.py`` gets ``--keep_3d`` and writes ``train/dumps3d/<exp>__<ckpt>__all3d.pkl``;
  * each ``train_rescore.py score`` call becomes ``rescore_3d.py`` on that dump with the same ``--ref`` / ``--cache``
    (aligned PredCls GT and full-GT cache of the run's feature family), ``--strict-gate`` (the job fails when the
    frame-convention gate fails), output ``train/score3d/<exp>__<ckpt>.json``;
  * on remote hosts the annotation PKLs come from the data mount (``--annot_dir "$(hp ...)"``).
The new job keeps the 2-D job's ``REQUIRES`` (the training run) and ``GPU_MEM_GB``. Run on the host that owns the
queue (the scripts are written to its pending/ with a .tmp + rename):

    python tools/rebuttal/train_eval3d_jobs.py --queue <queue root> --remote|--local --prefix 32 <exp> [<exp> ...]
"""
import argparse
import os
import re
import sys

ANNOT = "/data/rohith/ag/world4d_rel_annotations_worldbbox/test"
OUT3D = "/data3/rohith/ag/runs/rebuttal/train/score3d"
TOK = r'("[^"]*"|\S+)'


def find_eval(q, exp):
    name = f"31_trn_eval_{exp}.sh"
    for d in ("pending", "done"):
        if os.path.exists(f"{q}/{d}/{name}"):
            return f"{q}/{d}/{name}"
    for f in os.listdir(f"{q}/running"):
        if f.endswith("__" + name):
            return f"{q}/running/{f}"
    raise SystemExit(f"no 2-D eval job for {exp} in {q}")


def convert(src, exp, remote):
    out, n_dump, n_score, n_cfg = [], 0, 0, 0
    for line in src.split("\n"):
        if line.startswith("# TRAIN rebuttal round"):
            line = (f"# 3-D IoU re-score (user request, Nov 7): test-split dump with 3-D boxes (val-selected + final "
                    f"checkpoint) of {exp} and tools/rebuttal/rescore_3d.py on the same aligned GT as its 2-D score")
        elif "tools/rebuttal/train_dump.py" in line:
            assert "--keep_3d" not in line and line.count("/rebuttal/train/dumps") == 1, line
            line = line.replace("tools/rebuttal/train_dump.py", "tools/rebuttal/train_dump.py --keep_3d")
            line = line.replace("/rebuttal/train/dumps", "/rebuttal/train/dumps3d")
            n_dump += 1
        elif "tools/rebuttal/train_rescore.py score --mode sgdet" in line:
            m = re.search(rf"--dump {TOK} --ref {TOK} --cache {TOK} --out {TOK}\s*$", line)
            assert m, line
            dump, ref, cache = m.group(1), m.group(2), m.group(3)
            ck = re.search(r"__(best|checkpoint_\d+)__all\.pkl", dump)
            assert ck and "/train/dumps/" in dump, dump
            dump3 = dump.replace("/train/dumps/", "/train/dumps3d/").replace("__all.pkl", "__all3d.pkl")
            outd = f'"$(hp {OUT3D})"' if remote else OUT3D
            annot = f' --annot_dir "$(hp {ANNOT})"' if remote else ""
            head = line[:line.index("tools/rebuttal/train_rescore.py")]
            line = (f"{head}tools/rebuttal/rescore_3d.py --dump {dump3} --name {exp}__{ck.group(1)} --ref {ref} "
                    f"--cache {cache}{annot} --out {outd} --strict-gate")
            n_score += 1
        elif line.startswith("CFG=$(") and f" trn_eval_{exp})" in line:
            line = line.replace(f" trn_eval_{exp})", f" trn_eval3d_{exp})")
            n_cfg += 1
        elif "[trn-eval] done" in line:
            line = line.replace("[trn-eval] done", "[trn-eval3d] done")
        out.append(line)
    assert n_dump == 1 and n_score == 2 and n_cfg == (1 if remote else 0), (exp, n_dump, n_score, n_cfg)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--remote", action="store_true", help="cs94085 / cs94092 job (host.env, hp paths)")
    g.add_argument("--local", action="store_true", help="CS93371 job")
    ap.add_argument("--prefix", default="32")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("exps", nargs="+")
    a = ap.parse_args()
    for exp in a.exps:
        src_path = find_eval(a.queue, exp)
        text = convert(open(src_path).read(), exp, a.remote)
        dst = f"{a.queue}/pending/{a.prefix}_trn_eval3d_{exp}.sh"
        assert not os.path.exists(dst), dst
        print(f"{exp}: {src_path} -> {dst}")
        if a.dry:
            print(text)
            continue
        with open(dst + ".tmp", "w") as f:
            f.write(text)
        os.chmod(dst + ".tmp", 0o755)
        os.replace(dst + ".tmp", dst)


if __name__ == "__main__":
    sys.exit(main())
