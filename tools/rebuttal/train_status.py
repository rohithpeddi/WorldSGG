"""TRAIN: markdown status table of the rebuttal-round jobs (run on CS93371; read-only).

  python tools/rebuttal/train_status.py            # table to stdout
For every run: queue state of trn_<exp> and trn_eval_<exp>, host/gpu, epochs done, current val R@20 / best
(epoch), peak VRAM, and, once scored, covered / full WC R@20 and mR@20 (All, UO) for the val-selected and final
checkpoints.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_runs as tr  # noqa: E402

RESULTS = os.path.join(tr.REPO, "results")


def job_state(names, name, queue=tr.QUEUE):
    st = names.get(name)
    if st is None:
        return "-", ""
    where = ""
    if st == "running":
        f = glob.glob(f"{queue}/running/gpu*__*_{name}.sh")
        where = os.path.basename(f[0]).split("__")[0] if f else ""
    sj = f"{queue}/logs/{name}.status.json"
    if st in ("done", "failed") and os.path.exists(sj):
        s = json.load(open(sj))
        where = f"{s.get('host')}/gpu{s.get('gpu')}"
    return st, where


def epochs(exp):
    p = f"{RESULTS}/{exp}_metrics.jsonl"
    rows = []
    if os.path.exists(p):
        for line in open(p):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("type") == "header":
                rows = []          # keep the last run's rows only
            elif "epoch" in d:
                rows.append(d)
    return rows


def fmt_scores(exp, ck, mode):
    p = f"{tr.OUT}/scores/{exp}__{ck}.json"
    if not os.path.exists(p):
        return ""
    s = json.load(open(p))
    cov = s[f"{mode}|covered|sorted"]["wc"]
    full = s[f"{mode}|full|sorted" if mode == "predcls" else "sgdet|full_rule|sorted"]["wc"]
    f = lambda d, b, m: 100 * d[b][m]["20"] if isinstance(next(iter(d[b][m])), str) else 100 * d[b][m][20]  # noqa
    return (f"{f(cov, 'All', 'R'):.1f}/{f(cov, 'All', 'mR'):.1f} UO {f(cov, 'UO', 'R'):.1f}/{f(cov, 'UO', 'mR'):.1f}"
            f" | full {f(full, 'All', 'R'):.1f}/{f(full, 'All', 'mR'):.1f}")


REMOTE_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "HostName=10.176.140.100",
              "-o", "HostKeyAlias=cs94085.utdallas.edu", "cs94085"]
REMOTE_Q = "/data1/rohith/wsgg/queue"


def remote_names():
    """{job name: (state, where)} on cs94085 (live over ssh)."""
    import subprocess
    cmd = (f"for d in pending running done failed; do for f in $(ls {REMOTE_Q}/$d); do echo $d $f; done; done; "
           f"for f in {REMOTE_Q}/logs/trn_*.status.json; do echo STATUS $(cat $f); done")
    try:
        out = subprocess.run(REMOTE_SSH + [cmd], capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    res, status = {}, {}
    for line in out.splitlines():
        if line.startswith("STATUS "):
            try:
                s = json.loads(line[7:])
                status[s["job"]] = f"{s.get('host')}/gpu{s.get('gpu')}"
            except Exception:
                pass
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        d, f = parts
        g, b = (f.split("__", 1) + [""])[:2] if "__" in f else ("", f)
        if b.endswith(".sh") and b[:2].isdigit():
            res[b[3:-3]] = (d, f"cs94085/{g}" if g else "")
    for k, (d, w) in list(res.items()):
        if d in ("done", "failed"):
            res[k] = (d, status.get(k, w))
    return res


def main():
    names = tr.queued_names()
    remote = remote_names()
    print("| round | run | train job | where | epochs | val R@20 last / best (ep) | peak GB | eval job | "
          "best ckpt: sorted-WC R/mR@20 All, UO; full All | final ckpt (same) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for rnd, exp, row, mode, seed, ov in tr.run_list():
        if row in tr.REMOTE_ROWS and f"trn_{exp}" in remote:
            st, where = remote[f"trn_{exp}"]
            est, ewhere = remote.get(f"trn_eval_{exp}", ("-", ""))
        else:
            st, where = job_state(names, f"trn_{exp}")
            est, ewhere = job_state(names, f"trn_eval_{exp}")
            where = f"cs93371/{where}" if where and not where.startswith("cs") else where
        rows = epochs(exp)
        if rows:
            best = max(rows, key=lambda d: d.get("wc/R@20", 0))
            ep = f"{len(rows)}/20"
            val = f"{100 * rows[-1]['wc/R@20']:.1f} / {100 * best['wc/R@20']:.1f} ({best['epoch']})"
            peak = f"{max(d.get('peak_vram_gb', 0) for d in rows):.1f}"
        else:
            ep, val, peak = "0/20", "", ""
        print(f"| {rnd} | {exp} | {st} | {where} | {ep} | {val} | {peak} | {est} | "
              f"{fmt_scores(exp, 'best', mode)} | {fmt_scores(exp, 'checkpoint_19', mode)} |")


if __name__ == "__main__":
    main()
