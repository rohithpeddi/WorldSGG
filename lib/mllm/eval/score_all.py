"""
B7: score every MLLM run on the worldbbox test set and write the results tables.

    python -m lib.mllm.eval.score_all --out_prefix results/mllm_worldbbox_2026-09-22 \
        [--runs runs.json] [--subset /data3/rohith/ag/splits/test_worldbbox_graphs442.txt]

Runs default to every (method, model, mode) directory found under the configured
``outputs.{zero_shot,caption_all,rag_all,wsg_agent,track_a,track_b}`` roots; Track B
additionally gets an ``objects_pre`` (-critic) row.  Each run is scored on the
full split and, with ``--subset``, on the 442-video subset whose Stage-1 graphs
pre-existed (to check that filled graphs behave like the old ones).  Writes
``<out_prefix>.json`` (everything) and ``<out_prefix>.md`` (paper-ready tables),
and registers the JSON in the manifest as annotation-DEPENDENT.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.mllm.core.config_loader import get_path, load_config              # noqa: E402
from lib.mllm.eval.score_run import collect_records, score_records         # noqa: E402

logger = logging.getLogger("score_all")
METHODS = ("zero_shot", "caption_all", "rag_all", "wsg_agent", "track_a", "track_b")


def discover_runs(cfg: dict) -> List[Dict[str, Any]]:
    runs = []
    for m in METHODS:
        root = get_path(cfg, f"outputs.{m}")
        if not root or not Path(root).is_dir():
            continue
        for mode in ("predcls", "sgdet"):
            d = Path(root) / mode
            if not d.is_dir():
                continue
            for md in sorted(p for p in d.iterdir() if p.is_dir()):
                n = len(list(md.glob("*.pkl")))
                if n == 0:
                    continue
                runs.append({"method": m, "model": md.name, "mode": mode, "pred_dir": root, "n_files": n})
                if m == "track_b" and "nocritic" not in md.name:
                    runs.append({"method": m, "model": md.name, "mode": mode, "pred_dir": root, "n_files": n,
                                 "objects_key": "objects_pre"})
    return runs


def score_one(run: Dict[str, Any], cfg: dict, video_list: Optional[str], legacy: bool) -> Dict[str, Any]:
    col = collect_records(run["model"], run["mode"], run["pred_dir"], 0, cfg, video_list=video_list,
                          objects_key=run.get("objects_key", "objects"))
    res: Dict[str, Any] = {"n_videos": col["n_videos"], "n_split": col["n_split"], "n_missing": len(col["missing"]),
                           "n_errors": len(col["errors"]), "missing": col["missing"][:20]}
    if col["records"]:
        res.update(score_records(col["records"], run["mode"]))
    # the legacy evaluator only understands the vendored baseline PKL format
    if legacy and run["method"] in ("zero_shot", "caption_all", "rag_all", "wsg_agent"):
        try:
            from lib.mllm.eval.legacy_f1 import legacy_f1, summarize_legacy
            vids = None
            if video_list:
                vids = [Path(l.strip()).stem for l in open(video_list, encoding="utf-8") if l.strip()]
            res["legacy"] = summarize_legacy(legacy_f1(run["pred_dir"], run["mode"], run["model"], cfg, video_ids=vids))
        except Exception as e:  # noqa
            res["legacy"] = {"error": repr(e)}
    return res


def _g(d: Dict[str, Any], *ks, default=None):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


def md_tables(results: Dict[str, Any], subset_name: str) -> str:
    lines = []
    for scope in ("full", subset_name):
        rows = [(r, res.get(scope)) for r, res in results["runs"]]
        rows = [(r, s) for r, s in rows if s]
        if not rows:
            continue
        lines.append(f"\n### {scope} split\n")
        for mode in ("predcls", "sgdet"):
            rr = [(r, s) for r, s in rows if r["mode"] == mode]
            if not rr:
                continue
            lines.append(f"\n**{mode}** (all frames; R/mR in %, K=50 unless noted)\n")
            if mode == "predcls":
                lines.append("| method | model | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU nc mR@50 | OU-nt nc mR@50 | legacy uF1 | legacy corr-only uF1 |")
                lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for r, s in rr:
                    w = s.get("wsgg", {})
                    lines.append("| {m} | {mo} | {n}/{ns} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} | {i} |".format(
                        m=r["method"] + ("(-critic)" if r.get("objects_key") == "objects_pre" else ""), mo=r["model"],
                        n=s["n_videos"], ns=s["n_split"],
                        a=_pct(_g(w, "wc", "R", "20")), b=_pct(_g(w, "wc", "mR", "20")),
                        c=_pct(_g(w, "nc", "R", "50")), d=_pct(_g(w, "nc", "mR", "50")),
                        e=_pct(_g(w, "nc", "buckets", "buckets", "OO", "mR", "50")),
                        f=_pct(_g(w, "nc", "buckets", "buckets", "OU", "mR", "50")),
                        g=_pct(_g(w, "nc", "buckets", "ou_nontrivial", "drop_trivial", "mR", "50")),
                        h=_pct(_g(s, "legacy", "gt_plus_corrections", "micro_F1")),
                        i=_pct(_g(s, "legacy", "corrections_only", "micro_F1"))))
            else:
                lines.append("| method | model | videos | unloc nc R@50 | unloc nc mR@50 | IoU.15 nc R@50 | IoU.15 nc mR@50 | IoU.25 nc R@50 | IoU.25 nc mR@50 | slots w/ 3D | legacy uF1 | legacy obj F1 |")
                lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for r, s in rr:
                    l3 = s.get("loc3d", {})
                    cov = s.get("coverage", {})
                    frac = (cov.get("object_slots_with_pred_3d", 0) / cov["object_slots"]) if cov.get("object_slots") else None
                    lines.append("| {m} | {mo} | {n}/{ns} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} | {i} |".format(
                        m=r["method"] + ("(-critic)" if r.get("objects_key") == "objects_pre" else ""), mo=r["model"],
                        n=s["n_videos"], ns=s["n_split"],
                        a=_pct(_g(l3, "unloc", "nc", "R", "50")), b=_pct(_g(l3, "unloc", "nc", "mR", "50")),
                        c=_pct(_g(l3, "iou0.15", "nc", "R", "50")), d=_pct(_g(l3, "iou0.15", "nc", "mR", "50")),
                        e=_pct(_g(l3, "iou0.25", "nc", "R", "50")), f=_pct(_g(l3, "iou0.25", "nc", "mR", "50")),
                        g=_pct(frac), h=_pct(_g(s, "legacy", "gt_plus_corrections", "micro_F1")),
                        i=_pct(_g(s, "legacy", "object_detection", "f1"))))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_prefix", default=None)
    ap.add_argument("--runs", default=None, help="JSON list of runs (default: discover)")
    ap.add_argument("--subset", default="/data3/rohith/ag/splits/test_worldbbox_graphs442.txt")
    ap.add_argument("--no_legacy", action="store_true")
    ap.add_argument("--only", default=None, help="comma list of methods to score")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    runs = json.load(open(args.runs)) if args.runs else discover_runs(cfg)
    if args.only:
        keep = set(args.only.split(","))
        runs = [r for r in runs if r["method"] in keep]
    date = time.strftime("%Y-%m-%d")
    out_prefix = args.out_prefix or os.path.join(REPO, "results", f"mllm_worldbbox_{date}")
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    subset_name = Path(args.subset).stem if args.subset and Path(args.subset).exists() else None
    results: Dict[str, Any] = {"date": date, "annotation_version": None, "runs": []}
    try:
        from tools.cache_manifest import current_annotation_version
        results["annotation_version"] = current_annotation_version("test_worldbbox")
    except Exception:
        pass
    for r in runs:
        logger.info(f"scoring {r}")
        entry = {"full": score_one(r, cfg, None, not args.no_legacy)}
        if subset_name:
            entry[subset_name] = score_one(r, cfg, args.subset, not args.no_legacy)
        results["runs"].append((r, entry))
        with open(out_prefix + ".json", "w") as f:
            json.dump(results, f, indent=1)
    md = [f"# MLLM tracks on the worldbbox test set ({date})",
          f"annotation_version test_worldbbox = `{results['annotation_version']}`; split = 1,511 videos / 48,834 frames;",
          "protocol: lib/mllm/eval/README.md (stock WorldSGG evaluator for predcls, 3D-IoU matching for sgdet).",
          md_tables(results, subset_name or "subset")]
    with open(out_prefix + ".md", "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    try:
        from tools.cache_manifest import register
        register("mllm/results/" + Path(out_prefix).name, out_prefix + ".json",
                 inputs=[f"{r['method']}/{r['mode']}/{r['model']}" for r in runs],
                 annotation_version=results["annotation_version"], note="B7 scoring of all MLLM runs")
    except Exception as e:  # noqa
        logger.warning(f"manifest registration failed: {e!r}")
    print(open(out_prefix + ".md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
