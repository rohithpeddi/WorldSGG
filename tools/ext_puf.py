"""
PUF adapted to monocular per-timestamp WSGG -- driver (setup/EXT_PUF.md).
==========================================================================

    # 1. Pi3 geometry cache (both modes, one Pi3 load per video; CPU)
    python tools/ext_puf.py geom --out-dir /data3/rohith/ag/runs/ext/puf/geom --workers 8
    # 2. relation prior from the AG training annotations
    python tools/ext_puf.py prior --out /data3/rohith/ag/runs/ext/puf/prior/ag_train_prior.npz
    # 3. one arm over a front-end dump (W-DSGDetr++), all frames
    python tools/ext_puf.py run --mode predcls --arm puf_prior \
        --frontend /data3/rohith/ag/runs/rescore/dumps/w_dsgdetr_pp_predcls_resnet50__all.pkl \
        --geom-dir /data3/rohith/ag/runs/ext/puf/geom --prior <npz> --out <dump.pkl>
    # 4. score (reeval-style wc/nc all-frame JSON + bucketed breakdown + source / 3D stats)
    python tools/ext_puf.py score --dump <dump.pkl> --out-dir <dir>

``--videos-file`` restricts run / subset to a split list (e.g. the matched
150-video set); ``subset`` writes the front-end dump restricted the same way so
reference rows are scored on identical frames.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

GRAPH_ARMS = ("fross", "puf", "puf_prior", "puf_prior_vis")
LKS_ARMS = ("frontend", "lks", "lks_bi")


def _read_list(path):
    with open(path) as f:
        return [Path(l.strip()).stem.replace(".mp4", "") for l in f if l.strip()]


# ------------------------------------------------------------------ geom
def _geom_one(args):
    vid, out_dir, modes = args
    p = os.path.join(out_dir, f"{vid}.pkl")
    if os.path.exists(p):
        return vid, "skip", 0.0
    t0 = time.time()
    try:
        from lib.external.puf.geometry import build_video_geometry
        g = build_video_geometry(vid, modes=modes)
        tmp = p + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(g, f, protocol=4)
        os.replace(tmp, p)
        return vid, "ok", time.time() - t0
    except Exception as e:  # noqa: BLE001
        return vid, f"ERR {type(e).__name__}: {e}", time.time() - t0


def cmd_geom(a):
    vids = _read_list(a.split_file) if not a.videos else a.videos
    if a.limit:
        vids = vids[:a.limit]
    os.makedirs(a.out_dir, exist_ok=True)
    jobs = [(v, a.out_dir, tuple(a.modes)) for v in vids]
    n_ok = n_err = 0
    t0 = time.time()
    with mp.get_context("fork").Pool(a.workers, maxtasksperchild=20) as pool:
        for i, (vid, st, dt) in enumerate(pool.imap_unordered(_geom_one, jobs), 1):
            if st.startswith("ERR"):
                n_err += 1
                print(f"[geom] {vid} {st}", flush=True)
            else:
                n_ok += 1
            if i % 25 == 0 or i == len(jobs):
                print(f"[geom] {i}/{len(jobs)} ok={n_ok} err={n_err} last={vid} {dt:.1f}s "
                      f"elapsed={time.time() - t0:.0f}s", flush=True)
    print(f"[geom] done ok={n_ok} err={n_err}")


# ------------------------------------------------------------------ prior
def cmd_prior(a):
    from lib.external.puf.prior import fit_prior
    out = fit_prior(a.train_dir, a.out, workers=a.workers, limit=a.limit)
    np.set_printoptions(precision=3, suppress=True)
    print(f"[prior] {out['n_files']} train files -> {a.out}")
    print("[prior] pairs per class:", out["n_pairs"].astype(int))
    print("[prior] P_exist:", out["P_exist"])


# ------------------------------------------------------------------ run
_G = {}


def _group(records):
    by = OrderedDict()
    for r in records:
        by.setdefault(r["video_id"], []).append(r)
    for v in by:
        by[v].sort(key=lambda r: int(r["frame_t"]))
    return by


def _iou_stats(recs, gm):
    from lib.mllm.eval.iou3d import compute_iou_3d_obb
    st = defaultdict(list)
    for t, r in enumerate(recs):
        pc = r.get("pred_corners")
        if pc is None or gm is None or t >= len(gm["per_frame"]):
            continue
        gc = gm["per_frame"][t]["gt_corners"].astype(np.float32)
        vis = r["visibility_mask"].astype(bool)
        valid = r["valid_mask"].astype(bool)
        for o in range(min(len(pc), len(gc))):
            if not valid[o] or int(r["object_classes"][o]) <= 1:
                continue
            b = "vis" if vis[o] else "unvis"
            has_gt = np.all(np.isfinite(gc[o])) and np.abs(gc[o]).sum() > 0
            if not has_gt:
                continue
            st[b + "_n"].append(1)
            if not np.all(np.isfinite(pc[o])):
                st[b + "_iou"].append(0.0)
                continue
            st[b + "_cov"].append(1)
            try:
                st[b + "_iou"].append(float(compute_iou_3d_obb(pc[o].astype(np.float64), gc[o].astype(np.float64))))
            except Exception:  # noqa: BLE001
                st[b + "_iou"].append(0.0)
    return st


def _run_one(vid):
    from lib.external.puf.fusion import FusionConfig, run_video_graph, run_video_lks
    cfg: FusionConfig = _G["cfg"]
    recs = _G["by"][vid]
    info = {"video_id": vid, "T": len(recs)}
    if cfg.arm in LKS_ARMS:
        out, st = run_video_lks(recs, cfg)
        return vid, out, {**info, **st}
    gp = os.path.join(_G["geom_dir"], f"{vid}.pkl")
    gm, scale = None, 1.0
    if os.path.exists(gp):
        with open(gp, "rb") as f:
            g = pickle.load(f)
        gm = g["modes"].get(cfg.mode)
        scale = g.get("scale", 1.0)
        if gm is not None and len(gm["per_frame"]) != len(recs):
            info["geom_T_mismatch"] = (len(gm["per_frame"]), len(recs))
            gm = None
        elif gm is not None:
            bad = 0
            for t, r in enumerate(recs):
                lid = gm["per_frame"][t]["label_ids"]
                n = min(len(lid), int(r["valid_mask"].sum()))
                if not np.array_equal(lid[:n], r["object_classes"][:n]):
                    bad += 1
            info["label_mismatch_frames"] = bad
            if bad > len(recs) // 2:
                gm = None
    else:
        info["geom_missing"] = True
    out, st = run_video_graph(recs, gm, cfg, scale, _G.get("prior"))
    info.update(st)
    info["scale"] = scale
    if cfg.emit_boxes:
        info["iou"] = {k: (float(np.sum(v)) if k.endswith(("_n", "_cov")) else [float(np.sum(v)), len(v),
                           float(np.sum(np.array(v) >= 0.15)), float(np.sum(np.array(v) >= 0.25))])
                       for k, v in _iou_stats(out, gm).items()}
    return vid, out, info


def cmd_run(a):
    from lib.external.puf.fusion import FusionConfig
    t0 = time.time()
    with open(a.frontend, "rb") as f:
        blob = pickle.load(f)
    by = _group(blob["records"])
    if a.videos_file:
        keep = set(_read_list(a.videos_file))
        by = OrderedDict((v, r) for v, r in by.items() if v.replace(".mp4", "") in keep)
    if a.limit:
        by = OrderedDict(list(by.items())[:a.limit])
    print(f"[run] {a.arm} {a.mode}: {len(by)} videos, {sum(len(r) for r in by.values())} frames "
          f"(load {time.time() - t0:.0f}s)", flush=True)
    cfg = FusionConfig(arm=a.arm, mode=a.mode, edge_decay=a.edge_decay, lambda_birth=a.lambda_birth,
                       sigma_jsd=a.sigma_jsd, l2_gate=a.l2_gate, completion_threshold=a.completion_threshold,
                       emit_boxes=(a.arm in GRAPH_ARMS and not a.no_boxes))
    _G.update(by=by, cfg=cfg, geom_dir=a.geom_dir)
    if a.arm in ("puf_prior", "puf_prior_vis"):
        from lib.external.puf.prior import RelationPrior
        _G["prior"] = RelationPrior(a.prior, strength=a.prior_strength, sigma_d=a.sigma_d,
                                    use_spatial=not a.no_spatial_prior)
    results = {}
    infos = []
    with mp.get_context("fork").Pool(a.workers) as pool:
        for i, (vid, out, info) in enumerate(pool.imap_unordered(_run_one, list(by.keys()), chunksize=4), 1):
            results[vid] = out
            infos.append(info)
            if i % 200 == 0:
                print(f"[run] {i}/{len(by)} {time.time() - t0:.0f}s", flush=True)
    records = [r for v in by for r in results[v]]
    n_frames = len(records)
    meta = dict(blob["meta"])
    meta.update({"experiment": a.name or f"ext_puf_{a.arm}_{a.mode}", "method": f"ext_puf_{a.arm}",
                 "mode": a.mode, "frontend": a.frontend, "frontend_experiment": blob["meta"].get("experiment"),
                 "arm": a.arm, "cfg": cfg.__dict__, "n_videos": len(by), "n_frames": n_frames,
                 "videos_file": a.videos_file, "prior": a.prior, "prior_strength": a.prior_strength,
                 "sigma_d": a.sigma_d, "wall_s": time.time() - t0})
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump({"meta": meta, "records": records}, f, protocol=4)
    # run summary
    summ = {"meta": {k: v for k, v in meta.items() if k != "cfg"}, "cfg": cfg.__dict__}
    agg = defaultdict(float)
    iou = defaultdict(lambda: np.zeros(4))
    for inf in infos:
        for k in ("obs", "obs_nogeom", "births", "assoc", "nodes", "label_mismatch_frames"):
            agg[k] += inf.get(k, 0) or 0
        agg["geom_missing"] += 1 if inf.get("geom_missing") else 0
        agg["geom_T_mismatch"] += 1 if inf.get("geom_T_mismatch") else 0
        for k, v in (inf.get("iou") or {}).items():
            if isinstance(v, list):
                iou[k] += np.array(v)
            else:
                iou[k][0] += v
    summ["agg"] = dict(agg)
    summ["iou3d"] = {}
    for b in ("vis", "unvis"):
        n = iou[b + "_n"][0]
        s = iou[b + "_iou"]
        if n > 0:
            summ["iou3d"][b] = {"n_gt_slots": int(n), "coverage": float(iou[b + "_cov"][0] / n),
                                "mean_iou": float(s[0] / max(s[1], 1)), "acc@0.15": float(s[2] / max(s[1], 1)),
                                "acc@0.25": float(s[3] / max(s[1], 1))}
    src = np.concatenate([r["puf_source"][r["pair_valid"].astype(bool)] for r in records]) if records else []
    vis_pairs = np.concatenate([r["visibility_mask"][r["object_idx"][r["pair_valid"].astype(bool)]]
                                for r in records]) if records else []
    summ["pair_source_counts"] = {
        "observed": {int(k): int(v) for k, v in zip(*np.unique(src[vis_pairs.astype(bool)], return_counts=True))},
        "unobserved": {int(k): int(v) for k, v in zip(*np.unique(src[~vis_pairs.astype(bool)], return_counts=True))},
    }
    with open(a.out.replace(".pkl", "_run.json"), "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps({k: summ[k] for k in ("agg", "iou3d", "pair_source_counts")}, indent=1))
    print(f"[run] wrote {n_frames} frames / {len(by)} videos -> {a.out} ({time.time() - t0:.0f}s)")


# ------------------------------------------------------------------ subset
def cmd_subset(a):
    with open(a.frontend, "rb") as f:
        blob = pickle.load(f)
    keep = set(_read_list(a.videos_file))
    recs = [r for r in blob["records"] if r["video_id"].replace(".mp4", "") in keep]
    meta = dict(blob["meta"])
    meta.update({"n_frames": len(recs), "n_videos": len({r["video_id"] for r in recs}),
                 "videos_file": a.videos_file})
    if a.name:
        meta["experiment"] = a.name
    with open(a.out, "wb") as f:
        pickle.dump({"meta": meta, "records": recs}, f, protocol=4)
    print(f"[subset] {meta['n_videos']} videos {len(recs)} frames -> {a.out}")


# ------------------------------------------------------------------ score
def cmd_score(a):
    from dataloader.world_ag_dataset import (
        ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, OBJECT_CLASSES)
    from lib.supervised.evaluation_recall import BasicSceneGraphEvaluator, evaluate_wsgg_video
    from tools.bucketed_breakdown import process_dump

    pred = list(ATTENTION_RELATIONSHIPS) + list(SPATIAL_RELATIONSHIPS) + list(CONTACTING_RELATIONSHIPS)
    ks = [10, 20, 50, 100]
    t0 = time.time()
    with open(a.dump, "rb") as f:
        blob = pickle.load(f)
    mode = blob["meta"]["mode"]
    stem = a.name or blob["meta"].get("experiment") or Path(a.dump).stem
    evs = {c: BasicSceneGraphEvaluator(
        mode=mode, AG_object_classes=OBJECT_CLASSES, AG_all_predicates=pred,
        AG_attention_predicates=list(ATTENTION_RELATIONSHIPS),
        AG_spatial_predicates=list(SPATIAL_RELATIONSHIPS),
        AG_contacting_predicates=list(CONTACTING_RELATIONSHIPS),
        iou_threshold=0.5, save_file=os.devnull, constraint=cv) for c, cv in (("wc", "with"), ("nc", "no"))}
    for r in blob["records"]:
        for ev in evs.values():
            evaluate_wsgg_video(r, ev, mode=mode, verbose=False)
    res = {"experiment": stem, "mode": mode, "method": blob["meta"].get("method"),
           "n_frames": len(blob["records"]), "schemes": {"all": {}}}
    for c, ev in evs.items():
        s = ev.fetch_stats_json()
        res["schemes"]["all"][c] = {
            "R": {k: round(s["recall"].get(k, 0.0), 6) for k in ks},
            "mR": {k: round(s["mean_recall"].get(k, 0.0), 6) for k in ks},
            "hR": {k: round(s["harmonic_mean_recall"].get(k, 0.0), 6) for k in ks}}
    os.makedirs(os.path.join(a.out_dir, "reeval"), exist_ok=True)
    with open(os.path.join(a.out_dir, "reeval", f"reeval_{stem}.json"), "w") as f:
        json.dump(res, f, indent=2)
    del blob
    bk = process_dump(a.dump)
    bk["meta"]["experiment"] = stem
    with open(os.path.join(a.out_dir, f"bucketed_breakdown_{stem}.json"), "w") as f:
        json.dump([bk], f, indent=2)
    w, n = res["schemes"]["all"]["wc"], res["schemes"]["all"]["nc"]
    nt = bk["constraints"]["nc"].get("ou_nontrivial", {}).get("drop_trivial", {})
    b = bk["constraints"]["nc"]["buckets"]
    print(f"[score] {stem}: wc R@20 {100 * w['R'][20]:.1f} mR@20 {100 * w['mR'][20]:.1f} | nc R@50 "
          f"{100 * n['R'][50]:.1f} mR@50 {100 * n['mR'][50]:.1f} | OO R@20 {100 * (b['OO']['R'][20] or 0):.1f} "
          f"OU R@20 {100 * (b.get('OU', {}).get('R', {}).get(20) or 0):.1f} OU-nt R@20 "
          f"{100 * (nt.get('R', {}).get(20) or 0):.1f} mR@20 {100 * (nt.get('mR', {}).get(20) or 0):.1f} "
          f"({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("geom")
    g.add_argument("--split-file", default="/data3/rohith/ag/splits/test_worldbbox_1511.txt")
    g.add_argument("--videos", nargs="*", default=[])
    g.add_argument("--modes", nargs="+", default=["predcls", "sgdet"])
    g.add_argument("--out-dir", required=True)
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("prior")
    p.add_argument("--train-dir", default="/data/rohith/ag/world4d_rel_annotations/train")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0)
    r = sub.add_parser("run")
    r.add_argument("--frontend", required=True)
    r.add_argument("--mode", required=True, choices=["predcls", "sgdet"])
    r.add_argument("--arm", required=True, choices=list(LKS_ARMS + GRAPH_ARMS))
    r.add_argument("--geom-dir", default="/data3/rohith/ag/runs/ext/puf/geom")
    r.add_argument("--prior", default="/data3/rohith/ag/runs/ext/puf/prior/ag_train_prior.npz")
    r.add_argument("--prior-strength", type=float, default=1.0)
    r.add_argument("--sigma-d", type=float, default=0.5)
    r.add_argument("--no-spatial-prior", action="store_true")
    r.add_argument("--edge-decay", type=float, default=1.0)
    r.add_argument("--lambda-birth", type=float, default=0.4)
    r.add_argument("--sigma-jsd", type=float, default=0.3)
    r.add_argument("--l2-gate", type=float, default=1.5)
    r.add_argument("--completion-threshold", type=float, default=0.0)
    r.add_argument("--no-boxes", action="store_true")
    r.add_argument("--videos-file", default=None)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--name", default=None)
    r.add_argument("--out", required=True)
    s = sub.add_parser("subset")
    s.add_argument("--frontend", required=True)
    s.add_argument("--videos-file", required=True)
    s.add_argument("--name", default=None)
    s.add_argument("--out", required=True)
    sc = sub.add_parser("score")
    sc.add_argument("--dump", required=True)
    sc.add_argument("--out-dir", required=True)
    sc.add_argument("--name", default=None)
    a = ap.parse_args()
    {"geom": cmd_geom, "prior": cmd_prior, "run": cmd_run, "subset": cmd_subset, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
