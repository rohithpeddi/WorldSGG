"""Checkpoint selection without the test set (wqgS Q2, y9Am W3): score every saved epoch of one run, on the fly.

One process per run. Every saved ``checkpoint_<e>`` of the run is loaded once (4-11 M parameters each, all on one
GPU); each video is loaded once and run through every epoch; predictions are scored immediately with the common-GT
covered rules of ``tools/rescore_common_gt.py`` and never stored (per-epoch dumps would be ~96 GB).

Video sets
  * ``test``     the 1,511-video test split (``world4d_rel_annotations_worldbbox``), plus two fixed halves ``testA`` /
                 ``testB``: the videos sorted by sha1(video_id) hex digest, first 756 = A, the other 755 = B.
  * ``heldout``  the 223 AG test videos outside the split (``held_out_test_videos.txt``; never trained on), through an
                 annotation symlink dir (``--heldout-annot``; its PKLs are the OLDER annotation format, which is also
                 the training format, from ``world4d_rel_annotations/test`` or ``test_bak``). WorldWise+ / ++ have no
                 token features or grids for these videos, so their held-out set is empty (reported as such).

Ground truth (covered = objects that own a PredCls slot, as in Table 2)
  * PredCls: the record's own GT (identity slot map); on ``test`` it is checked against the reference dump.
  * SGDet: the PredCls GT of the same frame, from the run's own feature family (its PredCls slot boxes are the box
    space the SGDet slots are matched in); ``test``: the row's saved PredCls dump ``rescore_common_gt.DUMPS[row][0]``
    aligned by (video, frame_t) (relation GT checked against ``GT_REFERENCE``); ``heldout``: PredCls records built
    with the same loader (``predcls`` mode, the run's feature model, held-out annotations) aligned by frame name. The
    builder is checked on 2 test videos against the saved dump at start-up. Slot map: ``slot_map_common``.
Ranking: With Constraint in construction order (Table 2) and sorted by confidence (``rescore_rebuttal.wc_rows``), No
Constraint as in ``rescore_common_gt.ranked_rows``. Frame schemes: ``all`` frames and ``last`` frame per video
(the training-time selection criterion was WC R@20 on the last frame).

Gate: the epoch whose weights are ``best_model.pth`` must reproduce ``common_gt/results/<row>.json`` on ``test`` /
``all`` / construction (max |diff| written as ``gate_best_vs_common_gt``).

    WSGG_REPO=~/CODE/Scene4Cast_rebuttal $PY tools/rebuttal/epoch_sweep.py --config <cfg> --row <row> --out <json>
    $PY tools/rebuttal/epoch_sweep.py --report <dir with the 16 JSONs> [--md out.md]
"""
import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.setdefault("WSGG_REPO", os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import rescore_common_gt as cg  # noqa: E402

KS, BUCKETS = cg.KS, cg.BUCKETS
TEST_LIST = "/data3/rohith/ag/runs/rebuttal/infer/test1511_videos.txt"
HELDOUT_LIST = "/data3/rohith/ag/runs/rebuttal/splits/held_out_test_videos.txt"
HELDOUT_ANNOT = "/data3/rohith/ag/runs/rebuttal/annot_heldout"
GATE_VIDEOS = ("02DPI", "03PRW")
GT_FIELDS = ("pair_valid", "person_idx", "object_idx", "object_classes", "gt_attention", "gt_spatial",
             "gt_contacting", "bboxes_2d", "valid_mask", "visibility_mask")
SLOT_FIELDS = ("pair_valid", "person_idx", "object_idx", "object_classes", "bboxes_2d", "valid_mask")
BASELINES = ("w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg")
WORLDWISE = ("worldwise_dinov3l", "worldwise_plus_dinov3tok", "worldwise_pp_dinov3")


def halves(videos):
    ranked = sorted(videos, key=lambda v: hashlib.sha1(v.encode()).hexdigest())
    n = (len(ranked) + 1) // 2
    return sorted(ranked[:n]), sorted(ranked[n:])


# ----------------------------------------------------------------------------------------------------------------
# scoring core: identical arithmetic to rescore_common_gt.Scorer.add, with the per-frame GT work done once
# ----------------------------------------------------------------------------------------------------------------
def frame_static(rels, tags):
    key_to_gt = defaultdict(list)
    for g, r in enumerate(rels):
        key_to_gt[r].append(g)
    members = {"All": list(range(len(rels))), "OO": [], "UO": []}
    for g, t in enumerate(tags):
        if t in members:
            members[t].append(g)
    cnt = {b: Counter(rels[g][2] for g in members[b]) for b in BUCKETS}
    return key_to_gt, members, cnt


def hits_for(rows, m, key_to_gt):
    out, hit = {}, set()
    kmax = max(KS)
    cuts = set(KS)
    for i, (a, bb, p) in enumerate(rows[:kmax]):
        ga, gb = m.get(int(a), -1), m.get(int(bb), -1)
        if ga >= 0 and gb >= 0:
            hit.update(key_to_gt.get((ga, gb, int(p)), ()))
        if i + 1 in cuts:
            out[i + 1] = set(hit)
    for k in KS:          # fewer rows than k
        if k not in out:
            out[k] = set(hit)
    return out


def apply(sc, rels, members, cnt, hits_by_c):
    sc.n_frames += 1
    for b in BUCKETS:
        for g in members[b]:
            sc.tot[b][rels[g][2]] += 1
    for c, hk in hits_by_c.items():
        for k in KS:
            hit = hk[k]
            for b in BUCKETS:
                idx = members[b]
                if not idx:
                    continue
                h = [g for g in idx if g in hit]
                sc.frame_r[c][b][k].append(len(h) / len(idx))
                hc = Counter(rels[g][2] for g in h)
                for p_, n_ in hc.items():
                    sc.hit[c][b][k][p_] += n_
                for p_, n_ in cnt[b].items():
                    sc.frame_pr[c][b][k][p_].append(hc[p_] / n_)


def wc_sorted(rec):
    import rescore_rebuttal as rr  # noqa: E402  (imported lazily: pulls in the loader module)
    return rr.wc_rows(rec, "sorted")


def score_worker(qin, qout, mode):
    import rescore_rebuttal as rr  # noqa: F401,E402
    scorers = {}
    while True:
        msg = qin.get()
        if msg is None:
            break
        tags_sets, frames = msg
        for fr in frames:
            g, s = fr["g"], fr["s"]
            rels, tags = cg.ground_truth(g)
            if not rels:
                continue
            key_to_gt, members, cnt = frame_static(rels, tags)
            if mode == "predcls":
                m = {j: j for j in np.where(g["valid_mask"].astype(bool))[0].tolist()}
            else:
                s_ = dict(s)
                s_["pred_labels"] = s["object_classes"]
                m = cg.slot_map_common(s_, g, defaultdict(int))
            schemes = ("all", "last") if fr["last"] else ("all",)
            for e, (att, spa, con) in fr["dists"].items():
                rec = dict(s)
                rec["attention_distribution"], rec["spatial_distribution"], rec["contacting_distribution"] = att, spa, con
                if mode == "sgdet":
                    rec["pred_scores"] = np.ones(len(s["object_classes"]), dtype=np.float32)
                con_rows = cg.ranked_rows(rec, "with")
                nc_rows = cg.ranked_rows(rec, "no")
                srt_rows = wc_sorted(rec)
                h_con = {"wc": hits_for(con_rows, m, key_to_gt), "nc": hits_for(nc_rows, m, key_to_gt)}
                h_srt = {"wc": hits_for(srt_rows, m, key_to_gt)}
                for st in tags_sets:
                    for sch in schemes:
                        for order, hb in (("construction", h_con), ("sorted", h_srt)):
                            key = (st, sch, order, e)
                            if key not in scorers:
                                scorers[key] = cg.Scorer()
                            apply(scorers[key], rels, members, cnt, hb)
    out = {}
    for key, sc in scorers.items():
        summ = sc.summary()
        if key[2] == "sorted":
            summ.pop("nc")
        out[key] = summ
    qout.put(out)


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def restrict_video_list(video_ids, check_frames):
    """Process-local replacement of WorldAG._build_video_list: only ``video_ids`` with both PKLs (and, if
    ``check_frames``, >= 2 common frames, the original rule)."""
    from dataloader.world_ag_dataset import WorldAG
    wanted = sorted({v.replace(".mp4", "") for v in video_ids})

    def _build_video_list(self):
        keep = []
        for v in wanted:
            if not (self._feat_dir / f"{v}.pkl").exists():
                continue
            if not any((self._annot_dir / f"{v}{s}.pkl").exists() for s in ("", ".mp4")):
                continue
            if check_frames:
                common, _ = self._align_frames(self._load_feature_pkl(v), self._load_annotation_pkl(v))
                if len(common) < 2:
                    continue
            keep.append(v)
        self.video_list = sorted(keep)

    WorldAG._build_video_list = _build_video_list


def gt_records_from_batch(batch):
    """PredCls GT records (the dump's GT fields) of every frame of one loader item."""
    out = []
    for t in range(int(batch["T"])):
        r = {k: batch[k][t].numpy() for k in GT_FIELDS}
        r["video_id"], r["frame_t"] = batch["video_id"], t
        out.append(r)
    return out


def make_gt_dataset(conf, annot_dir_name, videos):
    """PredCls loader of the run's own feature family: its slot boxes are the box space the run's SGDet slots are
    matched in (ResNet-50 and DINOv3 PredCls PKLs carry different boxes)."""
    from dataloader.world_ag_dataset import WorldAG
    restrict_video_list(videos, check_frames=True)
    return WorldAG(phase="test", data_path=conf.data_path, mode="predcls", feature_model=conf.feature_model,
                   include_invisible=True, max_objects=getattr(conf, "max_objects", 64),
                   annot_dir_name=annot_dir_name, visibility_from_annotation=True)


def same_gt(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in GT_FIELDS if k != "bboxes_2d") and \
        np.allclose(np.asarray(a["bboxes_2d"]), np.asarray(b["bboxes_2d"]), atol=1e-3)


# ----------------------------------------------------------------------------------------------------------------
# sweep
# ----------------------------------------------------------------------------------------------------------------
def sweep(a):
    t0 = time.time()
    ctx = mp.get_context("fork")
    from wsgg_base import load_wsgg_config
    conf = load_wsgg_config(a.config)
    mode = conf.mode
    # scoring workers first (before CUDA is initialised in this process)
    qouts = ctx.Queue()
    qins = [ctx.Queue(maxsize=8) for _ in range(a.workers)]
    procs = [ctx.Process(target=score_worker, args=(q, qouts, mode), daemon=True) for q in qins]
    for p_ in procs:
        p_.start()

    import torch
    from torch.utils.data import DataLoader
    from dataloader.world_ag_dataset import world_collate_fn
    from tools.reeval_test import build_model, forward_all_frames, load_checkpoint, make_test_dataset

    exp_dir = os.path.join(conf.save_path, conf.experiment_name)
    epochs = sorted(int(p_.rsplit("_", 1)[1]) for p_ in glob.glob(f"{exp_dir}/checkpoint_*")
                    if os.path.isfile(f"{p_}/checkpoint_state.pth"))
    if a.epochs:
        epochs = [e for e in epochs if e in set(a.epochs)]
    best_state = torch.load(f"{exp_dir}/best_model.pth", map_location="cpu", weights_only=False)
    best_epoch = int(best_state.get("epoch"))
    del best_state
    owner = {e: i % a.workers for i, e in enumerate(epochs)}
    print(f"[sweep] {a.row} {mode} {conf.experiment_name}: epochs {epochs}, best_model = epoch {best_epoch}", flush=True)

    test_videos = [ln.strip() for ln in open(TEST_LIST) if ln.strip()]
    held_videos = [ln.strip().replace(".mp4", "") for ln in open(HELDOUT_LIST) if ln.strip()]
    if a.max_videos:
        test_videos, held_videos = test_videos[:a.max_videos], held_videos[:a.max_videos]
    half_a, half_b = halves(test_videos)
    half_of = {v: "testA" for v in half_a}
    half_of.update({v: "testB" for v in half_b})

    # datasets: test (the model's own loader), held-out (same loader, held-out annotation dir)
    restrict_video_list(test_videos, check_frames=False)
    ds_test = make_test_dataset(conf)
    assert ds_test.video_list == sorted(test_videos), (len(ds_test.video_list), len(test_videos))
    restrict_video_list(held_videos, check_frames=True)
    conf_h = load_wsgg_config(a.config)
    conf_h.test_annot_dir = a.heldout_annot
    ds_held = make_test_dataset(conf_h)
    assert str(ds_held._annot_dir) == os.path.join(a.heldout_annot, "test"), ds_held._annot_dir
    print(f"[sweep] test videos {len(ds_test)}, held-out videos {len(ds_held)} "
          f"(annotations {ds_held._annot_dir})", flush=True)

    # test-split GT = the saved PredCls dump of this row (same feature family; relation GT checked against the
    # common reference on every frame), and the held-out GT builder (checked on 2 test videos against it)
    from infer_score import load_reference
    ref, _, ref_index = load_reference(a.row)
    gt_check = {}
    held_gt = None
    if len(ds_held):
        ds_chk = make_gt_dataset(conf, conf.test_annot_dir, list(GATE_VIDEOS))
        n_eq = n_tot = 0
        for i in range(len(ds_chk)):
            for r in gt_records_from_batch(ds_chk[i]):
                n_tot += 1
                n_eq += int(same_gt(r, ref[ref_index[(r["video_id"], r["frame_t"])]]))
        gt_check["builder_vs_reference_frames"] = [n_eq, n_tot]
        assert n_eq == n_tot > 0, f"held-out GT builder does not reproduce the reference GT ({n_eq}/{n_tot})"
        held_gt = make_gt_dataset(conf, a.heldout_annot, ds_held.video_list)
        held_gt_index = {v: i for i, v in enumerate(held_gt.video_list)}
        print(f"[sweep] GT builder check {n_eq}/{n_tot} frames equal to the reference; "
              f"held-out GT videos {len(held_gt)}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    models = {}
    for e in epochs:
        mdl = build_model(conf, ds_test, device)
        load_checkpoint(mdl, conf, f"{exp_dir}/checkpoint_{e}/checkpoint_state.pth", device)
        mdl.eval()
        models[e] = mdl
    mem = torch.cuda.memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    print(f"[sweep] {len(models)} models on {device}; GPU memory allocated {mem:.2f} GB", flush=True)

    stats = defaultdict(int)
    for set_name, ds in (("test", ds_test), ("heldout", ds_held)):
        if not len(ds):
            stats[f"{set_name}_videos"] = 0
            continue
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=a.loader_workers, collate_fn=world_collate_fn,
                        prefetch_factor=2 if a.loader_workers else None, persistent_workers=False)
        tl = time.time()
        for vi, batch in enumerate(dl):
            vid, T = batch["video_id"], int(batch["T"])
            if T <= 0:
                continue
            # ground truth per frame
            if set_name == "test":
                gts = [ref[ref_index[(vid, t)]] for t in range(T)]
                assert (vid, T) not in ref_index and (vid, T - 1) in ref_index, (vid, T)
                tags_sets = ("test", half_of[vid])
            else:
                tags_sets = ("heldout",)
                gb = held_gt[held_gt_index[vid]] if vid in held_gt_index else None
                by_name = {}
                if gb is not None:
                    for t, r in enumerate(gt_records_from_batch(gb)):
                        by_name[gb["frame_names"][t]] = r
                gts = [by_name.get(fn) for fn in batch["frame_names"]]
            own = gt_records_from_batch(batch) if mode == "predcls" else None
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            dists = {}
            with torch.no_grad():
                for e, mdl in models.items():
                    pred = forward_all_frames(mdl, conf, b)
                    dists[e] = tuple(pred[k].float().cpu().numpy() for k in
                                     ("attention_distribution", "spatial_distribution", "contacting_distribution"))
            per_worker = [[] for _ in qins]
            for t in range(T):
                g = gts[t]
                if mode == "predcls":
                    if g is not None and not same_gt(own[t], g):
                        stats[f"{set_name}_predcls_gt_differs_from_reference_frames"] += 1
                    g = own[t]         # PredCls scores against its own GT (= the reference on the test split)
                if g is None:
                    stats[f"{set_name}_frames_without_gt"] += 1
                    continue
                s = {k: batch[k][t].numpy() for k in SLOT_FIELDS}
                stats[f"{set_name}_frames"] += 1
                for w in range(len(qins)):
                    de = {e: (d[0][t], d[1][t], d[2][t]) for e, d in dists.items() if owner[e] == w}
                    per_worker[w].append({"g": g, "s": s, "last": t == T - 1, "dists": de})
            for w, q in enumerate(qins):
                q.put((tags_sets, per_worker[w]))
            stats[f"{set_name}_videos"] += 1
            if vi % 100 == 0 or vi == len(ds) - 1:
                el = time.time() - tl
                print(f"[sweep] {set_name} {vi + 1}/{len(ds)} videos, {el / 60:.1f} min "
                      f"({el / (vi + 1):.2f} s/video)", flush=True)
    for q in qins:
        q.put(None)
    scores = {}
    for _ in procs:
        part = qouts.get()
        for (st, sch, order, e), summ in part.items():
            scores.setdefault(st, {}).setdefault(sch, {}).setdefault(order, {})[e] = summ
    for p_ in procs:
        p_.join()

    # gate: best epoch on test/all/construction vs the saved common-GT JSON
    gate = None
    saved_path = f"{cg.R}/common_gt/results/{a.row}.json"
    if os.path.exists(saved_path) and not a.max_videos and best_epoch in epochs:
        saved = json.load(open(saved_path))[mode if mode == "predcls" else "sgdet_common"]
        mine = scores["test"]["all"]["construction"][best_epoch]
        gate = 0.0
        for c in ("wc", "nc"):
            for bk in BUCKETS:
                for met in ("R", "mR", "pooled_R", "pooled_mR"):
                    for k in KS:
                        x, y = mine[c][bk][met][k], saved[c][bk][met][str(k)]
                        if x is None or y is None:
                            gate = gate if (x is None and y is None) else float("inf")
                            continue
                        gate = max(gate, abs(x - y))
    res = {"row": a.row, "mode": mode, "config": a.config, "experiment": conf.experiment_name, "exp_dir": exp_dir,
           "epochs": epochs, "best_epoch": best_epoch, "final_epoch": max(epochs),
           "gate_best_vs_common_gt": gate, "gate_reference": saved_path, "gt_check": gt_check,
           "stats": dict(stats), "halves": {"testA": half_a, "testB": half_b,
                                            "rule": "sorted by sha1(video_id) hex digest; first ceil(n/2) = testA"},
           "heldout_note": ("held-out annotations are the older world4d_rel_annotations format (also the training "
                            "format), symlinked from world4d_rel_annotations/test or test_bak"),
           "scores": scores, "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, default=str)
    print(f"[sweep] done {a.row} {mode}: gate max|diff| best epoch {best_epoch} vs common_gt = {gate} | "
          f"{dict(stats)} | {res['seconds'] / 60:.1f} min -> {a.out}", flush=True)
    if gate is not None and gate > 1e-3:
        print("[sweep] GATE FAIL", flush=True)
        return 1
    return 0


# ----------------------------------------------------------------------------------------------------------------
# report: selections and WorldWise-family vs best-baseline gaps
# ----------------------------------------------------------------------------------------------------------------
METRICS = {"WC R@20": ("wc", "All", "R"), "WC mR@20": ("wc", "All", "mR"), "NC R@20": ("nc", "All", "R"),
           "NC mR@20": ("nc", "All", "mR"), "NC UO R@20": ("nc", "UO", "R"), "NC UO mR@20": ("nc", "UO", "mR")}


def metric(summ, name, order_summ=None):
    c, b, m = METRICS[name]
    s = order_summ if (c == "wc" and order_summ is not None) else summ
    v = s[c][b][m]["20"] if "20" in s[c][b][m] else s[c][b][m][20]
    return None if v is None else 100.0 * v


def report(a):
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.report, "*.json"))):
        d = json.load(open(f))
        if "scores" in d:
            runs[(d["row"], d["mode"])] = d
    order = a.order
    out = {"order": order, "runs": {}}
    lines = [f"WC ranking for selection and WC cells: {order}. Values in %. Selections: best = best_model.pth "
             f"(test-selected, last frame, legacy evaluator); final = last epoch; held-out = argmax held-out WC R@20 "
             f"(all frames / last frame); cross-half = select on one test half, report on the other (mean of both "
             f"directions)."]
    for (row, mode), d in sorted(runs.items(), key=lambda x: (x[0][1], x[0][0])):
        sc = d["scores"]
        ep = [int(e) for e in d["epochs"]]

        def get(st, sch, e, name):
            con = sc[st][sch]["construction"][str(e)]
            srt = sc[st][sch]["sorted"][str(e)] if order == "sorted" else None
            return metric(con, name, srt)

        sel = {"best": d["best_epoch"], "final": d["final_epoch"]}
        if "heldout" in sc and d["stats"].get("heldout_videos", 0) >= 100:   # WW+/WW++: 0-1 held-out videos
            sel["heldout_all"] = max(ep, key=lambda e: get("heldout", "all", e, "WC R@20"))
            sel["heldout_last"] = max(ep, key=lambda e: get("heldout", "last", e, "WC R@20"))
        sel["test_all_oracle"] = max(ep, key=lambda e: get("test", "all", e, "WC R@20"))
        eA = max(ep, key=lambda e: get("testA", "all", e, "WC R@20"))
        eB = max(ep, key=lambda e: get("testB", "all", e, "WC R@20"))
        vals = {k: {m: get("test", "all", e, m) for m in METRICS} for k, e in sel.items()}
        vals["cross_half"] = {m: 0.5 * (get("testB", "all", eA, m) + get("testA", "all", eB, m)) for m in METRICS}
        sel["cross_half"] = f"A->{eA}, B->{eB}"
        out["runs"][f"{row}|{mode}"] = {"selected_epoch": sel, "values": vals,
                                        "gate_best_vs_common_gt": d.get("gate_best_vs_common_gt"),
                                        "heldout_videos": d["stats"].get("heldout_videos", 0)}
    # gaps: each WorldWise-family row minus the best baseline, per selection and metric
    gaps = {}
    for mode in ("predcls", "sgdet"):
        for selname in ("best", "final", "heldout_all", "heldout_last", "cross_half", "test_all_oracle"):
            base = [out["runs"].get(f"{r}|{mode}") for r in BASELINES]
            base = [b for b in base if b and selname in b["values"]]
            for ww in WORLDWISE:
                w = out["runs"].get(f"{ww}|{mode}")
                if not w or selname not in w["values"] or not base:
                    continue
                for m in ("WC mR@20", "NC UO R@20", "NC UO mR@20", "WC R@20"):
                    bb = max(b["values"][selname][m] for b in base)
                    gaps.setdefault(mode, {}).setdefault(selname, {}).setdefault(ww, {})[m] = \
                        round(w["values"][selname][m] - bb, 2)
    out["gaps_vs_best_baseline"] = gaps
    js = a.md.replace(".md", ".json") if a.md else os.path.join(a.report, "_report.json")
    with open(js, "w") as f:
        json.dump(out, f, indent=1)
    for mode in ("predcls", "sgdet"):
        lines.append(f"\n### {mode}\n")
        lines.append("| run | best | final | held-out (all) | held-out (last) | cross-half | gate |")
        lines.append("|---|---|---|---|---|---|---|")
        for k, r in out["runs"].items():
            if not k.endswith(mode):
                continue
            s = r["selected_epoch"]
            lines.append(f"| {k.split('|')[0]} | {s['best']} | {s['final']} | {s.get('heldout_all', '-')} | "
                         f"{s.get('heldout_last', '-')} | {s['cross_half']} | {r['gate_best_vs_common_gt']} |")
        for m in ("WC R@20", "WC mR@20", "NC UO R@20", "NC UO mR@20"):
            lines.append(f"\n{mode} {m} (test split, all frames, covered GT)\n")
            lines.append("| run | best | final | held-out (all) | held-out (last) | cross-half |")
            lines.append("|---|---|---|---|---|---|")
            for k, r in out["runs"].items():
                if not k.endswith(mode):
                    continue
                v = r["values"]
                cells = [f"{v[s][m]:.1f}" if s in v and v[s][m] is not None else "-"
                         for s in ("best", "final", "heldout_all", "heldout_last", "cross_half")]
                lines.append(f"| {k.split('|')[0]} | " + " | ".join(cells) + " |")
    sels = ("best", "final", "heldout_all", "heldout_last", "cross_half", "test_all_oracle")
    for mode in ("predcls", "sgdet"):
        for m in ("WC mR@20", "NC UO R@20", "NC UO mR@20", "WC R@20"):
            lines.append(f"\n{mode}: {m}, WorldWise-family row minus the best of the 5 baselines (same selection rule)\n")
            lines.append("| row | " + " | ".join(sels) + " |")
            lines.append("|---|" + "---|" * len(sels))
            for ww in WORLDWISE:
                cells = []
                for sn in sels:
                    v = gaps.get(mode, {}).get(sn, {}).get(ww, {}).get(m)
                    cells.append("-" if v is None else f"{v:+.1f}")
                lines.append(f"| {ww} | " + " | ".join(cells) + " |")
    txt = "\n".join(lines) + "\n"
    if a.md:
        with open(a.md, "w") as f:
            f.write(txt)
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--row", help="key of rescore_common_gt.DUMPS (for the gate JSON)")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, nargs="*", default=None)
    ap.add_argument("--max-videos", type=int, default=0, help="debug: first N videos of each set")
    ap.add_argument("--workers", type=int, default=4, help="scoring processes")
    ap.add_argument("--loader-workers", type=int, default=3)
    ap.add_argument("--heldout-annot", default=HELDOUT_ANNOT)
    ap.add_argument("--report", default=None, help="directory of sweep JSONs: print selections and gaps")
    ap.add_argument("--order", default="sorted", choices=["sorted", "construction"])
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    if a.report:
        report(a)
        return 0
    return sweep(a)


if __name__ == "__main__":
    sys.exit(main())
