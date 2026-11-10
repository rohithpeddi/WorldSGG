"""SGDet relation recall with slot -> object matching in 3-D (SRrf W1/Q2, wqgS Q1 part 2). CPU only.

Input: an SGDet dump written with ``tools/dump_predictions.py --keep-3d`` (1,511-video test split or a subset).
Each slot has ``bboxes_3d`` = the detector's input corners mapped camera -> canonical with the frame's camera pose
(``reeval_test.build_pred_pkl``); WorldWise++ dumps also have ``pred_corners_slot`` = those corners plus the learned
``slot_corner_head`` residual, mapped the same way (the "refined" box).

Ground truth: the common-GT relations of ``rescore_common_gt`` (covered = objects with a PredCls slot; full = every
annotated object, from the ``rescore_rebuttal`` GT cache), taken from the PredCls dump of the same feature family as
the SGDet dump (its 2-D boxes live in that family's box space; see ``infer_score.load_reference``). Ground-truth boxes are the annotation's ``corners_final``
(canonical frame), looked up per object exactly as the loader does (first object of that short label;
``person_info`` for the person); pseudo objects of the full inventory use their own ``object_info_list`` entry.

Matching rules (a slot is mapped to at most one GT object of its predicted class; visible and unobserved objects are
treated alike; there is no class-only fallback):
  * ``iou3d@tau``     oriented 3-D IoU (``lib/mllm/eval/iou3d.compute_iou_3d_obb``: floor-plane polygon IoU x
                      z-overlap, z-up) >= tau, tau in {0.10, 0.25, 0.50}; the best-IoU object wins;
  * ``centre@delta``  centre distance <= delta x the GT box diagonal, delta in {0.5, 1.0}; the nearest wins.
  * ``2d``            the common-GT 2-D rule (class + 2-D IoU >= 0.5 if visible, class only if unobserved), for the
                      side-by-side; ``covered`` = ``slot_map_common``, ``full`` = ``rescore_rebuttal.slot_map_full``.
Recall: ``rescore_common_gt.Scorer`` (per-frame R / mR, mR over 26 predicates), WC sorted by confidence
(``rescore_rebuttal.wc_rows(.., "sorted")``) and NC.

Gate (frame convention), always computed and written under ``gate``: for visible GT objects that a slot matches by
class + 2-D IoU >= 0.5, the median 3-D IoU of the slot's input box must be clearly above 0 (camera-frame corners score
~0 against canonical GT). ``--gate-only`` stops after it. The dump's ``corner_fallback`` counters (annotation-corner
fallback of the loader) are copied into the output.

Histograms (``hist``): 3-D IoU against the 2-D-rule-matched GT object, input box vs refined box (when present),
split by GT visibility (OO-side objects vs UO-side objects).

Caveats to state with any number: coordinates are reconstruction units, not metres; every unobserved test box is a
frozen copy of the object's nearest seen box (AUDIT DATA-10), not an independent 3-D annotation.

    WSGG_REPO=~/CODE/Scene4Cast_rebuttal $PY tools/rebuttal/rescore_3d.py --dump <pkl> --name <name> [--gate-only]
"""
import argparse
import json
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.setdefault("WSGG_REPO", os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import rescore_common_gt as cg  # noqa: E402
import rescore_rebuttal as rr  # noqa: E402
from infer_score import default_gt_row, load_reference  # noqa: E402
from lib.mllm.eval.iou3d import compute_iou_3d_obb  # noqa: E402
from dataloader.world_ag_dataset import OBJECT_CLASSES, _to_short  # noqa: E402

TAUS = (0.10, 0.25, 0.50)
DELTAS = (0.5, 1.0)
BINS = (0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.0000001)
PERSON = OBJECT_CLASSES.index("person")


def _ok(c):
    if c is None:
        return None
    c = np.asarray(c, dtype=np.float64)
    if c.shape != (8, 3) or not np.isfinite(c).all() or not np.abs(c).sum() > 0:
        return None
    return c


class GT3D:
    """GT corners per (record, GT object index), read from the annotation PKLs (one video in memory at a time)."""

    def __init__(self, annot_dir=rr.ANNOT):
        self.annot_dir = annot_dir
        self.vid, self.ann, self.frames = None, None, None

    def frame(self, vid, t):
        if vid != self.vid:
            with open(f"{self.annot_dir}/{vid}.mp4.pkl", "rb") as f:
                self.ann = pickle.load(f)
            self.frames = sorted(self.ann["frames"], key=lambda k: k.split("/")[-1])
            self.vid = vid
        return self.ann["frames"][self.frames[int(t)]]

    def objects(self, g, extra):
        """{gt index: (class id, corners or None, visible)} for covered slots and pseudo (uncovered) objects."""
        af = self.frame(g["video_id"], g["frame_t"])
        first = {}
        for o in af.get("object_info_list", []):
            first.setdefault(o.get("label", _to_short(o.get("class", ""))), o)
        out = {}
        vis = g["visibility_mask"].astype(bool)
        for j in np.where(g["valid_mask"].astype(bool))[0]:
            c = int(g["object_classes"][j])
            if c == PERSON:
                obj = af.get("person_info", {})
            else:
                obj = first.get(_to_short(OBJECT_CLASSES[c]))
            out[int(j)] = (c, _ok(obj.get("corners_final")) if obj is not None else None, bool(vis[j]))
        lst = af.get("object_info_list", [])
        for (o, cid) in sorted({(e[3], e[2]) for e in extra}):
            k = o - rr.PSEUDO
            obj = lst[k]
            out[o] = (cid, _ok(obj.get("corners_final")), bool(obj.get("visible", True)))
        return out


def used_slots(s):
    pv = s["pair_valid"].astype(bool)
    return sorted(set(s["person_idx"][pv].tolist()) | set(s["object_idx"][pv].tolist()))


def centre_diag(c):
    return c.mean(0), float(np.linalg.norm(c.max(0) - c.min(0)))


def map_3d(slots, labels, boxes, objs, allowed, pair_iou, pair_dist):
    """Per rule -> {slot: gt}. ``allowed``: set of GT indices in this inventory. Fills the pair caches."""
    maps = {f"iou3d@{t:.2f}": {} for t in TAUS}
    maps.update({f"centre@{d:.1f}": {} for d in DELTAS})
    by_class = defaultdict(list)
    for j, (c, corners, _) in objs.items():
        if j in allowed and corners is not None:
            by_class[c].append(j)
    for s in slots:
        b = boxes[s]
        if b is None:
            continue
        cands = by_class.get(int(labels[s]), [])
        if not cands:
            continue
        ious, dists = [], []
        for j in cands:
            key = (s, j)
            if key not in pair_iou:
                pair_iou[key] = compute_iou_3d_obb(b, objs[j][1])
                cb, _ = centre_diag(b)
                cg_, dg = centre_diag(objs[j][1])
                pair_dist[key] = (float(np.linalg.norm(cb - cg_)), dg)
            ious.append(pair_iou[key])
            dists.append(pair_dist[key])
        for t in TAUS:
            k = int(np.argmax(ious))
            if ious[k] >= t:
                maps[f"iou3d@{t:.2f}"][s] = cands[k]
        for d in DELTAS:
            ok = [(dist, j) for (dist, diag), j in zip(dists, cands) if dist <= d * diag]
            if ok:
                maps[f"centre@{d:.1f}"][s] = min(ok)[1]
    return maps


def hist(vals):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    h, _ = np.histogram(v, bins=BINS)
    return {"n": int(v.size), "median": float(np.median(v)), "mean": float(v.mean()),
            "frac_gt0": float((v > 0).mean()), "frac_ge_0.10": float((v >= 0.10).mean()),
            "frac_ge_0.25": float((v >= 0.25).mean()), "frac_ge_0.50": float((v >= 0.50).mean()),
            "bins": [round(b, 2) for b in BINS[:-1]] + [1.0], "counts": h.tolist()}


def load_explicit_reference(ref_path, cache_path):
    """TRAIN-round GT: a PredCls GT reference with the dump's slot layout and its full-GT cache (the --ref / --cache
    pair that ``train_rescore.py score`` uses for the same dump)."""
    ref = cg.load(ref_path)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    assert len(ref) == len(cache["keys"]), (len(ref), len(cache["keys"]))
    for i, r in enumerate(ref):
        assert (r["video_id"], int(r["frame_t"])) == tuple(cache["keys"][i]), (i, r["video_id"], r["frame_t"])
    return ref, cache, {tuple(k): i for i, k in enumerate(cache["keys"])}


def run(dump, name, out_dir, gate_only=False, max_records=0, gt_row=None, ref_path=None, cache_path=None,
        annot_dir=None):
    t0 = time.time()
    with open(dump, "rb") as f:
        d = pickle.load(f)
    meta, recs = d.get("meta", {}), d["records"]
    assert recs and "bboxes_3d" in recs[0], "dump has no bboxes_3d: write it with dump_predictions.py --keep-3d"
    sources = ["input"] + (["refined"] if "pred_corners_slot" in recs[0] else [])
    if ref_path:
        assert cache_path, "--ref needs --cache"
        ref, cache, index = load_explicit_reference(ref_path, cache_path)
        gt_row = None
    else:
        gt_row = gt_row or default_gt_row(meta)
        ref, cache, index = load_reference(gt_row)   # PredCls GT of the same feature family (its own 2-D box space)
    gt3d = GT3D(annot_dir or rr.ANNOT)
    rules = [f"iou3d@{t:.2f}" for t in TAUS] + [f"centre@{d:.1f}" for d in DELTAS]
    sc = {"2d|covered": cg.Scorer(), "2d|full_rule": cg.Scorer()}
    for src in sources:
        for inv in ("covered", "full"):
            for r in rules:
                sc[f"{src}|{inv}|{r}"] = cg.Scorer()
    st = defaultdict(int)
    gate = {src: [] for src in sources}
    hv = {f"{src}|{v}": [] for src in sources for v in ("visible", "unobserved")}
    gt_corner_check = [0, 0]
    diags = []
    for ri, s in enumerate(recs):
        if max_records and ri >= max_records:
            break
        i = index[(s["video_id"], int(s["frame_t"]))]
        g, ex, dr = ref[i], cache["extras"][i], set(cache["drops"][i])
        rels, tags = cg.ground_truth(g)
        frels = [r for gi, r in enumerate(rels) if gi not in dr] + [e[0] for e in ex]
        ftags = [t for gi, t in enumerate(tags) if gi not in dr] + [e[1] for e in ex]
        st["records"] += 1
        st["records_without_cam_pose"] += int(not s.get("has_cam_pose", True))
        objs = gt3d.objects(g, ex)
        slots = used_slots(s)
        labels = s.get("pred_labels", s["object_classes"])
        boxes = {}
        for src in sources:
            arr = s["bboxes_3d"] if src == "input" else s["pred_corners_slot"]
            boxes[src] = {sl: _ok(arr[sl]) for sl in slots}
        st["used_slots"] += len(slots)
        st["used_slots_without_input_box"] += sum(1 for sl in slots if boxes["input"][sl] is None)
        for j, (c, corners, vis) in objs.items():
            st["gt_objects"] += 1
            st["gt_objects_without_corners"] += int(corners is None)
            if corners is not None and j < rr.PSEUDO:
                diags.append(centre_diag(corners)[1])
        # 2-D rule maps (side-by-side reference, gate pairs, histogram pairs)
        st2 = defaultdict(int)
        m2c = cg.slot_map_common(s, g, st2) if rels or frels else {}
        m2f = rr.slot_map_full(s, g, ex) if frels else {}
        gvis = g["visibility_mask"].astype(bool)
        for sl, j in m2c.items():
            corners = objs.get(j, (None, None, None))[1]
            if corners is None:
                continue
            visible = bool(gvis[j])
            if visible:   # matched by class + 2-D IoU >= 0.5 (slot_map_common's visible branch)
                if _ok(s["gt_corners"][sl]) is not None:
                    gt_corner_check[0] += 1
                    gt_corner_check[1] += int(np.allclose(s["gt_corners"][sl], corners, atol=1e-4))
            for src in sources:
                b = boxes[src].get(sl)
                v = compute_iou_3d_obb(b, corners) if b is not None else 0.0
                hv[f"{src}|{'visible' if visible else 'unobserved'}"].append(v)
                if visible:
                    gate[src].append(v)
        if gate_only:
            continue
        nc = cg.ranked_rows(s, "no")
        srt = rr.wc_rows(s, "sorted")
        rows = {"wc": srt, "nc": nc}
        if rels:
            sc["2d|covered"].add(rels, tags, rows, m2c)
        if frels:
            sc["2d|full_rule"].add(frels, ftags, rows, m2f)
        covered_idx = set(objs) - {o for o in objs if o >= rr.PSEUDO}
        for src in sources:
            pair_iou, pair_dist = {}, {}
            for inv, allowed, R_, T_ in (("covered", covered_idx, rels, tags), ("full", set(objs), frels, ftags)):
                if not R_:
                    continue
                maps = map_3d(slots, labels, boxes[src], objs, allowed, pair_iou, pair_dist)
                for r in rules:
                    sc[f"{src}|{inv}|{r}"].add(R_, T_, rows, maps[r])
                    if inv == "full":
                        st[f"{src}|full|{r}|mapped_slots"] += len(maps[r])
                        st[f"{src}|full|{r}|mapped_to_unobserved"] += sum(1 for j in maps[r].values()
                                                                          if not objs[j][2])
    gate_out = {src: hist(v) for src, v in gate.items()}
    med = gate_out["input"].get("median", 0.0)
    gate_out["pass"] = bool(gate_out["input"].get("n", 0) > 0 and med > 0.02 and gate_out["input"]["frac_gt0"] > 0.5)
    gate_out["rule"] = "median 3-D IoU of input boxes on 2-D-matched (IoU>=0.5) visible pairs > 0.02 and >50% > 0"
    gate_out["gt_corners_equal_annotation"] = {"n": gt_corner_check[0], "equal": gt_corner_check[1]}
    res = {"name": name, "dump": dump, "meta": meta, "n_records": st["records"], "box_sources": sources,
           "gt_row": gt_row, "gt_dump": ref_path or cg.DUMPS[gt_row][0], "gt_cache": cache_path or rr.GT_CACHE,
           "corner_fallback": meta.get("corner_fallback"), "stats": dict(st), "gate": gate_out,
           "gt_box_diagonal_units": ({"median": float(np.median(diags)), "p10": float(np.percentile(diags, 10)),
                                      "p90": float(np.percentile(diags, 90))} if diags else None),
           "hist": {k: hist(v) for k, v in hv.items()},
           "caveats": ["coordinates are reconstruction units, not metres",
                       "every unobserved test box is a frozen copy of the nearest seen box (AUDIT DATA-10)",
                       "3-D IoU = floor-plane polygon IoU x z-overlap (z-up), lib/mllm/eval/iou3d.py"]}
    if not gate_only:
        for n, s_ in sc.items():
            res[n] = s_.summary()
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}{'_gate' if gate_only else ''}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=1, default=str)
    print(f"[{name}] gate {'PASS' if gate_out['pass'] else 'FAIL'}: input median IoU3D={med:.3f} "
          f"(n={gate_out['input'].get('n', 0)}), "
          + (f"refined median={gate_out['refined'].get('median', float('nan')):.3f} " if 'refined' in gate_out else "")
          + f"fallback={meta.get('corner_fallback')} no_pose_records={st['records_without_cam_pose']} "
          f"{res['seconds']}s -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="/data3/rohith/ag/runs/rebuttal/infer/score3d")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--strict-gate", action="store_true", help="exit 3 when the frame gate fails")
    ap.add_argument("--gt-row", default=None, help="key of rescore_common_gt.DUMPS whose PredCls dump gives the GT "
                                                   "(default: infer_score.default_gt_row)")
    ap.add_argument("--ref", default=None, help="explicit PredCls GT reference (TRAIN round: train/gt/predcls_*_aligned"
                                                ".pkl); replaces --gt-row")
    ap.add_argument("--cache", default=None, help="full-GT cache that goes with --ref (train/gt/full_gt_*_aligned.pkl)")
    ap.add_argument("--annot_dir", default=None, help=f"annotation PKLs with corners_final (default {rr.ANNOT})")
    a = ap.parse_args()
    r = run(a.dump, a.name, a.out, a.gate_only, a.max_records, a.gt_row, a.ref, a.cache, a.annot_dir)
    sys.exit(3 if a.strict_gate and not r["gate"]["pass"] else 0)
