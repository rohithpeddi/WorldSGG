"""V2: localisation of hidden objects in a V2 test dump. CPU only.

For every hidden slot-frame (valid, not visible) whose object has an annotated box (gt_corners), compares
  * the V2 decoded box (pred_corners_slot)                       -> "v2"
  * the copy reference: the slot's input box (bboxes_3d)        -> "copy" (PredCls: the annotated box itself, so
    the ceiling; SGDet: the held box from the nearest detection)
with the annotated box: oriented 3-D IoU (lib/mllm/eval/iou3d.py) >= 0.10 / 0.25 / 0.50 and centre distance / GT
diagonal, overall and by frames since the slot's nearest visible frame (1, 2-3, 4-10, >10, never visible in the
dump). Also the mean latent cosine (geo_cos) over the same slots.

    python tools/rebuttal/v2/v2_localize.py --dump <exp>__<ckpt>__all3d.pkl --out <json>
"""
import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from lib.mllm.eval.iou3d import compute_iou_3d_obb  # noqa: E402

TAUS = (0.10, 0.25, 0.50)


def ok_box(c):
    c = np.asarray(c, dtype=np.float64)
    return c.shape == (8, 3) and np.isfinite(c).all() and np.abs(c).sum() > 0


def gap_bucket(g):
    if g is None:
        return "never visible"
    return "1" if g <= 1 else "2-3" if g <= 3 else "4-10" if g <= 10 else ">10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.dump, "rb"))
    recs = d["records"]
    by_vid = defaultdict(list)
    for r in recs:
        by_vid[r["video_id"]].append(r)
    acc = defaultdict(lambda: defaultdict(list))        # bucket -> metric -> values
    for vid, rs in by_vid.items():
        rs = sorted(rs, key=lambda r: int(r["frame_t"]))
        vis = np.stack([r["visibility_mask"].astype(bool) & r["valid_mask"].astype(bool) for r in rs])   # (T, N)
        T = vis.shape[0]
        for t, r in enumerate(rs):
            hid = np.asarray(r["v2_hidden"]).astype(bool)
            for s in np.where(hid)[0]:
                gt = r["gt_corners"][s]
                if not ok_box(gt):
                    continue
                seen = np.where(vis[:, s])[0]
                gap = int(np.min(np.abs(seen - t))) if seen.size else None
                diag = float(np.linalg.norm(np.asarray(gt).max(0) - np.asarray(gt).min(0)))
                for name, key in (("v2", "pred_corners_slot"), ("copy", "bboxes_3d")):
                    b = r[key][s]
                    if not ok_box(b):
                        iou, cerr = 0.0, float("inf")
                    else:
                        iou = float(compute_iou_3d_obb(np.asarray(b, np.float64), np.asarray(gt, np.float64)))
                        cerr = float(np.linalg.norm(np.asarray(b).mean(0) - np.asarray(gt).mean(0))) / max(diag, 1e-6)
                    for bk in ("all", gap_bucket(gap)):
                        acc[bk][f"{name}_iou"].append(iou)
                        acc[bk][f"{name}_cerr"].append(cerr)
                if "geo_cos" in r:
                    for bk in ("all", gap_bucket(gap)):
                        acc[bk]["geo_cos"].append(float(r["geo_cos"][s]))
    out = {"dump": a.dump, "meta": d.get("meta"), "buckets": {}}
    for bk, m in acc.items():
        row = {"n": len(m["v2_iou"])}
        for name in ("v2", "copy"):
            iou = np.asarray(m[f"{name}_iou"])
            cerr = np.asarray(m[f"{name}_cerr"])
            row[name] = {f"iou>={t:.2f}": float((iou >= t).mean()) if iou.size else None for t in TAUS}
            row[name]["iou_median"] = float(np.median(iou)) if iou.size else None
            row[name]["centre_err_median_diag"] = float(np.median(cerr[np.isfinite(cerr)])) if np.isfinite(cerr).any() else None
        if m["geo_cos"]:
            row["geo_cos_mean"] = float(np.mean(m["geo_cos"]))
        out["buckets"][bk] = row
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    allr = out["buckets"].get("all", {})
    print(f"[v2_localize] {a.dump}: n={allr.get('n')} v2 {allr.get('v2')} copy {allr.get('copy')}")


if __name__ == "__main__":
    main()
