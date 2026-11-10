"""INFER probe (CPU, read-only): is the exported SGDet 3-D box frame right?

For a ``--keep-3d`` SGDet dump (e.g. a 2-video gate dump) and every visible GT object that a slot matches by class +
2-D IoU >= 0.5, compares the slot's box with the annotation ``corners_final`` under three hypotheses for the stored
``camera_poses[t]`` (raw = camera-frame detector corners, recovered from ``bboxes_3d`` with the pose):
  H1  pose is camera -> canonical (what build_pred_pkl assumes):  x = R raw + t   (= bboxes_3d)
  H2  pose is canonical -> camera:                                 x = R^T (raw - t)
  H3  no transform:                                                x = raw
Reports per hypothesis: median centre distance / GT diagonal, median 3-D IoU, fraction with IoU > 0, and the
verticality of the box (|z| component of the box's most vertical edge; 1 = an edge along canonical z), plus the
box-size ratio (slot diagonal / GT diagonal), which is frame-independent.

    $PY tools/rebuttal/infer_probe_frame.py --dump <gate dump>
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.setdefault("WSGG_REPO", os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "tools"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import rescore_common_gt as cg  # noqa: E402
import rescore_rebuttal as rr  # noqa: E402
from rescore_3d import GT3D, _ok, centre_diag  # noqa: E402
from infer_score import default_gt_row, load_reference  # noqa: E402
from lib.mllm.eval.iou3d import compute_iou_3d_obb  # noqa: E402


def verticality(c):
    c = np.asarray(c).reshape(8, 3)
    best = 0.0
    for i in range(8):
        for j in range(i + 1, 8):
            e = c[j] - c[i]
            n = np.linalg.norm(e)
            if n > 1e-9:
                best = max(best, abs(e[2]) / n)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--annot", default=rr.ANNOT)
    a = ap.parse_args()
    d = pickle.load(open(a.dump, "rb"))
    recs = d["records"]
    ref, _, idx = load_reference(default_gt_row(d["meta"]))   # PredCls GT boxes of the same feature family
    gt3d = GT3D(a.annot)
    poses = {}
    out = {h: {"cd": [], "iou": [], "vert": []} for h in ("H1", "H2", "H3")}
    gt_vert, size_ratio = [], []
    for s in recs:
        vid, t = s["video_id"], int(s["frame_t"])
        g = ref[idx[(vid, t)]]
        if vid not in poses:
            ann = pickle.load(open(f"{a.annot}/{vid}.mp4.pkl", "rb"))
            keys = [k.split("/")[-1] for k in ann.get("camera_frame_keys", [])]
            frames = sorted(ann["frames"], key=lambda k: k.split("/")[-1])
            k2i = {k: i for i, k in enumerate(keys)}
            cam = np.asarray(ann["camera_poses"], dtype=np.float64)
            poses[vid] = [cam[k2i[f.split("/")[-1]]] for f in frames]
        P = poses[vid][t]
        R, tr = P[:3, :3], P[:3, 3]
        objs = gt3d.objects(g, [])
        m = cg.slot_map_common(s, g, defaultdict(int))
        gvis = g["visibility_mask"].astype(bool)
        for sl, j in m.items():
            if not gvis[j]:
                continue
            gt = objs[j][1]
            x1 = _ok(s["bboxes_3d"][sl])
            if gt is None or x1 is None:
                continue
            raw = (x1 - tr) @ R          # R^T (x - t), row-vector form
            hyps = {"H1": x1, "H2": (raw - tr) @ R, "H3": raw}
            cgt, dgt = centre_diag(gt)
            gt_vert.append(verticality(gt))
            size_ratio.append(centre_diag(x1)[1] / max(dgt, 1e-9))
            for h, x in hyps.items():
                cx, _ = centre_diag(x)
                out[h]["cd"].append(np.linalg.norm(cx - cgt) / max(dgt, 1e-9))
                out[h]["iou"].append(compute_iou_3d_obb(x, gt))
                out[h]["vert"].append(verticality(x))
    n = len(size_ratio)
    print(f"pairs (visible, 2-D IoU>=0.5): {n}")
    print(f"GT verticality median {np.median(gt_vert):.3f}; slot/GT diagonal ratio median {np.median(size_ratio):.2f} "
          f"(p10 {np.percentile(size_ratio, 10):.2f}, p90 {np.percentile(size_ratio, 90):.2f})")
    for h, d in out.items():
        cd, iou, vert = map(np.asarray, (d["cd"], d["iou"], d["vert"]))
        print(f"{h}: centre dist / GT diag median {np.median(cd):.2f} (p25 {np.percentile(cd, 25):.2f}); "
              f"IoU3D median {np.median(iou):.3f} mean {iou.mean():.3f} frac>0 {(iou > 0).mean():.2f} "
              f"frac>=0.25 {(iou >= 0.25).mean():.2f}; verticality median {np.median(vert):.3f}")


if __name__ == "__main__":
    main()
