"""
Gate checks for pointing the test split at a different annotation folder
(e.g. world4d_rel_annotations_worldbbox). Uses the loader's own join
(_align_frames + _build_frame_tensors) so the numbers reflect exactly what
training / evaluation would see.

Reports, per annotation folder:
  * videos in features, in annotations, and in the loader's video_list
  * frames per video: feature / annotation / common
  * feature objects with no annotation match (-> zero corners, silent!)
  * annotation objects with no feature match (-> never evaluated)
  * valid slots with all-zero `corners`, and with all-zero `gt_bboxes_2d`
  * valid slots flagged invisible (visibility_mask == False)
  * person floor height: min z of the person GT corners (levelled floor -> ~0)

Exit code 1 if the zero-corner fraction exceeds --max_zero_corner_frac.

    python tools/check_worldbbox_alignment.py --data_path /data/rohith/ag \
        --mode predcls --feature_model dinov3l \
        --annot_dir world4d_rel_annotations_worldbbox \
        --ref_annot_dir world4d_rel_annotations --out results/gate_worldbbox_predcls.json
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader.world_ag_dataset import WorldAG, _to_short  # noqa: E402


def _pct(a, b):
    return 100.0 * a / b if b else 0.0


def check(data_path, mode, feature_model, annot_dir, limit=0, max_objects=64):
    ds = WorldAG(phase="test", data_path=data_path, mode=mode,
                 feature_model=feature_model, include_invisible=True,
                 max_objects=max_objects, annot_dir_name=annot_dir)
    feat_videos = {p.stem for p in ds._feat_dir.glob("*.pkl")}
    annot_videos = {p.name.replace(".pkl", "").replace(".mp4", "")
                    for p in ds._annot_dir.glob("*.pkl")}

    c = Counter()
    person_zmin = []
    feat_only_labels, ann_only_labels = Counter(), Counter()
    vids = ds.video_list[:limit] if limit else ds.video_list
    for vid in vids:
        feat = ds._load_feature_pkl(vid)
        ann = ds._load_annotation_pkl(vid)
        common, key_map = ds._align_frames(feat, ann)
        c["frames_feat"] += len(feat.get("frames", {}))
        c["frames_annot"] += len(ann.get("frames", {}))
        c["frames_common"] += len(common)
        for fr in common:
            ff = feat["frames"][fr]
            af = ann["frames"][key_map[fr]]
            f_labels = [_to_short(l) for l in ff.get("labels", [])]
            a_labels = [o.get("label", _to_short(o.get("class", "")))
                        for o in af.get("object_info_list", [])]
            fs, as_ = set(f_labels), set(a_labels)
            for l in fs - as_:
                feat_only_labels[l] += 1
            for l in as_ - fs:
                ann_only_labels[l] += 1
            c["feat_objects"] += len(f_labels)
            c["annot_objects"] += len(a_labels)
            c["feat_only"] += len(fs - as_)
            c["annot_only"] += len(as_ - fs)

            t = ds._build_frame_tensors(ff, af, max_objects)
            valid = t["valid_mask"]
            c["valid_slots"] += int(valid.sum())
            zero_c = valid & (t["corners"].abs().sum(dim=(1, 2)) == 0)
            zero_gtc = valid & (t["gt_corners"].abs().sum(dim=(1, 2)) == 0)
            zero_2d = valid & (t["gt_bboxes_2d"].abs().sum(dim=1) == 0)
            c["zero_corners"] += int(zero_c.sum())
            c["zero_gt_corners"] += int(zero_gtc.sum())
            c["zero_gt_bboxes_2d"] += int(zero_2d.sum())
            c["invisible_slots"] += int((valid & ~t["visibility_mask"]).sum())
            # 2D-box consistency: feature boxes vs annotation gt boxes (both
            # should be Pi3-space xyxy; predcls feature boxes ARE the GT boxes)
            has2d = valid & ~zero_2d & (t["bboxes_2d"].abs().sum(dim=1) > 0)
            if int(has2d.sum()) > 0:
                a, b = t["bboxes_2d"][has2d], t["gt_bboxes_2d"][has2d]
                lt = np.maximum(a[:, :2].numpy(), b[:, :2].numpy())
                rb = np.minimum(a[:, 2:].numpy(), b[:, 2:].numpy())
                wh = np.clip(rb - lt, 0, None); inter = wh[:, 0] * wh[:, 1]
                area = lambda x: np.clip((x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1]), 0, None)
                iou = inter / np.maximum(area(a.numpy()) + area(b.numpy()) - inter, 1e-6)
                c["iou2d_n"] += int(iou.size); c["iou2d_sum"] += float(iou.sum())
                c["iou2d_ge05"] += int((iou >= 0.5).sum())
            c["pairs"] += len(t["valid_raw_pair_indices"])
            if bool(valid[0]) and float(t["gt_corners"][0].abs().sum()) > 0:
                person_zmin.append(float(t["gt_corners"][0][:, 2].min()))
        c["videos"] += 1

    z = np.array(person_zmin) if person_zmin else np.zeros(0)
    rep = {
        "annot_dir": annot_dir, "mode": mode, "feature_model": feature_model,
        "videos_features": len(feat_videos), "videos_annotations": len(annot_videos),
        "videos_loader": len(ds.video_list), "videos_checked": c["videos"],
        "frames_feat": c["frames_feat"], "frames_annot": c["frames_annot"],
        "frames_common": c["frames_common"],
        "feat_objects": c["feat_objects"], "annot_objects": c["annot_objects"],
        "feat_only": c["feat_only"], "feat_only_pct": _pct(c["feat_only"], c["feat_objects"]),
        "annot_only": c["annot_only"], "annot_only_pct": _pct(c["annot_only"], c["annot_objects"]),
        "valid_slots": c["valid_slots"],
        "zero_corners": c["zero_corners"],
        "zero_corners_pct": _pct(c["zero_corners"], c["valid_slots"]),
        "zero_gt_corners": c["zero_gt_corners"],
        "zero_gt_corners_pct": _pct(c["zero_gt_corners"], c["valid_slots"]),
        "zero_gt_bboxes_2d": c["zero_gt_bboxes_2d"],
        "zero_gt_bboxes_2d_pct": _pct(c["zero_gt_bboxes_2d"], c["valid_slots"]),
        "invisible_slots": c["invisible_slots"],
        "invisible_pct": _pct(c["invisible_slots"], c["valid_slots"]),
        "pairs": c["pairs"],
        "gt2d_iou_mean": (c["iou2d_sum"] / c["iou2d_n"]) if c["iou2d_n"] else None,
        "gt2d_iou_ge05_pct": _pct(c["iou2d_ge05"], c["iou2d_n"]),
        "person_zmin_mean": float(z.mean()) if z.size else None,
        "person_zmin_p05": float(np.percentile(z, 5)) if z.size else None,
        "person_zmin_p50": float(np.percentile(z, 50)) if z.size else None,
        "person_zmin_p95": float(np.percentile(z, 95)) if z.size else None,
        "feat_only_top": feat_only_labels.most_common(10),
        "annot_only_top": ann_only_labels.most_common(10),
        "video_ids": sorted(ds.video_list),
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--feature_model", default="dinov3l")
    ap.add_argument("--annot_dir", default="world4d_rel_annotations_worldbbox")
    ap.add_argument("--ref_annot_dir", default=None,
                    help="old folder for a side-by-side column (optional)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_zero_corner_frac", type=float, default=0.01)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    reports = {"new": check(args.data_path, args.mode, args.feature_model,
                            args.annot_dir, args.limit)}
    if args.ref_annot_dir:
        reports["ref"] = check(args.data_path, args.mode, args.feature_model,
                               args.ref_annot_dir, args.limit)
        new_ids = set(reports["new"]["video_ids"])
        ref_ids = set(reports["ref"]["video_ids"])
        reports["video_set"] = {
            "new": len(new_ids), "ref": len(ref_ids),
            "new_minus_ref": sorted(new_ids - ref_ids),
            "ref_minus_new_count": len(ref_ids - new_ids),
        }

    keys = ["videos_features", "videos_annotations", "videos_loader", "frames_common",
            "feat_objects", "annot_objects", "feat_only_pct", "annot_only_pct",
            "valid_slots", "zero_corners_pct", "zero_gt_corners_pct",
            "zero_gt_bboxes_2d_pct", "invisible_pct", "pairs",
            "gt2d_iou_mean", "gt2d_iou_ge05_pct",
            "person_zmin_p05", "person_zmin_p50", "person_zmin_p95"]
    cols = [k for k in reports if k != "video_set"]
    header = "metric".ljust(28) + "".join(reports[k]["annot_dir"][-24:].rjust(26) for k in cols)
    print(header)
    for key in keys:
        row = key.ljust(28)
        for k in cols:
            v = reports[k][key]
            row += ("%26.3f" % v) if isinstance(v, float) else str(v).rjust(26)
        print(row)
    for k in cols:
        print("[%s] feat_only top: %s" % (k, reports[k]["feat_only_top"]))
        print("[%s] annot_only top: %s" % (k, reports[k]["annot_only_top"]))
    if "video_set" in reports:
        vs = reports["video_set"]
        print("video set: %d new vs %d ref; new-not-in-ref=%s; ref-not-in-new=%d"
              % (vs["new"], vs["ref"], vs["new_minus_ref"], vs["ref_minus_new_count"]))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        slim = {}
        for k, v in reports.items():
            slim[k] = {kk: vv for kk, vv in v.items() if kk != "video_ids"} if isinstance(v, dict) else v
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2)
        print("saved %s" % args.out)

    frac = reports["new"]["zero_corners"] / max(1, reports["new"]["valid_slots"])
    if frac > args.max_zero_corner_frac:
        print("GATE FAILED: %.2f%% valid slots have zero corners (> %.2f%%)"
              % (100 * frac, 100 * args.max_zero_corner_frac))
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
