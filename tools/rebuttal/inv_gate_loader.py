"""Round 2 (full inventory): loader gate. CPU only, read-only.

For N videos of a split, builds WorldAG with inventory="feature" and inventory="full" (both per_video_class) and checks
  1. every slot of the feature inventory is unchanged under the full inventory (class, visibility, feature bytes,
     3-D box) — appended entries only add slots;
  2. every pair of the feature inventory keeps its labels (by object class);
  3. appended slots: unobserved, zero appearance feature, zero 2-D box; PredCls 3-D box = annotation corners_final;
  4. PredCls: the person-object triplets of the full inventory = the annotation's triplets in every frame
     (test split; train split drops appended entries whose class has no annotated object, so the same holds);
  5. SGDet: the held entries do not depend on the annotation — rebuilding them with every annotation object
     removed gives the same appended slots (class, 3-D box) in every frame.
Prints counts and exits 1 on any failure.

    python tools/rebuttal/inv_gate_loader.py --mode predcls --phase test --feature_model resnet50 --n 40
"""
import argparse
import copy
import os
import random
import sys
from collections import Counter

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from dataloader.world_ag_dataset import (WorldAG, ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS,  # noqa: E402
                                         CONTACTING_RELATIONSHIPS, OBJECT_CLASSES, _to_short,
                                         INVENTORY_SOURCE, HELD_SOURCE)


def frame_triplets_from_item(it, t):
    """(object short label, predicate) multiset of frame t from the item's pairs."""
    out = Counter()
    pv = it["pair_valid"][t]
    for k in torch.where(pv)[0].tolist():
        o = int(it["object_idx"][t, k])
        lab = _to_short(OBJECT_CLASSES[int(it["object_classes"][t, o])])
        out[(lab, ATTENTION_RELATIONSHIPS[int(it["gt_attention"][t, k])])] += 1
        for i in torch.where(it["gt_spatial"][t, k] > 0)[0].tolist():
            out[(lab, SPATIAL_RELATIONSHIPS[i])] += 1
        for i in torch.where(it["gt_contacting"][t, k] > 0)[0].tolist():
            out[(lab, CONTACTING_RELATIONSHIPS[i])] += 1
    return out


def frame_triplets_from_annotation(af):
    out = Counter()
    seen = set()
    for obj in af.get("object_info_list", []):
        lab = obj.get("label", _to_short(obj.get("class", "")))
        if lab in seen:          # one instance per class is representable (first one, as the loader)
            continue
        seen.add(lab)
        att = [r for r in (obj.get("attention_relationship") or []) if r in ATTENTION_RELATIONSHIPS][:1]
        out[(lab, att[0] if att else "looking_at")] += 1
        for r in obj.get("spatial_relationship") or []:
            if r in SPATIAL_RELATIONSHIPS:
                out[(lab, r)] += 1
        for r in obj.get("contacting_relationship") or []:
            if r in CONTACTING_RELATIONSHIPS:
                out[(lab, r)] += 1
    return out


def slot_sig(it, t, s):
    return (int(it["object_classes"][t, s]), bool(it["visibility_mask"][t, s]),
            hash(it["visual_features"][t, s].numpy().tobytes()), hash(it["corners"][t, s].numpy().tobytes()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["predcls", "sgdet"])
    ap.add_argument("--phase", default="test")
    ap.add_argument("--feature_model", default="resnet50")
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    annot = "world4d_rel_annotations_worldbbox" if a.phase == "test" else "world4d_rel_annotations"
    vis_ann = a.mode == "predcls"
    kw = dict(phase=a.phase, data_path=a.data_path, mode=a.mode, feature_model=a.feature_model,
              include_invisible=True, max_objects=64, annot_dir_name=annot, slot_alignment="per_video_class",
              visibility_from_annotation=vis_ann)
    vids = sorted({p.split(".")[0] for p in os.listdir(f"{a.data_path}/features/roi_features/{a.mode}/{a.feature_model}/{a.phase}")})
    random.Random(1).shuffle(vids)
    old = WorldAG(inventory="feature", video_ids=vids[:a.n * 2], **kw)
    new = WorldAG(inventory="full", video_ids=old.video_list[:a.n], **kw)
    c, bad = Counter(), Counter()
    for vi, vid in enumerate(new.video_list):
        io, inew = old[old.video_list.index(vid)], new[vi]
        assert io["frame_names"] == inew["frame_names"]
        for t in range(io["T"]):
            c["frames"] += 1
            so = Counter(slot_sig(io, t, s) for s in torch.where(io["valid_mask"][t])[0].tolist())
            sn_all = torch.where(inew["valid_mask"][t])[0].tolist()
            sn = Counter(slot_sig(inew, t, s) for s in sn_all)
            if so - sn:
                bad["old_slot_changed"] += 1
            c["slots_old"] += sum(so.values())
            c["slots_new"] += sum(sn.values())
            # appended slots
            for s in sn_all:
                sig = slot_sig(inew, t, s)
                if so[sig] > 0:
                    so[sig] -= 1
                    continue
                c["appended"] += 1
                if bool(inew["visibility_mask"][t, s]):
                    bad["appended_visible"] += 1
                if inew["visual_features"][t, s].abs().sum() > 0 or inew["bboxes_2d"][t, s].abs().sum() > 0:
                    bad["appended_nonzero_feature_or_box"] += 1
            to, tn = frame_triplets_from_item(io, t), frame_triplets_from_item(inew, t)
            if to - tn:
                bad["old_pair_label_lost"] += 1
            c["triplets_old"] += sum(to.values())
            c["triplets_new"] += sum(tn.values())
            if a.mode == "predcls":
                ta = frame_triplets_from_annotation(inew["gt_annotations"][t])
                c["triplets_annotation"] += sum(ta.values())
                if ta - tn:
                    bad["annotation_triplet_missing"] += 1
                c["triplets_new_not_in_annotation"] += sum((tn - ta).values())
        if a.mode == "sgdet":
            # 5. held entries must not depend on the annotation
            ff = new._load_feature_pkl(vid)
            ad = new._load_annotation_pkl(vid)
            cf, key = new._align_frames(ff, ad)
            e1 = new._expand_inventory(ff["frames"], cf, ad["frames"], key)
            blank = copy.deepcopy(ad["frames"])
            for v in blank.values():
                v["object_info_list"] = []
            e2 = new._expand_inventory(ff["frames"], cf, blank, key)
            for f in cf:
                if [str(x) for x in e1[f]["labels"]] != [str(x) for x in e2[f]["labels"]] or \
                        not np.array_equal(e1[f]["boxes_3d"], e2[f]["boxes_3d"]):
                    bad["held_depends_on_annotation"] += 1
                c["held_entries"] += sum(1 for s in e1[f]["sources"] if s == HELD_SOURCE)
    print(f"[inv-gate] {a.mode}/{a.phase}/{a.feature_model}: {len(new.video_list)} videos {dict(c)}")
    print(f"[inv-gate] loader stats {dict(new._inventory_stats)}")
    print(f"[inv-gate] failures {dict(bad) or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
