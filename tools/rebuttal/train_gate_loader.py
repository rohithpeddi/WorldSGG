"""TRAIN gate: WorldAG loader options (CPU only, read-only).

Part ``default``: with every new option at its default, ``WorldAG.__getitem__`` returns exactly what the loader
returned before this round (the snapshot ``<orig_repo>/dataloader/world_ag_dataset.py``): every tensor equal in
value and dtype, every list / dict equal. Checked for a few videos per (mode, feature model, split), and for SGDet
also with ``visibility_from_annotation=False``. A second check builds the new loader through its ``video_ids``
filter (the real ``_build_video_list``) and compares the resulting list with the requested ids.

Part ``align``: ``slot_alignment="per_video_class"`` against the default layout of the same videos:
  * slot class drift: a slot holding different classes in different frames (consecutive valid frames, and any
    two valid frames); must be 0 with alignment;
  * each frame's content is unchanged: the multiset of node records (class, visibility, feature, boxes, corners)
    equals the default one minus the dropped duplicate entries (``slot_stats``), and the multiset of pair records
    (both endpoint records, the three label vectors, pair source, union feature) is equal exactly, so every
    frame's ground-truth triplet set is unchanged;
  * the person (class 1) sits in slot 0 whenever present.

  python tools/rebuttal/train_gate_loader.py --part default --orig_repo /data3/rohith/ag/runs/rebuttal/train/gate/orig_repo
  python tools/rebuttal/train_gate_loader.py --part align --n 40
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from dataloader.world_ag_dataset import WorldAG  # noqa: E402

DATA = "/data/rohith/ag"
SPLITS = "/data3/rohith/ag/runs/rebuttal/splits"
ANNOT = {"train": "world4d_rel_annotations", "test": "world4d_rel_annotations_worldbbox"}
COMBOS = [(m, f) for m in ("predcls", "sgdet") for f in ("resnet50", "dinov3l", "dinov3_tok")]


def pick_videos(mode, feat, phase, n, offset=0):
    """Videos with features and annotations: val750 order for train, sorted feature files for test."""
    fdir = f"{DATA}/features/roi_features/{mode}/{feat}/{phase}"
    adir = f"{DATA}/{ANNOT[phase]}/{phase}"
    have = {p[:-4] for p in os.listdir(fdir) if p.endswith(".pkl")}
    if phase == "train":
        order = [v.strip() for v in open(f"{SPLITS}/val750.txt") if v.strip()]
    else:
        order = sorted(have)
    out = []
    for v in order[offset:]:
        if v in have and (os.path.exists(f"{adir}/{v}.mp4.pkl") or os.path.exists(f"{adir}/{v}.pkl")):
            out.append(v)
        if len(out) >= n:
            break
    return out


def fixed_list_cls(base, vids):
    """Subclass whose video list is ``vids`` (skips the full-split PKL scan)."""
    def _build(self):
        self.video_list = list(vids)
    return type(f"Fixed{base.__name__}", (base,), {"_build_video_list": _build})


def kw(mode, feat, phase, **extra):
    d = dict(phase=phase, data_path=DATA, mode=mode, feature_model=feat, include_invisible=True,
             max_objects=64, annot_dir_name=ANNOT[phase])
    d.update(extra)
    return d


def same(a, b, path, errs):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.dtype != b.dtype \
                or a.shape != b.shape or not torch.equal(a, b):
            errs.append(path)
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            errs.append(f"{path}: keys {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            same(a[k], b[k], f"{path}.{k}", errs)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            errs.append(f"{path}: len {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]", errs)
    elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            errs.append(path)
    else:
        if a != b:
            errs.append(path)


def part_default(args):
    spec = importlib.util.spec_from_file_location("orig_world_ag", f"{args.orig_repo}/dataloader/world_ag_dataset.py")
    orig = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orig)
    report = {}
    for mode, feat in COMBOS:
        for phase in ("train", "test"):
            vids = pick_videos(mode, feat, phase, args.n_default)
            for vis in ([None, False] if mode == "sgdet" else [None]):
                extra = {} if vis is None else {"visibility_from_annotation": vis}
                a = fixed_list_cls(orig.WorldAG, vids)(**kw(mode, feat, phase, **extra))
                b = fixed_list_cls(WorldAG, vids)(**kw(mode, feat, phase, **extra))
                errs = []
                for i in range(len(vids)):
                    same(a[i], b[i], f"{vids[i]}", errs)
                key = f"{mode}/{feat}/{phase}" + ("" if vis is None else "/vis_off")
                report[key] = {"videos": vids, "n_mismatch": len(errs), "mismatch": errs[:10]}
                print(key, "videos", len(vids), "mismatches", len(errs), errs[:3], flush=True)
    # the real video_ids filter (full _build_video_list on a restricted candidate set)
    vids = pick_videos("predcls", "resnet50", "train", 5)
    ds = WorldAG(**kw("predcls", "resnet50", "train", video_ids=vids + ["NOT_A_VIDEO"]))
    report["video_ids_filter"] = {"requested": vids, "got": ds.video_list, "ok": sorted(vids) == ds.video_list}
    print("video_ids filter ok:", report["video_ids_filter"]["ok"], flush=True)
    report["pass"] = all(v.get("n_mismatch", 0) == 0 for v in report.values() if isinstance(v, dict)) \
        and report["video_ids_filter"]["ok"]
    return report


def node_rec(it, t, j):
    return (int(it["object_classes"][t, j]), bool(it["visibility_mask"][t, j]),
            it["visual_features"][t, j].numpy().tobytes(), it["bboxes_2d"][t, j].numpy().tobytes(),
            it["gt_bboxes_2d"][t, j].numpy().tobytes(), it["corners"][t, j].numpy().tobytes(),
            it["gt_corners"][t, j].numpy().tobytes())


def pair_recs(it, t):
    out = Counter()
    uf = it.get("union_features")
    for k in np.where(it["pair_valid"][t].numpy())[0]:
        p, o = int(it["person_idx"][t, k]), int(it["object_idx"][t, k])
        out[(node_rec(it, t, p), node_rec(it, t, o), int(it["gt_attention"][t, k]),
             it["gt_spatial"][t, k].numpy().tobytes(), it["gt_contacting"][t, k].numpy().tobytes(),
             int(it["pair_source"][t, k]), uf[t, k].numpy().tobytes() if uf is not None else b"")] += 1
    return out


def drift(it):
    """(consecutive valid-frame pairs, of which class changed, slots, slots with > 1 class)."""
    v = it["valid_mask"].numpy()
    c = it["object_classes"].numpy()
    both = v[1:] & v[:-1]
    n_pairs = int(both.sum())
    n_change = int((both & (c[1:] != c[:-1])).sum())
    n_slots = n_multi = 0
    for j in range(v.shape[1]):
        cls = set(c[v[:, j], j].tolist())
        if cls:
            n_slots += 1
            n_multi += int(len(cls) > 1)
    return n_pairs, n_change, n_slots, n_multi


def part_align(args):
    report = {}
    ok = True
    for mode, feat in COMBOS:
        for phase in ("train", "test"):
            vids = pick_videos(mode, feat, phase, args.n, offset=args.n_default)
            vis = False if mode == "sgdet" else None
            extra = {} if vis is None else {"visibility_from_annotation": vis}
            a = fixed_list_cls(WorldAG, vids)(**kw(mode, feat, phase, **extra))
            b = fixed_list_cls(WorldAG, vids)(**kw(mode, feat, phase, slot_alignment="per_video_class", **extra))
            st = Counter()
            for i, vid in enumerate(vids):
                x, y = a[i], b[i]
                for name, it in (("default", x), ("aligned", y)):
                    n_pairs, n_change, n_slots, n_multi = drift(it)
                    st[f"{name}_consec_pairs"] += n_pairs
                    st[f"{name}_consec_class_change"] += n_change
                    st[f"{name}_slots"] += n_slots
                    st[f"{name}_slots_multiclass"] += n_multi
                    st[f"{name}_videos_with_drift"] += int(n_multi > 0)
                ss = y["slot_stats"]
                st["dropped_entries"] += ss["dropped_entries"]
                st["videos_with_dropped"] += int(ss["dropped_identities"] > 0)
                st["N_default"] += int(x["N_max"])
                st["N_aligned"] += int(y["N_max"])
                assert x["T"] == y["T"] and x["frame_names"] == y["frame_names"] and x["K_max"] == y["K_max"]
                for t in range(x["T"]):
                    st["frames"] += 1
                    nx = Counter(node_rec(x, t, j) for j in np.where(x["valid_mask"][t].numpy())[0])
                    ny = Counter(node_rec(y, t, j) for j in np.where(y["valid_mask"][t].numpy())[0])
                    if ny - nx:
                        st["frames_node_extra"] += 1
                    st["node_entries_missing"] += sum((nx - ny).values())
                    px, py = pair_recs(x, t), pair_recs(y, t)
                    st["pairs"] += sum(px.values())
                    if px != py:
                        st["frames_pair_mismatch"] += 1
                    pslots = [j for j in np.where(y["valid_mask"][t].numpy())[0] if int(y["object_classes"][t, j]) == 1]
                    if pslots and 0 not in pslots:
                        st["frames_person_not_slot0"] += 1
                    for name in ("camera_poses",):
                        if (x[name] is None) != (y[name] is None) or (x[name] is not None and not torch.equal(x[name][t], y[name][t])):
                            st["frames_camera_mismatch"] += 1
            st = dict(st)
            st["default_consec_drift_rate"] = st["default_consec_class_change"] / max(st["default_consec_pairs"], 1)
            st["aligned_consec_drift_rate"] = st["aligned_consec_class_change"] / max(st["aligned_consec_pairs"], 1)
            good = (st.get("aligned_consec_class_change", 0) == 0 and st.get("aligned_slots_multiclass", 0) == 0
                    and st.get("frames_pair_mismatch", 0) == 0 and st.get("frames_node_extra", 0) == 0
                    and st.get("node_entries_missing", 0) == st.get("dropped_entries", 0)
                    and st.get("frames_person_not_slot0", 0) == 0 and st.get("frames_camera_mismatch", 0) == 0)
            st["pass"] = good
            ok &= good
            key = f"{mode}/{feat}/{phase}"
            report[key] = st
            print(key, json.dumps(st), flush=True)
    report["pass"] = ok
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["default", "align"], required=True)
    ap.add_argument("--orig_repo", default="/data3/rohith/ag/runs/rebuttal/train/gate/orig_repo")
    ap.add_argument("--n_default", type=int, default=3)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    t0 = time.time()
    rep = part_default(args) if args.part == "default" else part_align(args)
    rep["seconds"] = round(time.time() - t0, 1)
    print("PASS" if rep["pass"] else "FAIL", rep["seconds"], "s")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(rep, f, indent=1, default=str)
