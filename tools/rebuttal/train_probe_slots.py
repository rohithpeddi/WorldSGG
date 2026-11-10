"""TRAIN probe (read-only): per-frame slot layout facts that the per-video slot
alignment relies on. CPU only.

  python tools/rebuttal/train_probe_slots.py --data_path /data/rohith/ag --n 300
"""
import argparse, os, pickle, random, json
from collections import Counter, defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data_path", default="/data/rohith/ag")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", default=None)
a = ap.parse_args()
random.seed(0)
res = {}
for mode in ["predcls", "sgdet"]:
    for fm in ["resnet50", "dinov3l"]:
        for split in ["train", "test"]:
            d = f"{a.data_path}/features/roi_features/{mode}/{fm}/{split}"
            if not os.path.isdir(d):
                continue
            vids = sorted(os.listdir(d))
            random.shuffle(vids)
            vids = vids[:a.n]
            c = Counter()
            nkeys, nmaxf, nkeys0 = [], [], []
            for v in vids:
                with open(f"{d}/{v}", "rb") as f:
                    x = pickle.load(f)
                keys = set(); mx = 0
                for fr, ff in x["frames"].items():
                    lids = [int(l) for l in ff.get("label_ids", [])]
                    srcs = list(ff.get("sources", []))
                    sc = ff.get("scores", None)
                    n = len(lids); mx = max(mx, n)
                    c["frames"] += 1
                    if n == 0:
                        c["empty_frame"] += 1; continue
                    if lids[0] != 1: c["pos0_not_person"] += 1
                    if 1 not in lids: c["no_person"] += 1
                    if lids.count(1) > 1: c["multi_person"] += 1
                    cnt = Counter(lids)
                    if any(v2 > 1 for k2, v2 in cnt.items() if k2 != 1): c["dup_class_frame"] += 1
                    if mode == "predcls":
                        by = defaultdict(set)
                        for l, s in zip(lids, srcs): by[l].add(s)
                        if any(len(s) > 1 for s in by.values()): c["class_both_gt_gdino"] += 1
                        c.update("src_" + s for s in srcs)
                    if sc is not None and n > 2:
                        s1 = np.asarray(sc[1:], dtype=float)
                        if np.any(np.diff(s1) > 1e-9): c["scores_not_sorted_after0"] += 1
                    # rank by score within class (stable by position)
                    order = sorted(range(n), key=lambda i: (-(float(sc[i]) if sc is not None else 0.0), i))
                    rank = {}; seen = Counter()
                    for i in order:
                        rank[i] = seen[lids[i]]; seen[lids[i]] += 1
                    first_pos = {}
                    for i, l in enumerate(lids):
                        first_pos.setdefault(l, i)
                    for l, i in first_pos.items():
                        if rank[i] != 0: c["first_pos_not_rank0"] += 1
                    for i, l in enumerate(lids):
                        keys.add((l, rank[i]))
                    for (p, o) in ff.get("pair_indices", []):
                        c["pairs"] += 1
                        if int(o) not in first_pos or int(p) not in first_pos: c["pair_unmapped"] += 1
                nkeys.append(len(keys)); nmaxf.append(mx)
                nkeys0.append(len({k for k in keys if k[1] == 0}))
            nk = np.array(nkeys); nm = np.array(nmaxf)
            r = dict(c)
            r.update(videos=len(vids), nkeys_mean=float(nk.mean()), nkeys_p99=float(np.percentile(nk, 99)),
                     nkeys_max=int(nk.max()), nmaxframe_mean=float(nm.mean()), nmaxframe_max=int(nm.max()),
                     videos_nkeys_gt64=int((nk > 64).sum()), nkeys_rank0_max=int(max(nkeys0)))
            key = f"{mode}/{fm}/{split}"
            res[key] = r
            print(key, json.dumps(r), flush=True)
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
