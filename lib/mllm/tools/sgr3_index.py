#!/usr/bin/env python3
"""
SGR3-style visual reference retrieval: bank, index and offline queries.
=======================================================================

Reimplements the *retrieval strategy* of SGR3 (arXiv 2603.04614; no public
code) for the WorldRAG (``rag_all``) MLLM pipeline.  See
``setup/EXT_SGR3_RETRIEVAL.md`` for the design and every deviation.

The retrieval runs offline in its own env (``/data3/rohith/ag/envs/sgr3``: a
``--system-site-packages`` venv over ``wsg`` + ``faiss-cpu``), so nothing is
installed into the vLLM env.  The MLLM runner
(``lib.mllm.methods.rag_sgr3.runner``) only reads the JSON it writes.

Sub-commands (all paths default under ``/data3/rohith/ag/cache/sgr3``)::

    bank    train-split reference scene graphs + leakage check
    embed   SigLIP2 patch embeddings of the train frames, key-frame filtering
    index   FAISS IVF index over the key-frame patch embeddings
    query   per test frame: weighted patch voting -> ranked reference scenes

Values stored in the bank are the WorldBBox *train* scene graphs
(``world4d_rel_annotations/train``) in the Action Genome vocabulary.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

logger = logging.getLogger("sgr3_index")

AG_ROOT = "/data/rohith/ag"
TRAIN_ANN = f"{AG_ROOT}/world4d_rel_annotations/train"
TEST_ANN = f"{AG_ROOT}/world4d_rel_annotations_worldbbox/test"
FRAMES = f"{AG_ROOT}/frames_annotated"
TEST_1511 = "/data3/rohith/ag/splits/test_worldbbox_1511.txt"
AG_SPLITS = f"{AG_ROOT}/video_splits.json"
OUT = "/data3/rohith/ag/cache/sgr3"
MODEL_ID = "google/siglip2-base-patch16-224"
N_PATCH = 196          # 14 x 14 patches at 224 px, patch 16
DIM = 768


def _stem(v: str) -> str:
    return Path(v).stem if v.endswith(".mp4") else v


def _read_list(p: str) -> List[str]:
    return [_stem(l.strip()) for l in open(p, encoding="utf-8") if l.strip()]


# ---------------------------------------------------------------------------
# bank
# ---------------------------------------------------------------------------

def frame_graph(fr: dict) -> List[dict]:
    """One reference scene graph = the frame's person->object edges."""
    objs = []
    for o in fr.get("object_info_list", []) or []:
        label = o.get("label") or o.get("class")
        if not label:
            continue
        rels = []
        for key in ("attention_relationship", "contacting_relationship", "spatial_relationship"):
            for r in (o.get(key) or []):
                if r not in rels:
                    rels.append(str(r))
        objs.append({"label": str(label), "visible": bool(o.get("visible", True)), "rels": rels})
    return objs


def cmd_bank(a):
    os.makedirs(a.out, exist_ok=True)
    test_ids = set(_read_list(a.test_list))
    ag_test = set()
    if os.path.exists(AG_SPLITS):
        sp = json.load(open(AG_SPLITS))
        for k in ("test", "Test", "val"):
            if isinstance(sp.get(k), list):
                ag_test |= {_stem(v) for v in sp[k]}
    train_files = sorted(f for f in os.listdir(a.train_ann) if f.endswith(".mp4.pkl"))
    train_ids = [_stem(f.replace(".pkl", "")) for f in train_files]
    overlap_1511 = sorted(set(train_ids) & test_ids)
    overlap_ag = sorted(set(train_ids) & ag_test)
    logger.info(f"train pkls={len(train_ids)} test1511={len(test_ids)} ag_test={len(ag_test)} "
                f"overlap1511={len(overlap_1511)} overlap_agtest={len(overlap_ag)}")
    excluded = set(overlap_1511) | set(overlap_ag)
    bank: Dict[str, dict] = {}
    n_frames = 0
    for f, vid in zip(train_files, train_ids):
        if vid in excluded:
            continue
        d = pickle.load(open(os.path.join(a.train_ann, f), "rb"))
        frames = {}
        for fk in sorted(d.get("frames", {})):
            g = frame_graph(d["frames"][fk])
            if not g:
                continue
            img = os.path.join(FRAMES, fk)
            if not os.path.exists(img):
                continue
            frames[fk] = g
        if frames:
            bank[vid] = frames
            n_frames += len(frames)
    bank_ids = set(bank)
    assert not (bank_ids & test_ids), "LEAK: test_worldbbox_1511 ids in the bank"
    assert not (bank_ids & ag_test), "LEAK: AG test-split ids in the bank"
    leak = {"train_pkls": len(train_ids), "bank_videos": len(bank_ids), "bank_frames": n_frames,
            "test1511": len(test_ids), "ag_test": len(ag_test),
            "raw_overlap_test1511": overlap_1511, "raw_overlap_agtest": overlap_ag,
            "bank_overlap_test1511": 0, "bank_overlap_agtest": 0,
            "assert": "passed"}
    pickle.dump(bank, open(os.path.join(a.out, "bank_graphs.pkl"), "wb"), protocol=4)
    json.dump(leak, open(os.path.join(a.out, "leakage.json"), "w"), indent=2)
    logger.info(f"bank: {len(bank_ids)} videos, {n_frames} frames -> {a.out}; leakage {leak['assert']}")


# ---------------------------------------------------------------------------
# SigLIP2 patch embeddings
# ---------------------------------------------------------------------------

class Encoder:
    def __init__(self, device="cuda"):
        import torch
        from transformers import AutoModel, AutoImageProcessor
        self.torch = torch
        self.device = device
        m = AutoModel.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        self.vis = m.vision_model.to(device).eval()
        self.proc = AutoImageProcessor.from_pretrained(MODEL_ID)

    def pixel(self, pil_list):
        return self.proc(images=pil_list, return_tensors="pt")["pixel_values"]

    def __call__(self, pixel_values):
        """(B,3,224,224) -> L2-normalised patch embeddings (B,196,768) float16 numpy."""
        torch = self.torch
        with torch.no_grad():
            out = self.vis(pixel_values=pixel_values.to(self.device, torch.float16))
            p = out.last_hidden_state.float()
            p = torch.nn.functional.normalize(p, dim=-1)
        return p.half().cpu().numpy()


def token_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Late-interaction (MaxSim) similarity of two frames' patch sets, mean over a's patches."""
    s = a.astype(np.float32) @ b.astype(np.float32).T
    return float(s.max(axis=1).mean())


def select_keyframes(P: np.ndarray, thr: float, cap: int) -> List[int]:
    """Stream-order key-frame filter (SGR3 §III-A): drop a frame whose token-wise
    similarity to any kept frame exceeds ``thr``; keep at most ``cap``."""
    keep: List[int] = []
    for i in range(len(P)):
        if all(token_sim(P[i], P[j]) <= thr for j in keep):
            keep.append(i)
    if len(keep) > cap:                           # spread the cap over the video
        idx = np.linspace(0, len(keep) - 1, cap).round().astype(int)
        keep = [keep[i] for i in sorted(set(idx.tolist()))]
    return keep


class _VideoFrames:
    """torch Dataset: one item = all (bank) frames of one train video."""

    def __init__(self, items, proc):
        self.items, self.proc = items, proc

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image
        vid, keys = self.items[i]
        ims = []
        ok = []
        for k in keys:
            try:
                ims.append(Image.open(os.path.join(FRAMES, k)).convert("RGB"))
                ok.append(k)
            except Exception:
                pass
        px = self.proc(images=ims, return_tensors="pt")["pixel_values"] if ims else None
        return vid, ok, px


def cmd_embed(a):
    import torch
    bank = pickle.load(open(os.path.join(a.out, "bank_graphs.pkl"), "rb"))
    items = [(v, sorted(bank[v])) for v in sorted(bank)]
    if a.limit:
        items = items[: a.limit]
    enc = Encoder()
    ds = _VideoFrames(items, enc.proc)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, shuffle=False, num_workers=a.workers)
    cap_rows = sum(min(len(k), a.cap) for _, k in items)
    mm_path = os.path.join(a.out, f"kf_patches{a.suffix}.f16")
    mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(cap_rows, N_PATCH, DIM))
    kf_keys: List[str] = []
    stats = []
    t0 = time.time()
    for n, (vid, keys, px) in enumerate(dl):
        if px is None:
            continue
        P = np.concatenate([enc(px[i:i + 64]) for i in range(0, len(px), 64)], 0)
        keep = select_keyframes(P, a.thr, a.cap)
        r0 = len(kf_keys)
        mm[r0:r0 + len(keep)] = P[keep]
        kf_keys.extend(keys[i] for i in keep)
        stats.append((len(keys), len(keep)))
        if n % 200 == 0:
            el = time.time() - t0
            logger.info(f"embed {n}/{len(items)} videos, {len(kf_keys)} key frames, "
                        f"{el:.0f}s ({el / max(n, 1):.2f}s/video)")
    mm.flush()
    del mm
    json.dump({"keys": kf_keys, "rows_alloc": cap_rows, "thr": a.thr, "cap": a.cap, "model": MODEL_ID,
               "frames_in": int(sum(s[0] for s in stats)), "kf_out": len(kf_keys),
               "videos": len(stats)},
              open(os.path.join(a.out, f"kf_meta{a.suffix}.json"), "w"))
    fi = np.array([s[0] for s in stats]); ko = np.array([s[1] for s in stats])
    logger.info(f"key frames: {ko.sum()} of {fi.sum()} frames; per video mean {ko.mean():.2f} "
                f"median {np.median(ko):.0f} max {ko.max()} ({time.time() - t0:.0f}s)")


def cmd_calib(a):
    """Distribution of within-video token-wise similarities (to set --thr)."""
    import torch
    bank = pickle.load(open(os.path.join(a.out, "bank_graphs.pkl"), "rb"))
    rng = np.random.RandomState(0)
    vids = rng.choice(sorted(bank), size=min(a.limit or 40, len(bank)), replace=False)
    enc = Encoder()
    ds = _VideoFrames([(v, sorted(bank[v])) for v in vids], enc.proc)
    sims_adj, sims_other = [], []
    Ps = []
    for i in range(len(ds)):
        _, keys, px = ds[i]
        P = enc(px)
        Ps.append(P)
        for j in range(1, len(P)):
            sims_adj.append(token_sim(P[j], P[j - 1]))
    for i in range(len(Ps) - 1):
        sims_other.append(token_sim(Ps[i][0], Ps[i + 1][0]))
    q = lambda x: np.percentile(x, [5, 25, 50, 75, 95]).round(3).tolist()
    logger.info(f"adjacent annotated frames (same video) pct5/25/50/75/95: {q(sims_adj)}")
    logger.info(f"first frames of different videos        pct5/25/50/75/95: {q(sims_other)}")
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        ks = [len(select_keyframes(P, thr, 10 ** 6)) for P in Ps]
        nf = [len(P) for P in Ps]
        logger.info(f"thr={thr}: kept {np.sum(ks)}/{np.sum(nf)} frames, per video mean {np.mean(ks):.2f}")


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def cmd_index(a):
    import faiss
    faiss.omp_set_num_threads(a.threads)
    meta = json.load(open(os.path.join(a.out, f"kf_meta{a.suffix}.json")))
    n_kf = meta["kf_out"]
    mm = np.memmap(os.path.join(a.out, f"kf_patches{a.suffix}.f16"), dtype=np.float16, mode="r",
                   shape=(meta["rows_alloc"], N_PATCH, DIM))
    N = n_kf * N_PATCH
    flat = mm[:n_kf].reshape(N, DIM)
    rng = np.random.RandomState(0)
    tr = flat[np.sort(rng.choice(N, size=min(a.n_train, N), replace=False))].astype(np.float32)
    quant = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIVFScalarQuantizer(quant, DIM, a.nlist, faiss.ScalarQuantizer.QT_8bit,
                                          faiss.METRIC_INNER_PRODUCT)
    t0 = time.time()
    index.train(tr)
    logger.info(f"trained IVF{a.nlist},SQ8 on {len(tr)} vectors ({time.time() - t0:.0f}s)")
    step = 500_000
    for s in range(0, N, step):
        index.add(flat[s:s + step].astype(np.float32))
        logger.info(f"added {min(s + step, N)}/{N} ({time.time() - t0:.0f}s)")
    faiss.write_index(index, os.path.join(a.out, f"ivf_sq8{a.suffix}.faiss"))
    logger.info(f"index written: {index.ntotal} patch vectors from {n_kf} key frames")


# ---------------------------------------------------------------------------
# query
# ---------------------------------------------------------------------------

def patch_weights(P: np.ndarray, tau: float) -> np.ndarray:
    """SGR3 eq.: mu_i = mean_t!=i cos(p_i,p_t); w = softmax(-mu/tau)."""
    X = P.astype(np.float32)
    S = X @ X.T
    n = len(X)
    mu = (S.sum(1) - np.diag(S)) / (n - 1)
    z = -mu / tau
    z = z - z.max()
    w = np.exp(z)
    return w / w.sum()


def frame_scores(D: np.ndarray, I: np.ndarray, w: np.ndarray) -> Dict[int, float]:
    """Score(I_q, I_f) = sum_{i in Omega_f} w_i a_f[i],  a_f[i] = max sim of patch i's
    kNN hits that fall in bank frame f (Omega_f = query patches with >=1 hit in f)."""
    P, k = I.shape
    valid = I >= 0
    fr = np.where(valid, I // N_PATCH, -1)
    qi = np.repeat(np.arange(P), k)
    f = fr.reshape(-1)
    d = D.reshape(-1)
    m = f >= 0
    qi, f, d = qi[m], f[m], d[m]
    key = f.astype(np.int64) * P + qi
    order = np.lexsort((-d, key))
    key, d, f, qi = key[order], d[order], f[order], qi[order]
    first = np.ones(len(key), bool)
    first[1:] = key[1:] != key[:-1]
    f, qi, d = f[first], qi[first], d[first]           # max per (frame, query patch)
    contrib = w[qi] * d
    uf, inv = np.unique(f, return_inverse=True)
    sc = np.bincount(inv, weights=contrib)
    return dict(zip(uf.tolist(), sc.tolist()))


def cmd_query(a):
    import faiss
    import torch
    from PIL import Image
    faiss.omp_set_num_threads(a.threads)
    meta = json.load(open(os.path.join(a.out, f"kf_meta{a.suffix}.json")))
    kf_keys = meta["keys"]
    kf_vid = [_stem(k.split("/")[0]) for k in kf_keys]
    index = faiss.read_index(os.path.join(a.out, f"ivf_sq8{a.suffix}.faiss"))
    index.nprobe = a.nprobe
    bank_ids = set(kf_vid)
    vids = _read_list(a.video_list)
    assert not (bank_ids & set(vids)), "LEAK: query video in the bank"
    assert not (bank_ids & set(_read_list(TEST_1511))), "LEAK: test ids in the bank"
    enc = Encoder()
    out_dir = os.path.join(a.out, f"retrieval{a.suffix}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    for n, vid in enumerate(vids):
        dst = os.path.join(out_dir, f"{vid}.json")
        if os.path.exists(dst) and not a.overwrite:
            continue
        d = pickle.load(open(os.path.join(TEST_ANN, f"{vid}.mp4.pkl"), "rb"))
        fkeys = sorted(d["frames"])
        ims = [Image.open(os.path.join(FRAMES, k)).convert("RGB") for k in fkeys]
        px = enc.pixel(ims)
        P = np.concatenate([enc(px[i:i + 64]) for i in range(0, len(px), 64)], 0)
        per_frame: List[Dict[int, float]] = []
        for j in range(len(P)):
            D, I = index.search(P[j].astype(np.float32), a.knn)
            per_frame.append(frame_scores(D, I, patch_weights(P[j], a.tau)))
        res = {}
        for t, fk in enumerate(fkeys):
            win = [u for u in range(t - a.half_window, t + a.half_window + 1) if 0 <= u < len(fkeys)]
            scene = {}                 # scene -> sum_q max_f Score(q, f)
            fsum = {}                  # bank frame -> sum_q Score(q, f)
            for u in win:
                best = {}
                for f, s in per_frame[u].items():
                    v = kf_vid[f]
                    if s > best.get(v, -1.0):
                        best[v] = s
                    fsum[f] = fsum.get(f, 0.0) + s
                for v, s in best.items():
                    scene[v] = scene.get(v, 0.0) + s
            top = sorted(scene.items(), key=lambda x: -x[1])[: a.top_scenes]
            scenes = []
            for v, s in top:
                fr = sorted(((kf_keys[f], sc) for f, sc in fsum.items() if kf_vid[f] == v),
                            key=lambda x: -x[1])[: a.top_frames]
                scenes.append({"video": v, "score": round(s, 5),
                               "frames": [[k, round(sc, 5)] for k, sc in fr]})
            res[os.path.basename(fk)] = {"window": [os.path.basename(fkeys[u]) for u in win],
                                         "scenes": scenes}
        json.dump({"video": vid, "params": {"knn": a.knn, "nprobe": a.nprobe, "tau": a.tau,
                                            "half_window": a.half_window, "model": MODEL_ID},
                   "frames": res}, open(dst, "w"))
        if n % 10 == 0:
            el = time.time() - t0
            logger.info(f"query {n + 1}/{len(vids)} {vid}: {len(fkeys)} frames ({el:.0f}s)")
    logger.info(f"query done: {len(vids)} videos -> {out_dir} ({time.time() - t0:.0f}s)")


def cmd_diag(a):
    """Retrieval sanity check against the test GT (never used by any arm):
    per (test frame, GT object), is the object's class present in the merged
    graph of the top-k retrieved scenes?  Compared with k random bank scenes."""
    bank = pickle.load(open(os.path.join(a.out, "bank_graphs.pkl"), "rb"))
    bank_vids = sorted(bank)
    rng = np.random.RandomState(0)
    rdir = os.path.join(a.out, f"retrieval{a.suffix}")
    ks = (1, 3, 5)
    hit = {k: [] for k in ks}
    hit_rand = {k: [] for k in ks}
    hit_unseen = {k: [] for k in ks}
    jac = []
    for vid in _read_list(a.video_list):
        r = json.load(open(os.path.join(rdir, f"{vid}.json")))["frames"]
        d = pickle.load(open(os.path.join(TEST_ANN, f"{vid}.mp4.pkl"), "rb"))
        for fk, fr in d["frames"].items():
            ent = r.get(os.path.basename(fk))
            if not ent:
                continue
            gt = frame_graph(fr)
            gt_labels = {o["label"] for o in gt}
            for k in ks:
                labs = set()
                for s in ent["scenes"][:k]:
                    for rk, _ in s["frames"][:3]:
                        labs |= {o["label"] for o in bank[s["video"]].get(rk, [])}
                rl = set()
                for v in rng.choice(bank_vids, size=k, replace=False):
                    for rk in list(bank[v])[:3]:
                        rl |= {o["label"] for o in bank[v][rk]}
                for o in gt:
                    hit[k].append(o["label"] in labs)
                    hit_rand[k].append(o["label"] in rl)
                    if not o["visible"]:
                        hit_unseen[k].append(o["label"] in labs)
                if k == 1:
                    jac.append(len(gt_labels & labs) / max(len(gt_labels | labs), 1))
    res = {f"k{k}": {"gt_object_in_ref": round(float(np.mean(hit[k])), 4),
                     "random_scenes": round(float(np.mean(hit_rand[k])), 4),
                     "unseen_gt_object_in_ref": round(float(np.mean(hit_unseen[k])), 4) if hit_unseen[k] else None,
                     "n_gt_objects": len(hit[k])} for k in ks}
    res["top1_label_jaccard"] = round(float(np.mean(jac)), 4)
    json.dump(res, open(os.path.join(a.out, f"diag{a.suffix}.json"), "w"), indent=2)
    logger.info(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bank", "calib", "embed", "index", "query", "diag"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--train_ann", default=TRAIN_ANN)
    ap.add_argument("--test_list", default=TEST_1511)
    ap.add_argument("--video_list", default="/data3/rohith/ag/splits/test_worldbbox_thinking150.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.75, help="key-frame filter: token-wise sim above this = redundant")
    ap.add_argument("--cap", type=int, default=8, help="max key frames per train video")
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--n_train", type=int, default=400_000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--knn", type=int, default=64, help="nearest neighbours per query patch")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--half_window", type=int, default=1, help="query window W = t +- this many annotated frames")
    ap.add_argument("--top_scenes", type=int, default=10)
    ap.add_argument("--top_frames", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    {"bank": cmd_bank, "calib": cmd_calib, "embed": cmd_embed, "index": cmd_index,
     "query": cmd_query, "diag": cmd_diag}[a.cmd](a)


if __name__ == "__main__":
    main()
