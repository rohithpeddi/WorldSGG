#!/usr/bin/env python3
"""Bundle the semantic-annotation intermediates of one Action Genome video.

Runs on the data server (CS93371).  Reads only; writes a self-contained bundle
(``bundle.json`` + a few frame PNGs) that ``panels_semantic.py`` renders
locally.  No GPU, no model calls.

Sources (all real, produced by WorldSceneGraphAnnotationTool/backend/pseudo):

* Coarse event graphs, one pickle per video, a list of per-clip dicts
  ``{clip_metadata{annotated_frame,start_frame,end_frame}, subtitle, graph}``
  (``process_ag_graphs.py``).  Three VLMs are kept under
  ``0_backup/graphs/{qwen3vl,kimikvl,internvl}``.  The legacy key ``qwen3vl``
  loaded ``Qwen/Qwen2.5-VL-7B-Instruct`` (core/vgent.py MODEL_MAP before the
  2026-05-03 rename), ``kimikvl`` -> Kimi-VL-A3B-Instruct, ``internvl`` ->
  InternVL2_5-8B.
* RAG all-objects predictions with per-label Yes/No verification
  ``{label, yes_prob}`` (``process_ag_rag_all.py``), the March-2026 runs under
  ``0_backup/mllms/rag_all_objects_results/predcls/<model>``.
* RAG missing-objects predictions (``process_ag_rag.py``) under
  ``0_backup/mllms/rag_results`` / ``stuff/rag_results``.
* Human corrections of the unobserved-object labels (test split,
  ``wsg_corrections/<vid>.pkl``): ``raw_response`` is the VLM output that
  seeded the annotation tool, ``attention/contacting/spatial`` are the
  compiled (latest) human labels.
* Final 2D relationship labels (``wsg_2d_augmentations/<vid>.mp4.pkl``):
  GT for observed objects (source "gt") + human-corrected labels for
  unobserved objects (source "correction").

Usage (server)::

    ~/anaconda3/envs/scene4cast/bin/python extract_semantic_pipeline.py \
        --video 00T1E --targets 000273,000137
"""

import argparse
import json
import os
import pickle
import re
import shutil
from pathlib import Path

import numpy as np

AG = Path("/data/rohith/ag")
OUT_ROOT = Path("/data3/rohith/ag/runs/semantic_pipeline")

GRAPH_MODELS = {
    # key on disk -> (display name, HF id actually loaded)
    "qwen3vl": ("Qwen2.5-VL-7B", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "kimikvl": ("Kimi-VL-A3B", "moonshotai/Kimi-VL-A3B-Instruct"),
    "internvl": ("InternVL2.5-8B", "OpenGVLab/InternVL2_5-8B"),
}

RAG_ALL = {
    "qwen3vl": AG / "0_backup/mllms/rag_all_objects_results/predcls/qwen3vl",
    "kimikvl": AG / "0_backup/mllms/rag_all_objects_results/predcls/kimikvl",
    "internvl": AG / "0_backup/mllms/rag_all_objects_results/predcls/internvl",
    # Pragya re-run, produced with --skip-verification (every yes_prob == 1.0)
    "qwen25vl_7b_pragya": Path("/data2/rohith/ag/mllms_pragya/rag_all_results/predcls/qwen25vl_7b"),
}
RAG_MISSING = {
    "qwen3vl": AG / "0_backup/mllms/rag_results/predcls/qwen3vl",
    "kimikvl": AG / "stuff/rag_results/predcls/kimikvl",
}
RAG_ALL_LOG = AG / "0_backup/mllms/rag_all_objects_results/ag_rag_all_objects_predcls_{m}.log"


def to_py(o):
    """Recursively convert numpy / tuples / sets into JSON-friendly types."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def parse_raw(raw):
    """Parse a raw VLM JSON answer (strings or lists) -> dict of lists."""
    if not raw:
        return None
    txt = re.sub(r"```(json)?", "", raw).strip()
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ("attention", "contacting", "spatial"):
        v = d.get(k, [])
        out[k] = [v] if isinstance(v, str) else list(v or [])
    return out


def graph_to_json(clips):
    out = []
    for c in clips:
        G = c["graph"]
        out.append({
            "clip_metadata": c["clip_metadata"],
            "subtitle": c["subtitle"] if isinstance(c["subtitle"], str) else " ".join(c["subtitle"]),
            "nodes": [{"id": int(n), **to_py(d)} for n, d in G.nodes(data=True)],
            "edges": [[int(u), int(v), to_py(d)] for u, v, d in G.edges(data=True)],
        })
    return out


def merged_entity_graph(clips):
    """Replicates ActionGenomeRAGAllObjectsProcessor.load_precomputed_graphs:
    clip nodes re-indexed with an offset, entity keys = text before the first
    comma of every entity/action/scene string, lower-cased."""
    ent = {}
    n_nodes = n_edges = 0
    for c in clips:
        G = c["graph"]
        off = n_nodes
        for nid, d in G.nodes(data=True):
            for s in d.get("entities", []) + d.get("actions", []) + d.get("scenes", []):
                key = s.split(",")[0].lower().strip()
                ent.setdefault(key, set()).add(nid + off)
        n_nodes += G.number_of_nodes()
        n_edges += G.number_of_edges()
    return {"n_nodes": n_nodes, "n_edges": n_edges,
            "entity_keys": {k: sorted(v) for k, v in sorted(ent.items())}}


def log_excerpt(model, vid):
    p = Path(str(RAG_ALL_LOG).format(m=model))
    if not p.exists():
        return []
    lines, grab = [], False
    with open(p, errors="replace") as f:
        for line in f:
            if f"{vid}.mp4" in line:
                grab = True
            if grab and (f"{vid}.mp4" in line or "Step" in line or "Dedup" in line
                         or "Bulk Rel Verification" in line):
                lines.append(line.rstrip()[:240])
            if grab and "Saved" in line and f"{vid}.mp4" in line:
                break
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--targets", default="000273,000137",
                    help="annotated frame stems whose VLM clip frames are copied")
    args = ap.parse_args()
    vid = args.video
    out = OUT_ROOT / vid
    (out / "frames").mkdir(parents=True, exist_ok=True)

    bundle = {"video": vid, "sources": {}}

    # ---- final labels (GT observed + human-corrected unobserved) ----------
    aug_p = AG / "wsg_2d_augmentations" / f"{vid}.mp4.pkl"
    aug = load(aug_p)
    bundle["sources"]["final_labels"] = str(aug_p)
    final = {}
    for k in sorted(aug["frames"]):
        fr = aug["frames"][k]
        final[k.split("/")[-1].replace(".png", "")] = {
            "person_bbox": to_py(fr.get("person_bbox")),
            "objects": [to_py(o) for o in fr["objects"]],
        }
    bundle["final_labels"] = final
    frames = sorted(final)
    bundle["annotated_frames"] = frames

    # ---- human corrections ------------------------------------------------
    cp = AG / "wsg_corrections" / f"{vid}.pkl"
    if cp.exists():
        c = load(cp)
        bundle["sources"]["corrections"] = str(cp)
        corr = {}
        for f, d in sorted(c["frames"].items()):
            corr[f] = [{
                "object": p["missing_object"],
                "seed": parse_raw(p.get("raw_response")),
                "seed_raw": p.get("raw_response"),
                "human": {"attention": [p["attention"]] if isinstance(p["attention"], str) else p["attention"],
                          "contacting": p["contacting"], "spatial": p["spatial"]},
            } for p in d["predictions"]]
        bundle["corrections"] = corr

    # ---- event graphs -----------------------------------------------------
    bundle["graphs"] = {}
    for m, (disp, hf) in GRAPH_MODELS.items():
        gp = AG / "0_backup/graphs" / m / f"{vid}.mp4.pkl"
        if not gp.exists():
            continue
        clips = load(gp)
        bundle["graphs"][m] = {"display": disp, "hf_id": hf, "path": str(gp),
                               "clips": graph_to_json(clips),
                               "merged": merged_entity_graph(clips)}

    # ---- RAG predictions with verification --------------------------------
    bundle["rag_all"] = {}
    for m, d in RAG_ALL.items():
        p = d / f"{vid}.mp4.pkl"
        if p.exists():
            r = load(p)
            bundle["rag_all"][m] = {"path": str(p), "model_name": r["model_name"],
                                    "video_objects": r["video_objects"],
                                    "frames": to_py(r["frames"])}
    bundle["rag_missing"] = {}
    for m, d in RAG_MISSING.items():
        p = d / f"{vid}.mp4.pkl"
        if p.exists():
            r = load(p)
            bundle["rag_missing"][m] = {"path": str(p), "frames": to_py(r["frames"])}
    bundle["rag_all_logs"] = {m: log_excerpt(m, vid) for m in ("qwen3vl", "kimikvl")}

    # ---- frames -----------------------------------------------------------
    fdir = AG / "frames" / f"{vid}.mp4"
    copied = []
    for f in frames:
        shutil.copy2(fdir / f"{f}.png", out / "frames" / f"{f}.png")
        copied.append(f)
    clip_meta = {c["clip_metadata"]["annotated_frame"]: c["clip_metadata"]
                 for c in bundle["graphs"].get("internvl", bundle["graphs"].get("qwen3vl"))["clips"]}
    bundle["target_clips"] = {}
    for t in args.targets.split(","):
        cm = clip_meta[int(t)]
        # process_ag_rag_all.py (Feb/Mar 2026): clip = frames start..end, step 2
        idx = [i for i in range(cm["start_frame"], cm["end_frame"] + 1, 2)
               if (fdir / f"{i:06d}.png").exists()]
        for i in idx:
            dst = out / "frames" / f"{i:06d}.png"
            if not dst.exists():
                shutil.copy2(fdir / f"{i:06d}.png", dst)
        bundle["target_clips"][t] = {"clip_metadata": cm, "clip_frames": [f"{i:06d}" for i in idx]}
    from PIL import Image
    bundle["image_size"] = list(Image.open(out / "frames" / f"{frames[0]}.png").size)

    with open(out / "bundle.json", "w") as f:
        json.dump(to_py(bundle), f, indent=1)
    print("wrote", out, "frames:", len(os.listdir(out / "frames")))


if __name__ == "__main__":
    main()
