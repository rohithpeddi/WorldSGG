"""Bundle one video's MLLM run outputs for the architecture-figure panels (server, CPU).

Reads a sandbox run (``lib.mllm`` runners driven by a config that redirects every output into
one directory, see ``scripts/paper_figures/dark/README.md``) and writes a self-contained bundle
of plain JSON / PNG / NPZ that ``scripts/paper_figures/dark/mllm_panels.py`` renders locally:

    video.json               ground truth and geometry per annotated frame (canonical frame)
    frames_annotated/*.png   the annotated key frames
    raw_thumbs.npz           every raw frame, downscaled (segment film strip)
    graph.json               the Stage-1 event graph (per segment: interval, caption, nodes)
    zero_shot_<mode>.json    U-WSGG-Zero predictions (the runner's pickle, as JSON)
    caption_all_<mode>.json  U-WSGG-Sub predictions
    rag_all_<mode>.json      Graph-RAG predictions
    capture_<method>_<mode>.json/.npz
                             what the capture wrappers record during those runs: every P4 prompt
                             text, the P6 discovery prompt / answer, the V and Q(f) tensors and,
                             for rag_all, the keywords, retrieval scores and context blocks
    track_a_<mode>.json      Track A predictions;  track_b_<mode>.json  Track B (+ critic log)
    cache/                   BEV base / meta / cloud, GDino detections, lifted OBBs
    payload/<mode>/          the exact Track A inputs of every frame: images 1-4, prompt, id table,
                             and the Track B prompt (retrieved context inserted before "Task:")

    WSGG_MLLM_CONFIG=<sandbox>/config.yaml ~/anaconda3/envs/wsg/bin/python \\
        scripts/paper_figures/dump_mllm_intermediates.py --video 00T1E \\
        --capture <sandbox>/capture --out /data3/rohith/ag/runs/intermediates/00T1E/mllm_bundle
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
REPO = os.environ.get("MLLM_REPO", os.path.expanduser("~/CODE/Scene4Cast_mllm"))
sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from lib.mllm.core.config_loader import get_path, load_config  # noqa: E402
from lib.mllm.data.worldbbox import WorldBBoxTestSet  # noqa: E402
from lib.mllm.methods.track_a_prompt.runner import TrackAContext, build_payload  # noqa: E402
from lib.mllm.methods.track_b_agent.runner import load_stage1, query_graph, with_context  # noqa: E402


def plain(v):
    """JSON-safe copy (numpy -> lists / scalars, sets -> lists, keys -> str)."""
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer, np.floating, np.bool_)):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plain(obj), f)


def arr(x):
    return None if x is None else np.asarray(x, np.float64).tolist()


def write_runs(cfg, vid: str, out: Path, args) -> None:
    """Every run output (the pickles, as JSON) and every capture of the sandbox."""
    def copy_run(method, mode, model, name):
        p = Path(get_path(cfg, f"outputs.{method}")) / mode / model / f"{vid}.mp4.pkl"
        if not p.exists():
            print(f"  missing {p}")
            return
        with open(p, "rb") as f:
            dump_json(pickle.load(f), out / name)

    for mode in ("predcls", "sgdet"):
        for method in ("zero_shot", "caption_all", "rag_all"):
            copy_run(method, mode, args.rag_model, f"{method}_{mode}.json")
            cap = Path(args.capture) / f"{method}_{mode}"
            for suf, name in ((".json", f"capture_{method}_{mode}.json"), ("_tensors.npz", f"capture_{method}_{mode}.npz")):
                src = cap / f"{vid}{suf}"
                if src.exists():
                    shutil.copy2(src, out / name)
                else:
                    print(f"  missing {src}")
        copy_run("track_a", mode, args.track_model, f"track_a_{mode}.json")
        copy_run("track_b", mode, args.track_model, f"track_b_{mode}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--capture", required=True, help="capture dir written by the capture wrappers (capture_*.py)")
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of {gt,frames,graph,runs,caches,payload} to (re)write")
    ap.add_argument("--graph_model", default="qwen25vl_7b")
    ap.add_argument("--rag_model", default="qwen25vl_7b")
    ap.add_argument("--track_model", default="qwen3vl_8b")
    ap.add_argument("--context_frames", type=int, default=2)
    ap.add_argument("--thumb_w", type=int, default=90)
    args = ap.parse_args()
    cfg = load_config()
    vid = Path(args.video).stem
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    parts = set(args.only.split(",")) if args.only else {"gt", "frames", "graph", "runs", "caches", "payload"}
    ts = WorldBBoxTestSet(cfg)
    v = ts.load(vid)

    if "runs" in parts:
        write_runs(cfg, vid, out, args)
    if not parts - {"runs"}:
        return

    # ---- ground truth + geometry ------------------------------------------------------
    frames = []
    for fr in v.frames:
        frames.append({
            "key": fr.key, "file": fr.file, "frame_num": fr.frame_num, "pi3_index": fr.pi3_index,
            "camera_pose_final": arr(fr.camera_pose_final),
            "person_bbox_2d": arr(fr.person_bbox_2d), "person_corners_final": arr(fr.person_corners_final),
            "objects": [{"label": o.label, "cls": o.cls, "bbox_2d": arr(o.bbox_2d), "visible": o.visible,
                         "observed": o.observed, "source": o.source, "attention": o.attention,
                         "spatial": o.spatial, "contacting": o.contacting,
                         "corners_final": arr(o.corners_final)} for o in fr.objects],
        })
    S = v.pi3_num_frames
    dump_json({
        "video_id": vid, "annotation": str(ts._annot_path(vid).resolve()), "orig_size": v.orig_size,
        "pi3_size": v.pi3_size, "pi3_start": v.pi3_start, "sampled_frames_idx": v.sampled_frames_idx,
        "pi3_frame_numbers": v.pi3_frame_numbers(),
        "camera_poses_pi3_final": [arr(v.camera_pose_final_for_pi3(k)) for k in range(S)],
        "frames": frames,
    }, out / "video.json")

    # ---- frames ---------------------------------------------------------------------------
    fa = Path(get_path(cfg, "ag_frames_annotated")) / f"{vid}.mp4"
    dst = out / "frames_annotated"
    dst.mkdir(exist_ok=True)
    for p in sorted(fa.glob("*.png")):
        shutil.copy2(p, dst / p.name)
    raw_dir = Path(get_path(cfg, "ag_frames")) / f"{vid}.mp4"
    names = sorted((p for p in raw_dir.iterdir() if p.suffix in (".png", ".jpg")), key=lambda p: int(p.stem))
    thumbs = []
    for p in names:
        im = Image.open(p).convert("RGB")
        h = int(round(im.height * args.thumb_w / im.width))
        thumbs.append(np.asarray(im.resize((args.thumb_w, h), Image.LANCZOS)))
    np.savez_compressed(out / "raw_thumbs.npz", frames=np.array([int(p.stem) for p in names]),
                        thumbs=np.stack(thumbs))

    # ---- Stage-1 graph ------------------------------------------------------------------
    gp = Path(get_path(cfg, "graphs")) / args.graph_model / f"{vid}.mp4.pkl"
    with open(gp, "rb") as f:
        clips = pickle.load(f)
    graph = []
    for c in clips:
        g = c.get("graph")
        nodes = [] if g is None else [{"id": int(n), **{k: d.get(k) for k in ("entities", "actions", "scenes",
                                                                             "captions", "sampled_indices")}}
                                      for n, d in g.nodes(data=True)]
        graph.append({"clip_metadata": c.get("clip_metadata"), "caption": c.get("caption") or c.get("subtitle"),
                      "nodes": nodes, "edges": [] if g is None else [[int(a), int(b)] for a, b in g.edges()]})
    dump_json({"source": str(gp), "clips": graph}, out / "graph.json")

    # ---- perception caches ------------------------------------------------------------
    cdir = out / "cache"
    cdir.mkdir(exist_ok=True)
    bev = Path(get_path(cfg, "worldbbox.caches.bev"))
    for suf in ("_bev.png", "_meta.json", "_cloud.npz"):
        shutil.copy2(bev / f"{vid}{suf}", cdir / suf[1:])
    shutil.copy2(Path(get_path(cfg, "worldbbox.caches.detections")) / f"{vid}.json", cdir / "detections.json")
    shutil.copy2(Path(get_path(cfg, "worldbbox.caches.lifted3d")) / f"{vid}.json", cdir / "lifted3d.json")

    # ---- the exact Track A / B inputs of every frame ------------------------------------
    stage1 = load_stage1(vid, get_path(cfg, "graphs"))
    for mode in ("predcls", "sgdet"):
        ctx = TrackAContext(v, cfg, mode)
        pdir = out / "payload" / mode
        pdir.mkdir(parents=True, exist_ok=True)
        for fr in v.frames:
            p = build_payload(ctx, fr, args.context_frames)
            stem = Path(fr.file).stem
            for j, im in enumerate(p["images"]):
                im.save(pdir / f"{stem}_img{j}.png")
            (pdir / f"{stem}_prompt_a.txt").write_text(p["text"], encoding="utf-8")
            retrieved = query_graph(stage1, fr.frame_num) if stage1 else ""
            (pdir / f"{stem}_prompt_b.txt").write_text(with_context(p["text"], retrieved), encoding="utf-8")
            dump_json({"objects": [{k: o.get(k) for k in ("id", "label", "bbox", "obb", "visible", "score", "corners")}
                                   for o in p["objects"]],
                       "person_corners": p["person_corners"], "retrieved": retrieved,
                       "n_images": len(p["images"])}, pdir / f"{stem}_objects.json")
        ctx.save()
    n = sum(1 for _ in out.rglob("*") if _.is_file())
    print(f"{vid}: bundle with {n} files -> {out}")


if __name__ == "__main__":
    main()
