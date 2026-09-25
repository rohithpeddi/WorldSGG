"""
SceneGraphVLM (arXiv 2605.13667) as an external baseline on the worldbbox test set.
===================================================================================

The released AG checkpoint (Qwen3.5-0.8B, SFT + GRPO) is run with the authors' own
GEN-prompt driver (``lib/external/scenegraphvlm/infer_swift_gen_prompt.py``, vendored
verbatim from markus0440/SceneGraphVLM@a6197da): frame t is prompted with the model's
own TOON graph of frame t-1 of the same video (``--prev-source model``), frame 0 with
the no-context prompt.  Prompts are built with the authors' ``sft_to_jsonl_ag.py``
functions, frames are resized to 640x480 (bilinear) exactly like
``prepare_original_ag_sft.py``.  Nothing about the model or its decoding is changed.

Three steps, each resumable (a step skips outputs that already exist unless --force):

    # 1. frames + Swift JSONL (any env with PIL/numpy, e.g. wsg)
    python -m lib.mllm.methods.scenegraphvlm.runner prepare --video_list <split> --tag t150
    # 2. inference (sgvlm env: vLLM 0.30 + ms-swift 4.5), GPU 0
    CUDA_VISIBLE_DEVICES=0 <sgvlm>/bin/python -m lib.mllm.methods.scenegraphvlm.runner infer --tag t150
    # 3. TOON -> per-video PKLs read by lib/mllm/eval/dump_adapter.py (+ parse/mapping stats)
    python -m lib.mllm.methods.scenegraphvlm.runner convert --tag t150

Frames = every annotated frame of the worldbbox annotation (``WorldBBoxVideo.frames``,
the same frames every other MLLM runner is scored on), in frame order; the previous
frame of the chain is the previous annotated frame (as in the authors' AG test JSONL).

Output layout under ``outputs.scenegraphvlm`` (default /data3/rohith/ag/runs/mllm/scenegraphvlm/):

    jsonl/<tag>.jsonl                 Swift chat rows (content = observed-GT TOON, only used
                                      by the authors' own evaluator for a sanity check)
    raw/<model>/<tag>.jsonl           the authors' driver output (``predict`` = TOON)
    sgdet/<model>/<vid>.mp4.pkl       ``{"frames": {frame_file: {"objects": {short_label:
                                      {"attention","spatial","contacting": [(label, score)],
                                      "bbox_2d": xyxy ORIGINAL px, "score"}},
                                      "person_bbox_2d", "objects_norel", "raw_response"}}}``
    stats/<model>__<tag>.json         parse rate, unmapped labels, objects/frame, ...

Only an sgdet-style mode exists: the model proposes its own objects and 2D boxes.
There is no predcls mode -- the released model has no input for the current frame's
object list (its only conditioning is the previous frame's graph), so GT-object
conditioning would be a different, untrained prompt.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

logger = logging.getLogger("scenegraphvlm")

TARGET_W, TARGET_H = 640, 480
DEFAULT_ROOT = "/data3/rohith/ag/runs/mllm/scenegraphvlm/"
DEFAULT_FRAMES = "/data3/rohith/ag/cache/mllm/sgvlm_frames/"
DEFAULT_CKPT = "/data3/rohith/hf/sgvlm/checkpoints/AG"
DEFAULT_MODEL = "sgvlm_ag"
DEFAULT_PY = "/data3/datasets/OakInk_pilot/conda_envs/sgvlm/bin/python"
VENDORED = Path(REPO) / "lib" / "external" / "scenegraphvlm"

# ---------------------------------------------------------------------------
# vocabulary mapping (model names -> our 36 short labels / 26 predicates)
# ---------------------------------------------------------------------------
_ATT = ["looking_at", "not_looking_at", "unsure"]
_SPA = ["above", "beneath", "in_front_of", "behind", "on_the_side_of", "in"]
_CON = ["carrying", "covered_by", "drinking_from", "eating", "have_it_on_the_back",
        "holding", "leaning_on", "lying_on", "not_contacting", "other_relationship",
        "sitting_on", "standing_on", "touching", "twisting", "wearing", "wiping", "writing_on"]
_OBJ_SYNONYMS = {  # extra surface forms -> AG full name (logged when used)
    "closet": "closet/cabinet", "cabinet": "closet/cabinet", "cup": "cup/glass/bottle",
    "glass": "cup/glass/bottle", "bottle": "cup/glass/bottle", "paper": "paper/notebook",
    "notebook": "paper/notebook", "phone": "phone/camera", "camera": "phone/camera",
    "sofa": "sofa/couch", "couch": "sofa/couch", "tv": "television",
}


def _cfg_root(cfg=None) -> str:
    try:
        from lib.mllm.core.config_loader import get_path, load_config
        cfg = cfg or load_config()
        return get_path(cfg, "outputs.scenegraphvlm") or DEFAULT_ROOT
    except Exception:  # noqa -- the sgvlm env may not have the lib.mllm deps
        return DEFAULT_ROOT


def _norm_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("_", " "))


def map_object(name: str) -> Tuple[Optional[str], str]:
    """-> (short label in our vocab | None, how: exact|synonym|unmapped)."""
    from lib.mllm.data.worldbbox import OBJECT_CLASSES, to_short
    n = _norm_name(name)
    full = {c: c for c in OBJECT_CLASSES[1:]}
    if n in full:
        return to_short(n), "exact"
    n2 = n.replace("-", "/").replace(" / ", "/")
    if n2 in full:
        return to_short(n2), "exact"
    if n in _OBJ_SYNONYMS:
        return to_short(_OBJ_SYNONYMS[n]), "synonym"
    return None, "unmapped"


def map_predicate(p: str) -> Optional[str]:
    q = re.sub(r"[\s\-]+", "_", str(p).strip().lower())
    return q if q in _ATT or q in _SPA or q in _CON else None


# ---------------------------------------------------------------------------
# TOON parsing (same line grammar as the authors' metrics/sgbench parser, but
# lenient on boxes: clipped to the 640x480 canvas instead of dropping the object)
# ---------------------------------------------------------------------------
_RE_OBJ_LINE = re.compile(
    r"^\s*(\d+)\s*,\s*([^,\[\]]+?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
    r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", re.MULTILINE)
_RE_REL_LINE = re.compile(
    r"^\s*(\d+)\s*,\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*,\s*\[([^\]]*)\]\s*,\s*(\d+)\s*$", re.MULTILINE)
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def parse_toon(text: str) -> Dict[str, Any]:
    t = _RE_THINK.sub("", text or "")
    t = t.replace("<answer>", "\n").replace("</answer>", "\n")
    objs: Dict[int, Dict[str, Any]] = {}
    for m in _RE_OBJ_LINE.finditer(t):
        oid = int(m.group(1))
        if oid in objs:
            continue
        x1, y1, x2, y2 = (float(m.group(i)) for i in range(3, 7))
        x1, x2 = sorted((min(max(x1, 0.0), TARGET_W), min(max(x2, 0.0), TARGET_W)))
        y1, y2 = sorted((min(max(y1, 0.0), TARGET_H), min(max(y2, 0.0), TARGET_H)))
        box = [x1, y1, x2, y2] if (x2 > x1 and y2 > y1) else None
        objs[oid] = {"name": m.group(2).strip(), "box": box}
    rels = []
    for m in _RE_REL_LINE.finditer(t):
        sp = lambda s: [x.strip() for x in s.split(",") if x.strip()]  # noqa: E731
        rels.append({"subj": int(m.group(1)), "obj": int(m.group(5)),
                     "attention": sp(m.group(2)), "spatial": sp(m.group(3)), "contacting": sp(m.group(4))})
    return {"objects": objs, "rel_pairs": rels,
            "has_obj_header": bool(re.search(r"obj\[\d+\]\{id,name", t)),
            "has_rel_header": bool(re.search(r"rel_pairs\[\d+\]\{", t))}


# ---------------------------------------------------------------------------
# step 1: prepare
# ---------------------------------------------------------------------------

def _video_ids(video_list: str) -> List[str]:
    return [Path(l.strip()).stem.replace(".mp4", "") for l in open(video_list, encoding="utf-8") if l.strip()]


def _gt_toon(fr, sx: float, sy: float) -> str:
    """Observed-GT TOON of one frame, built like prepare_original_ag_sft.py (visible objects
    with a box; person = id 1), only for the authors' own evaluator (sanity check)."""
    from lib.external.scenegraphvlm.sft_to_jsonl_ag import format_assistant_like_psg
    objects, rels = [], []
    sc = lambda b: [int(round(b[0] * sx)), int(round(b[1] * sy)), int(round(b[2] * sx)), int(round(b[3] * sy))]  # noqa
    pid = None
    if fr.person_bbox_2d is not None:
        pid = 1
        objects.append((1, "person", sc(fr.person_bbox_2d)))
    for o in fr.objects:
        if not o.observed or o.bbox_2d is None:
            continue
        oid = len(objects) + 1
        objects.append((oid, o.cls, sc(o.bbox_2d)))
        if pid is not None and (o.attention or o.spatial or o.contacting):
            rels.append((pid, o.attention, o.spatial, o.contacting, oid))
    lines = [f"obj[{len(objects)}]{{id,name,x1,y1,x2,y2}}:"]
    lines += [f"  {i},{n.replace(',', '-')},{b[0]},{b[1]},{b[2]},{b[3]}" for i, n, b in objects]
    lines.append(f"rel_pairs[{len(rels)}]{{subj,attention,spatial,contacting,obj}}:")
    lines += [f"  {s},[{','.join(a)}],[{','.join(p)}],[{','.join(c)}],{o}" for s, a, p, c, o in rels]
    return format_assistant_like_psg("\n".join(lines))


def _resize_one(args) -> Tuple[str, Optional[str]]:
    src, dst = args
    if os.path.exists(dst):
        return dst, None
    try:
        from PIL import Image
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            if im.size == (TARGET_W, TARGET_H):
                im.save(dst + ".tmp.png")
            else:
                im.resize((TARGET_W, TARGET_H), Image.BILINEAR).save(dst + ".tmp.png")
        os.replace(dst + ".tmp.png", dst)
        return dst, None
    except Exception as e:  # noqa
        return dst, repr(e)


def cmd_prepare(args):
    from multiprocessing import Pool
    from lib.external.scenegraphvlm.sft_to_jsonl_ag import (
        IMG_PREFIX, USER_PROMPT_FIRST, build_user_prompt_follow)
    from lib.mllm.core.config_loader import load_config
    from lib.mllm.data.worldbbox import WorldBBoxTestSet
    cfg = load_config()
    root = Path(args.root or _cfg_root(cfg))
    (root / "jsonl").mkdir(parents=True, exist_ok=True)
    out = root / "jsonl" / f"{args.tag}.jsonl"
    if out.exists() and not args.force:
        logger.info(f"exists: {out}")
        return
    ts = WorldBBoxTestSet(cfg)
    vids = _video_ids(args.video_list)
    if args.limit:
        vids = vids[:args.limit]
    rows, jobs, meta = [], [], {}
    for vid in vids:
        v = ts.load(vid)
        ow, oh = v.orig_size
        sx, sy = TARGET_W / ow, TARGET_H / oh
        prev_body = None
        for fr in v.frames:
            src = str(v.frame_path(fr.file))
            dst = os.path.join(args.frames_root, v.vid_mp4, fr.file)
            jobs.append((src, dst))
            body = _gt_toon(fr, sx, sy)
            user = USER_PROMPT_FIRST if prev_body is None else build_user_prompt_follow(prev_body)
            rows.append({"messages": [{"role": "user", "content": IMG_PREFIX + user},
                                      {"role": "assistant", "content": "<answer>\n" + body + "\n</answer>\n"}],
                         "images": [dst]})
            meta[dst] = {"video_id": vid, "frame_file": fr.file, "orig_size": [ow, oh]}
            prev_body = body
    logger.info(f"{len(vids)} videos, {len(rows)} frames; resizing to {args.frames_root}")
    errs = []
    with Pool(args.workers) as pool:
        for i, (dst, err) in enumerate(pool.imap_unordered(_resize_one, jobs, chunksize=32)):
            if err:
                errs.append((dst, err))
            if (i + 1) % 5000 == 0:
                logger.info(f"  {i + 1}/{len(jobs)} frames")
    if errs:
        raise SystemExit(f"{len(errs)} frames failed to resize, e.g. {errs[:3]}")
    with open(str(out) + ".tmp", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(str(out) + ".tmp", out)
    with open(root / "jsonl" / f"{args.tag}.meta.json", "w") as f:
        json.dump({"videos": vids, "frames": meta}, f)
    logger.info(f"wrote {out} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# step 2: infer (the authors' driver, unmodified)
# ---------------------------------------------------------------------------

def cmd_infer(args):
    root = Path(args.root or DEFAULT_ROOT)
    jl = root / "jsonl" / f"{args.tag}.jsonl"
    out_dir = root / "raw" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / f"{args.tag}.jsonl").exists() and not args.force:
        logger.info(f"exists: {out_dir / (args.tag + '.jsonl')}")
        return
    cmd = [args.python, str(VENDORED / "launch.py"),   # = infer_swift_gen_prompt.py + model_type shim
           "--model", args.ckpt, "--test-jsonl", str(jl), "--output-dir", str(out_dir),
           "--run-name", args.tag, "--infer-backend", "vllm", "--batch-size", str(args.batch_size),
           "--max-new-tokens", str(args.max_new_tokens), "--temperature", "0.0",
           "--max-model-len", "8192", "--gpu-memory-utilization", str(args.gpu_memory_utilization),
           "--dataset-tag", "ag_worldbbox_test", "--checkpoint-step", "released",
           "--response-prefix", "<answer>\n", "--prev-source", args.prev_source, "--force",
           # ms-swift 4.5 matches both qwen3_5 and qwen3_8 templates; the checkpoint is qwen3_5
           "--template-type", "qwen3_5"]
    env = dict(os.environ)
    env.setdefault("IMAGE_MAX_TOKEN_NUM", "1024")
    logger.info("running: " + " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    logger.info(f"driver exited rc={rc} after {time.time() - t0:.0f}s")
    if rc != 0:
        raise SystemExit(rc)


# ---------------------------------------------------------------------------
# step 3: convert
# ---------------------------------------------------------------------------

def _read_raw(path: Path) -> Dict[str, Dict[str, Any]]:
    """image path -> last record (the driver writes error rows mid-run and every row at the end)."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r.get("images") or [""])[0]] = r
    return out


def frame_prediction(text: str, orig_size, stats: Counter, unm_obj: Counter, unm_pred: Counter,
                     syn_obj: Counter) -> Dict[str, Any]:
    ow, oh = orig_size
    fx, fy = ow / TARGET_W, oh / TARGET_H
    p = parse_toon(text)
    stats["frames"] += 1
    if not (text or "").strip():
        stats["empty_response"] += 1
    if p["objects"]:
        stats["frames_parsed"] += 1
    if p["rel_pairs"]:
        stats["frames_with_rel_rows"] += 1
    if p["has_obj_header"] and not p["has_rel_header"]:
        stats["frames_missing_rel_header"] += 1
    person_ids, id2lab = [], {}
    person_box = None
    boxes: Dict[str, List[float]] = {}
    unmapped_names = []
    for oid, o in p["objects"].items():
        stats["objects_raw"] += 1
        if o["box"] is None:
            stats["objects_degenerate_box"] += 1
        box = None if o["box"] is None else [o["box"][0] * fx, o["box"][1] * fy, o["box"][2] * fx, o["box"][3] * fy]
        if _norm_name(o["name"]) == "person":
            person_ids.append(oid)
            if person_box is None:
                person_box = box
            continue
        lab, how = map_object(o["name"])
        if lab is None:
            unm_obj[_norm_name(o["name"])] += 1
            stats["objects_unmapped"] += 1
            unmapped_names.append(_norm_name(o["name"]))
            continue
        if how == "synonym":
            syn_obj[_norm_name(o["name"])] += 1
        id2lab[oid] = lab
        if lab not in boxes or boxes[lab] is None:
            boxes[lab] = box
    objects: Dict[str, Dict[str, Any]] = {}
    for r in p["rel_pairs"]:
        stats["rel_rows"] += 1
        if r["subj"] not in person_ids:
            stats["rel_rows_nonperson_subj"] += 1
            continue
        lab = id2lab.get(r["obj"])
        if lab is None:
            stats["rel_rows_dangling_or_unmapped_obj"] += 1
            continue
        e = objects.setdefault(lab, {"attention": [], "spatial": [], "contacting": [],
                                     "bbox_2d": boxes.get(lab), "score": 1.0})
        for head in ("attention", "spatial", "contacting"):
            have = {l for l, _ in e[head]}
            for lbl in r[head]:
                q = map_predicate(lbl)
                if q is None:
                    unm_pred[str(lbl).strip().lower()] += 1
                    stats["predicates_unmapped"] += 1
                    continue
                if q in have:
                    continue
                if (head == "attention" and q not in _ATT) or (head == "spatial" and q not in _SPA) or \
                        (head == "contacting" and q not in _CON):
                    stats["predicates_wrong_head"] += 1
                    continue
                e[head].append((q, 1.0 - 1e-4 * len(e[head])))   # model order, as the authors' evaluator
                have.add(q)
                stats["predicates_kept"] += 1
    norel = sorted({l for l in id2lab.values() if l not in objects})
    stats["objects_mapped_unique"] += len(set(id2lab.values()))
    stats["objects_with_relations"] += len(objects)
    stats["objects_norel"] += len(norel)
    return {"objects": objects, "objects_norel": norel, "objects_unmapped": unmapped_names,
            "person_bbox_2d": person_box, "raw_response": text}


def cmd_convert(args):
    root = Path(args.root or _cfg_root())
    meta = json.load(open(root / "jsonl" / f"{args.tag}.meta.json"))
    raw = _read_raw(root / "raw" / args.model / f"{args.tag}.jsonl")
    out_dir = root / "sgdet" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    stats, unm_obj, unm_pred, syn_obj = Counter(), Counter(), Counter(), Counter()
    per_video: Dict[str, Dict[str, Any]] = {}
    missing_rows, errors = 0, 0
    for img, m in meta["frames"].items():
        r = raw.get(img)
        if r is None:
            missing_rows += 1
            text = ""
        else:
            text = r.get("predict") or ""
            if r.get("predict_error"):
                errors += 1
        fp = frame_prediction(text, m["orig_size"], stats, unm_obj, unm_pred, syn_obj)
        per_video.setdefault(m["video_id"], {})[m["frame_file"]] = fp
    for vid in meta["videos"]:
        rec = {"video_id": f"{vid}.mp4", "mode": "sgdet", "model_name": args.model,
               "method": "scenegraphvlm", "frames": per_video.get(vid, {}),
               "meta": {"tag": args.tag, "prev_source": "model", "checkpoint": "zenodo 20511274 checkpoints/AG"}}
        with open(out_dir / f"{vid}.mp4.pkl", "wb") as f:
            pickle.dump(rec, f, protocol=4)
    n = max(1, stats["frames"])
    summary = {
        "tag": args.tag, "model": args.model, "videos": len(meta["videos"]), "frames": stats["frames"],
        "raw_rows_missing": missing_rows, "driver_errors": errors,
        "parse_rate": stats["frames_parsed"] / n,
        "frames_with_rel_rows": stats["frames_with_rel_rows"] / n,
        "objects_raw_per_frame": stats["objects_raw"] / n,
        "objects_mapped_per_frame": stats["objects_mapped_unique"] / n,
        "objects_with_relations_per_frame": stats["objects_with_relations"] / n,
        "object_unmapped_rate": stats["objects_unmapped"] / max(1, stats["objects_raw"]),
        "predicate_unmapped_rate": stats["predicates_unmapped"] / max(1, stats["predicates_unmapped"] + stats["predicates_kept"]),
        "counts": dict(stats), "unmapped_objects": dict(unm_obj.most_common(50)),
        "synonym_objects": dict(syn_obj.most_common(50)),
        "unmapped_predicates": dict(unm_pred.most_common(50)),
    }
    (root / "stats").mkdir(exist_ok=True)
    sp = root / "stats" / f"{args.model}__{args.tag}.json"
    with open(sp, "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("counts",)}, indent=1))
    logger.info(f"wrote {len(meta['videos'])} pkls to {out_dir}; stats {sp}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "infer", "convert"):
        s = sub.add_parser(name)
        s.add_argument("--tag", required=True)
        s.add_argument("--root", default=None)
        s.add_argument("--model", default=DEFAULT_MODEL)
        s.add_argument("--force", action="store_true")
        if name == "prepare":
            s.add_argument("--video_list", required=True)
            s.add_argument("--frames_root", default=DEFAULT_FRAMES)
            s.add_argument("--workers", type=int, default=8)
            s.add_argument("--limit", type=int, default=0)
        if name == "infer":
            s.add_argument("--ckpt", default=DEFAULT_CKPT)
            s.add_argument("--python", default=DEFAULT_PY)
            s.add_argument("--batch_size", type=int, default=64)
            s.add_argument("--max_new_tokens", type=int, default=2048)
            s.add_argument("--gpu_memory_utilization", type=float, default=0.85)
            s.add_argument("--prev_source", default="model", choices=["model", "gt"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    {"prepare": cmd_prepare, "infer": cmd_infer, "convert": cmd_convert}[args.cmd](args)


if __name__ == "__main__":
    main()
