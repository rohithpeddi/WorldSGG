"""
Track A -- prompt-engineered localised WSGG with a (thinking) VLM (B5).
=======================================================================

    CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.track_a_prompt.runner \
        --model_name qwen3vl_8b_thinking --mode predcls --video_list <split> [--limit N]
    python -m lib.mllm.methods.track_a_prompt.runner --mode sgdet --dry_run --limit 2   # no GPU: writes prompts+images

Per annotated frame one VLM call with: the marked target frame (set-of-mark ids),
``--context_frames`` unmarked context frames, and the marked BEV (B4 caches), plus a
text table of the same objects with metric OBBs.  predcls: ids = the frame's GT
objects (boxes given, relations asked).  sgdet: ids = GDino proposals with their
lifted OBBs; the model returns objects (kept / new) with OBBs and relations.

Every call goes through ``lib/mllm/tools/llm_cache.py`` (content hash), so a
re-run after an annotation revision only recomputes prompts whose content
changed.  Output: ``<outputs.track_a>/<mode>/<model>/<video>.mp4.pkl`` =
``{"video_id", "mode", "model_name", "frames": {frame_file: {"objects": {label:
{"attention", "spatial", "contacting", "corners", "obb", "score", "src"}},
"raw_response", "prompt_key"}}}`` -- read by ``lib/mllm/eval/dump_adapter.py``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.mllm.core.config_loader import get_path, get_vllm_engine_settings, load_config  # noqa: E402
from lib.mllm.data.geometry import corners_to_obb, obb_to_corners                   # noqa: E402
from lib.mllm.data.worldbbox import (                                                # noqa: E402
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, NAME_TO_IDX, SPATIAL_RELATIONSHIPS,
    WorldBBoxTestSet, WorldBBoxVideo, to_short,
)
from lib.mllm.methods.track_a_prompt.prompts import predcls_prompt, sgdet_prompt      # noqa: E402
from lib.mllm.tools.bev import get_bev, mark_bev                                      # noqa: E402
from lib.mllm.tools.detections import get_detections                                 # noqa: E402
from lib.mllm.tools.lift import LiftCache                                            # noqa: E402
from lib.mllm.tools.llm_cache import CachedVLM, LLMCache                             # noqa: E402
from lib.mllm.tools.marks import mark_frame                                          # noqa: E402

logger = logging.getLogger("track_a")
_ATT, _SPA, _CON = set(ATTENTION_RELATIONSHIPS), set(SPATIAL_RELATIONSHIPS), set(CONTACTING_RELATIONSHIPS)


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def obb_dict(corners) -> Optional[Dict[str, Any]]:
    if corners is None or not np.any(corners):
        return None
    c, s, yaw = corners_to_obb(np.asarray(corners))
    return {"center": [round(float(v), 3) for v in c], "size": [round(float(v), 3) for v in s],
            "yaw_deg": round(math.degrees(yaw), 1)}


def camera_dict(pose: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if pose is None:
        return None
    P = np.asarray(pose, np.float64)
    fwd = P[:3, :3] @ np.array([0, 0, 1.0])
    return {"x": float(P[0, 3]), "y": float(P[1, 3]), "z": float(P[2, 3]),
            "heading_deg": float(math.degrees(math.atan2(fwd[1], fwd[0])))}


def _norm(lab: Any) -> str:
    return str(lab).strip().lower().replace(" ", "_")


# ---------------------------------------------------------------------------
# per-frame payload
# ---------------------------------------------------------------------------

class TrackAContext:
    """Per-video handles on the B4 caches."""

    def __init__(self, video: WorldBBoxVideo, cfg: dict, mode: str):
        self.video, self.mode = video, mode
        self.bev_img, self.bev_meta, _ = get_bev(video, get_path(cfg, "worldbbox.caches.bev"))
        self.extent = {"x0": self.bev_meta["x0"], "y0": self.bev_meta["y0"],
                       "x1": self.bev_meta["x0"] + self.bev_meta["width"] / self.bev_meta["px_per_m"],
                       "y1": self.bev_meta["y0"] + self.bev_meta["height"] / self.bev_meta["px_per_m"]}
        self.dets = None
        self.lift = None
        if mode == "sgdet":
            self.dets = get_detections(video, get_path(cfg, "annotations.detection_dynamic"),
                                       get_path(cfg, "worldbbox.caches.detections"))
            self.lift = LiftCache(video, get_path(cfg, "worldbbox.caches.lifted3d"))
        self._img_cache: Dict[str, Image.Image] = {}

    def frame_image(self, frame_file: str) -> Image.Image:
        if frame_file not in self._img_cache:
            self._img_cache[frame_file] = Image.open(self.video.frame_path(frame_file)).convert("RGB")
        return self._img_cache[frame_file]

    def save(self):
        if self.lift is not None:
            self.lift.save()


def build_payload(ctx: TrackAContext, fr, n_context: int, ctx_side: int = 320,
                  bev_side: int = 640, target_side: int = 640) -> Dict[str, Any]:
    v = ctx.video
    frames = v.frames
    # ---- objects / ids ----
    objs: List[Dict[str, Any]] = []
    if ctx.mode == "predcls":
        seen = set()
        for o in fr.objects:
            if o.label in seen:
                continue
            seen.add(o.label)
            objs.append({"id": len(objs) + 1, "label": o.label, "bbox": None if o.bbox_2d is None else o.bbox_2d.tolist(),
                         "corners": o.corners_final, "obb": obb_dict(o.corners_final), "visible": o.observed})
        person_corners = fr.person_corners_final
        person_bbox = None if fr.person_bbox_2d is None else fr.person_bbox_2d.tolist()
    else:
        dets = ctx.dets["frames"].get(fr.file, [])
        person_dets = [d for d in dets if d["label"] == "person"]
        person_bbox = None
        person_corners = None
        if person_dets:
            pd = max(person_dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
            person_bbox = pd["bbox"]
            r = ctx.lift.get(fr.file, "person", pd["bbox_pi3"])
            person_corners = None if r is None else np.asarray(r["corners"])
        for d in sorted((d for d in dets if d["label"] != "person"), key=lambda d: -d["score"]):
            r = ctx.lift.get(fr.file, d["label"], d["bbox_pi3"])
            corners = None if r is None else np.asarray(r["corners"])
            objs.append({"id": len(objs) + 1, "label": d["label"], "bbox": d["bbox"], "score": d["score"],
                         "corners": corners, "obb": obb_dict(corners), "visible": True})
    # ---- images ----
    target = mark_frame(ctx.frame_image(fr.file), objs, person_bbox=person_bbox, max_side=target_side)
    others = [f for f in frames if f.file != fr.file]
    if n_context and others:
        idx = np.unique(np.linspace(0, len(others) - 1, min(n_context, len(others))).round().astype(int))
        ctx_imgs = []
        for i in idx:
            im = ctx.frame_image(others[i].file)
            s = min(1.0, ctx_side / max(im.size))
            ctx_imgs.append(im.resize((int(im.width * s), int(im.height * s)), Image.BILINEAR) if s < 1 else im)
    else:
        ctx_imgs = []
    cam_pose = fr.camera_pose_final
    if cam_pose is None and fr.pi3_index is not None:
        cam_pose = v.camera_pose_final_for_pi3(fr.pi3_index)
    bev = mark_bev(ctx.bev_img, ctx.bev_meta, objs, camera_pose=cam_pose, person_corners=person_corners)
    if max(bev.size) > bev_side:
        sc = bev_side / max(bev.size)
        bev = bev.resize((int(bev.width * sc), int(bev.height * sc)), Image.BILINEAR)
    images = [target] + ctx_imgs + [bev]
    cam = camera_dict(cam_pose)
    pobb = obb_dict(person_corners)
    if ctx.mode == "predcls":
        text = predcls_prompt(fr.file, objs, pobb, cam, len(ctx_imgs), True)
    else:
        text = sgdet_prompt(fr.file, objs, pobb, cam, len(ctx_imgs), True, ctx.extent)
    return {"text": text, "images": images, "objects": objs, "person_corners": person_corners}


# ---------------------------------------------------------------------------
# response parsing
# ---------------------------------------------------------------------------

def salvage_objects(text: str) -> Optional[dict]:
    """Recover the COMPLETE entries of a truncated ``{"objects": [{...}, {...}, ...``
    response.  The models answer with one JSON array, so a response cut off at
    ``max_new_tokens`` is unparseable and the whole frame would be dropped even
    though every entry before the cut is valid."""
    i = text.find('"objects"')
    if i == -1:
        return None
    i = text.find("[", i)
    if i == -1:
        return None
    items, j, n = [], i + 1, len(text)
    while j < n:
        if text[j] != "{":
            if text[j] == "]":
                break
            j += 1
            continue
        depth, k, closed = 0, j, False
        while k < n:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    break
            k += 1
        if not closed:
            break                      # the truncated tail entry
        try:
            items.append(json.loads(text[j:k + 1]))
        except Exception:
            pass
        j = k + 1
    return {"objects": items, "truncated": True} if items else None


def extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    text = text.replace("```json", "```").strip()
    cands = re.findall(r"```(.*?)```", text, re.S) or []
    cands.append(text)
    for c in cands:
        c = c.strip()
        i = c.find("{")
        while i != -1:
            depth = 0
            for j in range(i, len(c)):
                if c[j] == "{":
                    depth += 1
                elif c[j] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            d = json.loads(c[i:j + 1])
                            if isinstance(d, dict) and "objects" in d:
                                return d
                        except Exception:
                            pass
                        break
            i = c.find("{", i + 1)
    for c in cands:                    # truncated response -> keep what is complete
        d = salvage_objects(c)
        if d is not None:
            return d
    return None


def parse_objects(d: Optional[dict], payload: Dict[str, Any], mode: str) -> Dict[str, Dict[str, Any]]:
    """-> {label: {attention, spatial, contacting, corners, obb, score, src}} (first per label wins)."""
    by_id = {o["id"]: o for o in payload["objects"]}
    out: Dict[str, Dict[str, Any]] = {}
    if not d:
        return out
    for item in d.get("objects", []) or []:
        if not isinstance(item, dict):
            continue
        oid = item.get("id")
        try:
            oid = int(oid)
        except (TypeError, ValueError):
            oid = None
        ref = by_id.get(oid) if oid is not None else None
        if mode == "predcls":
            if ref is None:
                continue
            label, corners, obb, src = ref["label"], ref["corners"], ref["obb"], "gt"
        else:
            label = to_short(_norm(item.get("label", ref["label"] if ref else "")))
            if label not in NAME_TO_IDX or label == "person":
                continue
            corners, obb, src = (ref["corners"], ref["obb"], "proposal") if ref else (None, None, "new")
            c, s = item.get("center"), item.get("size")
            if isinstance(c, list) and isinstance(s, list) and len(c) == 3 and len(s) == 3:
                try:
                    yaw = math.radians(float(item.get("yaw_deg", 0.0)))
                    corners = obb_to_corners([float(x) for x in c], [abs(float(x)) for x in s], yaw)
                    obb = obb_dict(corners)
                    src = "model" if ref is None else "proposal+model"
                except (TypeError, ValueError):
                    pass
            if corners is None:
                continue                      # no localisation -> cannot be scored in 3D
        att = _norm(item.get("attention", ""))
        att = [att] if att in _ATT else []
        spa = [_norm(x) for x in (item.get("spatial") or []) if _norm(x) in _SPA]
        con = [_norm(x) for x in (item.get("contacting") or []) if _norm(x) in _CON]
        if label in out:
            continue
        out[label] = {"attention": att, "spatial": spa, "contacting": con,
                      "corners": None if corners is None else np.asarray(corners, np.float32).tolist(),
                      "obb": obb, "score": float(ref.get("score", 1.0)) if ref else 1.0, "src": src}
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def load_model(cfg: dict, model_name: str, tp: int, max_model_len: int, temperature: float,
               top_p: float, seed: int, max_images: int):
    from lib.mllm.core.vgent import Vgent
    vllm_cfg = get_vllm_engine_settings(cfg)

    class Args:
        pass
    a = Args()
    a.model_name = model_name
    a.tensor_parallel_size = tp
    a.use_vllm = True
    a.model_weights_dir = get_path(cfg, "model_weights") or None
    a.gpu_memory_utilization = vllm_cfg.get("gpu_memory_utilization", 0.9)
    a.max_model_len = max_model_len
    a.max_num_seqs = vllm_cfg.get("max_num_seqs", 64)
    a.enable_chunked_prefill = vllm_cfg.get("enable_chunked_prefill", True)
    a.dtype = vllm_cfg.get("dtype", "bfloat16")
    a.temperature, a.top_p, a.seed, a.max_images = temperature, top_p, seed, max_images
    a.fps, a.chunk_size, a.total_pixels = 1, 128, 128000
    a.n_retrieval, a.n_refine, a.uniform_frame = 20, 5, 450
    return Vgent(a).model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="qwen3vl_8b_thinking")
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--video_list", default=None)
    ap.add_argument("--video_id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--context_frames", type=int, default=2)
    ap.add_argument("--videos_per_batch", type=int, default=4,
                    help="pool the prompts of this many videos into one vLLM generate() call")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--tag", default="", help="suffix for the output model dir (prompt/ablation variants)")
    ap.add_argument("--dry_run", action="store_true", help="build prompts + images only (no model), dump to logs/track_a_dry/")
    ap.add_argument("--status", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = load_config()
    ts = WorldBBoxTestSet(cfg)
    ids = ts.video_ids
    if args.video_list:
        keep = {Path(l.strip()).stem for l in open(args.video_list) if l.strip()}
        ids = [v for v in ids if v in keep]
    if args.video_id:
        ids = [Path(args.video_id).stem]
    if args.limit:
        ids = ids[: args.limit]
    out_dir = Path(get_path(cfg, "outputs.track_a")) / args.mode / (args.model_name + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status or str(Path(get_path(cfg, "outputs.logs") or "/data3/rohith/ag/logs") /
                                     f"track_a_{args.mode}_{args.model_name}{args.tag}.status.json")
    cache = LLMCache(get_path(cfg, "worldbbox.caches.llm_cache"))
    vlm = None
    if not args.dry_run:
        model = load_model(cfg, args.model_name, args.tensor_parallel_size, args.max_model_len,
                           args.temperature, args.top_p, args.seed, max_images=args.context_frames + 2)
        vlm = CachedVLM(model, args.model_name, cache, temperature=args.temperature, strip_thinking=False)
    dry_dir = Path("/data3/rohith/ag/logs/track_a_dry") / args.mode
    t0 = time.time()
    n_done = n_skip = n_fail = 0
    def flush(group):
        """One generate() over the prompts of several videos, then parse + save per video."""
        nonlocal n_done, n_fail
        prompts = [{"text": p["text"], "images": p["images"], "max_new_tokens": args.max_new_tokens}
                   for _, payloads in group for _, p in payloads]
        responses = vlm.generate(prompts)
        pos = 0
        for vid, payloads in group:
            resp_v = responses[pos:pos + len(payloads)]
            pos += len(payloads)
            try:
                frames_out: Dict[str, Any] = {}
                n_parsed = 0
                for (fr, p), resp in zip(payloads, resp_v):
                    d = extract_json(resp)
                    objs = parse_objects(d, p, args.mode)
                    n_parsed += d is not None
                    frames_out[fr.file] = {"objects": objs, "raw_response": resp,
                                           "ids": {o["id"]: o["label"] for o in p["objects"]},
                                           "person_corners": None if p["person_corners"] is None
                                           else np.asarray(p["person_corners"], np.float32).tolist()}
                rec = {"video_id": f"{vid}.mp4", "mode": args.mode, "model_name": args.model_name + args.tag,
                       "track": "A", "context_frames": args.context_frames, "frames": frames_out,
                       "n_frames": len(frames_out), "n_parsed": n_parsed}
                out_path = out_dir / f"{vid}.mp4.pkl"
                tmp = out_path.with_suffix(".pkl.tmp")
                with open(tmp, "wb") as f:
                    pickle.dump(rec, f)
                os.replace(tmp, out_path)
                n_done += 1
                logger.info(f"{vid}: {len(frames_out)} frames, {n_parsed} parsed, cache hits={cache.hits} "
                            f"misses={cache.misses}, {time.time() - t0:.0f}s")
            except Exception as e:  # noqa
                n_fail += 1
                logger.exception(f"[{vid}] save failed: {e!r}")

    group: List[Tuple[str, list]] = []
    for i, vid in enumerate(ids):
        out_path = out_dir / f"{vid}.mp4.pkl"
        if out_path.exists() and not args.dry_run:
            n_skip += 1
            continue
        try:
            video = ts.load(vid)
            ctx = TrackAContext(video, cfg, args.mode)
            payloads = [(fr, build_payload(ctx, fr, args.context_frames)) for fr in video.frames]
            ctx.save()
            if args.dry_run:
                d = dry_dir / vid
                d.mkdir(parents=True, exist_ok=True)
                for fr, p in payloads[:3]:
                    for j, im in enumerate(p["images"]):
                        im.save(d / f"{fr.file[:-4]}_img{j}.png")
                    (d / f"{fr.file[:-4]}_prompt.txt").write_text(p["text"], encoding="utf-8")
                logger.info(f"[{vid}] dry run: {len(payloads)} frames -> {d}")
                n_done += 1
                continue
            group.append((vid, payloads))
            if len(group) >= args.videos_per_batch:
                logger.info(f"[{i + 1}/{len(ids)}] generating for {len(group)} videos "
                            f"({sum(len(p) for _, p in group)} prompts)")
                flush(group)
                group = []
        except Exception as e:  # noqa
            n_fail += 1
            logger.exception(f"[{vid}] failed: {e!r}")
        if (i + 1) % 5 == 0 or i + 1 == len(ids):
            with open(status_path, "w") as f:
                json.dump({"state": "running" if i + 1 < len(ids) else "done", "done": n_done, "skipped": n_skip,
                           "failed": n_fail, "total": len(ids), "cache_hits": cache.hits,
                           "cache_misses": cache.misses, "seconds": round(time.time() - t0)}, f)
    if group:
        flush(group)
    with open(status_path, "w") as f:
        json.dump({"state": "done", "done": n_done, "skipped": n_skip, "failed": n_fail, "total": len(ids),
                   "cache_hits": cache.hits, "cache_misses": cache.misses, "seconds": round(time.time() - t0)}, f)
    logger.info(f"finished: done={n_done} skipped={n_skip} failed={n_fail} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
