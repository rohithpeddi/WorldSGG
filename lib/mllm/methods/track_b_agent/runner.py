"""
Track B -- agentic localised WSGG: tool loop + geometric critic + repair (B6).
==============================================================================

    CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.track_b_agent.runner \
        --model_name qwen3vl_8b_thinking --mode sgdet --video_list <split> [--limit N]

Per annotated frame the agent runs a fixed plan (docs/ICLR_PLAN.md WS2 Track B:
plan -> perceive -> relate -> verify -> repair -> emit) whose tools are the B4
caches and lib/mllm/tools:

  perceive   detect (GDino boxes), lift_to_3d (Pi3 OBBs), get_camera_pose,
             render_bev (marked map)             -- Track A's payload builder
  query_graph the Stage-1 graph/captions of the clips around the frame
             (what the person did before / after)    -- retrieval context
  relate     VLM call 1 -> proposal JSON (objects with OBBs + predicates)
  verify     check_geometry / check_schema (lib/mllm/methods/track_b_agent/critic.py)
  repair     VLM call 2 with the violation list (only when the critic fires)
  emit       parsed graph in the Track output format

The tools are called by the driver in this fixed order (an 8B model choosing
tools freely per frame would not fit the budget); what the model *sees* differs
from Track A only by the retrieved context, and the critic+repair round is the
ablatable component.  Both arms of the ablation come out of one run:
``frames[f]["objects"]`` (after repair, "+critic") and ``frames[f]["objects_pre"]``
(the first proposal, "-critic"); ``--emit pre`` scores the latter through the
usual adapter without re-running anything.

Output: ``<outputs.track_b>/<mode>/<model><tag>/<video>.mp4.pkl`` with the Track A
schema plus ``objects_pre``, ``violations_pre``, ``violations_post``, ``n_calls``.
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.mllm.core.config_loader import get_path, load_config                       # noqa: E402
from lib.mllm.data.worldbbox import WorldBBoxTestSet                                 # noqa: E402
from lib.mllm.methods.track_a_prompt.runner import (                                 # noqa: E402
    TrackAContext, build_payload, extract_json, load_model, parse_objects,
)
from lib.mllm.methods.track_b_agent.critic import check_geometry, summarize          # noqa: E402
from lib.mllm.tools.llm_cache import CachedVLM, LLMCache                             # noqa: E402

logger = logging.getLogger("track_b")


# ---------------------------------------------------------------------------
# query_graph tool: Stage-1 captions / entities around the frame
# ---------------------------------------------------------------------------

def load_stage1(video_id: str, graph_dir: str, model: str = "qwen25vl_7b") -> List[Dict[str, Any]]:
    p = Path(graph_dir) / model / f"{Path(video_id).stem}.mp4.pkl"
    if not p.exists():
        return []
    try:
        with open(p, "rb") as f:
            clips = pickle.load(f)
        return clips if isinstance(clips, list) else []
    except Exception:
        return []


def query_graph(clips: List[Dict[str, Any]], frame_num: int, window: int = 2, max_chars: int = 900) -> str:
    """Captions of the clips whose annotated frame is nearest to ``frame_num`` (before/at/after)."""
    if not clips:
        return ""
    items = []
    for c in clips:
        meta = c.get("clip_metadata", {}) or {}
        af = meta.get("annotated_frame")
        cap = c.get("subtitle") or c.get("caption") or ""
        if af is None or not cap:
            continue
        ents = []
        g = c.get("graph")
        try:
            for _, nd in g.nodes(data=True):
                ents += [e.split(",")[0].strip() for e in nd.get("entities", [])][:6]
        except Exception:
            pass
        items.append((int(af), str(cap).strip(), sorted(set(ents))[:8]))
    if not items:
        return ""
    items.sort()
    idx = min(range(len(items)), key=lambda i: abs(items[i][0] - frame_num))
    lo, hi = max(0, idx - window), min(len(items), idx + window + 1)
    lines = []
    for af, cap, ents in items[lo:hi]:
        when = "at the target frame" if af == items[idx][0] else ("earlier" if af < frame_num else "later")
        s = f"- [{when}, frame {af}] {cap}"
        if ents:
            s += f" (entities: {', '.join(ents)})"
        lines.append(s)
    txt = "\n".join(lines)
    return txt[:max_chars]


def with_context(text: str, retrieved: str) -> str:
    if not retrieved:
        return text
    block = ("Retrieved context from a scene graph built over the whole video (may be noisy):\n"
             f"{retrieved}\n\n")
    marker = "Task:"
    i = text.find(marker)
    return text[:i] + block + text[i:] if i != -1 else block + text


def repair_prompt(base_text: str, proposal_json: str, violations: List[Dict[str, Any]]) -> str:
    v = "\n".join(f"- {x['msg']}" for x in violations[:12])
    return (base_text + "\n\nYour previous answer was:\n" + proposal_json +
            "\n\nA geometric checker found these problems with it:\n" + v +
            "\n\nFix them: keep every correct object and relationship, correct the box or the predicate that is "
            "inconsistent with the 3D layout (boxes must rest on or above the floor, contact predicates need the "
            "object next to the person, 'above'/'beneath' must match the heights). Answer with ONLY the corrected "
            "JSON in the same format.")


def _objs_for_critic(objs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {k: {"attention": v.get("attention"), "spatial": v.get("spatial"), "contacting": v.get("contacting"),
                "corners": None if v.get("corners") is None else np.asarray(v["corners"])} for k, v in objs.items()}


def _proposal_json(objs: Dict[str, Dict[str, Any]], mode: str, ids: Dict[str, int]) -> str:
    items = []
    for lab, o in objs.items():
        it: Dict[str, Any] = {"id": ids.get(lab, "new")}
        if mode == "sgdet":
            it["label"] = lab
            if o.get("obb"):
                it["center"], it["size"], it["yaw_deg"] = o["obb"]["center"], o["obb"]["size"], o["obb"]["yaw_deg"]
        it["attention"] = (o.get("attention") or [""])[0]
        it["contacting"] = o.get("contacting") or []
        it["spatial"] = o.get("spatial") or []
        items.append(it)
    return json.dumps({"objects": items})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="qwen3vl_8b_thinking")
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--video_list", default=None)
    ap.add_argument("--video_id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--context_frames", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--no_critic", action="store_true", help="ablation: skip verify/repair (single call)")
    ap.add_argument("--no_retrieval", action="store_true", help="ablation: no query_graph context")
    ap.add_argument("--max_repairs", type=int, default=1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry_run", action="store_true")
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
    tag = args.tag + ("_nocritic" if args.no_critic else "") + ("_noretr" if args.no_retrieval else "")
    out_dir = Path(get_path(cfg, "outputs.track_b")) / args.mode / (args.model_name + tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status or str(Path(get_path(cfg, "outputs.logs") or "/data3/rohith/ag/logs") /
                                     f"track_b_{args.mode}_{args.model_name}{tag}.status.json")
    graph_dir = get_path(cfg, "graphs")
    cache = LLMCache(get_path(cfg, "worldbbox.caches.llm_cache"))
    vlm = None
    if not args.dry_run:
        model = load_model(cfg, args.model_name, args.tensor_parallel_size, args.max_model_len,
                           args.temperature, args.top_p, args.seed, max_images=args.context_frames + 2)
        vlm = CachedVLM(model, args.model_name, cache, temperature=args.temperature, strip_thinking=False)
    t0 = time.time()
    n_done = n_skip = n_fail = 0
    tot_calls = tot_viol_pre = tot_viol_post = tot_frames = 0
    for i, vid in enumerate(ids):
        out_path = out_dir / f"{vid}.mp4.pkl"
        if out_path.exists() and not args.dry_run:
            n_skip += 1
            continue
        try:
            video = ts.load(vid)
            ctx = TrackAContext(video, cfg, args.mode)
            clips = [] if args.no_retrieval else load_stage1(vid, graph_dir)
            payloads: List[Tuple[Any, Dict[str, Any]]] = []
            for fr in video.frames:
                p = build_payload(ctx, fr, args.context_frames)
                p["retrieved"] = query_graph(clips, fr.frame_num) if clips else ""
                p["text"] = with_context(p["text"], p["retrieved"])
                payloads.append((fr, p))
            ctx.save()
            if args.dry_run:
                d = Path("/data3/rohith/ag/logs/track_b_dry") / args.mode / vid
                d.mkdir(parents=True, exist_ok=True)
                for fr, p in payloads[:2]:
                    (d / f"{fr.file[:-4]}_prompt.txt").write_text(p["text"], encoding="utf-8")
                logger.info(f"[{vid}] dry run: {len(payloads)} frames, retrieval={'yes' if clips else 'no'} -> {d}")
                n_done += 1
                continue
            # ---- relate: proposal call for every frame (batched) ----
            prompts = [{"text": p["text"], "images": p["images"], "max_new_tokens": args.max_new_tokens}
                       for _, p in payloads]
            responses = vlm.generate(prompts)
            n_calls = len(prompts)
            frames_out: Dict[str, Any] = {}
            pending: List[Tuple[int, Dict[str, Any]]] = []
            for j, ((fr, p), resp) in enumerate(zip(payloads, responses)):
                d = extract_json(resp)
                objs = parse_objects(d, p, args.mode)
                pc = p["person_corners"]
                viol = [] if args.no_critic else check_geometry(_objs_for_critic(objs), pc, ctx.extent)
                frames_out[fr.file] = {
                    "objects_pre": objs, "raw_response_pre": resp, "violations_pre": viol,
                    "objects": objs, "raw_response": resp, "violations_post": viol,
                    "ids": {o["id"]: o["label"] for o in p["objects"]},
                    "person_corners": None if pc is None else np.asarray(pc, np.float32).tolist(),
                    "retrieved": p["retrieved"], "n_calls": 1, "parsed": d is not None,
                }
                if viol and d is not None:
                    pending.append((j, {"text": repair_prompt(p["text"], _proposal_json(objs, args.mode, {o["label"]: o["id"] for o in p["objects"]}), viol),
                                        "images": p["images"], "max_new_tokens": args.max_new_tokens}))
            # ---- verify -> repair round(s) ----
            for _round in range(args.max_repairs):
                if not pending:
                    break
                resps = vlm.generate([q for _, q in pending])
                n_calls += len(pending)
                nxt = []
                for (j, q), resp in zip(pending, resps):
                    fr, p = payloads[j]
                    d = extract_json(resp)
                    if d is None:
                        continue                 # keep the previous answer
                    objs = parse_objects(d, p, args.mode)
                    viol = check_geometry(_objs_for_critic(objs), p["person_corners"], ctx.extent)
                    fo = frames_out[fr.file]
                    fo.update({"objects": objs, "raw_response": resp, "violations_post": viol,
                               "n_calls": fo["n_calls"] + 1})
                    if viol:
                        nxt.append((j, {"text": repair_prompt(p["text"], _proposal_json(objs, args.mode, {o["label"]: o["id"] for o in p["objects"]}), viol),
                                        "images": p["images"], "max_new_tokens": args.max_new_tokens}))
                pending = nxt
            rec = {"video_id": f"{vid}.mp4", "mode": args.mode, "model_name": args.model_name + tag, "track": "B",
                   "critic": not args.no_critic, "retrieval": bool(clips), "context_frames": args.context_frames,
                   "frames": frames_out, "n_frames": len(frames_out), "n_calls": n_calls,
                   "n_parsed": sum(f["parsed"] for f in frames_out.values()),
                   "violations_pre": sum(len(f["violations_pre"]) for f in frames_out.values()),
                   "violations_post": sum(len(f["violations_post"]) for f in frames_out.values())}
            tmp = out_path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(rec, f)
            os.replace(tmp, out_path)
            n_done += 1
            tot_calls += n_calls
            tot_frames += len(frames_out)
            tot_viol_pre += rec["violations_pre"]
            tot_viol_post += rec["violations_post"]
            logger.info(f"[{i + 1}/{len(ids)}] {vid}: {len(frames_out)} frames, {n_calls} calls, violations "
                        f"{rec['violations_pre']}->{rec['violations_post']}, cache hits={cache.hits} "
                        f"misses={cache.misses}, {time.time() - t0:.0f}s")
        except Exception as e:  # noqa
            n_fail += 1
            logger.exception(f"[{vid}] failed: {e!r}")
        if (i + 1) % 5 == 0 or i + 1 == len(ids):
            with open(status_path, "w") as f:
                json.dump({"state": "running" if i + 1 < len(ids) else "done", "done": n_done, "skipped": n_skip,
                           "failed": n_fail, "total": len(ids), "calls": tot_calls, "frames": tot_frames,
                           "violations_pre": tot_viol_pre, "violations_post": tot_viol_post,
                           "cache_hits": cache.hits, "cache_misses": cache.misses,
                           "seconds": round(time.time() - t0)}, f)
    logger.info(f"finished: done={n_done} skipped={n_skip} failed={n_fail} calls={tot_calls} "
                f"violations {tot_viol_pre}->{tot_viol_post} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
