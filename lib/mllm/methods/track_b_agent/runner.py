"""
Track B -- agentic localised WSGG: tool loop + geometric critic + repair (B6).
==============================================================================

    CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.track_b_agent.runner \
        --model_name qwen3vl_8b --mode sgdet --video_list <split> [--limit N]

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
(the first proposal, "-critic"); ``score_run --objects_key objects_pre`` scores
the latter through the usual adapter without re-running anything.

Prompts of ``--videos_per_batch`` videos are pooled into one vLLM call (proposal
round, then repair round) so small videos do not starve the batch.

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
from lib.mllm.methods.track_b_agent.critic import check_geometry                     # noqa: E402
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
    return "\n".join(lines)[:max_chars]


def with_context(text: str, retrieved: str) -> str:
    if not retrieved:
        return text
    block = ("Retrieved context from a scene graph built over the whole video (may be noisy):\n"
             f"{retrieved}\n\n")
    i = text.find("Task:")
    return text[:i] + block + text[i:] if i != -1 else block + text


def repair_prompt(base_text: str, proposal_json: str, violations: List[Dict[str, Any]]) -> str:
    v = "\n".join(f"- {x['msg']}" for x in violations[:12])
    return (base_text + "\n\nYour previous answer was:\n" + proposal_json +
            "\n\nA geometric checker found these problems with it:\n" + v +
            "\n\nFix them: keep every correct object and relationship, correct the box or the predicate that is "
            "inconsistent with the 3D layout (boxes must rest on or above the floor, contact predicates need the "
            "object next to the person, 'above'/'beneath' must match the heights). Answer with ONLY the corrected "
            "JSON in the same compact format.")


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


class Stats:
    def __init__(self):
        self.done = self.skip = self.fail = 0
        self.calls = self.frames = self.viol_pre = self.viol_post = 0


def process_group(group: List[Tuple[str, Any, list]], vlm: CachedVLM, args, out_dir: Path, cache: LLMCache,
                  st: Stats, t0: float) -> None:
    """group: [(vid, ctx, [(frame, payload), ...]), ...] -> proposal round, critic, repair round(s), save."""
    flat = [(gi, j, fr, p) for gi, (_, _, payloads) in enumerate(group) for j, (fr, p) in enumerate(payloads)]
    prompts = [{"text": p["text"], "images": p["images"], "max_new_tokens": args.max_new_tokens} for _, _, _, p in flat]
    responses = vlm.generate(prompts)
    n_calls = [len(payloads) for _, _, payloads in group]
    fouts: List[Dict[str, Any]] = [{} for _ in group]
    pending: List[Tuple[int, int, Dict[str, Any]]] = []
    for (gi, j, fr, p), resp in zip(flat, responses):
        ctx = group[gi][1]
        d = extract_json(resp)
        objs = parse_objects(d, p, args.mode)
        pc = p["person_corners"]
        viol = [] if args.no_critic else check_geometry(_objs_for_critic(objs), pc, ctx.extent)
        fouts[gi][fr.file] = {
            "objects_pre": objs, "raw_response_pre": resp, "violations_pre": viol,
            "objects": objs, "raw_response": resp, "violations_post": viol,
            "ids": {o["id"]: o["label"] for o in p["objects"]},
            "person_corners": None if pc is None else np.asarray(pc, np.float32).tolist(),
            "retrieved": p.get("retrieved", ""), "n_calls": 1, "parsed": d is not None,
        }
        if viol and d is not None:
            ids = {o["label"]: o["id"] for o in p["objects"]}
            pending.append((gi, j, {"text": repair_prompt(p["text"], _proposal_json(objs, args.mode, ids), viol),
                                    "images": p["images"], "max_new_tokens": args.max_new_tokens}))
    for _round in range(args.max_repairs):
        if not pending:
            break
        resps = vlm.generate([q for _, _, q in pending])
        nxt = []
        for (gi, j, q), resp in zip(pending, resps):
            n_calls[gi] += 1
            fr, p = group[gi][2][j]
            ctx = group[gi][1]
            d = extract_json(resp)
            if d is None:
                continue                      # keep the previous answer
            objs = parse_objects(d, p, args.mode)
            viol = check_geometry(_objs_for_critic(objs), p["person_corners"], ctx.extent)
            fo = fouts[gi][fr.file]
            fo.update({"objects": objs, "raw_response": resp, "violations_post": viol, "n_calls": fo["n_calls"] + 1})
            if viol:
                ids = {o["label"]: o["id"] for o in p["objects"]}
                nxt.append((gi, j, {"text": repair_prompt(p["text"], _proposal_json(objs, args.mode, ids), viol),
                                    "images": p["images"], "max_new_tokens": args.max_new_tokens}))
        pending = nxt
    for gi, (vid, ctx, payloads) in enumerate(group):
        fo = fouts[gi]
        rec = {"video_id": f"{vid}.mp4", "mode": args.mode, "model_name": args.model_name + args.tag_full,
               "track": "B", "critic": not args.no_critic, "retrieval": bool(ctx.clips),
               "context_frames": args.context_frames, "frames": fo, "n_frames": len(fo), "n_calls": n_calls[gi],
               "n_parsed": sum(f["parsed"] for f in fo.values()),
               "violations_pre": sum(len(f["violations_pre"]) for f in fo.values()),
               "violations_post": sum(len(f["violations_post"]) for f in fo.values())}
        out_path = out_dir / f"{vid}.mp4.pkl"
        tmp = out_path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(rec, f)
        os.replace(tmp, out_path)
        st.done += 1
        st.calls += n_calls[gi]
        st.frames += len(fo)
        st.viol_pre += rec["violations_pre"]
        st.viol_post += rec["violations_post"]
        logger.info(f"{vid}: {len(fo)} frames, {n_calls[gi]} calls, violations "
                    f"{rec['violations_pre']}->{rec['violations_post']}, cache hits={cache.hits} "
                    f"misses={cache.misses}, {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="qwen3vl_8b")
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--video_list", default=None)
    ap.add_argument("--video_id", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--context_frames", type=int, default=2)
    ap.add_argument("--videos_per_batch", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--max_model_len", type=int, default=24576)
    ap.add_argument("--temperature", type=float, default=0.2)
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
    args.tag_full = args.tag + ("_nocritic" if args.no_critic else "") + ("_noretr" if args.no_retrieval else "")
    out_dir = Path(get_path(cfg, "outputs.track_b")) / args.mode / (args.model_name + args.tag_full)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.status or str(Path(get_path(cfg, "outputs.logs") or "/data3/rohith/ag/logs") /
                                     f"track_b_{args.mode}_{args.model_name}{args.tag_full}.status.json")
    graph_dir = get_path(cfg, "graphs")
    cache = LLMCache(get_path(cfg, "worldbbox.caches.llm_cache"))
    vlm = None
    if not args.dry_run:
        model = load_model(cfg, args.model_name, args.tensor_parallel_size, args.max_model_len,
                           args.temperature, args.top_p, args.seed, max_images=args.context_frames + 2)
        vlm = CachedVLM(model, args.model_name, cache, temperature=args.temperature, strip_thinking=False)
    t0 = time.time()
    st = Stats()

    def write_status(state):
        with open(status_path, "w") as f:
            json.dump({"state": state, "done": st.done, "skipped": st.skip, "failed": st.fail, "total": len(ids),
                       "calls": st.calls, "frames": st.frames, "violations_pre": st.viol_pre,
                       "violations_post": st.viol_post, "cache_hits": cache.hits, "cache_misses": cache.misses,
                       "seconds": round(time.time() - t0)}, f)

    group: List[Tuple[str, Any, list]] = []
    for i, vid in enumerate(ids):
        out_path = out_dir / f"{vid}.mp4.pkl"
        if out_path.exists() and not args.dry_run:
            st.skip += 1
            continue
        try:
            video = ts.load(vid)
            ctx = TrackAContext(video, cfg, args.mode)
            ctx.clips = [] if args.no_retrieval else load_stage1(vid, graph_dir)
            payloads = []
            for fr in video.frames:
                p = build_payload(ctx, fr, args.context_frames)
                p["retrieved"] = query_graph(ctx.clips, fr.frame_num) if ctx.clips else ""
                p["text"] = with_context(p["text"], p["retrieved"])
                payloads.append((fr, p))
            ctx.save()
            ctx._img_cache.clear()
            if args.dry_run:
                d = Path("/data3/rohith/ag/logs/track_b_dry") / args.mode / vid
                d.mkdir(parents=True, exist_ok=True)
                for fr, p in payloads[:2]:
                    (d / f"{fr.file[:-4]}_prompt.txt").write_text(p["text"], encoding="utf-8")
                logger.info(f"[{vid}] dry run: {len(payloads)} frames, retrieval={'yes' if ctx.clips else 'no'} -> {d}")
                st.done += 1
                continue
            group.append((vid, ctx, payloads))
            if len(group) >= args.videos_per_batch:
                logger.info(f"[{i + 1}/{len(ids)}] proposal round for {len(group)} videos "
                            f"({sum(len(p) for _, _, p in group)} prompts)")
                process_group(group, vlm, args, out_dir, cache, st, t0)
                group = []
        except Exception as e:  # noqa
            st.fail += 1
            logger.exception(f"[{vid}] failed: {e!r}")
            group = []
        if (i + 1) % 5 == 0 or i + 1 == len(ids):
            write_status("running")
    if group:
        try:
            process_group(group, vlm, args, out_dir, cache, st, t0)
        except Exception as e:  # noqa
            st.fail += len(group)
            logger.exception(f"final group failed: {e!r}")
    write_status("done")
    logger.info(f"finished: done={st.done} skipped={st.skip} failed={st.fail} calls={st.calls} "
                f"violations {st.viol_pre}->{st.viol_post} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
