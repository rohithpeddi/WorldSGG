#!/usr/bin/env python3
"""
rag_sgr3: retrieval arms inside WorldRAG (``rag_all``) at a fixed context budget
===============================================================================

Phase 4 of ``docs/EXTERNAL_BASELINES_PLAN.md``; design and deviations in
``setup/EXT_SGR3_RETRIEVAL.md``.  Everything except the retrieved context is
the ``rag_all`` predcls pipeline unchanged: the same per-object prompt, the same
visual input (target frame + annotated context frames), the same object
inventory, the same ``max_new_tokens`` and sampling, ``--skip-verification``.

Arms (``--arm``)::

    none     R0  no retrieved context (zero-shot inside the rag_all prompt)
    bge      R1  ours: BGE-large text retrieval over the video's own Stage-1
                 graph, top-1 node (``rag_all`` Steps 1-2), truncated to the budget
    sgr3     R2  SGR3-style visual retrieval of *train-split* reference scene
                 graphs (``lib/mllm/tools/sgr3_index.py``), top-``--k`` scenes,
                 filled up to the budget
    hybrid   R3  R1 (first half of the budget) + R2 (the rest)

``--budget`` is the retrieved-context token cap (Qwen3-VL tokenizer, header
included), the same for every arm.  ``--context_only`` builds and logs the
contexts without the answer call (used to measure R1's context length).

Per-prompt contexts and their token counts are written next to the run:
``<out_root>/ctx/<model+tag>/<video>.json``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from lib.mllm.core.config_loader import get_inference_defaults, get_path, load_config, resolve_model_path
from lib.mllm.core.logger_utils import setup_logging
from lib.mllm.methods.rag_all.runner import ActionGenomeRAGAllObjectsProcessor

logger = logging.getLogger(__name__)

SGR3_CACHE = "/data3/rohith/ag/cache/sgr3"
R1_HEADER = "Relevant scene context from video analysis:\n"
R2_HEADER = "Reference scene graphs from visually similar videos ((unseen) = object not visible):\n"
_NORM = {"closet/cabinet": "closet", "cup/glass/bottle": "cup", "paper/notebook": "paper",
         "sofa/couch": "sofa", "phone/camera": "phone"}


def _norm(label: str) -> str:
    s = str(label).lower().strip()
    return _NORM.get(s, s)


class _ContextOnly(Exception):
    pass


class SGR3RAGProcessor(ActionGenomeRAGAllObjectsProcessor):
    """``rag_all`` with a pluggable, budget-matched retrieved context."""

    arm: str = "bge"
    budget: int = 0
    k_scenes: int = 1
    frames_per_scene: int = 3
    context_only: bool = False
    random_scenes: bool = False
    sgr3_dir: str = SGR3_CACHE
    retrieval_suffix: str = ""
    ctx_root: Optional[Path] = None

    # ------------------------------------------------------------------ setup
    def setup_arm(self):
        from transformers import AutoTokenizer
        cfg = load_config()
        self.tok = AutoTokenizer.from_pretrained(resolve_model_path(cfg, self.args.model_name))
        self.bank = None
        if self.arm in ("sgr3", "hybrid"):
            with open(os.path.join(self.sgr3_dir, "bank_graphs.pkl"), "rb") as f:
                self.bank = pickle.load(f)
            leak = json.load(open(os.path.join(self.sgr3_dir, "leakage.json")))
            assert leak.get("assert") == "passed", "SGR3 bank failed the leakage check"
            logger.info(f"SGR3 bank: {len(self.bank)} train videos (leakage check {leak['assert']})")
        self._ret_cache: Dict[str, Any] = {}
        logger.info(f"arm={self.arm} budget={self.budget} k={self.k_scenes} "
                    f"frames/scene={self.frames_per_scene} context_only={self.context_only}")

    def ntok(self, s: str) -> int:
        return len(self.tok(s, add_special_tokens=False)["input_ids"]) if s else 0

    def truncate(self, s: str, n: int) -> str:
        if not s or n <= 0:
            return ""
        ids = self.tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) <= n:
            return s
        n_body = n - self.ntok("\n\n")
        out = self.tok.decode(ids[:max(n_body, 0)]).rstrip()
        while out and self.ntok(out + "\n\n") > n:    # decode/encode round-trip can grow
            out = out[:-1].rstrip()
        return out + "\n\n" if out else ""

    def load_precomputed_graphs(self, video_id: str):
        """Parent loader, minus the clip intervals: the per-annotation clips only
        feed Step-3 node checks and verification, both unused here (skipping
        them saves ~45 s of frame I/O per video and changes no prompt)."""
        video_graph, entity_graph, captions, _ = super().load_precomputed_graphs(video_id)
        return video_graph, entity_graph, captions, []

    # ------------------------------------------------------------- R1 (BGE)
    def r1_contexts(self, objects: List[str], prompts: List[str], video_graph, entity_graph,
                    captions, video_inputs, embedding_cache) -> Dict[str, str]:
        """``rag_all`` Steps 1-2 + the top-1 node context, exactly as in the parent.
        (Step 3 node checks only choose verification clips, unused under
        --skip-verification, so they are skipped.)"""
        kw = self.vgent.batch_extract_keywords(prompts)
        out: Dict[str, str] = {}
        for obj, prompt, (query_list, llm_info) in zip(objects, prompts, kw):
            retrieved = self.vgent.retrieve_nodes_with_cache(
                prompt, query_list, video_inputs, candidates=[], video_graph=video_graph,
                entity_graph=entity_graph, captions=captions, llm_info=llm_info,
                embedding_cache=embedding_cache,
            )
            nodes = retrieved.get("nodes", [])
            ctx = ""
            if nodes and video_graph is not None and nodes[0] in video_graph.nodes:
                nd = video_graph.nodes[nodes[0]]
                parts = nd.get("entities", []) + nd.get("actions", []) + nd.get("scenes", [])
                if parts:
                    ctx = R1_HEADER + "; ".join(parts) + "\n\n"
            out[obj] = ctx
        return out

    # ------------------------------------------------------------ R2 (SGR3)
    def _retrieval(self, video_id: str) -> Dict[str, Any]:
        stem = Path(video_id).stem
        if stem not in self._ret_cache:
            p = os.path.join(self.sgr3_dir, f"retrieval{self.retrieval_suffix}", f"{stem}.json")
            if not os.path.exists(p):
                logger.warning(f"SGR3 retrieval missing for {stem}: {p}")
                self._ret_cache[stem] = {}
            else:
                self._ret_cache[stem] = json.load(open(p)).get("frames", {})
        return self._ret_cache[stem]

    def _scene_units(self, scene: Dict[str, Any], obj: str) -> List[Tuple[str, bool, str]]:
        """Merged (deduplicated) edges of one reference scene, query object first.

        Primary block: the scene's ``frames_per_scene`` top-ranked retrieved frames
        (SGR3's merged E_ref).  Padding block (only reached when the primary
        block is shorter than the budget): the scene's remaining retrieved
        frames, then its other annotated frames, nearest in time to the top
        frame first.  Both blocks stay inside the same reference video."""
        frames = self.bank.get(scene["video"], {})
        ranked = [fk for fk, _ in scene["frames"]]
        top = ranked[: self.frames_per_scene]
        anchor = ranked[0] if ranked else None

        def _idx(fk):
            m = re.search(r"(\d+)\.png$", fk)
            return int(m.group(1)) if m else 0

        rest = ranked[self.frames_per_scene:]
        others = [fk for fk in frames if fk not in set(ranked)]
        if anchor is not None:
            others.sort(key=lambda fk: abs(_idx(fk) - _idx(anchor)))
        q = _norm(obj)
        seen = set()
        units: List[Tuple[str, bool, str]] = []
        for block in (top, rest + others):
            merged: Dict[str, Dict[str, Any]] = {}
            order: List[str] = []
            for fk in block:
                for o in frames.get(fk, []):
                    lab = o["label"]
                    if lab not in merged:
                        merged[lab] = {"visible": o["visible"], "rels": []}
                        order.append(lab)
                    merged[lab]["visible"] = merged[lab]["visible"] or o["visible"]
                    for r in o["rels"]:
                        if r not in merged[lab]["rels"]:
                            merged[lab]["rels"].append(r)
            order.sort(key=lambda lab: 0 if _norm(lab) == q else 1)      # stable
            for lab in order:
                for r in merged[lab]["rels"]:
                    if (lab, r) not in seen:
                        seen.add((lab, r))
                        units.append((lab, merged[lab]["visible"], r))
        return units

    @staticmethod
    def _serialize(header: str, picked: List[List[Tuple[str, bool, str]]]) -> str:
        lines = []
        for si, units in enumerate(picked):
            if not units:
                continue
            by_obj: Dict[str, List[str]] = {}
            vis: Dict[str, bool] = {}
            for lab, v, r in units:
                by_obj.setdefault(lab, []).append(r)
                vis[lab] = vis.get(lab, False) or v
            body = "; ".join(f"{lab}{'' if vis[lab] else ' (unseen)'}: {', '.join(rs)}"
                             for lab, rs in by_obj.items())
            lines.append(f"Reference {si + 1}: person -> {body}")
        return (header + "\n".join(lines) + "\n\n") if lines else ""

    def r2_context(self, video_id: str, frame_file: str, obj: str, budget: int) -> Tuple[str, int]:
        """Fill the budget round-robin over the top-k retrieved scenes, edge by edge."""
        if budget <= 0:
            return "", 0
        ent = self._retrieval(video_id).get(os.path.basename(frame_file))
        if not ent:
            return "", 0
        scenes = ent.get("scenes", [])[: self.k_scenes]
        if self.random_scenes:
            # control: k random bank videos per (video, frame), same serialization and budget
            import random
            rng = random.Random(f"{Path(video_id).stem}/{os.path.basename(frame_file)}")
            if not hasattr(self, "_bank_ids"):
                self._bank_ids = sorted(self.bank)
            scenes = [{"video": v, "frames": [[fk, 0.0] for fk in sorted(self.bank[v])[:5]]}
                      for v in rng.sample(self._bank_ids, self.k_scenes)]
        units = [self._scene_units(s, obj) for s in scenes]
        picked: List[List[Tuple[str, bool, str]]] = [[] for _ in units]
        ptr = [0] * len(units)
        active = [bool(u) for u in units]
        best = ""
        while any(active):
            for si in range(len(units)):
                if not active[si]:
                    continue
                if ptr[si] >= len(units[si]):
                    active[si] = False
                    continue
                picked[si].append(units[si][ptr[si]])
                cand = self._serialize(R2_HEADER, picked)
                if self.ntok(cand) > budget:
                    picked[si].pop()
                    active[si] = False
                else:
                    best = cand
                    ptr[si] += 1
        return best, sum(1 for p in picked if p)

    # ------------------------------------------------------- the answer step
    def _batch_all_video_queries(self, all_entries, video_inputs, captions, video_graph, entity_graph,
                                 embedding_cache, frame_to_clip=None, video_id=None, query_ctx_map=None):
        if not all_entries:
            return [], [], []
        n = len(all_entries)
        if video_inputs is None or video_inputs[0] is None:
            return [None] * n, [None] * n, [""] * n
        B = self.budget

        r1: Dict[str, str] = {}
        if self.arm in ("bge", "hybrid"):
            objs, prompts = [], []
            for _, q, _ in all_entries:
                if q["object"] not in objs:
                    objs.append(q["object"])
                    prompts.append(q["prompt"])
            r1 = self.r1_contexts(objs, prompts, video_graph, entity_graph, captions,
                                  video_inputs, embedding_cache)

        contexts: List[str] = []
        log: List[Dict[str, Any]] = []
        cache: Dict[Tuple[str, str], Tuple[str, Dict[str, Any]]] = {}
        for frame_stem, q, _ in all_entries:
            key = (frame_stem, q["object"])
            if key not in cache:
                obj = q["object"]
                info: Dict[str, Any] = {}
                if self.arm == "none":
                    ctx = ""
                elif self.arm == "bge":
                    raw = r1.get(obj, "")
                    info["r1_raw_tokens"] = self.ntok(raw)
                    ctx = self.truncate(raw, B) if B > 0 else raw
                elif self.arm == "sgr3":
                    ctx, info["scenes_used"] = self.r2_context(video_id, frame_stem, obj, B)
                elif self.arm == "hybrid":
                    raw = r1.get(obj, "")
                    info["r1_raw_tokens"] = self.ntok(raw)
                    part1 = self.truncate(raw, B // 2) if raw else ""
                    rest = B - self.ntok(part1)
                    part2, info["scenes_used"] = self.r2_context(video_id, frame_stem, obj, rest)
                    ctx = part1 + part2
                else:
                    raise ValueError(self.arm)
                info["ctx_tokens"] = self.ntok(ctx)
                cache[key] = (ctx, info)
            ctx, info = cache[key]
            contexts.append(ctx)
            log.append({"frame": frame_stem, "object": q["object"], **info, "context": ctx})

        stem = Path(video_id or "unknown").stem
        if self.ctx_root is not None:
            self.ctx_root.mkdir(parents=True, exist_ok=True)
            with open(self.ctx_root / f"{stem}.json", "w") as f:
                json.dump({"video": stem, "arm": self.arm, "budget": B, "k": self.k_scenes,
                           "entries": log}, f)
        toks = [e["ctx_tokens"] for e in log]
        logger.info(f"[{stem}] contexts: {n} prompts, tokens mean={sum(toks) / max(n, 1):.1f} "
                    f"max={max(toks) if toks else 0} empty={sum(1 for t in toks if t == 0)}")
        if self.context_only:
            raise _ContextOnly()

        batch_prompts = []
        for idx, (frame_stem, q, _) in enumerate(all_entries):
            qv = query_ctx_map[frame_stem] if query_ctx_map and frame_stem in query_ctx_map else video_inputs[0]
            ctx = contexts[idx]
            batch_prompts.append({
                "text": ctx + q["prompt"] if ctx else q["prompt"],
                "video_inputs": [qv],
                "max_new_tokens": getattr(self, "gen_max_tokens", None) or 128,
            })
        self._log_prompts(video_id or "unknown", batch_prompts, tag=f"rag_sgr3_{self.arm}")
        logger.info(f"  [answer] {len(batch_prompts)} prompts (arm={self.arm}) …")
        clip_tensors = [video_inputs[0]] * n      # verification is skipped for every arm
        try:
            responses = self._chunked_batch_response(batch_prompts)
            return [r.strip() if r else None for r in responses], clip_tensors, contexts
        except Exception as e:
            logger.error(f"Batched final-answer error: {e}")
            return [None] * n, clip_tensors, contexts

    def process_video(self, video_id: str):
        try:
            return super().process_video(video_id)
        except _ContextOnly:
            logger.info(f"[{video_id}] context-only: contexts written, answer skipped")


def main():
    cfg = load_config()
    inf = get_inference_defaults(cfg)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default=inf.get("default_model", "qwen3vl_8b"))
    ap.add_argument("--arm", required=True, choices=["none", "bge", "sgr3", "hybrid"])
    ap.add_argument("--budget", type=int, default=0, help="retrieved-context token cap (0 = uncapped)")
    ap.add_argument("--k", type=int, default=1, help="SGR3 reference scenes (R2-k)")
    ap.add_argument("--frames_per_scene", type=int, default=3, help="top frames merged per reference scene")
    ap.add_argument("--sgr3_dir", default=SGR3_CACHE)
    ap.add_argument("--retrieval_suffix", default="")
    ap.add_argument("--out_root", default="/data3/rohith/ag/runs/mllm/rag_sgr3")
    ap.add_argument("--context_only", action="store_true")
    ap.add_argument("--random_scenes", action="store_true",
                    help="control for R2: k random train videos instead of the retrieved ones")
    ap.add_argument("--mode", default="predcls", choices=["predcls", "sgdet"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--video_list", default=None)
    ap.add_argument("--video_id", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--tag", required=True, help="output dir suffix, e.g. _R2k1")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    setup_logging(str(out_root), f"rag_sgr3_{args.mode}_{args.model_name}{args.tag}.log")

    p = SGR3RAGProcessor(
        ag_root_directory=get_path(cfg, "ag_root"),
        output_dir=str(out_root),
        graph_dir=get_path(cfg, "graphs"),
        model_name=args.model_name,
        split=args.split,
        tensor_parallel_size=args.tensor_parallel_size,
        use_vllm=True,
        mode=args.mode,
        model_weights_dir=get_path(cfg, "model_weights") or None,
    )
    p.skip_verification = True
    p.print_prompts = False
    p.gen_max_tokens = args.max_new_tokens
    p.run_tag = args.tag
    p.args.temperature = args.temperature
    p.args.top_p = args.top_p
    p.arm, p.budget, p.k_scenes = args.arm, args.budget, args.k
    p.frames_per_scene, p.sgr3_dir, p.retrieval_suffix = args.frames_per_scene, args.sgr3_dir, args.retrieval_suffix
    p.context_only = args.context_only
    p.random_scenes = args.random_scenes
    p.ctx_root = out_root / "ctx" / (args.model_name + args.tag)
    p.setup_arm()
    p.run(limit=args.limit, video_id=args.video_id, video_list=args.video_list)


if __name__ == "__main__":
    main()
