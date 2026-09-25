"""
Backbone-controlled "previous-frame graph" ablation (SceneGraphVLM idea, our backbone).
====================================================================================

Question (docs/EXTERNAL_BASELINES_PLAN.md, Phase 3 item 4): does conditioning each
frame on the preceding frame's predicted graph help, with the backbone held fixed?
SceneGraphVLM changes backbone, training and prompt at once; this isolates the prompt.

Two arms, both = the vendored ``zero_shot`` unlocalized per-object prompt on
Qwen3-VL-8B, predcls, ``--skip-verification``, temperature 0.2 (same settings as the
``rag_all`` std150 cell):

  * ``noctx``     the plain zero_shot prompt (the arm-1 baseline on Qwen3-VL-8B);
  * ``prevgraph`` the same prompt plus a block with the previous annotated frame's
                  predicted graph, serialized as one line per object, capped at a fixed
                  character budget (``--budget_chars``, default 600 ~ 150 tokens; lines
                  are dropped from the end once the cap is reached, the first frame gets
                  "(no previous frame)").  Every prompt of a frame gets the same block.

The previous graph is **the noctx arm's own prediction for frame t-1** (two-pass), not
an autoregressive chain of the prevgraph arm: the vendored runner batches every frame
of a video in one call, and a true chain would need per-frame sequential calls
(~30x the wall time).  The block therefore carries exactly what SceneGraphVLM's
``--prev-source model`` carries at t (a model graph of t-1), minus error compounding
through the chain.  Stated as a deviation in the results note.

    CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.prev_graph.runner --arm noctx \
        --video_list /data3/rohith/ag/splits/test_worldbbox_thinking150.txt
    CUDA_VISIBLE_DEVICES=0 python -m lib.mllm.methods.prev_graph.runner --arm prevgraph \
        --video_list /data3/rohith/ag/splits/test_worldbbox_thinking150.txt

Outputs ``/data3/rohith/ag/runs/mllm/prev_graph/<arm>/predcls/<model>/<vid>.pkl`` (the
vendored baseline PKL format; score with ``score_run --method prev_graph --pred_dir
/data3/rohith/ag/runs/mllm/prev_graph/<arm>/``).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from lib.mllm.core.config_loader import get_path, load_config                # noqa: E402
from lib.mllm.core.logger_utils import setup_logging                         # noqa: E402
from lib.mllm.methods.zero_shot import runner as zs                          # noqa: E402

logger = logging.getLogger(__name__)
ROOT = "/data3/rohith/ag/runs/mllm/prev_graph/"

PREV_BLOCK = (
    "\nFor temporal context, this was the predicted scene graph at the PREVIOUS annotated "
    "frame of this video (one line per object: attention; contacting; spatial). The scene may "
    "have changed since then; answer for the target frame.\n{block}\n"
)


def _labels(v) -> List[str]:
    items = v if isinstance(v, list) else [v]
    out = []
    for it in items:
        lab = it.get("label") if isinstance(it, dict) else it
        if lab and str(lab).lower() != "unknown":
            out.append(str(lab))
    return out


def serialize_frame(preds: List[Dict[str, Any]], budget_chars: int) -> str:
    lines, used = [], 0
    for p in preds:
        ln = (f"- {p.get('object')}: {','.join(_labels(p.get('attention'))) or '-'}; "
              f"{','.join(_labels(p.get('contacting'))) or '-'}; {','.join(_labels(p.get('spatial'))) or '-'}")
        if used + len(ln) + 1 > budget_chars:
            break
        lines.append(ln)
        used += len(ln) + 1
    return "\n".join(lines) if lines else "(no relations predicted)"


class PrevGraphProcessor(zs.ActionGenomeZeroShotProcessor):
    arm: str = "noctx"
    prev_root: Optional[Path] = None
    budget_chars: int = 600
    _blocks: List[str] = []
    _block_lens: List[int] = []

    def generate_relationship_queries(self, objects):     # instance method on purpose
        qs = zs.ActionGenomeZeroShotProcessor.generate_relationship_queries(objects)
        if self.arm != "prevgraph":
            return qs
        block = self._blocks.pop(0) if self._blocks else "(no previous frame)"
        self._block_lens.append(len(block))
        for q in qs:   # insert the block right before the three questions
            q["prompt"] = q["prompt"].replace("\nYou must answer three questions:",
                                              PREV_BLOCK.format(block=block) + "\nYou must answer three questions:", 1)
        return qs

    def process_video(self, video_id: str):
        self._blocks = []
        if self.arm == "prevgraph":
            vid = video_id if video_id.endswith(".mp4") else f"{video_id}.mp4"
            src = self.prev_root / "predcls" / self.args.model_name / f"{Path(vid).stem}.pkl"
            if not src.exists():
                src = self.prev_root / "predcls" / self.args.model_name / f"{vid}.pkl"
            if not src.exists():
                logger.warning(f"[{video_id}] no noctx predictions at {src}; skipping")
                return
            prev = pickle.load(open(src, "rb"))
            try:
                stems = sorted(self.ag_data.get_final_data_lite(vid).get("bbox_frames", {}).keys())
            except (FileNotFoundError, ValueError):
                stems = []
            frames = prev.get("frames", {})
            self._blocks = ["(no previous frame)"] + [
                serialize_frame((frames.get(s) or {}).get("predictions", []), self.budget_chars)
                for s in stems[:-1]]
        super().process_video(video_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["noctx", "prevgraph"])
    ap.add_argument("--model_name", default="qwen3vl_8b")
    ap.add_argument("--video_list", required=True)
    ap.add_argument("--budget_chars", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--root", default=ROOT)
    args = ap.parse_args()
    cfg = load_config()
    out = os.path.join(args.root, args.arm)
    setup_logging(out, f"prev_graph_{args.arm}_{args.model_name}.log")
    p = PrevGraphProcessor(
        ag_root_directory=get_path(cfg, "ag_root"), output_dir=out, graph_dir=get_path(cfg, "graphs"),
        model_name=args.model_name, split="test", tensor_parallel_size=1, use_vllm=True, mode="predcls",
        model_weights_dir=get_path(cfg, "model_weights") or None)
    p.skip_verification = True
    p.arm = args.arm
    p.prev_root = Path(args.root) / "noctx"
    p.budget_chars = args.budget_chars
    p._block_lens = []
    p.args.temperature = args.temperature
    p.run(limit=args.limit, video_list=args.video_list)
    if p._block_lens:
        import numpy as np
        L = np.array(p._block_lens)
        stats = {"n_blocks": int(len(L)), "mean_chars": float(L.mean()), "max_chars": int(L.max()),
                 "budget_chars": args.budget_chars}
        json.dump(stats, open(os.path.join(out, f"block_stats_{args.model_name}.json"), "w"), indent=1)
        logger.info(f"prev-graph block stats: {stats}")


if __name__ == "__main__":
    main()
