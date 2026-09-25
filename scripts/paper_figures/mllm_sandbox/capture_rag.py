"""Run ``lib.mllm.methods.rag_all.runner`` unchanged while recording what its pickles do not keep.

Recorded per video (``<capture_dir>/rag_all_<mode>/<video>.json`` + ``_tensors.npz``):
  * P2 keyword extraction: the reasoning prompt, raw response, parsed ``llm_info`` and keyword list;
  * retrieval per object query: the final query list (keywords + the P4 question, as the runner
    appends it), the mean BGE cosine of every entity key and every graph node (recomputed from the
    runner's own embedding cache, so identical to what it thresholded at 0.5 and ranked), and the
    top-20 nodes it returned;
  * the context block prepended to each (frame, object) prompt, the P4 prompt per object, the raw
    responses, the merged graph (node texts, source key frame, entity index) and the captions;
  * the visual inputs: the whole-video tensor V and the Q(f) tensors (target frame + key-frame context).

Only wrappers are installed; every call goes to the original implementation first.

usage: python capture_rag.py <capture_dir> <rag_all runner args ...>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

CAP_DIR = Path(sys.argv[1])
sys.argv = [sys.argv[0]] + sys.argv[2:]
REPO = os.path.expanduser("~/CODE/Scene4Cast_mllm")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402

import lib.mllm.core.vgent as VG  # noqa: E402
import lib.mllm.methods.rag_all.runner as R  # noqa: E402
from lib.mllm.core.retrieval import precompute_embeddings  # noqa: E402

STATE = {}


def _reset(video=None):
    STATE.clear()
    STATE.update(video=video, kw=[], ret=[], batch=None, tensors=None)


def _u8(t):
    """(T, C, H, W) float in [0, 255] -> (T, H, W, C) uint8."""
    return t.detach().float().clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def _jsonable(v):
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


# --- P2: same body as Vgent.batch_extract_keywords, plus the raw responses -------------------
def batch_extract_keywords(self, prompts, candidates_list=None):
    if candidates_list is None:
        candidates_list = [[] for _ in prompts]
    reason_prompts = [
        {"text": VG.REASONING_PROMPT.format(query=p, candidates=c), "max_new_tokens": 256}
        for p, c in zip(prompts, candidates_list)
    ]
    responses = self.model.mllm_batch_response(reason_prompts)
    results = []
    for p, rp, resp, cands in zip(prompts, reason_prompts, responses, candidates_list):
        llm_info = None
        try:
            llm_info = json.loads(resp.replace("```json", "").replace("```", "").strip())
        except (json.JSONDecodeError, TypeError):
            pass
        qlist = llm_info["keywords"] if llm_info is not None and "keywords" in llm_info else []
        qlist = list(set(qlist + cands))
        results.append((qlist, llm_info))
        STATE["kw"].append({"prompt": p, "reason_prompt": rp["text"], "raw_response": resp,
                            "llm_info": _jsonable(llm_info), "keywords": list(qlist)})
    return results


VG.Vgent.batch_extract_keywords = batch_extract_keywords

# --- retrieval: original call, then the scores it thresholded / ranked ------------------------
_orig_retrieve = VG.Vgent.retrieve_nodes_with_cache


def retrieve_nodes_with_cache(self, question, query_list, video_inputs, candidates, video_graph,
                              entity_graph, captions, llm_info, embedding_cache=None):
    out = _orig_retrieve(self, question, query_list, video_inputs, candidates, video_graph,
                         entity_graph, captions, llm_info, embedding_cache=embedding_cache)
    rec = {"question": question, "query_list": list(query_list), "nodes": [int(n) for n in out.get("nodes", [])]}
    try:
        ec = embedding_cache or {}
        with torch.no_grad():
            qe = precompute_embeddings(list(query_list), self.embedding_model, self.embedding_tokenizer)
            if qe is not None and ec.get("entity_key_embeddings") is not None:
                s = torch.mean(qe @ ec["entity_key_embeddings"].T, dim=0)
                rec["entity_keys"] = list(ec["entity_keys"])
                rec["entity_sims"] = s.float().cpu().tolist()
            if qe is not None and ec.get("node_content_embeddings") is not None:
                s = torch.mean(qe @ ec["node_content_embeddings"].T, dim=0)
                rec["node_ids"] = [int(n) for n in ec["node_ids"]]
                rec["node_sims"] = s.float().cpu().tolist()
    except Exception as e:  # noqa: BLE001 - the capture must never break the run
        rec["score_error"] = repr(e)
    STATE["ret"].append(rec)
    return out


VG.Vgent.retrieve_nodes_with_cache = retrieve_nodes_with_cache

# --- the per-video batch: contexts, prompts, responses, graph, visual tensors ----------------
_orig_batch = R.ActionGenomeRAGAllObjectsProcessor._batch_all_video_queries


def _batch_all_video_queries(self, all_entries, video_inputs, captions, video_graph, entity_graph,
                             embedding_cache, frame_to_clip=None, video_id=None, query_ctx_map=None):
    out = _orig_batch(self, all_entries, video_inputs, captions, video_graph, entity_graph, embedding_cache,
                      frame_to_clip=frame_to_clip, video_id=video_id, query_ctx_map=query_ctx_map)
    try:
        raw, _clips, node_ctx = out
        nodes = []
        for n, d in video_graph.nodes(data=True):
            nodes.append({"id": int(n), "source_annotated_frame": d.get("source_annotated_frame"),
                          "entities": list(d.get("entities") or []), "actions": list(d.get("actions") or []),
                          "scenes": list(d.get("scenes") or []), "captions": d.get("captions")})
        STATE["batch"] = {
            "entries": [[fs, q["object"], int(fi)] for fs, q, fi in all_entries],
            "prompts_by_object": {q["object"]: q["prompt"] for _, q, _ in all_entries},
            "node_contexts": list(node_ctx),
            "raw_responses": list(raw),
            "captions": [[int(a), str(t)] for a, t in (captions or [])],
            "graph_nodes": _jsonable(nodes),
            "graph_edges": [[int(u), int(v)] for u, v in video_graph.edges()],
            "entity_graph": {k: sorted(int(x) for x in v) for k, v in entity_graph.items()},
            "clip_frames": sorted(int(k) for k in (frame_to_clip or {})),
            "video_v_shape": list(video_inputs[0].shape),
            "max_new_tokens": getattr(self, "gen_max_tokens", None) or 128,
        }
        tens = {"video_v": _u8(video_inputs[0])}
        if query_ctx_map:
            stems = sorted(query_ctx_map)
            tens["q_context"] = _u8(query_ctx_map[stems[0]][1:])
            tens["q_targets"] = np.stack([_u8(query_ctx_map[s][:1])[0] for s in stems])
            STATE["batch"]["q_stems"] = stems
            STATE["batch"]["q_shape"] = list(query_ctx_map[stems[0]].shape)
        STATE["tensors"] = tens
    except Exception as e:  # noqa: BLE001
        STATE["batch_error"] = repr(e)
    return out


R.ActionGenomeRAGAllObjectsProcessor._batch_all_video_queries = _batch_all_video_queries

# --- per video: reset, run, dump ------------------------------------------------------------
_orig_process = R.ActionGenomeRAGAllObjectsProcessor.process_video


def process_video(self, video_id):
    _reset(video_id)
    try:
        return _orig_process(self, video_id)
    finally:
        d = CAP_DIR / f"rag_all_{self.mode}"
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(video_id).stem
        rec = {"video_id": video_id, "mode": self.mode, "model_name": self.args.model_name,
               "keywords": STATE["kw"], "retrieval": STATE["ret"], "batch": STATE["batch"],
               "batch_error": STATE.get("batch_error")}
        with open(d / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(rec), f)
        if STATE.get("tensors"):
            np.savez_compressed(d / f"{stem}_tensors.npz", **STATE["tensors"])
        print(f"[capture] {stem} {self.mode}: {len(STATE['kw'])} keyword calls, "
              f"{len(STATE['ret'])} retrievals -> {d}", flush=True)


R.ActionGenomeRAGAllObjectsProcessor.process_video = process_video

if __name__ == "__main__":
    _reset()
    R.main()
