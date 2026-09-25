"""Run the ``zero_shot`` or ``caption_all`` runner unchanged while recording what its pickles drop.

Recorded per video (``<capture_dir>/<method>_<mode>/<video>.json`` + ``_tensors.npz``):
  * the exact P4 text of every (frame, object) prompt of the one batched call (for ``caption_all``
    this includes the per-frame caption prefix, sorted by distance to the target frame), in the
    runner's own order, mapped back to (frame, object);
  * the visual inputs: the whole-video tensor V (SGDet discovery) and the query tensors Q(f)
    (target frame + the shared annotated context), as uint8 arrays;
  * SGDet object discovery: the P6 estimation prompt and raw response, the parsed candidate set,
    and the P5 Yes/No prompts (their scores are in the pickle's ``estimation_meta``);
  * ``caption_all``: the caption list and clip intervals read from the Stage-1 pickle.

Only wrappers are installed; every call goes to the original implementation first.

usage: python capture_ctx.py <capture_dir> <zero_shot|caption_all> <runner args ...>
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

CAP_DIR = Path(sys.argv[1])
METHOD = sys.argv[2]
sys.argv = [sys.argv[0]] + sys.argv[3:]
REPO = os.path.expanduser("~/CODE/Scene4Cast_mllm")
sys.path.insert(0, REPO)
os.chdir(REPO)

import torch  # noqa: E402

import lib.mllm.base_processor as B  # noqa: E402

if METHOD == "zero_shot":
    import lib.mllm.methods.zero_shot.runner as R
    PROC = R.ActionGenomeZeroShotProcessor
elif METHOD == "caption_all":
    import lib.mllm.methods.caption_all.runner as R
    PROC = R.ActionGenomeCaptionAllObjectsProcessor
else:
    raise SystemExit(f"unknown method {METHOD!r}")

STATE = {}


def _reset(video=None):
    STATE.clear()
    STATE.update(video=video, video_v=None, q_context=None, q_map=None, prompts=None, estimation=None,
                 object_checks=None, captions=None, clip_intervals=None)


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


# --- the whole-video tensor V: the first load_video_clip call of process_video ------------------
_orig_load_clip = B.ActionGenomeBaseProcessor.load_video_clip


def load_video_clip(self, image_paths):
    out = _orig_load_clip(self, image_paths)
    if STATE.get("video_v") is None and out is not None:
        STATE["video_v"] = _u8(out[0])
        STATE["video_v_paths"] = [Path(p).name for p in image_paths]
    return out


B.ActionGenomeBaseProcessor.load_video_clip = load_video_clip

# --- the shared annotated context C and the per-frame Q(f) = [f ; C] ---------------------------
_orig_ctx = B.ActionGenomeBaseProcessor._build_annotated_context


def _build_annotated_context(self, bbox_frames, video_id, max_context_frames=15):
    out = _orig_ctx(self, bbox_frames, video_id, max_context_frames)
    if out is not None:
        STATE["q_context"] = _u8(out)
        stems = sorted(bbox_frames.keys())
        if len(stems) > max_context_frames:
            idx = np.linspace(0, len(stems) - 1, max_context_frames, dtype=int)
            stems = [stems[i] for i in idx]
        STATE["context_stems"] = stems
    return out


B.ActionGenomeBaseProcessor._build_annotated_context = _build_annotated_context

_orig_qmap = B.ActionGenomeBaseProcessor._build_query_context_map


def _build_query_context_map(self, bbox_frames, video_id, annotated_context):
    out = _orig_qmap(self, bbox_frames, video_id, annotated_context)
    STATE["q_map"] = out
    return out


B.ActionGenomeBaseProcessor._build_query_context_map = _build_query_context_map

# --- SGDet discovery: the P6 prompt / response and the P5 Yes/No prompts -----------------------
_orig_estimate = B.ActionGenomeBaseProcessor.estimate_objects_from_captions


def estimate_objects_from_captions(self, captions, video_inputs):
    model = self.vgent.model
    orig = model.mllm_response
    rec = {}

    def spy(text, vi, *a, **kw):
        resp = orig(text, vi, *a, **kw)
        rec.update(prompt=text, raw_response=resp, max_new_tokens=kw.get("max_new_tokens", a[0] if a else None),
                   video_shape=list(vi[0].shape) if vi and hasattr(vi[0], "shape") else None)
        return resp
    model.mllm_response = spy
    try:
        out = _orig_estimate(self, captions, video_inputs)
    finally:
        model.mllm_response = orig
    rec["parsed"] = sorted(out)
    rec["captions_given"] = [[int(a), str(t)] for a, t in (captions or [])]
    STATE["estimation"] = rec
    return out


B.ActionGenomeBaseProcessor.estimate_objects_from_captions = estimate_objects_from_captions

_orig_verify_obj = B.ActionGenomeBaseProcessor.verify_objects_batch


def verify_objects_batch(self, objects, video_inputs, captions=None):
    model = self.vgent.model
    orig = model.mllm_yes_no_batch
    rec = {}

    def spy(prompts):
        rec["prompts"] = [p["text"] for p in prompts]
        return orig(prompts)
    model.mllm_yes_no_batch = spy
    try:
        out = _orig_verify_obj(self, objects, video_inputs, captions=captions)
    finally:
        model.mllm_yes_no_batch = orig
    rec["scores"] = _jsonable(out)
    STATE["object_checks"] = rec
    return out


B.ActionGenomeBaseProcessor.verify_objects_batch = verify_objects_batch

# --- the batched P4 prompts, in the runner's order, mapped back to (frame, object) --------------
_orig_log = B.ActionGenomeBaseProcessor._log_prompts


def _log_prompts(self, video_id, batch_prompts, tag=""):
    _orig_log(self, video_id, batch_prompts, tag)
    qmap = STATE.get("q_map") or {}
    by_id = {id(t): stem for stem, t in qmap.items()}
    recs = []
    for p in batch_prompts:
        vi = p.get("video_inputs") or [None]
        stem = by_id.get(id(vi[0]))
        text = p.get("text", "")
        i = text.find('The object "')
        obj = text[i + 12: text.find('"', i + 12)] if i != -1 else None
        recs.append({"frame": stem, "object": obj, "text": text, "max_new_tokens": p.get("max_new_tokens"),
                     "shape": list(vi[0].shape) if vi[0] is not None and hasattr(vi[0], "shape") else None})
    STATE["prompts"] = recs
    STATE["tag"] = tag


B.ActionGenomeBaseProcessor._log_prompts = _log_prompts

if METHOD == "caption_all":
    _orig_caps = PROC.load_captions

    def load_captions(self, video_id):
        captions, intervals = _orig_caps(self, video_id)
        STATE["captions"] = [[int(a), str(t)] for a, t in captions]
        STATE["clip_intervals"] = _jsonable(intervals)
        return captions, intervals
    PROC.load_captions = load_captions

# --- per video: reset, run, dump ------------------------------------------------------------
_orig_process = PROC.process_video


def process_video(self, video_id):
    _reset(video_id)
    try:
        return _orig_process(self, video_id)
    finally:
        d = CAP_DIR / f"{METHOD}_{self.mode}"
        d.mkdir(parents=True, exist_ok=True)
        stem = Path(video_id).stem
        rec = {"video_id": video_id, "method": METHOD, "mode": self.mode, "model_name": self.args.model_name,
               "skip_verification": bool(self.skip_verification), "tag": STATE.get("tag"),
               "prompts": STATE.get("prompts"), "estimation": STATE.get("estimation"),
               "object_checks": STATE.get("object_checks"), "captions": STATE.get("captions"),
               "clip_intervals": STATE.get("clip_intervals"), "context_stems": STATE.get("context_stems"),
               "video_v_paths": STATE.get("video_v_paths"),
               "chunk_size": B.ActionGenomeBaseProcessor.BATCH_CHUNK_SIZE}
        tens = {}
        if STATE.get("video_v") is not None:
            tens["video_v"] = STATE["video_v"]
        if STATE.get("q_context") is not None:
            tens["q_context"] = STATE["q_context"]
        qmap = STATE.get("q_map") or {}
        if qmap:
            stems = sorted(qmap)
            tens["q_targets"] = np.stack([_u8(qmap[s][:1])[0] for s in stems])
            rec["q_stems"] = stems
            rec["q_shape"] = list(qmap[stems[0]].shape)
        with open(d / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(_jsonable(rec), f)
        if tens:
            np.savez_compressed(d / f"{stem}_tensors.npz", **tens)
        n = len(STATE.get("prompts") or [])
        print(f"[capture] {stem} {METHOD} {self.mode}: {n} prompts, tensors {sorted(tens)} -> {d}", flush=True)


PROC.process_video = process_video

if __name__ == "__main__":
    _reset()
    R.main()
