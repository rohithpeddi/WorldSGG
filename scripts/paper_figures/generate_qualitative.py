"""Qualitative WorldSGG figures: 2-D overlays, 3-D point clouds and scene graphs.

Unlike the three architecture generators in this directory, this one is
*empirical*: every box, relation and colour comes from the locked worldbbox
test annotations and from prediction dumps that already exist on disk.  It
fabricates nothing -- a method that has no prediction for a video, a frame or
an object is drawn as **absent**, never filled in.

It runs where the data is (the CS93371 box), CPU only, and reads every path
through ``lib/mllm/core/config_loader`` exactly as the rest of ``lib/mllm``
does, so it makes no assumption about the host operating system.

Three figure families per video and category (``<cat>`` is ``training`` or
``mllm``, see ``CATEGORIES`` below):

``<vid>_<cat>_<mode>_overlay2d``
                       ground-truth boxes and relations on real frames, one row
                       per method, colour-coded correct / partial / missed /
                       absent, at a handful of keyframes.
``<vid>_<cat>_<mode>_scene3d_f*``
                       the Pi-3 dynamic scene as a coloured point cloud in the
                       canonical floor frame, with ground-truth and predicted
                       oriented 3-D boxes, from two oblique viewpoints and a
                       bird's-eye view.  Track A, Track B and WorldWise++ have a
                       3-D head and are drawn; every other method in the run is
                       named on the figure as having **no 3-D output**, so an
                       absent box is never read as a failed detection.  No
                       method's *input* corners are ever drawn as a prediction.
``<vid>_<cat>_<mode>_scenegraph_f*``
                       the scene graph itself, ground truth beside each
                       method, with every predicate coloured by outcome.

The paper runs two different styles of experiment, so each video gets **two**
complete sets of figures rather than one mixed set (``--category``):

``training``   the supervised lineage -- WorldWise++, WorldWise+, WorldWise,
               W-DSGDetr++.  Only WorldWise++ has a 3-D head, so it is the only
               predicted box in that category's 3-D panel.
``mllm``       the MLLM tracks -- ``zero_shot``, ``caption_all``, ``rag_all``
               (unlocalized) and Track A, Track B (localized).  The two
               localized tracks are the predicted boxes in its 3-D panel.

Ground truth is drawn in both categories, so either figure stands on its own,
and the category key goes into every filename so the two sets never collide.

Usage (on the server, from the repository root)::

    python scripts/paper_figures/generate_qualitative.py --video 12XD3 --category training
    python scripts/paper_figures/generate_qualitative.py --video AQQQ5 --category mllm \\
        --frames 4 --formats pdf png --dpi 300

``--methods`` still overrides the category's method list when a one-off figure
is wanted.  ``--output-dir`` defaults to ``outputs/paper_figures/qualitative``
inside the repository.  Each run replaces only the files it writes.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from lib.mllm.core.config_loader import get_path, load_config            # noqa: E402
from lib.mllm.data.geometry import corners_to_obb, obb_to_corners        # noqa: E402
from lib.mllm.data.worldbbox import (                                    # noqa: E402
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, OBJECT_CLASSES,
    SPATIAL_RELATIONSHIPS, WorldBBoxTestSet, WorldBBoxVideo,
)
from lib.mllm.eval.dump_adapter import (                                 # noqa: E402
    build_records, load_run_pkl, prediction_lookup,
)
from lib.mllm.eval.recall3d import evaluate_record_3d                    # noqa: E402
from lib.mllm.eval.score_run import make_evaluator                       # noqa: E402
from lib.supervised.evaluation_recall import evaluate_from_dict          # noqa: E402

PRED_NAMES = list(ATTENTION_RELATIONSHIPS) + list(SPATIAL_RELATIONSHIPS) + list(CONTACTING_RELATIONSHIPS)
N_ATT, N_SPA = len(ATTENTION_RELATIONSHIPS), len(SPATIAL_RELATIONSHIPS)

# ---------------------------------------------------------------------------
# Palette -- shares the ink and the pastel fills of common.py (architecture
# figures), plus the four outcome colours this figure family needs.
# ---------------------------------------------------------------------------
INK = "#182B3A"
OUTCOME = {
    "hit": "#2E7D4F",        # every ground-truth predicate of this object recalled
    "partial": "#C98A1B",    # some recalled
    "miss": "#C0392B",       # none recalled although the method spoke
    "absent": "#8A97A0",     # the method emitted nothing for this object
}
OUTCOME_LABEL = {"hit": "every GT predicate recalled", "partial": "partly recalled",
                 "miss": "none recalled", "absent": "no prediction / no feature slot"}
GT_OBSERVED = "#20558A"
GT_UNOBSERVED = "#78609A"
PERSON = "#182B3A"
BOX3D = {"gt": "#182B3A", "track_a": "#1F77B4", "track_b": "#D1611F",
         "worldwise_pp": "#2E8B57"}


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str            # printed on the figure
    track: str            # "1 training-based" | "2 unlocalized MLLM" | "3 localized MLLM"
    kind: str             # "dump" (supervised) | "mllm"
    backbone: str         # printed on the figure, so the reader sees the confound
    dump: str = ""        # {mode}-templated path under cache_root, for kind="dump"
    method_dir: str = ""  # config key under paths.outputs, for kind="mllm"
    model: str = ""       # model sub-directory, for kind="mllm"
    emits_3d: bool = False
    # kind="dump" only: a second, 3-D-carrying dump of the same checkpoint written
    # with ``tools/dump_predictions.py --keep-3d``.  It holds the *predicted* corners
    # (``pred_corners_slot``), which the monolithic dumps drop.  When it covers the
    # video at hand it replaces ``dump`` outright, so relations and boxes always come
    # from one forward pass.
    dump3d: str = ""
    # what the 3-D box of this method actually is, printed in the 3-D legend so a
    # reader never mistakes a refinement for a from-scratch prediction
    box3d_note: str = ""


METHODS: Dict[str, MethodSpec] = {s.key: s for s in [
    # --- track 1: training-based (dumps from tools/dump_predictions.py) ------
    MethodSpec("worldwise_pp", "WorldWise++", "1 training-based", "dump", "dinov3",
               dump="runs/worldwise_pp/score/dumps/worldwise_pp_dinov3_{mode}__all.pkl",
               dump3d="runs/worldwise_pp/score/dumps/qualitative/"
                      "worldwise_pp_dinov3_{mode}_3d__all.pkl",
               emits_3d=True,
               box3d_note="slot refinement of the detector's input corners"),
    MethodSpec("worldwise_pp_nodet", "WorldWise++ (no det.)", "1 training-based", "dump", "dinov3",
               dump="runs/worldwise_pp/score/dumps/worldwise_pp_dinov3_nodet_{mode}__all.pkl"),
    MethodSpec("worldwise_plus", "WorldWise+", "1 training-based", "dump", "dinov3tok",
               dump="runs/worldformer/score/dumps/worldformer_c1_dinov3tok_{mode}__all.pkl"),
    MethodSpec("worldwise", "WorldWise", "1 training-based", "dump", "dinov3l",
               dump="runs/rescore/dumps/worldwise_{mode}_dinov3l__all.pkl"),
    MethodSpec("w_dsgdetr_pp", "W-DSGDetr++", "1 training-based", "dump", "resnet50",
               dump="runs/rescore/dumps/w_dsgdetr_pp_{mode}_resnet50__all.pkl"),
    MethodSpec("w_dsgdetr", "W-DSGDetr", "1 training-based", "dump", "resnet50",
               dump="runs/rescore/dumps/w_dsgdetr_{mode}_resnet50__all.pkl"),
    MethodSpec("w_sttran", "W-STTran", "1 training-based", "dump", "resnet50",
               dump="runs/rescore/dumps/w_sttran_{mode}_resnet50__all.pkl"),
    MethodSpec("w_usg", "W-USG", "1 training-based", "dump", "resnet50",
               dump="runs/rescore/dumps/w_usg_{mode}_resnet50__all.pkl"),
    # --- track 2: unlocalized MLLM ------------------------------------------
    MethodSpec("zero_shot", "zero_shot (frames only)", "2 unlocalized MLLM", "mllm", "qwen25vl_7b",
               method_dir="zero_shot", model="qwen25vl_7b"),
    MethodSpec("caption_all", "caption_all (captions)", "2 unlocalized MLLM", "mllm", "qwen25vl_7b",
               method_dir="caption_all", model="qwen25vl_7b"),
    MethodSpec("rag_all", "Graph-RAG (rag_all)", "2 unlocalized MLLM", "mllm", "qwen25vl_7b",
               method_dir="rag_all", model="qwen25vl_7b"),
    # --- track 3: localized MLLM (the only track that emits 3-D boxes) ------
    MethodSpec("track_a", "Track A (marked frames + BEV)", "3 localized MLLM", "mllm", "qwen3vl_8b",
               method_dir="track_a", model="qwen3vl_8b", emits_3d=True),
    MethodSpec("track_b", "Track B (tool loop + critic)", "3 localized MLLM", "mllm", "qwen3vl_8b",
               method_dir="track_b", model="qwen3vl_8b", emits_3d=True),
]}

DEFAULT_METHODS = ["worldwise_pp", "worldwise", "w_dsgdetr_pp",
                   "rag_all", "zero_shot", "track_a", "track_b"]


# ---------------------------------------------------------------------------
# Categories -- the paper runs two different styles of experiment, and mixing
# them into one figure asks the reader to compare rows that were never measured
# the same way (different supervision, different backbones, different sgdet
# matcher).  Each category therefore gets its own complete set of figures, and
# ground truth is drawn in both so either figure stands on its own.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CategorySpec:
    key: str              # short, stable, goes in every filename
    label: str            # printed on the figure
    blurb: str            # one line under the title
    methods: Tuple[str, ...]


CATEGORIES: Dict[str, CategorySpec] = {c.key: c for c in [
    CategorySpec(
        "training", "training-based",
        "supervised lineage; GT shown for reference",
        ("worldwise_pp", "worldwise_plus", "worldwise", "w_dsgdetr_pp")),
    CategorySpec(
        "mllm", "MLLM tracks",
        "unlocalized (zero_shot / caption_all / rag_all) and localized (Track A / Track B); "
        "GT shown for reference",
        ("zero_shot", "caption_all", "rag_all", "track_a", "track_b")),
]}


# ---------------------------------------------------------------------------
# Records: one uniform per-frame container for every method
# ---------------------------------------------------------------------------
def _slot_labels(rec: Dict[str, Any]) -> List[str]:
    """Short labels per slot.  MLLM records carry them; supervised dumps do not."""
    if rec.get("slot_labels"):
        return [str(x) for x in rec["slot_labels"]]
    from lib.mllm.data.worldbbox import to_short
    return [to_short(OBJECT_CLASSES[int(c)]) if 0 <= int(c) < len(OBJECT_CLASSES) else "__pad__"
            for c in rec["object_classes"]]


def load_mllm_records(spec: MethodSpec, video: WorldBBoxVideo, mode: str, cfg: dict
                      ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    root = get_path(cfg, f"outputs.{spec.method_dir}")
    path = Path(root) / mode / spec.model / f"{video.video_id}.mp4.pkl"
    if not path.exists():
        alt = Path(root) / mode / spec.model / f"{video.video_id}.pkl"
        path = alt if alt.exists() else path
    if not path.exists():
        return None, str(path)
    preds = prediction_lookup(load_run_pkl(str(path)))
    return build_records(video, preds, mode=mode), str(path)


def _slice_dump(path: Path, video_id: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        blob = pickle.load(f)
    records = [r for r in (blob["records"] if isinstance(blob, dict) else blob)
               if r.get("video_id") == video_id or
               str(r.get("video_id", "")).replace(".mp4", "") == video_id]
    del blob
    return records


def load_dump_records(spec: MethodSpec, video_id: str, mode: str, cfg: dict, cache_dir: Path
                      ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """Slice one video out of a monolithic ``*__all.pkl``, caching the slice.

    The dumps are 70-310 MB each; the per-video cache makes re-runs cheap and
    keeps the figure script usable without re-running any model.

    When the method has a ``dump3d`` (a small re-run of the *same* checkpoint
    written with ``--keep-3d``) and that dump covers this video, it is used
    instead of the monolithic one: it carries the same relation distributions
    plus the model's predicted 3-D corners, so the 3-D panel and the relation
    panels never disagree about which forward pass they describe.
    """
    root = Path(get_path(cfg, "cache_root"))
    if spec.dump3d:
        p3 = root / spec.dump3d.format(mode=mode)
        cache3 = cache_dir / f"{video_id}__{spec.key}__{mode}__3d.pkl"
        if cache3.exists():
            with open(cache3, "rb") as f:
                return pickle.load(f), str(p3)
        if p3.exists():
            records = _slice_dump(p3, video_id)
            if records:
                cache3.parent.mkdir(parents=True, exist_ok=True)
                with open(cache3, "wb") as f:
                    pickle.dump(records, f)
                return records, str(p3)
    path = root / spec.dump.format(mode=mode)
    if not path.exists():
        return None, str(path)
    cache = cache_dir / f"{video_id}__{spec.key}__{mode}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f), str(path)
    records = _slice_dump(path, video_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as f:
        pickle.dump(records, f)
    return records, str(path)


def load_records(spec: MethodSpec, video: WorldBBoxVideo, mode: str, cfg: dict, cache_dir: Path
                 ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    if spec.kind == "mllm":
        return load_mllm_records(spec, video, mode, cfg)
    return load_dump_records(spec, video.video_id, mode, cfg, cache_dir)


# ---------------------------------------------------------------------------
# Matching -- which ground-truth triplets a method recalled
# ---------------------------------------------------------------------------
def gt_triplet_keys(rec: Dict[str, Any], labels: Sequence[str]) -> List[Tuple[str, str]]:
    """``(object label, predicate name)`` per ground-truth triplet.

    Built in exactly the order used by ``evaluate_record_3d`` and
    ``evaluation_recall.evaluate_wsgg_video`` (attention, then spatial, then
    contacting, pair by pair), so a ``pred_to_gt`` index means the same thing
    here as it does inside the real metric.
    """
    keys: List[Tuple[str, str]] = []
    pair_valid = rec["pair_valid"].astype(bool)
    object_idx = rec["object_idx"].astype(int)
    for k in np.where(pair_valid)[0]:
        lab = labels[int(object_idx[k])]
        keys.append((lab, PRED_NAMES[int(rec["gt_attention"][k])]))
        for s in np.where(rec["gt_spatial"][k] > 0.5)[0]:
            keys.append((lab, PRED_NAMES[N_ATT + int(s)]))
        for c in np.where(rec["gt_contacting"][k] > 0.5)[0]:
            keys.append((lab, PRED_NAMES[N_ATT + N_SPA + int(c)]))
    return keys


def _wsgg_pred_to_gt(rec: Dict[str, Any], evaluator, mode: str):
    """``pred_to_gt`` for a supervised dump record.

    Mirrors ``evaluation_recall.evaluate_wsgg_video`` (which builds the same
    two entries but returns nothing) and then calls the real
    ``evaluate_from_dict``, so the matcher, the ranking and the 2-D IoU 0.5
    triplet test are the metric's own.
    """
    pair_valid = rec["pair_valid"].astype(bool)
    if not pair_valid.any():
        return None
    person_idx, object_idx = rec["person_idx"], rec["object_idx"]
    gt_classes = rec["object_classes"].astype(np.int64)
    gt_boxes = rec["bboxes_2d"].astype(np.float32)
    spa_off, con_off = N_ATT, N_ATT + N_SPA
    gt_relations: List[List[int]] = []
    for k in np.where(pair_valid)[0]:
        p, o = int(person_idx[k]), int(object_idx[k])
        gt_relations.append([p, o, int(rec["gt_attention"][k])])
        for s in np.where(rec["gt_spatial"][k] > 0.5)[0]:
            gt_relations.append([o, p, spa_off + int(s)])
        for c in np.where(rec["gt_contacting"][k] > 0.5)[0]:
            gt_relations.append([p, o, con_off + int(c)])
    if not gt_relations:
        return None
    vp, vo = person_idx[pair_valid], object_idx[pair_valid]
    rels_i = np.concatenate([np.stack([vp, vo], 1), np.stack([vo, vp], 1), np.stack([vp, vo], 1)], 0)
    att = rec["attention_distribution"][pair_valid]
    spa = rec["spatial_distribution"][pair_valid]
    con = rec["contacting_distribution"][pair_valid]
    K, n_c = len(att), con.shape[1]
    rel_scores = np.concatenate([
        np.concatenate([att, np.zeros((K, N_SPA + n_c))], 1),
        np.concatenate([np.zeros((K, N_ATT)), spa, np.zeros((K, n_c))], 1),
        np.concatenate([np.zeros((K, N_ATT + N_SPA)), con], 1)], 0)
    if mode == "predcls":
        pred_boxes, pred_classes = gt_boxes.copy(), gt_classes.copy()
        obj_scores = np.ones(len(gt_classes), np.float32)
    else:
        gt_boxes = rec.get("gt_bboxes_2d", gt_boxes).astype(np.float32)
        pred_boxes = rec.get("bboxes_2d", gt_boxes).astype(np.float32)
        pred_classes = rec.get("pred_labels", gt_classes).astype(np.int64)
        obj_scores = rec.get("pred_scores", np.ones(len(gt_classes))).astype(np.float32)
    gt_entry = {"gt_classes": gt_classes, "gt_boxes": gt_boxes,
                "gt_relations": np.array(gt_relations, dtype=np.int64)}
    pred_entry = {"pred_boxes": pred_boxes, "pred_classes": pred_classes,
                  "pred_rel_inds": rels_i, "obj_scores": obj_scores, "rel_scores": rel_scores}
    pred_to_gt, _, _ = evaluate_from_dict(
        gt_entry, pred_entry, evaluator.mode, evaluator.result_dict,
        iou_thresh=evaluator.iou_threshold, method=evaluator.constraint,
        threshold=evaluator.semi_threshold, num_rel=evaluator.num_rel)
    return pred_to_gt


def recalled_keys(spec: MethodSpec, rec: Dict[str, Any], labels: Sequence[str], mode: str,
                  k_at: int, iou_thr: float, constraint: str) -> Optional[set]:
    """Ground-truth ``(label, predicate)`` keys this method recalled at R@``k_at``.

    ``None`` means the frame has no ground-truth triplet at all.
    """
    evaluator = make_evaluator(mode, constraint)
    try:
        if spec.kind == "mllm":
            thr = iou_thr if (mode == "sgdet" and spec.emits_3d) else 0.0
            p2g = evaluate_record_3d(rec, evaluator, iou_thr=thr, mode=mode)
        else:
            p2g = _wsgg_pred_to_gt(rec, evaluator, mode)
    except Exception:                                    # a malformed frame must not kill the figure
        return set()
    keys = gt_triplet_keys(rec, labels)
    if p2g is None:
        return None if not keys else set()
    if not len(p2g):
        return set()
    match = reduce(np.union1d, p2g[:k_at])
    return {keys[int(i)] for i in np.atleast_1d(match) if int(i) < len(keys)}


def predicted_keys(rec: Dict[str, Any], labels: Sequence[str], constraint: str,
                   max_per_head: int = 3) -> Dict[str, List[str]]:
    """What the method actually said, per object, in the protocol's own reading.

    With constraint the evaluator keeps exactly one predicate per (pair, head)
    -- ``rel_scores.argmax(1)`` in ``evaluate_from_dict`` and in
    ``evaluate_record_3d`` alike -- so that is what the figure shows.  With no
    constraint several predicates per head survive the ranking, so the top
    ``max_per_head`` non-zero ones are shown instead.  Only pairs the method
    actually emitted (``pred_pair_valid``) contribute.
    """
    out: Dict[str, List[str]] = {}
    ppv = rec.get("pred_pair_valid", rec["pair_valid"]).astype(bool)
    object_idx = rec["object_idx"].astype(int)
    heads = (("attention_distribution", 0), ("spatial_distribution", N_ATT),
             ("contacting_distribution", N_ATT + N_SPA))
    for k in np.where(ppv)[0]:
        lab = labels[int(object_idx[k])]
        said: List[str] = []
        for field_name, offset in heads:
            scores = np.asarray(rec[field_name][k], dtype=np.float64)
            if scores.size == 0 or scores.max() <= 0:
                continue
            if constraint == "with":
                said.append(PRED_NAMES[offset + int(scores.argmax())])
            else:
                idx = np.argsort(-scores)[:max_per_head]
                said.extend(PRED_NAMES[offset + int(i)] for i in idx if scores[i] > 0)
        out[lab] = said
    return out


# ---------------------------------------------------------------------------
# Per-video assembly
# ---------------------------------------------------------------------------
@dataclass
class MethodView:
    spec: MethodSpec
    source: str
    available: bool
    records: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # frame file -> record
    labels: Dict[str, List[str]] = field(default_factory=dict)         # frame file -> slot labels
    recalled: Dict[str, Optional[set]] = field(default_factory=dict)   # frame file -> keys
    said: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


def build_views(video: WorldBBoxVideo, specs: Sequence[MethodSpec], mode: str, cfg: dict,
                cache_dir: Path, k_at: int, iou_thr: float, constraint: str) -> Dict[str, MethodView]:
    frame_files = [f.file for f in video.frames]
    views: Dict[str, MethodView] = {}
    for spec in specs:
        records, source = load_records(spec, video, mode, cfg, cache_dir)
        view = MethodView(spec=spec, source=source, available=bool(records))
        if records:
            for t, rec in enumerate(records):
                ff = rec.get("frame_file") or (frame_files[t] if t < len(frame_files) else f"t{t}")
                labels = _slot_labels(rec)
                view.records[ff] = rec
                view.labels[ff] = labels
                view.recalled[ff] = recalled_keys(spec, rec, labels, mode, k_at, iou_thr, constraint)
                view.said[ff] = predicted_keys(rec, labels, constraint)
        views[spec.key] = view
    return views


def gt_relations_of(frame) -> Dict[str, List[str]]:
    """``{object label: [predicate, ...]}`` straight from the annotation."""
    out: Dict[str, List[str]] = {}
    for o in frame.objects:
        if o.label in out:
            continue
        out[o.label] = list(o.attention) + list(o.spatial) + list(o.contacting)
    return out


def outcome_for(view: MethodView, frame_file: str, label: str, gt_preds: Sequence[str]
                ) -> Tuple[str, int, int]:
    """``(outcome, recalled, total)`` for one object in one frame.

    ``total`` counts the distinct ground-truth predicates of that object.  With
    constraint the evaluator keeps one predicate per head, so an object with
    several spatial or contacting labels cannot reach ``recalled == total``;
    the figures therefore print the fraction next to the label rather than
    letting the colour alone carry it.
    """
    total = len(set(gt_preds))
    if not view.available or frame_file not in view.records:
        return "absent", 0, total
    if label not in view.said.get(frame_file, {}):
        return "absent", 0, total
    hits = view.recalled.get(frame_file) or set()
    got = len({p for p in gt_preds if (label, p) in hits})
    if total == 0:
        return "absent", 0, 0
    return ("hit" if got == total else ("partial" if got else "miss")), got, total


def pick_frames(video: WorldBBoxVideo, n: int, explicit: Sequence[int]) -> List[int]:
    """Indices into ``video.frames``: explicit frame numbers, or a spread over
    the frames whose relation set changed the most (with a Pi-3 index, so the
    3-D panels have a camera)."""
    frames = video.frames
    if explicit:
        wanted = set(int(x) for x in explicit)
        idx = [t for t, f in enumerate(frames) if f.frame_num in wanted]
        if idx:
            return idx
    rel_sets = [set((o.label, r) for o in f.objects
                    for r in list(o.attention) + list(o.spatial) + list(o.contacting))
                for f in frames]
    churn = [0] + [len(a ^ b) for a, b in zip(rel_sets[:-1], rel_sets[1:])]
    order = sorted(range(len(frames)), key=lambda t: (-churn[t], t))
    chosen: List[int] = []
    for t in order:
        if len(chosen) >= n:
            break
        if any(abs(t - c) < max(1, len(frames) // (2 * max(n, 1))) for c in chosen):
            continue
        chosen.append(t)
    while len(chosen) < min(n, len(frames)):
        for t in range(len(frames)):
            if t not in chosen:
                chosen.append(t)
                break
    return sorted(chosen)[:n]


# ---------------------------------------------------------------------------
# Figure 1 -- 2-D overlays
# ---------------------------------------------------------------------------
_TEXT_W_CACHE: Dict[Tuple[str, float, bool], float] = {}


def text_width_in(s: str, size: float, bold: bool = False) -> float:
    """Rendered width of ``s`` in inches, measured rather than guessed.

    ``TextPath`` lays the string out with the real font metrics at ``size``
    points, so this is what matplotlib will actually draw.  The figures use it
    to size the row-label gutter: a label must never be allowed to run under
    the first image column, and the method names are fixed by the paper, so the
    canvas has to accommodate them rather than the other way round.
    """
    if not s:
        return 0.0
    key = (s, size, bold)
    if key not in _TEXT_W_CACHE:
        fp = FontProperties(size=size, weight="bold" if bold else "normal")
        _TEXT_W_CACHE[key] = float(TextPath((0, 0), s, prop=fp).get_extents().width) / 72.0
    return _TEXT_W_CACHE[key]


def wrap_to_width(s: str, size: float, bold: bool, max_in: float) -> List[str]:
    """Greedy word wrap at a *measured* width.  A single word wider than
    ``max_in`` is never split -- it is returned whole and the caller's gutter
    grows to fit it, which keeps the "no text crosses into the image" rule
    absolute instead of best-effort."""
    lines: List[str] = []
    current = ""
    for word in s.split(" "):
        trial = f"{current} {word}".strip()
        if current and text_width_in(trial, size, bold) > max_in:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def _draw_box(ax, xyxy, colour, text, dashed=False, lw=1.4, fs=6.2):
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=colour,
                           linewidth=lw, linestyle="--" if dashed else "-", zorder=3))
    ax.text(x0 + 1, max(y0 - 2, 6), text, fontsize=fs, color="white", va="bottom", ha="left",
            zorder=4, bbox=dict(boxstyle="square,pad=0.14", facecolor=colour, edgecolor="none"))


def matcher_note(spec: MethodSpec, mode: str, iou_thr: float) -> str:
    """How a triplet of this method is matched, in the words of the protocol.

    SGDet is not comparable across tracks (ICLR_THREE_TRACKS.md section 5), so
    every figure states the matcher next to the method instead of leaving the
    reader to assume one.
    """
    if mode == "predcls":
        return "predcls, GT boxes supplied"
    if spec.kind == "dump":
        return "sgdet, 2-D IoU 0.5"
    if spec.emits_3d:
        return f"sgdet, 3-D IoU {iou_thr:g}"
    return "sgdet, class-only (emits no box)"


def raw_frame_path(video: WorldBBoxVideo, cfg: dict, frame_file: str) -> Path:
    """The untouched extracted frame.

    ``WorldBBoxVideo.frame_path`` prefers ``ag_frames_annotated``, which may
    already carry drawn boxes; a qualitative figure must start from the raw
    pixels, so ``paths.ag_frames`` is tried first here.
    """
    for root in (get_path(cfg, "ag_frames"), get_path(cfg, "ag_frames_annotated")):
        if not root:
            continue
        p = Path(root) / video.vid_mp4 / frame_file
        if p.exists():
            return p
    return video.frame_path(frame_file)


def figure_overlay2d(video: WorldBBoxVideo, views: Dict[str, MethodView], specs: Sequence[MethodSpec],
                     frame_idx: Sequence[int], mode: str, k_at: int, cfg: dict,
                     iou_thr: float, category: Optional[CategorySpec] = None) -> plt.Figure:
    frames = [video.frames[t] for t in frame_idx]
    n_rows, n_cols = 1 + len(specs), len(frames)
    cell_w, cell_h = 2.35, 2.35

    # --- row labels, laid out before the canvas exists ----------------------
    # The method names are the paper's and may not be shortened, so the left
    # gutter is sized to the longest line that will actually be drawn: every
    # line is first wrapped at LABEL_MAX_IN, then measured, and the gutter is
    # the widest survivor plus padding.  Nothing can cross into the image area.
    TITLE_FS, SUB_FS = 8.2, 6.4
    LABEL_X_IN, LABEL_PAD_IN, LABEL_MAX_IN = 0.06, 0.20, 2.30
    n_gt_objects = len({o.label for o in frames[0].objects})
    row_text: List[Tuple[List[str], List[str]]] = [
        (wrap_to_width("Ground truth", TITLE_FS, True, LABEL_MAX_IN),
         wrap_to_width(f"worldbbox test • {n_gt_objects} objects", SUB_FS, False, LABEL_MAX_IN))]
    for spec in specs:
        sub_src = [f"track {spec.track} • {spec.backbone}", matcher_note(spec, mode, iou_thr)]
        if not views[spec.key].available:
            sub_src.append("NO OUTPUT FOR THIS VIDEO")
        sub_lines: List[str] = []
        for piece in sub_src:
            sub_lines.extend(wrap_to_width(piece, SUB_FS, False, LABEL_MAX_IN))
        row_text.append((wrap_to_width(spec.label, TITLE_FS, True, LABEL_MAX_IN), sub_lines))
    widest = max([text_width_in(l, TITLE_FS, True) for t, _ in row_text for l in t]
                 + [text_width_in(l, SUB_FS) for _, s in row_text for l in s] + [0.0])
    gutter = max(1.55, LABEL_X_IN + widest + LABEL_PAD_IN)

    fig_w = gutter + cell_w * n_cols
    fig_h = 1.0 + cell_h * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(n_rows, n_cols, left=gutter / fig_w, right=0.995,
                          top=1 - 0.76 / fig_h, bottom=0.22 / fig_h,
                          wspace=0.04, hspace=0.06)
    images = {}
    for fr in frames:
        images[fr.file] = plt.imread(str(raw_frame_path(video, cfg, fr.file)))

    def row_label(y, title_lines, sub_lines):
        """Title block growing up from ``y``, sub-block growing down from it, so
        a wrapped title never collides with the sub-block below it."""
        x, gap = LABEL_X_IN / fig_w, 0.05 / fig_h
        fig.text(x, y + gap, "\n".join(title_lines), fontsize=TITLE_FS, fontweight="bold",
                 color=INK, va="bottom", ha="left", linespacing=1.3)
        fig.text(x, y - gap, "\n".join(sub_lines), fontsize=SUB_FS, color="#516271",
                 va="top", ha="left", linespacing=1.4)

    for r in range(n_rows):
        spec = None if r == 0 else specs[r - 1]
        for c, fr in enumerate(frames):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(images[fr.file])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#C8D0D6")
            gt_rel = gt_relations_of(fr)
            offscreen: List[Tuple[str, str]] = []
            if fr.person_bbox_2d is not None:
                _draw_box(ax, fr.person_bbox_2d, PERSON, "person")
            seen = set()
            for o in fr.objects:
                if o.label in seen:
                    continue
                seen.add(o.label)
                if spec is None:
                    colour = GT_OBSERVED if o.observed else GT_UNOBSERVED
                    tag = o.label if o.observed else f"{o.label} (unobs.)"
                    dashed = not o.observed
                else:
                    oc, got, total = outcome_for(views[spec.key], fr.file, o.label,
                                                 gt_rel.get(o.label, []))
                    colour, dashed = OUTCOME[oc], oc == "absent"
                    tag = f"{o.label} — absent" if oc == "absent" else f"{o.label} {got}/{total}"
                if o.bbox_2d is not None:
                    _draw_box(ax, o.bbox_2d, colour, tag, dashed=dashed)
                else:
                    offscreen.append((tag, colour))
            if offscreen:
                txt = "no 2-D box: " + ", ".join(t for t, _ in offscreen)
                ax.text(0.015, 0.02, txt, transform=ax.transAxes, fontsize=5.9, va="bottom",
                        ha="left", color=INK, wrap=True,
                        bbox=dict(boxstyle="square,pad=0.22", facecolor="white", alpha=0.82,
                                  edgecolor="#C8D0D6", linewidth=0.5))
            if r == 0:
                ax.set_title(f"frame {fr.frame_num:06d}", fontsize=7.2, color=INK, pad=3)
            if c == 0:
                bbox = ax.get_position()
                row_label(bbox.y0 + bbox.height / 2, *row_text[r])
    cat = f"{category.label}  •  " if category else ""
    fig.suptitle(f"{video.video_id}  •  {cat}{mode}  •  per-method relation recall at R@{k_at}\n"
                 f"the fraction on each box is (recalled / ground-truth predicates)",
                 fontsize=9.6, fontweight="bold", color=INK, y=1 - 0.06 / fig_h,
                 va="top", linespacing=1.45)
    handles = [Line2D([], [], color=GT_OBSERVED, lw=2, label="GT, observed"),
               Line2D([], [], color=GT_UNOBSERVED, lw=2, ls="--", label="GT, unobserved")]
    handles += [Line2D([], [], color=OUTCOME[k], lw=2, ls="--" if k == "absent" else "-",
                       label=OUTCOME_LABEL[k]) for k in ("hit", "partial", "miss", "absent")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=6.8,
               bbox_to_anchor=(0.5, 0.0005))
    return fig


# ---------------------------------------------------------------------------
# Figure 2 -- 3-D point cloud with ground-truth and predicted boxes
# ---------------------------------------------------------------------------
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]


def canonical_corners(corners) -> Optional[np.ndarray]:
    """Re-order any floor-parallel 8-corner box into ``obb_to_corners`` order."""
    c = np.asarray(corners, dtype=np.float64)
    if c.size != 24 or not np.any(c):
        return None
    c = c.reshape(8, 3)
    try:
        centre, size, yaw = corners_to_obb(c)
        return obb_to_corners(centre, size, yaw)
    except Exception:
        return c


def _draw_box3d(ax, corners, colour, lw=1.1, ls="-", alpha=1.0):
    c = canonical_corners(corners)
    if c is None:
        return
    for i, j in BOX_EDGES:
        ax.plot(*zip(c[i], c[j]), color=colour, lw=lw, ls=ls, alpha=alpha, zorder=6)


def _equal_3d(ax, pts):
    lo, hi = pts.min(0), pts.max(0)
    ctr, rad = (lo + hi) / 2, max((hi - lo).max() / 2, 0.5)
    ax.set_xlim(ctr[0] - rad, ctr[0] + rad)
    ax.set_ylim(ctr[1] - rad, ctr[1] + rad)
    ax.set_zlim(ctr[2] - rad, ctr[2] + rad)


def predicted_corners(spec: MethodSpec, view: MethodView, frame_file: str
                      ) -> Dict[str, np.ndarray]:
    """``{object label: (8,3) predicted corners}`` in the canonical frame.

    Two record layouts carry a 3-D box and they are read differently:

    * localized MLLM dumps -- ``has_pred_3d`` / ``pred_corners``, one row per slot;
    * WorldWise++ supervised dumps -- ``pred_corners_slot``, written by
      ``tools/reeval_test.build_pred_pkl`` from the model's ``det.slot_corners``
      and already mapped to the canonical frame there.

    ``bboxes_3d`` is deliberately **not** consulted: it is the detector's input
    corner set, not a prediction, and a method with no 3-D head returns ``{}``.
    """
    rec = view.records.get(frame_file)
    if rec is None:
        return {}
    labels = view.labels.get(frame_file, [])
    out: Dict[str, np.ndarray] = {}
    if spec.kind == "mllm":
        if "pred_corners" not in rec:
            return {}
        for i, lab in enumerate(labels):
            if rec["has_pred_3d"][i] and np.any(rec["pred_corners"][i]):
                out[lab] = np.asarray(rec["pred_corners"][i])
        return out
    corners = rec.get("pred_corners_slot")
    if corners is None:
        return {}
    valid = rec.get("valid_mask")
    for i, lab in enumerate(labels):
        if lab == "__pad__" or i >= len(corners):
            continue
        if valid is not None and i < len(valid) and not bool(valid[i]):
            continue
        if np.any(corners[i]):
            out.setdefault(lab, np.asarray(corners[i]))
    return out


def figure_scene3d(video: WorldBBoxVideo, views: Dict[str, MethodView], specs: Sequence[MethodSpec],
                   t: int, cloud: Dict[str, np.ndarray], max_points: int, iou_thr: float,
                   mode: str, category: Optional[CategorySpec] = None
                   ) -> Tuple[plt.Figure, Dict[str, Any]]:
    from lib.mllm.eval.iou3d import compute_iou_3d_obb

    fr = video.frames[t]

    def carries_3d(spec: MethodSpec) -> bool:
        """True only if the loaded records actually hold predicted corners.

        A method with a 3-D head whose 3-D-carrying dump is missing must not be
        drawn as though it predicted nothing, so it is reported separately.
        """
        key = "pred_corners" if spec.kind == "mllm" else "pred_corners_slot"
        return any(key in rec for rec in views[spec.key].records.values())

    loc = [s for s in specs if s.emits_3d and views[s.key].available and carries_3d(s)]
    # methods in this run that have no 3-D head at all -- named on the figure so the
    # absence of a box is read as "no 3-D output", never as a missed detection
    no3d = [s for s in specs if not s.emits_3d]
    # a 3-D head whose predictions are not in the dump at hand: neither of the above
    missing3d = [s for s in specs if s.emits_3d and s not in loc]
    xyz, rgb = cloud["xyz"], cloud["rgb"]
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    gt_boxes: List[Tuple[str, np.ndarray, bool]] = []
    if fr.person_corners_final is not None and np.any(fr.person_corners_final):
        gt_boxes.append(("person", np.asarray(fr.person_corners_final), True))
    seen = set()
    for o in fr.objects:
        if o.label in seen or o.corners_final is None or not np.any(o.corners_final):
            continue
        seen.add(o.label)
        gt_boxes.append((o.label, np.asarray(o.corners_final), o.observed))

    pred_boxes: Dict[str, Dict[str, np.ndarray]] = {
        s.key: predicted_corners(s, views[s.key], fr.file) for s in loc}

    def iou_row(boxes: Dict[str, np.ndarray]) -> Dict[str, float]:
        row = {}
        for lab, c, _ in gt_boxes:
            p = boxes.get(lab)
            if p is None:
                row[lab] = float("nan")
            else:
                try:
                    row[lab] = float(compute_iou_3d_obb(np.asarray(c), p))
                except Exception:
                    row[lab] = 0.0
        return row

    ious: Dict[str, Dict[str, float]] = {s.key: iou_row(pred_boxes[s.key]) for s in loc}

    # For a method whose 3-D box is a refinement of an input box, also score the
    # *input* corners, so the meta says how far the learned delta moved them.  This
    # is provenance only -- the input boxes are never drawn and never labelled as a
    # prediction on the figure.
    input_ious: Dict[str, Dict[str, float]] = {}
    for s in loc:
        rec = views[s.key].records.get(fr.file)
        if rec is None or rec.get("bboxes_3d") is None:
            continue
        labels = views[s.key].labels.get(fr.file, [])
        valid = rec.get("valid_mask")
        raw: Dict[str, np.ndarray] = {}
        for i, lab in enumerate(labels):
            if lab == "__pad__" or i >= len(rec["bboxes_3d"]):
                continue
            if valid is not None and i < len(valid) and not bool(valid[i]):
                continue
            if np.any(rec["bboxes_3d"][i]):
                raw.setdefault(lab, np.asarray(rec["bboxes_3d"][i]))
        input_ious[s.key] = iou_row(raw)

    views3d = [("oblique view A", 22, -62), ("oblique view B", 22, 34), ("bird's-eye view", 89, -90)]
    # The caption block (one 3-D IoU line per method, including the methods that have
    # no 3-D head) and the legend both grow with the number of methods, so the canvas
    # grows with them instead of letting them land on the point cloud.
    n_handles = 3 + len(loc) + len(no3d)
    ncol = min(3, max(n_handles, 1))
    legend_rows = int(np.ceil(n_handles / ncol))
    n_lines = max(len(loc) + len(no3d) + len(missing3d), 1)
    PANEL_IN, LINE_IN, LEG_IN, PAD_IN = 3.95, 0.155, 0.175, 0.10
    foot_in = n_lines * LINE_IN + legend_rows * LEG_IN + 3 * PAD_IN
    fig_h = PANEL_IN + foot_in
    fig = plt.figure(figsize=(10.8, fig_h), facecolor="white")
    all_pts = [xyz] + [c for _, c, _ in gt_boxes] + \
              [p for d in pred_boxes.values() for p in d.values()]
    all_pts = np.concatenate([np.asarray(a).reshape(-1, 3) for a in all_pts], 0)
    for i, (name, elev, azim) in enumerate(views3d):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.set_proj_type("ortho")
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=rgb / 255.0, s=1.1, linewidths=0,
                   alpha=0.75, zorder=1, rasterized=True)
        for lab, c, observed in gt_boxes:
            _draw_box3d(ax, c, BOX3D["gt"], lw=1.25, ls="-" if observed else (0, (3, 2)))
            top = np.asarray(c).reshape(8, 3)
            ax.text(*top.mean(0)[:2], top[:, 2].max() + 0.04, lab, fontsize=5.4,
                    color=BOX3D["gt"], ha="center", va="bottom", zorder=8)
        for s in loc:
            for lab, p in pred_boxes.get(s.key, {}).items():
                _draw_box3d(ax, p, BOX3D.get(s.key, "#7F7F7F"), lw=1.0, ls=(0, (4, 2)), alpha=0.95)
        cam = fr.camera_pose_final
        if cam is not None:
            o3 = np.asarray(cam)[:3, 3]
            d3 = np.asarray(cam)[:3, :3] @ np.array([0.0, 0.0, 0.45])
            ax.plot(*zip(o3, o3 + d3), color="#C0392B", lw=1.6, zorder=7)
            ax.scatter(*o3, color="#C0392B", s=14, zorder=7)
        _equal_3d(ax, all_pts)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(name, fontsize=8, color=INK, pad=-2)
        ax.set_xlabel("x (m)", fontsize=6, labelpad=-6)
        ax.set_ylabel("y (m)", fontsize=6, labelpad=-6)
        ax.tick_params(labelsize=5, pad=-2)
        ax.set_box_aspect((1, 1, 1))
        if elev >= 85:                       # bird's-eye: the z axis is edge-on and unreadable
            for tick in ax.get_zticklabels():
                tick.set_visible(False)
        else:
            ax.set_zlabel("z (m)", fontsize=6, labelpad=-6)
        ax.set_facecolor("white")
        ax.grid(True, linewidth=0.3, alpha=0.4)

    handles = [Line2D([], [], color=BOX3D["gt"], lw=1.6, label="GT 3-D box (observed)"),
               Line2D([], [], color=BOX3D["gt"], lw=1.6, ls=(0, (3, 2)), label="GT 3-D box (unobserved)")]
    handles += [Line2D([], [], color=BOX3D.get(s.key, "#7F7F7F"), lw=1.6, ls=(0, (4, 2)),
                       label=f"{s.label} — predicted"
                             + (f" ({s.box3d_note})" if s.box3d_note else ""))
                for s in loc]
    handles.append(Line2D([], [], color="#C0392B", lw=1.6, label="camera at this frame"))
    for s in no3d:
        handles.append(Line2D([], [], color="none", lw=0,
                              label=f"{s.label} — no 3-D output (no 3-D head)"))
    lines = []
    for s in loc:
        parts = [f"{lab} {ious[s.key][lab]:.2f}" if np.isfinite(ious[s.key][lab]) else f"{lab} —"
                 for lab, _, _ in gt_boxes]
        lines.append(f"{s.label} 3-D IoU:  " + "   ".join(parts))
    for s in no3d:
        lines.append(f"{s.label} 3-D IoU:  no 3-D head — this method predicts no box")
    for s in missing3d:
        lines.append(f"{s.label} 3-D IoU:  has a 3-D head, but this dump carries no "
                     f"predicted corners")
    if not lines:
        lines = ["no method in this figure emits a 3-D box"]
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.5, PAD_IN / fig_h))
    fig.text(0.5, (2 * PAD_IN + legend_rows * LEG_IN) / fig_h, "\n".join(lines),
             fontsize=6.6, color=INK, ha="center", va="bottom", linespacing=1.55)
    cat = f"{category.label}  •  " if category else ""
    fig.suptitle(f"{video.video_id}  •  frame {fr.frame_num:06d}  •  {cat}{mode}  •  "
                 f"Pi-3 dynamic scene, canonical floor frame  •  match at 3-D IoU {iou_thr:g}",
                 fontsize=9.6, fontweight="bold", color=INK, y=1 - 0.07 / fig_h)
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - 0.46 / fig_h,
                        bottom=foot_in / fig_h, wspace=0.0)
    def _round(row):
        return {lab: (None if not np.isfinite(v) else round(float(v), 4)) for lab, v in row.items()}

    meta = {"frame": fr.frame_num, "frame_file": fr.file, "pi3_index": fr.pi3_index,
            "category": category.key if category else None,
            "n_cloud_points": int(len(xyz)),
            "gt_boxes": [lab for lab, _, _ in gt_boxes],
            "iou3d": {k: _round(row) for k, row in ious.items()},
            # provenance for the "refinement, not a from-scratch prediction" claim:
            # the same IoU computed against the detector's *input* corners, which the
            # figure never draws
            "iou3d_input_corners": {k: _round(row) for k, row in input_ious.items()},
            "no_3d_head": [s.key for s in no3d],
            "3d_head_but_no_predictions_in_dump": [s.key for s in missing3d],
            "box3d_note": {s.key: s.box3d_note for s in loc if s.box3d_note}}
    return fig, meta


# ---------------------------------------------------------------------------
# Figure 3 -- the scene graph itself
# ---------------------------------------------------------------------------
PRED_STEP = 0.34          # vertical units between two predicate lines on an edge
EDGE_LABEL_T = 0.62       # where along the person->object edge the predicate block sits


def panel_height(n_objects: int, max_rows: int) -> float:
    """Vertical extent in data units that fits ``n_objects`` without collisions.

    Consecutive predicate blocks sit ``EDGE_LABEL_T`` of the way along their
    edges, so they are that fraction of the object spacing apart; the spacing is
    grown until the tallest block fits between two of them.  The figure is then
    scaled by the same amount, which keeps the type size constant instead of
    squeezing more rows into a fixed canvas.
    """
    block = PRED_STEP * max(max_rows, 1) + 0.35
    spacing = max(1.05, block / EDGE_LABEL_T)
    return max(10.0, 2.9 + spacing * max(n_objects - 1, 0) + 1.2)


def _graph_panel(ax, title, subtitle, objects, edges, height, note=""):
    """``objects``: [(key, display, colour)]; ``edges``: {key: [(predicate, colour, style)]}."""
    ax.set_xlim(0, 10); ax.set_ylim(0, height); ax.axis("off")
    ax.text(5, height - 0.28, title, fontsize=8.2, fontweight="bold", color=INK,
            ha="center", va="center")
    ax.text(5, height - 0.78, subtitle, fontsize=6.2, color="#516271", ha="center", va="top",
            linespacing=1.35)
    n = max(len(objects), 1)
    ys = np.linspace(height - 1.9, 1.0, n) if n > 1 else np.array([height / 2])
    py = float(ys.mean())
    ax.add_patch(FancyBboxPatch((0.25, py - 0.42), 1.9, 0.84,
                                boxstyle="round,pad=0,rounding_size=0.2", linewidth=1.0,
                                edgecolor=PERSON, facecolor="#EDF1F4", zorder=3))
    ax.text(1.2, py, "person", fontsize=7.4, fontweight="bold", color=INK, ha="center", va="center", zorder=4)
    for (key, display, colour), y in zip(objects, ys):
        ax.add_patch(FancyBboxPatch((7.35, y - 0.4), 2.55, 0.8,
                                    boxstyle="round,pad=0,rounding_size=0.2", linewidth=1.1,
                                    edgecolor=colour, facecolor="white", zorder=3))
        ax.text(8.62, y, display, fontsize=6.4, color=colour, ha="center", va="center", zorder=4)
        ax.plot([2.2, 7.3], [py, y], color="#C8D0D6", lw=0.8, zorder=1)
        preds = edges.get(key, [])
        if not preds:
            ax.text(4.75, py + EDGE_LABEL_T * (y - py), "(no relation predicted)",
                    fontsize=5.8, color=OUTCOME["absent"],
                    style="italic", ha="center", va="center", zorder=5,
                    bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="none"))
            continue
        step = PRED_STEP
        y0 = py + EDGE_LABEL_T * (y - py) + step * (len(preds) - 1) / 2
        for i, (name, colour_e, style) in enumerate(preds):
            ax.text(4.75, y0 - i * step, name, fontsize=5.8, color=colour_e, ha="center", va="center",
                    style=style, zorder=5,
                    bbox=dict(boxstyle="square,pad=0.16", facecolor="white", alpha=0.92,
                              edgecolor="none"))
    if note:
        ax.text(5, 0.22, note, fontsize=5.9, color="#516271", ha="center", va="center")


def figure_scenegraph(video: WorldBBoxVideo, views: Dict[str, MethodView], specs: Sequence[MethodSpec],
                      t: int, mode: str, k_at: int, iou_thr: float,
                      category: Optional[CategorySpec] = None) -> plt.Figure:
    fr = video.frames[t]
    gt_rel = gt_relations_of(fr)
    order = [o.label for o in fr.objects]
    labels = list(dict.fromkeys(order))
    n_panels = 1 + len(specs)
    # every panel of one figure shares a height, sized by the busiest edge anywhere
    panels: List[Tuple[str, str, list, dict, str]] = []
    observed = {o.label: o.observed for o in fr.objects}
    gt_objects = [(l, l, GT_OBSERVED if observed.get(l, True) else GT_UNOBSERVED) for l in labels]
    gt_edges = {l: [(p, INK, "normal") for p in gt_rel.get(l, [])] for l in labels}
    panels.append(("Ground truth", "worldbbox annotation", gt_objects, gt_edges,
                   "purple = unobserved in this frame"))
    for i, spec in enumerate(specs, start=1):
        view = views[spec.key]
        hits = view.recalled.get(fr.file) or set()
        said = view.said.get(fr.file, {})
        objects, edges = [], {}
        for l in labels:
            oc, got, total = outcome_for(view, fr.file, l, gt_rel.get(l, []))
            objects.append((l, f"{l}  {got}/{total}", OUTCOME[oc]))
            rows: List[Tuple[str, str, str]] = []
            for p in said.get(l, []):
                rows.append((p, OUTCOME["hit"] if (l, p) in hits else OUTCOME["miss"], "normal"))
            for p in gt_rel.get(l, []):
                if (l, p) not in hits and p not in said.get(l, []):
                    rows.append((f"{p} (missed)", OUTCOME["absent"], "italic"))
            edges[l] = rows
        sub = (f"track {spec.track} • {spec.backbone}"
               "\n" + matcher_note(spec, mode, iou_thr))
        note = "" if view.available else "no prediction file for this video"
        panels.append((spec.label, sub, objects, edges, note))
    max_rows = max((len(rows) for _, _, _, edges, _ in panels for rows in edges.values()),
                   default=1)
    height = panel_height(len(labels), max_rows)
    unit_in = 0.43
    # The colour key rides on a second title line: a category figure has half the
    # panels of the old mixed one, so a single line no longer fits the canvas.
    fig_h = 0.80 + height * unit_in
    fig = plt.figure(figsize=(2.55 * n_panels, fig_h), facecolor="white")
    top = 1 - 0.68 / fig_h
    gs = fig.add_gridspec(1, n_panels, left=0.005, right=0.995, top=top, bottom=0.012, wspace=0.03)
    for i, (title, sub, objects, edges, note) in enumerate(panels):
        _graph_panel(fig.add_subplot(gs[0, i]), title, sub, objects, edges, height, note=note)
    cat = f"{category.label}  •  " if category else ""
    fig.suptitle(f"{video.video_id}  •  frame {fr.frame_num:06d}  •  {cat}{mode}  •  scene graph\n"
                 f"green = recalled at R@{k_at}, red = predicted but not matched, "
                 f"grey italic = ground-truth predicate the method missed",
                 fontsize=8.6, fontweight="bold", color=INK, y=1 - 0.06 / fig_h,
                 va="top", linespacing=1.5)
    return fig


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def save(fig: plt.Figure, name: str, out_dir: Path, formats: Sequence[str], dpi: int) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        target = out_dir / f"{name}.{fmt}"
        fig.savefig(target, dpi=dpi, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        written.append(str(target))
        print(target)
    plt.close(fig)
    return written


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", nargs="+", required=True, help="worldbbox test video id(s), e.g. 12XD3")
    p.add_argument("--category", default=None, choices=tuple(CATEGORIES),
                   help="which style of experiment this figure is about: 'training' (the "
                        "supervised lineage) or 'mllm' (the unlocalized and localized MLLM "
                        "tracks).  Sets the default --methods, puts the category key in every "
                        "filename and records it in the meta, so the two sets never collide. "
                        "Ground truth is drawn in both, so either figure stands alone.")
    p.add_argument("--methods", nargs="+", default=None,
                   help=f"method keys, any of: {', '.join(METHODS)}.  Defaults to the "
                        f"--category method list, or to the mixed legacy set when no "
                        f"category is given")
    p.add_argument("--mode", default="sgdet", choices=("predcls", "sgdet"))
    p.add_argument("--figures", nargs="+", default=["overlay2d", "scenegraph", "scene3d"],
                   choices=("overlay2d", "scenegraph", "scene3d"),
                   help="which figure families to write; the 2-D montage is a raster of real "
                        "frames and is normally written as PNG only, the other two as PDF")
    p.add_argument("--frames", type=int, default=4, help="number of keyframes for the 2-D overlay")
    p.add_argument("--frame-numbers", nargs="*", type=int, default=[],
                   help="explicit annotated frame numbers instead of the automatic choice")
    p.add_argument("--graph-frames", type=int, default=2, help="how many keyframes get a graph figure")
    p.add_argument("--scene3d-frames", type=int, default=2, help="how many keyframes get a 3-D figure")
    p.add_argument("--k", type=int, default=20, help="R@K used for the correct/missed colouring")
    p.add_argument("--constraint", default="with", choices=("with", "no"))
    p.add_argument("--iou3d", type=float, default=0.15, help="3-D IoU for localized sgdet matching")
    p.add_argument("--cloud", default="dense", choices=("dense", "cache"),
                   help="dense: re-accumulate the Pi-3 scene at 1.5 cm; "
                        "cache: reuse the 3 cm BEV cache the tracks were prompted with")
    p.add_argument("--cloud-points", type=int, default=90000,
                   help="points kept per 3-D panel after the cloud is built")
    p.add_argument("--output-dir", type=Path,
                   default=REPO / "outputs" / "paper_figures" / "qualitative")
    p.add_argument("--cache-dir", type=Path, default=None,
                   help="where per-video slices of the supervised dumps are kept "
                        "(default: <outputs.dumps>/qualitative/cache)")
    p.add_argument("--formats", nargs="+", choices=("pdf", "svg", "png"), default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--config", default=None, help="path to the mllm YAML config")
    return p


def scene_cloud(video: WorldBBoxVideo, cfg: dict, dense: bool) -> Tuple[Dict[str, np.ndarray], str]:
    """The coloured Pi-3 cloud in the canonical frame.

    ``cache`` reuses the BEV cache under ``paths.worldbbox.caches.bev`` (voxel
    3 cm, every second pixel), which is what Track A/B were prompted with.
    ``dense`` re-accumulates the same scene at 1.5 cm from every pixel, which
    is what a printed figure needs; it reads the Pi-3 ``.npz`` and is CPU only.
    """
    from lib.mllm.tools.bev import build_cloud, get_bev

    bev_dir = get_path(cfg, "worldbbox.caches.bev")
    if not dense:
        _, _, cloud = get_bev(video, bev_dir)
        return cloud, f"BEV cache {bev_dir}"
    cloud = build_cloud(video, conf_min=0.1, stride=1, voxel=0.015, max_frames=60)
    return cloud, f"rebuilt from {video.pi3_path} (stride 1, voxel 1.5 cm)"


def run_video(video_id: str, args, cfg: dict, test_set: WorldBBoxTestSet) -> Dict[str, Any]:

    category = CATEGORIES[args.category] if args.category else None
    # every figure of a run carries its category in the name, so the training-based
    # set and the MLLM set of the same video/mode never overwrite one another
    stem = f"{video_id}_{category.key}_{args.mode}" if category else f"{video_id}_{args.mode}"
    specs = [METHODS[k] for k in args.methods]
    video = test_set.load(video_id)
    cache_dir = args.cache_dir or Path(get_path(cfg, "outputs.dumps")) / "qualitative" / "cache"
    out_dir = args.output_dir / video_id
    views = build_views(video, specs, args.mode, cfg, Path(cache_dir), args.k, args.iou3d,
                        args.constraint)
    frame_idx = pick_frames(video, args.frames, args.frame_numbers)
    meta: Dict[str, Any] = {
        "video_id": video_id, "mode": args.mode, "k": args.k, "constraint": args.constraint,
        "category": category.key if category else None,
        "category_label": category.label if category else None,
        "iou3d": args.iou3d, "n_annotated_frames": len(video.frames),
        "objects": video.video_objects(),
        "keyframes": [video.frames[t].frame_num for t in frame_idx],
        "methods": {s.key: {"label": s.label, "track": s.track, "backbone": s.backbone,
                            "source": views[s.key].source, "available": views[s.key].available,
                            "emits_3d": s.emits_3d} for s in specs},
        "figures": {},
    }

    if "overlay2d" in args.figures:
        fig = figure_overlay2d(video, views, specs, frame_idx, args.mode, args.k, cfg, args.iou3d,
                               category=category)
        meta["figures"]["overlay2d"] = save(fig, f"{stem}_overlay2d", out_dir,
                                            args.formats, args.dpi)

    for t in (frame_idx[:args.graph_frames] if "scenegraph" in args.figures else []):
        fig = figure_scenegraph(video, views, specs, t, args.mode, args.k, args.iou3d,
                                category=category)
        name = f"{stem}_scenegraph_f{video.frames[t].frame_num:06d}"
        meta["figures"].setdefault("scenegraph", []).extend(
            save(fig, name, out_dir, args.formats, args.dpi))

    scene3d_idx = frame_idx[:args.scene3d_frames] if "scene3d" in args.figures else []
    cloud, cloud_source = (scene_cloud(video, cfg, args.cloud == "dense")
                           if scene3d_idx else ({}, "not built"))
    meta["cloud_source"] = cloud_source
    for t in scene3d_idx:
        fig, m3 = figure_scene3d(video, views, specs, t, cloud, args.cloud_points,
                                 args.iou3d, args.mode, category=category)
        name = f"{stem}_scene3d_f{video.frames[t].frame_num:06d}"
        meta["figures"].setdefault("scene3d", []).extend(
            save(fig, name, out_dir, args.formats, args.dpi))
        meta.setdefault("scene3d_detail", []).append(m3)

    out_dir.mkdir(parents=True, exist_ok=True)
    meta["figure_families"] = list(args.figures)
    # a run may write only some families (raster montage and vector panels are
    # normally two calls with different formats); keep what the other call wrote
    previous = out_dir / f"{stem}_meta.json"
    if previous.exists():
        try:
            old = json.loads(previous.read_text(encoding="utf-8"))
            for family, value in (old.get("figures") or {}).items():
                merged = list(meta["figures"].get(family, [])) + list(value)
                meta["figures"][family] = sorted(dict.fromkeys(merged))
            meta["figure_families"] = sorted(set(meta["figure_families"])
                                             | set(old.get("figure_families") or []))
            if not meta.get("scene3d_detail") and old.get("scene3d_detail"):
                meta["scene3d_detail"] = old["scene3d_detail"]
        except Exception:
            pass
    (out_dir / f"{stem}_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(out_dir / f"{stem}_meta.json")
    return meta


def main() -> None:
    args = build_parser().parse_args()
    if args.methods is None:
        args.methods = (list(CATEGORIES[args.category].methods) if args.category
                        else list(DEFAULT_METHODS))
    unknown = [k for k in args.methods if k not in METHODS]
    if unknown:
        raise SystemExit(f"unknown method(s) {unknown}; choose from {sorted(METHODS)}")
    cfg = load_config(args.config)
    test_set = WorldBBoxTestSet(cfg)
    for video_id in args.video:
        run_video(video_id, args, cfg, test_set)


if __name__ == "__main__":
    main()
