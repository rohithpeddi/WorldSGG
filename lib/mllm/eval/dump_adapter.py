"""
MLLM run -> WorldSGG dump records (B2 of docs/EXECUTION_PLAN_WORLDBBOX.md §3)
==============================================================================

Turns any MLLM output for the worldbbox test set into per-frame records in the
exact format of ``tools/dump_predictions.py`` so that
``lib/supervised/evaluation_recall.py::evaluate_wsgg_video`` (R/mR@K, with/no
constraint) and ``evaluation_recall_bucketed.py`` (OO/OU buckets) run unchanged,
plus a few extra keys for the 3D regimes of ``lib/mllm/eval/recall3d.py``.

Slot layout per frame (N_max = 1 + #objects):
    slot 0            person (always observed)
    slots 1..n_gt     GT objects of the frame (worldbbox annotation order,
                      unique by short label -- same rule as WorldAG)
    slots n_gt+1..    (sgdet only) objects the model predicted that are not in
                      the frame's GT

Standard keys (dump_predictions format): attention/spatial/contacting_distribution
(K,C), gt_attention (K,), gt_spatial (K,6), gt_contacting (K,17), pair_valid,
person_idx, object_idx (K,), object_classes (N,), bboxes_2d / gt_bboxes_2d (N,4)
in **Pi-3 space** (WorldAG rule; objects without a 2D box get the full-frame
placeholder for both GT and pred so PredCls IoU is 1), valid_mask,
visibility_mask (N,), pred_labels, pred_scores, video_id, frame_t, is_last.

Extra keys: pred_pair_valid (K,) -- pairs the model actually emitted;
gt_corners / pred_corners (N,8,3) canonical frame (zeros when unknown);
has_pred_3d (N,); slot_labels (list of short labels); frame_file.

Input formats accepted by :func:`prediction_lookup`:
  * vendored baseline PKL (``frames[frame_file]["predictions"]`` with scored
    dicts ``{"label", "yes_prob"}`` or plain strings), predcls and sgdet;
  * Track A/B JSON-like dict ``{"frames": {frame_file: {"objects": {label:
    {"attention": [...], "spatial": [...], "contacting": [...],
    "obb": {"center", "size", "yaw"} | "corners": [...]}}}}}``.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from lib.mllm.data.geometry import obb_to_corners
from lib.mllm.data.worldbbox import (
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, SPATIAL_RELATIONSHIPS,
    NAME_TO_IDX, WorldBBoxVideo, to_short,
)

_ATT = {r: i for i, r in enumerate(ATTENTION_RELATIONSHIPS)}
_SPA = {r: i for i, r in enumerate(SPATIAL_RELATIONSHIPS)}
_CON = {r: i for i, r in enumerate(CONTACTING_RELATIONSHIPS)}
N_ATT, N_SPA, N_CON = len(_ATT), len(_SPA), len(_CON)


def _norm_rel(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_")


def _scored(value, multi: bool) -> List[Tuple[str, float]]:
    """Normalise a vendored ``attention``/``contacting``/``spatial`` field."""
    out: List[Tuple[str, float]] = []
    if value is None:
        return out
    items = value if isinstance(value, list) else [value]
    for it in items:
        if isinstance(it, dict):
            lab, p = it.get("label"), it.get("yes_prob", 1.0)
        elif isinstance(it, (tuple, list)) and len(it) == 2:
            lab, p = it
        else:
            lab, p = it, 1.0
        if lab is None:
            continue
        lab = _norm_rel(lab)
        if lab in ("unknown", ""):
            continue
        try:
            p = float(p)
        except (TypeError, ValueError):
            p = 1.0
        out.append((lab, p))
        if not multi:
            break
    return out


def load_run_pkl(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def prediction_lookup(record: Dict[str, Any], objects_key: str = "objects") -> Dict[str, Dict[str, Dict[str, Any]]]:
    """``{frame_file: {short_label: {"attention": [(lab,p)], "spatial": [...],
    "contacting": [...], "corners": (8,3)|None, "score": float}}}``.
    ``objects_key``: which per-frame dict to read for Track outputs ("objects" =
    final answer, "objects_pre" = Track B's first proposal, the -critic arm)."""
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    frames = record.get("frames", {}) or {}
    obj_scores = ((record.get("estimation_meta") or {}).get("object_scores") or {})
    for fkey, fdata in frames.items():
        ffile = Path(fkey).name
        if not ffile.endswith(".png"):
            ffile = f"{ffile}.png"
        entry: Dict[str, Dict[str, Any]] = {}
        preds = fdata.get("predictions")
        if isinstance(preds, list):                      # vendored baseline format
            for p in preds:
                lab = to_short(str(p.get("object", "")).strip().lower())
                if not lab:
                    continue
                entry[lab] = {
                    "attention": _scored(p.get("attention"), multi=False),
                    "spatial": _scored(p.get("spatial"), multi=True),
                    "contacting": _scored(p.get("contacting"), multi=True),
                    "corners": _corners_of(p),
                    "score": float((obj_scores.get(lab) or {}).get("yes_prob", 1.0)),
                }
        else:                                            # Track A/B format
            objs = fdata.get(objects_key, fdata.get("objects", {})) or {}
            if isinstance(objs, list):
                objs = {o.get("label"): o for o in objs}
            for lab, p in objs.items():
                lab = to_short(str(lab).strip().lower())
                entry[lab] = {
                    "attention": _scored(p.get("attention"), multi=False),
                    "spatial": _scored(p.get("spatial"), multi=True),
                    "contacting": _scored(p.get("contacting"), multi=True),
                    "corners": _corners_of(p),
                    "score": float(p.get("score", 1.0)),
                }
        out[ffile] = entry
    return out


def _corners_of(p: Dict[str, Any]) -> Optional[np.ndarray]:
    c = p.get("corners", p.get("corners_final"))
    if c is not None:
        c = np.asarray(c, dtype=np.float64)
        return c.reshape(8, 3) if c.size == 24 else None
    obb = p.get("obb")
    if isinstance(obb, dict) and all(k in obb for k in ("center", "size")):
        try:
            return obb_to_corners(obb["center"], obb["size"], float(obb.get("yaw", 0.0)))
        except Exception:
            return None
    return None


def _dist(pairs: List[Tuple[str, float]], index: Dict[str, int], n: int) -> np.ndarray:
    d = np.zeros(n, dtype=np.float32)
    for lab, p in pairs:
        i = index.get(lab)
        if i is not None:
            d[i] = max(d[i], p)
    return d


def build_records(video: WorldBBoxVideo, preds: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
                  mode: str = "predcls", person_pred_corners: Optional[Dict[str, np.ndarray]] = None
                  ) -> List[Dict[str, Any]]:
    """One dump record per annotated frame of ``video``."""
    W, H = video.pi3_size if video.pi3_path.exists() else video.orig_size
    sx, sy = video.bbox_scale if video.pi3_path.exists() else (1.0, 1.0)
    placeholder = np.array([0.0, 0.0, float(W), float(H)], dtype=np.float32)
    preds = preds or {}
    frames = video.frames
    T = len(frames)
    records: List[Dict[str, Any]] = []
    for t, fr in enumerate(frames):
        fp = preds.get(fr.file, {})
        # ---- slots ----
        labels: List[str] = ["person"]
        gt_objs = []
        seen = set()
        for o in fr.objects:
            if o.label in seen:
                continue
            seen.add(o.label)
            gt_objs.append(o)
            labels.append(o.label)
        n_gt = len(gt_objs)
        extra = []
        if mode == "sgdet":
            for lab in sorted(fp.keys()):
                if lab not in seen and lab != "person" and lab in NAME_TO_IDX:
                    extra.append(lab)
                    labels.append(lab)
        N = len(labels)
        object_classes = np.array([NAME_TO_IDX.get(l, 0) for l in labels], dtype=np.int64)
        valid_mask = np.ones(N, dtype=bool)
        visibility = np.ones(N, dtype=bool)
        gt_boxes = np.tile(placeholder, (N, 1)).astype(np.float32)
        gt_corners = np.zeros((N, 8, 3), dtype=np.float32)
        pred_corners = np.zeros((N, 8, 3), dtype=np.float32)
        has_pred_3d = np.zeros(N, dtype=bool)
        pred_scores = np.ones(N, dtype=np.float32)
        if fr.person_bbox_2d is not None:
            gt_boxes[0] = np.array([fr.person_bbox_2d[0] * sx, fr.person_bbox_2d[1] * sy,
                                    fr.person_bbox_2d[2] * sx, fr.person_bbox_2d[3] * sy])
        if fr.person_corners_final is not None:
            gt_corners[0] = fr.person_corners_final
        pc = (person_pred_corners or {}).get(fr.file)
        if pc is not None:
            pred_corners[0] = pc
            has_pred_3d[0] = True
        elif mode == "predcls" and fr.person_corners_final is not None:
            pred_corners[0] = fr.person_corners_final
            has_pred_3d[0] = True
        for i, o in enumerate(gt_objs, start=1):
            visibility[i] = o.observed
            if o.bbox_2d is not None:
                gt_boxes[i] = np.array([o.bbox_2d[0] * sx, o.bbox_2d[1] * sy,
                                        o.bbox_2d[2] * sx, o.bbox_2d[3] * sy])
            if o.corners_final is not None:
                gt_corners[i] = o.corners_final
        # predicted 3D boxes (any slot the model emitted); predcls falls back to GT corners
        for i, lab in enumerate(labels):
            if i == 0:
                continue
            p = fp.get(lab)
            if p is not None and p.get("corners") is not None:
                pred_corners[i] = p["corners"]
                has_pred_3d[i] = True
            elif mode == "predcls" and i <= n_gt and gt_objs[i - 1].corners_final is not None:
                pred_corners[i] = gt_objs[i - 1].corners_final
                has_pred_3d[i] = True
            if p is not None:
                pred_scores[i] = float(p.get("score", 1.0))
        # ---- pairs ----
        K = N - 1
        person_idx = np.zeros(K, dtype=np.int64)
        object_idx = np.arange(1, N, dtype=np.int64)
        pair_valid = np.zeros(K, dtype=bool)
        pred_pair_valid = np.zeros(K, dtype=bool)
        gt_att = np.zeros(K, dtype=np.int64)
        gt_spa = np.zeros((K, N_SPA), dtype=np.float32)
        gt_con = np.zeros((K, N_CON), dtype=np.float32)
        att_d = np.zeros((K, N_ATT), dtype=np.float32)
        spa_d = np.zeros((K, N_SPA), dtype=np.float32)
        con_d = np.zeros((K, N_CON), dtype=np.float32)
        for k in range(K):
            lab = labels[k + 1]
            if k < n_gt:
                o = gt_objs[k]
                pair_valid[k] = True
                gt_att[k] = next((_ATT[r] for r in o.attention if r in _ATT), 0)
                for r in o.spatial:
                    if r in _SPA:
                        gt_spa[k, _SPA[r]] = 1.0
                for r in o.contacting:
                    if r in _CON:
                        gt_con[k, _CON[r]] = 1.0
            p = fp.get(lab)
            if p is not None:
                pred_pair_valid[k] = True
                att_d[k] = _dist(p["attention"], _ATT, N_ATT)
                spa_d[k] = _dist(p["spatial"], _SPA, N_SPA)
                con_d[k] = _dist(p["contacting"], _CON, N_CON)
        records.append({
            "video_id": video.vid_mp4, "frame_file": fr.file, "frame_t": t, "is_last": t == T - 1,
            "attention_distribution": att_d, "spatial_distribution": spa_d,
            "contacting_distribution": con_d,
            "gt_attention": gt_att, "gt_spatial": gt_spa, "gt_contacting": gt_con,
            "pair_valid": pair_valid, "pred_pair_valid": pred_pair_valid,
            "person_idx": person_idx, "object_idx": object_idx,
            "object_classes": object_classes, "bboxes_2d": gt_boxes.copy(),
            "gt_bboxes_2d": gt_boxes, "valid_mask": valid_mask, "visibility_mask": visibility,
            "pred_labels": object_classes.copy(), "pred_scores": pred_scores,
            "gt_corners": gt_corners, "pred_corners": pred_corners, "has_pred_3d": has_pred_3d,
            "slot_labels": labels, "n_gt_objects": n_gt,
        })
    return records
