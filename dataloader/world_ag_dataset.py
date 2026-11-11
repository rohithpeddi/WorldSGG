"""
World Scene Graph Dataset
==========================

PyTorch Dataset that loads pre-extracted ROI features and pre-built
world4D relationship annotations per video.

Two PKL sources per video:

1. **Features**:
   ``<data_path>/features/roi_features/<mode>/<feature_model>/<phase>/<video>.pkl``
   — ROI features, bboxes, labels, pair_indices, union_features.

2. **Annotations**:
   ``<data_path>/world4d_rel_annotations/<phase>/<video>.pkl``
   — GT+RAG relationships, 3D corners, camera poses, visibility.

A video is valid only if present in **both** directories.

All outputs are pre-padded tensors (T, N_max, ...) and (T, K_max, ...),
eliminating ragged list formats that previously caused for-loops in
models and loss functions.

Usage::

    from dataloader.world_ag_dataset import WorldAG, world_collate_fn
    from torch.utils.data import DataLoader

    dataset = WorldAG(
        phase="train",
        data_path="/data/rohith/ag",
        mode="predcls",
        feature_model="dinov2b",
    )
    loader = DataLoader(dataset, batch_size=1, collate_fn=world_collate_fn)

    for batch in loader:
        print(batch["video_id"])
        print(batch["visual_features"].shape)  # (T, N_max, D)
        print(batch["gt_spatial"].shape)        # (T, K_max, 6)
"""

import logging
import math
import os
import pickle
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relationship label vocabularies (canonical ordering)
# ---------------------------------------------------------------------------

ATTENTION_RELATIONSHIPS = ["looking_at", "not_looking_at", "unsure"]
SPATIAL_RELATIONSHIPS = [
    "above", "beneath", "in_front_of", "behind", "on_the_side_of", "in",
]
CONTACTING_RELATIONSHIPS = [
    "carrying", "covered_by", "drinking_from", "eating",
    "have_it_on_the_back", "holding", "leaning_on", "lying_on",
    "not_contacting", "other_relationship", "sitting_on", "standing_on",
    "touching", "twisting", "wearing", "wiping", "writing_on",
]

NUM_ATTENTION = len(ATTENTION_RELATIONSHIPS)    # 3
NUM_SPATIAL = len(SPATIAL_RELATIONSHIPS)         # 6
NUM_CONTACTING = len(CONTACTING_RELATIONSHIPS)  # 17

# Build reverse lookup dicts
_ATT_LABEL_TO_IDX = {r: i for i, r in enumerate(ATTENTION_RELATIONSHIPS)}
_SPA_LABEL_TO_IDX = {r: i for i, r in enumerate(SPATIAL_RELATIONSHIPS)}
_CON_LABEL_TO_IDX = {r: i for i, r in enumerate(CONTACTING_RELATIONSHIPS)}

# Short-form ↔ full AG name mappings (for matching between PKLs)
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet", "cup/glass/bottle": "cup",
    "paper/notebook": "paper", "sofa/couch": "sofa",
    "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}

# Object class list (AG vocabulary, 1-indexed; 0 = background)
OBJECT_CLASSES = [
    "__background__", "person", "bag", "bed", "blanket", "book", "box",
    "broom", "chair", "closet/cabinet", "clothes", "cup/glass/bottle",
    "dish", "door", "doorknob", "doorway", "floor", "food", "groceries",
    "laptop", "light", "medicine", "mirror", "paper/notebook",
    "phone/camera", "picture", "pillow", "refrigerator", "sandwich",
    "shelf", "shoe", "sofa/couch", "table", "television", "towel",
    "vacuum", "window",
]
NAME_TO_IDX = {name: idx for idx, name in enumerate(OBJECT_CLASSES) if idx > 0}
# Also register short forms
for _short, _full in LABEL_DENORMALIZE_MAP.items():
    NAME_TO_IDX[_short] = NAME_TO_IDX[_full]


SLOT_ALIGNMENTS = ("none", "per_video_class")
PERSON_CLASS_ID = 1

# Object inventory (rebuttal round 2, Nov 2026):
#   "feature" (historical) — a frame's objects are the entries of its feature PKL. Most unobserved
#              objects have no entry, so they get no slot, no relationship label and no prediction.
#   "full"     — every object gets a slot in every frame where it exists, and missing objects are
#              natural masks (unobserved slots: zero appearance feature, zero 2-D box, completed by
#              each method's own mechanism):
#                PredCls: every annotated object of the frame without a feature entry is appended
#                         (source "inventory", class + 3-D box from the annotation; PredCls supplies
#                         annotated objects and boxes).
#                SGDet:   no ground truth is used for the inventory. Every object class established in the
#                         video (one detection scoring >= held_min_score, default 0.2; its rank-0 identity)
#                         is appended in each frame where it is not detected (source "held"), with the 3-D
#                         box of its nearest detected frame.
#              Each appended entry gets one person-object pair; its labels come from the annotation
#              like every other pair. In the training split, appended entries without an annotated
#              object of their class in that frame get no pair (no supervision without a label).
INVENTORIES = ("feature", "full")
INVENTORY_SOURCE = "inventory"
HELD_SOURCE = "held"
UNOBSERVED_SOURCES = ("rag", "gdino", "correction", INVENTORY_SOURCE, HELD_SOURCE)

# Per-node tensors built by WorldAG._build_frame_tensors (permuted by slot alignment)
_NODE_KEYS = ("visual_features", "corners", "gt_corners", "bboxes_2d", "gt_bboxes_2d",
              "valid_mask", "visibility_mask", "object_classes", "gt_box_valid")
# Per-edge lists built by WorldAG._build_frame_tensors (filtered together)
_EDGE_KEYS = ("person_indices", "object_indices", "att_labels", "spa_multi_hot",
              "con_multi_hot", "pair_sources", "valid_raw_pair_indices")


def _to_short(label: str) -> str:
    """Full AG name → short form. E.g. 'closet/cabinet' → 'closet'."""
    return LABEL_NORMALIZE_MAP.get(label, label)


def _compute_target_size(orig_w: int, orig_h: int, pixel_limit: int = 255000,
                         patch_size: int = 14):
    """Pi-3 resize target (identical to the ROI feature extractor /
    ag_dataset_3d.py): aspect-preserving, multiples of patch_size,
    total pixels <= pixel_limit."""
    scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1
    w_target, h_target = orig_w * scale, orig_h * scale
    k = round(w_target / patch_size)
    m = round(h_target / patch_size)
    while (k * patch_size) * (m * patch_size) > pixel_limit:
        if k / m > w_target / h_target:
            k -= 1
        else:
            m -= 1
    return max(1, k) * patch_size, max(1, m) * patch_size


def _png_size(path: str):
    """(w, h) from a PNG header without decoding the image."""
    import struct
    with open(path, "rb") as f:
        head = f.read(24)
    if len(head) >= 24 and head[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", head[16:24])
        return int(w), int(h)
    from PIL import Image  # fallback for non-PNG frames
    with Image.open(path) as im:
        return im.size


def _multi_hot(rel_strings: List[str], label_to_idx: Dict[str, int],
               num_classes: int) -> torch.Tensor:
    """Convert a list of relationship label strings to a multi-hot float tensor.

    Args:
        rel_strings: e.g. ["above", "in"]
        label_to_idx: maps label string → index
        num_classes: size of output tensor

    Returns:
        (num_classes,) float tensor with 1.0 at active indices.
    """
    target = torch.zeros(num_classes)
    for r in rel_strings:
        idx = label_to_idx.get(r, -1)
        if idx >= 0:
            target[idx] = 1.0
    return target


def _attention_label_to_idx(rel_strings: List[str]) -> int:
    """Convert attention relationship strings to a single class index.

    Attention is single-label: take the first valid label.
    """
    for r in rel_strings:
        idx = _ATT_LABEL_TO_IDX.get(r, -1)
        if idx >= 0:
            return idx
    return 0  # default to "looking_at"


# ---------------------------------------------------------------------------
# World Scene Graph Dataset
# ---------------------------------------------------------------------------

class WorldAG(Dataset):
    """
    PyTorch Dataset that loads ROI features and world4D relationship
    annotations per video, producing pre-padded tensors.

    Each ``__getitem__`` returns a dict with all tensors padded to
    ``(T, N_max, ...)`` for per-node data and ``(T, K_max, ...)`` for
    per-edge data, along with validity masks.
    """

    def __init__(
        self,
        phase: str,
        data_path: str,
        mode: str = "predcls",
        feature_model: str = "dinov2b",
        include_invisible: bool = True,
        max_objects: int = 64,
        annot_dir_name: str = "world4d_rel_annotations",
        visibility_from_annotation: Optional[bool] = None,
        slot_alignment: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        inventory: Optional[str] = None,
        held_min_score: Optional[float] = None,
    ):
        """
        Args:
            phase: "train" or "test"
            data_path: Root directory of Action Genome dataset
            mode: "predcls" or "sgdet"
            feature_model: Feature model directory name (e.g. "dinov2b")
            include_invisible: If True, include RAG-predicted objects
            max_objects: Maximum number of objects per frame (N_max cap)
            annot_dir_name: Annotation folder under ``data_path`` (e.g.
                "world4d_rel_annotations" or the WorldBBox release set
                "world4d_rel_annotations_worldbbox"; the loader reads
                ``<data_path>/<annot_dir_name>/<phase>/<video>.pkl``)
            visibility_from_annotation: If True (the historical behaviour),
                a slot's visibility flag is overwritten with the matched
                annotation object's ``visible`` flag. In SGDet that copies
                ground-truth visibility onto detections at test time. False
                keeps the feature-source rule only (``gt``/``detector`` =
                visible; ``rag``/``gdino``/``correction`` = unobserved).
                None reads the environment variable
                ``WSGG_VISIBILITY_FROM_ANNOTATION`` ("0" = False; default True).
            slot_alignment: ``"none"`` (the historical layout: slot n is the
                n-th entry of each frame's feature list, so a slot can hold a
                different object in every frame) or ``"per_video_class"``:
                every video gets one fixed slot per object identity, constant
                across frames, identity = (AG class id, rank among the
                frame's same-class entries). Rank 0 is the class's first
                entry in the frame (the one the class's pairs attach to);
                further same-class entries (SGDet duplicates) are ranked by
                detector score. The person (class 1, rank 0) is slot 0.
                Absent identities are ``valid_mask`` False in that frame.
                Pairs, labels, union features and every per-node tensor move
                with their entry, so each frame's content is unchanged; only
                the slot order differs. If a video has more identities than
                ``max_objects``, the lowest-priority duplicates (rank >= 1,
                lowest max score) are dropped (they carry no pairs).
                None reads the environment variable ``WSGG_SLOT_ALIGNMENT``
                (default "none").
            video_ids: optional list of video ids (with or without ".mp4");
                the split is restricted to these videos before the per-video
                frame check. None = every video of the split.
            inventory: ``"feature"`` (historical: the feature PKL entries) or
                ``"full"`` (every object gets a slot; see ``INVENTORIES``).
                ``"full"`` requires ``slot_alignment="per_video_class"`` so a
                missing object keeps its slot. None reads the environment
                variable ``WSGG_INVENTORY`` (default "feature").
            held_min_score: SGDet with inventory="full": an object class is
                held only once one of its detections in the video scores at
                least this much (default 0.2; env ``WSGG_HELD_MIN_SCORE``).
        """
        super().__init__()

        if inventory is None:
            inventory = os.environ.get("WSGG_INVENTORY", "feature") or "feature"
        if inventory not in INVENTORIES:
            raise ValueError(f"inventory must be one of {INVENTORIES}, got {inventory!r}")
        self._inventory = inventory
        if held_min_score is None:
            held_min_score = float(os.environ.get("WSGG_HELD_MIN_SCORE", "0.2"))
        self._held_min_score = float(held_min_score)

        if visibility_from_annotation is None:
            visibility_from_annotation = os.environ.get(
                "WSGG_VISIBILITY_FROM_ANNOTATION", "1") != "0"
        self._visibility_from_annotation = bool(visibility_from_annotation)
        if slot_alignment is None:
            slot_alignment = os.environ.get("WSGG_SLOT_ALIGNMENT", "none") or "none"
        if slot_alignment not in SLOT_ALIGNMENTS:
            raise ValueError(f"slot_alignment must be one of {SLOT_ALIGNMENTS}, got {slot_alignment!r}")
        self._slot_alignment = slot_alignment
        if self._inventory == "full" and slot_alignment != "per_video_class":
            raise ValueError('inventory="full" requires slot_alignment="per_video_class"')
        self._inventory_stats = Counter()
        self._video_ids = None
        if video_ids is not None:
            self._video_ids = {str(v).strip().replace(".mp4", "") for v in video_ids if str(v).strip()}
        self._phase = phase
        self._data_path = Path(data_path)
        self._mode = mode
        self._feature_model = feature_model
        self._include_invisible = include_invisible
        self._max_objects = max_objects
        self._annot_dir_name = annot_dir_name
        # Original frame size per video (for scaling annotation 2D boxes,
        # stored in original-frame pixels, into Pi-3 feature space)
        self._orig_size_cache: Dict[str, Optional[tuple]] = {}
        self._warned_no_frames = False

        # Directories
        self._feat_dir = (
            self._data_path / "features" / "roi_features"
            / mode / feature_model / phase
        )
        self._annot_dir = (
            self._data_path / annot_dir_name / phase
        )

        # Expose vocabularies for model construction
        self.attention_relationships = ATTENTION_RELATIONSHIPS
        self.spatial_relationships = SPATIAL_RELATIONSHIPS
        self.contacting_relationships = CONTACTING_RELATIONSHIPS
        self.object_classes = OBJECT_CLASSES

        # Build video list: intersection of feature & annotation PKLs
        self.video_list: List[str] = []
        self._build_video_list()

        logger.info(
            f"[WorldAG][{phase}] mode={mode}, features={feature_model}, "
            f"annotations={annot_dir_name}, {len(self.video_list)} videos, "
            f"visibility_from_annotation={self._visibility_from_annotation}"
            + (f", slot_alignment={self._slot_alignment}" if self._slot_alignment != "none" else "")
            + (f", inventory={self._inventory}" if self._inventory != "feature" else "")
            + (f", video_ids={len(self._video_ids)}" if self._video_ids is not None else "")
        )

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_video_list(self):
        """Discover videos present in both feature and annotation dirs,
        keeping only those with >= 2 common frames."""
        if not self._feat_dir.exists():
            raise FileNotFoundError(
                f"Feature directory not found: {self._feat_dir}"
            )
        if not self._annot_dir.exists():
            raise FileNotFoundError(
                f"Annotation directory not found: {self._annot_dir}"
            )

        feat_videos = {p.stem for p in self._feat_dir.glob("*.pkl")}
        # Annotation PKLs may be named <video_id>.pkl where video_id
        # includes ".mp4" suffix, so stem = "001YG.mp4"
        annot_videos = set()
        for p in self._annot_dir.glob("*.pkl"):
            name = p.name.replace(".pkl", "")
            annot_videos.add(name)

        # Intersection: try matching with and without .mp4 suffix
        candidates = set()
        for vid in feat_videos:
            if vid in annot_videos:
                candidates.add(vid)
            elif vid + ".mp4" in annot_videos:
                candidates.add(vid)

        if self._video_ids is not None:
            n_before = len(candidates)
            candidates = {v for v in candidates if v.replace(".mp4", "") in self._video_ids}
            n_absent = len(self._video_ids - {v.replace(".mp4", "") for v in candidates})
            logger.info(
                f"[WorldAG][{self._phase}] video_ids filter: {len(candidates)}/{n_before} "
                f"candidates kept; {n_absent} listed ids have no feature+annotation pair"
            )

        if not candidates:
            logger.warning(
                f"No common videos found!\n"
                f"  Features ({len(feat_videos)}): {list(feat_videos)[:5]}...\n"
                f"  Annotations ({len(annot_videos)}): {list(annot_videos)[:5]}..."
            )
            self.video_list = []
            return

        # Filter: keep only videos with >= 2 common frames (parallelized)
        def _check_video(vid):
            """Load both PKLs and count common frames. Returns (vid, n_common)."""
            try:
                feat_data = self._load_feature_pkl(vid)
                annot_data = self._load_annotation_pkl(vid)
                common_frames, _ = self._align_frames(feat_data, annot_data)
                return vid, len(common_frames)
            except Exception as e:
                logger.warning(f"Error checking video {vid}: {e}")
                return vid, 0

        valid = []
        skipped = 0
        num_workers = min(8, len(candidates))  # I/O-bound: threads are fine

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_check_video, vid): vid
                for vid in sorted(candidates)
            }
            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc=f"Validating {self._phase} videos",
            ):
                vid, n_common = future.result()
                if n_common >= 2:
                    valid.append(vid)
                else:
                    skipped += 1

        if skipped > 0:
            logger.info(
                f"Skipped {skipped} videos with < 2 common frames"
            )

        self.video_list = sorted(valid)

    # ------------------------------------------------------------------
    # PKL loading helpers
    # ------------------------------------------------------------------

    def _load_feature_pkl(self, video_id: str) -> Dict[str, Any]:
        """Load the ROI feature PKL for a video."""
        path = self._feat_dir / f"{video_id}.pkl"
        with open(path, "rb") as f:
            return pickle.load(f)

    def _load_annotation_pkl(self, video_id: str) -> Dict[str, Any]:
        """Load the world4D relationship annotation PKL for a video."""
        # Try with and without .mp4 suffix
        for suffix in ["", ".mp4"]:
            path = self._annot_dir / f"{video_id}{suffix}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    return pickle.load(f)
        raise FileNotFoundError(
            f"Annotation PKL not found for video: {video_id}"
        )

    # ------------------------------------------------------------------
    # Frame alignment
    # ------------------------------------------------------------------

    def _align_frames(
        self,
        feat_data: Dict[str, Any],
        annot_data: Dict[str, Any],
    ) -> List[str]:
        """Find common frames between feature and annotation PKLs.

        Returns sorted list of frame filenames present in both.
        """
        feat_frames = set(feat_data.get("frames", {}).keys())
        annot_frames_raw = set(annot_data.get("frames", {}).keys())

        # Annotation frame keys are "video_id/frame.png"; extract filename
        annot_frame_to_key = {}
        for key in annot_frames_raw:
            frame_file = key.split("/")[-1] if "/" in key else key
            annot_frame_to_key[frame_file] = key

        common = sorted(feat_frames & set(annot_frame_to_key.keys()))
        return common, annot_frame_to_key

    # ------------------------------------------------------------------
    # Per-frame tensor construction
    # ------------------------------------------------------------------

    def _bbox_scale(self, video_id: str, frame_file: str,
                    feat_frame: Dict[str, Any]):
        """(sx, sy) mapping annotation ``bbox_2d`` (original frame pixels)
        to the Pi-3 space of the feature PKL boxes. Target size comes from
        the feature PKL (``target_size``) or is recomputed; original size is
        read from the frame PNG header once per video."""
        if video_id not in self._orig_size_cache:
            size = None
            for sub in ("frames_annotated", "frames"):
                for suffix in (".mp4", ""):
                    p = self._data_path / sub / f"{video_id}{suffix}" / frame_file
                    if p.exists():
                        try:
                            size = _png_size(str(p))
                        except Exception:
                            size = None
                        break
                if size is not None:
                    break
            if size is None and not self._warned_no_frames:
                logger.warning(
                    f"[WorldAG] frame images not found under {self._data_path}/frames_annotated; "
                    f"annotation 2D boxes are used unscaled (gt_bboxes_2d may mismatch feature boxes)")
                self._warned_no_frames = True
            self._orig_size_cache[video_id] = size
        size = self._orig_size_cache[video_id]
        if size is None:
            return (1.0, 1.0)
        ow, oh = size
        ts = feat_frame.get("target_size", None)
        if ts is None:
            tw, th = _compute_target_size(ow, oh)
        else:
            tw, th = int(ts[0]), int(ts[1])
        return (tw / ow, th / oh)

    def _feat_target_size(self, video_id: str, frame_file: str, feat_frame: Dict[str, Any]):
        """(W, H) of the feature PKL's box space for one frame (stored
        ``target_size``, else recomputed like ``_bbox_scale``)."""
        ts = feat_frame.get("target_size", None)
        if ts is not None:
            return float(ts[0]), float(ts[1])
        self._bbox_scale(video_id, frame_file, feat_frame)   # fills the size cache
        size = self._orig_size_cache.get(video_id)
        if size is None:
            raise ValueError(f"no target_size and no frame image for {video_id}/{frame_file}")
        tw, th = _compute_target_size(*size)
        return float(tw), float(th)

    def _build_frame_tensors(
        self,
        feat_frame: Dict[str, Any],
        annot_frame: Dict[str, Any],
        max_N: int,
        bbox_scale=(1.0, 1.0),
    ):
        """Build padded tensors for a single frame.

        3D corners and object labels are sourced differently by mode:
          - **predcls**: GT corners from annotation ``corners_final``,
            GT labels from annotation-derived ``label_ids``.
          - **sgdet**: Detector-predicted corners from ``boxes_3d`` in the
            features PKL, detector-predicted labels from ``label_ids``.

        Pair indices always come from the features PKL, which stores
        GT pairs (predcls) or detector-matched + supply pairs (sgdet).

        Args:
            feat_frame: feature PKL entry for this frame
            annot_frame: annotation PKL entry for this frame
            max_N: N_max for padding

        Returns:
            Dict of node and edge tensors for this frame.
        """
        # --- Feature data ---
        roi_features = feat_frame.get("roi_features", np.zeros((0, 1024)))
        if isinstance(roi_features, np.ndarray):
            roi_features = torch.from_numpy(roi_features).float()
        feat_labels = feat_frame.get("labels", [])
        feat_label_ids = feat_frame.get("label_ids", [])
        feat_sources = feat_frame.get("sources", [])
        bboxes_xyxy = feat_frame.get("bboxes_xyxy", np.zeros((0, 4)))
        if isinstance(bboxes_xyxy, np.ndarray):
            bboxes_xyxy = torch.from_numpy(bboxes_xyxy).float()
        feat_pair_indices = feat_frame.get("pair_indices", [])

        N_feat = len(feat_labels)
        D = roi_features.shape[-1] if roi_features.numel() > 0 else 1024

        # --- Annotation data (always needed for relationship GT labels) ---
        person_info = annot_frame.get("person_info", {})
        object_info_list = annot_frame.get("object_info_list", [])

        # Build label→annotation map for matching (used by PredCls and
        # as fallback for SGDet supply boxes)
        annot_by_short_label = {}
        for obj in object_info_list:
            short = obj.get("label", _to_short(obj.get("class", "")))
            if short not in annot_by_short_label:
                annot_by_short_label[short] = obj

        # SGDet: build detection-position → GT annotation mapping via
        # assigned_labels.  assigned_labels[det_pos] is the GT class ID
        # that the IoU matcher assigned to each detection during feature
        # extraction.  We convert GT class ID → short label → annotation.
        # This is robust to unmatched GT objects (which detector_found_idx
        # skips, creating sparse ordering issues).
        det_pos_to_annot = {}  # detection position → annotation dict
        if self._mode == "sgdet":
            assigned_labels = feat_frame.get("assigned_labels", [])
            for det_pos, gt_class_id in enumerate(assigned_labels):
                gt_class_id = int(gt_class_id)
                if gt_class_id == 0:
                    # No GT match for this detection (unmatched)
                    continue
                elif gt_class_id == 1:
                    # Person class
                    det_pos_to_annot[det_pos] = person_info
                else:
                    # Object class: convert GT class ID to short label
                    if gt_class_id < len(OBJECT_CLASSES):
                        gt_label = OBJECT_CLASSES[gt_class_id]
                    else:
                        gt_label = ""
                    gt_short = _to_short(gt_label)
                    annot_obj = annot_by_short_label.get(gt_short)
                    if annot_obj is not None:
                        det_pos_to_annot[det_pos] = annot_obj

        # --- SGDet: load detector-predicted 3D boxes from features PKL ---
        feat_boxes_3d = None
        if self._mode == "sgdet":
            raw_3d = feat_frame.get("boxes_3d", None)
            if raw_3d is not None:
                if isinstance(raw_3d, np.ndarray):
                    feat_boxes_3d = torch.from_numpy(
                        raw_3d.astype(np.float32)
                    )
                else:
                    feat_boxes_3d = raw_3d.float()
                # Expected shape: (N_feat, 8, 3)

        # --- Build per-object arrays ---
        N = min(N_feat, max_N)

        visual_features = torch.zeros(max_N, D)
        corners = torch.zeros(max_N, 8, 3)
        gt_corners = torch.zeros(max_N, 8, 3)  # real GT 3D corners (FINAL space)
        bboxes_2d = torch.zeros(max_N, 4)
        gt_bboxes_2d = torch.zeros(max_N, 4)  # real GT annotation boxes
        valid_mask = torch.zeros(max_N, dtype=torch.bool)
        visibility_mask = torch.zeros(max_N, dtype=torch.bool)
        object_classes = torch.zeros(max_N, dtype=torch.long)
        # True where gt_corners is the annotated box of a *visible* object that this slot really is
        # (PredCls: the slot's own visible object; SGDet: an IoU-matched detection). Used by V2's 3-D loss
        # on masked items; never true for unobserved objects or label-fallback matches.
        gt_box_valid = torch.zeros(max_N, dtype=torch.bool)

        sx, sy = float(bbox_scale[0]), float(bbox_scale[1])
        box_scale = np.array([sx, sy, sx, sy], dtype=np.float32)

        for i in range(N):
            visual_features[i] = roi_features[i]
            if i < bboxes_xyxy.shape[0]:
                bboxes_2d[i] = bboxes_xyxy[i]
                # Default GT box = the feature box (detector / supply box in
                # sgdet); overridden below by the annotation box when the
                # object has one (unobserved objects have none).
                gt_bboxes_2d[i] = bboxes_xyxy[i]
            valid_mask[i] = True

            # Object class label (GT for predcls, detector for sgdet —
            # both stored in feat_label_ids by their extraction scripts)
            class_idx = feat_label_ids[i] if i < len(feat_label_ids) else 0
            object_classes[i] = class_idx

            # Source determines visibility
            src = feat_sources[i] if i < len(feat_sources) else "gt"
            is_visible = src not in UNOBSERVED_SOURCES
            visibility_mask[i] = is_visible

            # Skip invisible if requested
            if not is_visible and not self._include_invisible:
                valid_mask[i] = False
                continue
            # inventory="full", PredCls training: unobserved object never visible in the video (no box)
            if i in feat_frame.get("_drop_entries", ()):
                valid_mask[i] = False
                continue

            # --- 3D corners: mode-dependent source ---
            # Resolve annotation object for this position
            if self._mode == "sgdet":
                # SGDet: use detector_found_idx mapping first, then
                # fall back to label string matching for supply boxes
                annot_obj = det_pos_to_annot.get(i)
                if annot_obj is None:
                    label_str = feat_labels[i] if i < len(feat_labels) else ""
                    short_label = _to_short(label_str)
                    annot_obj = annot_by_short_label.get(short_label)
            else:
                label_str = feat_labels[i] if i < len(feat_labels) else ""
                short_label = _to_short(label_str)
                annot_obj = annot_by_short_label.get(short_label)

            # Store real GT annotation bbox (for SGDet localization eval)
            if annot_obj is not None:
                # PKLs store the 2D box under "bbox_2d" (legacy key "bbox"),
                # in ORIGINAL frame pixels -> scale into Pi-3 feature space
                gt_box = annot_obj.get("bbox_2d", annot_obj.get("bbox", None))
                if gt_box is not None:
                    gb = np.asarray(gt_box, dtype=np.float32).reshape(-1)
                    if gb.size >= 4:
                        gt_bboxes_2d[i] = torch.from_numpy(gb[:4] * box_scale)
                # Store real GT 3D corners (FINAL space, for 3D IoU eval)
                gt_c = annot_obj.get("corners_final", None)
                if gt_c is not None:
                    gt_c = np.asarray(gt_c, dtype=np.float32)
                    if gt_c.shape == (8, 3):
                        gt_corners[i] = torch.from_numpy(gt_c)
                        matched = self._mode != "sgdet" or i in det_pos_to_annot
                        if matched and is_visible and annot_obj.get("visible", True):
                            gt_box_valid[i] = True

            if self._mode == "sgdet":
                # SGDet: use detector-predicted 3D corners from features PKL
                if feat_boxes_3d is not None and i < feat_boxes_3d.shape[0]:
                    box3d = feat_boxes_3d[i]
                    if box3d.shape == (8, 3):
                        corners[i] = box3d
                else:
                    # Fallback: detector did not produce 3D for this object,
                    # try annotation corners (e.g. for GT-supplied boxes)
                    if annot_obj is not None:
                        c = annot_obj.get("corners_final", None)
                        if c is not None:
                            c = np.asarray(c, dtype=np.float32)
                            if c.shape == (8, 3):
                                corners[i] = torch.from_numpy(c)
            else:
                # PredCls: use GT 3D corners from annotation PKL
                over = feat_frame.get("_corners_override", {})
                if i in over:
                    # inventory="full", training: unobserved object -> box of its nearest visible frame;
                    # the annotation's unobserved box and 2-D box are not used anywhere
                    corners[i] = torch.from_numpy(np.asarray(over[i], dtype=np.float32))
                    gt_corners[i] = corners[i]
                    gt_bboxes_2d[i] = 0.0
                elif annot_obj is not None:
                    c = annot_obj.get("corners_final", None)
                    if c is not None:
                        c = np.asarray(c, dtype=np.float32)
                        if c.shape == (8, 3):
                            corners[i] = torch.from_numpy(c)

            # Update visibility from annotation if available (applies to both modes).
            # In SGDet this copies ground-truth visibility onto detections; the
            # flag ``visibility_from_annotation=False`` disables it.
            if (self._visibility_from_annotation and annot_obj is not None
                    and not annot_obj.get("visible", True)):
                visibility_mask[i] = False

        # --- Person GT bbox and GT 3D corners (slot 0) ---
        person_bbox = person_info.get("bbox_2d", person_info.get("person_bbox", None))
        if person_bbox is not None:
            pb = np.asarray(person_bbox, dtype=np.float32).reshape(-1)
            if pb.size >= 4:
                gt_bboxes_2d[0] = torch.from_numpy(pb[:4] * box_scale)
        person_gt_corners = person_info.get("corners_final", None)
        if person_gt_corners is not None:
            person_gt_corners = np.asarray(person_gt_corners, dtype=np.float32)
            if person_gt_corners.shape == (8, 3):
                gt_corners[0] = torch.from_numpy(person_gt_corners)

        # --- Person 3D corners (slot 0): mode-dependent ---
        if self._mode == "sgdet":
            # SGDet: person 3D from detector prediction (slot 0 in boxes_3d)
            if feat_boxes_3d is not None and feat_boxes_3d.shape[0] > 0:
                if feat_boxes_3d[0].shape == (8, 3):
                    corners[0] = feat_boxes_3d[0]
            else:
                # Fallback to GT person corners if detector didn't provide
                person_corners = person_info.get("corners_final", None)
                if person_corners is not None:
                    person_corners = np.asarray(person_corners, dtype=np.float32)
                    if person_corners.shape == (8, 3):
                        corners[0] = torch.from_numpy(person_corners)
        else:
            # PredCls: person 3D from GT annotation
            person_corners = person_info.get("corners_final", None)
            if person_corners is not None:
                person_corners = np.asarray(person_corners, dtype=np.float32)
                if person_corners.shape == (8, 3):
                    corners[0] = torch.from_numpy(person_corners)

        # --- Build per-edge arrays ---
        # pair_indices from the features PKL contains (person_label_id,
        # object_label_id) tuples — i.e. class IDs, NOT positional indices.
        # We must map them to positional indices in the feat arrays.
        label_id_to_pos = {}
        for i in range(N):
            lid = feat_label_ids[i] if i < len(feat_label_ids) else 0
            if lid not in label_id_to_pos:
                label_id_to_pos[lid] = i

        K = len(feat_pair_indices)

        person_indices = []
        object_indices = []
        att_labels = []
        spa_multi_hot = []
        con_multi_hot = []
        pair_sources = []
        valid_raw_pair_indices = []  # track which raw pair index each valid pair came from

        for raw_k, (p_label_id, o_label_id) in enumerate(feat_pair_indices):
            # Convert class IDs to positional indices
            p_pos = label_id_to_pos.get(p_label_id, None)
            o_pos = label_id_to_pos.get(o_label_id, None)

            if p_pos is None or o_pos is None:
                continue
            if p_pos >= max_N or o_pos >= max_N:
                continue
            if not valid_mask[p_pos] or not valid_mask[o_pos]:
                continue

            # Relationship GT labels always from annotation PKL.
            # For SGDet: use detector_found_idx mapping (IoU-based GT match)
            # instead of label string matching (detector may mislabel).
            if self._mode == "sgdet":
                annot_obj = det_pos_to_annot.get(o_pos)
                if annot_obj is None:
                    # Fallback for supply boxes: label string matching
                    obj_label = feat_labels[o_pos] if o_pos < len(feat_labels) else ""
                    short_label = _to_short(obj_label)
                    annot_obj = annot_by_short_label.get(short_label)
            else:
                obj_label = feat_labels[o_pos] if o_pos < len(feat_labels) else ""
                short_label = _to_short(obj_label)
                annot_obj = annot_by_short_label.get(short_label)

            # inventory="full", training split: an appended (inventory / held) entry with no
            # annotated object of its class in this frame has no label, so it gets no pair.
            # (inventory="full": also GroundingDINO entries of a class the annotation does not list)
            o_src = feat_sources[o_pos] if o_pos < len(feat_sources) else "gt"
            if (annot_obj is None and self._phase == "train" and self._inventory == "full"
                    and o_src in UNOBSERVED_SOURCES):
                continue

            person_indices.append(p_pos)
            object_indices.append(o_pos)
            valid_raw_pair_indices.append(raw_k)

            if annot_obj is not None:
                att_rel = annot_obj.get("attention_relationship", [])
                spa_rel = annot_obj.get("spatial_relationship", [])
                con_rel = annot_obj.get("contacting_relationship", [])
                obj_source = annot_obj.get("source", "gt")
            else:
                att_rel = []
                spa_rel = []
                con_rel = []
                obj_source = "gt"

            att_labels.append(_attention_label_to_idx(att_rel))
            spa_multi_hot.append(
                _multi_hot(spa_rel, _SPA_LABEL_TO_IDX, NUM_SPATIAL)
            )
            con_multi_hot.append(
                _multi_hot(con_rel, _CON_LABEL_TO_IDX, NUM_CONTACTING)
            )
            pair_sources.append(0 if obj_source == "gt" else 1)

        K_valid = len(person_indices)

        return {
            # Node tensors (max_N, ...)
            "visual_features": visual_features,
            "corners": corners,
            "gt_corners": gt_corners,
            "bboxes_2d": bboxes_2d,
            "gt_bboxes_2d": gt_bboxes_2d,
            "valid_mask": valid_mask,
            "visibility_mask": visibility_mask,
            "object_classes": object_classes,
            "gt_box_valid": gt_box_valid,
            # Edge data (lists, will be padded to K_max later)
            "person_indices": person_indices,
            "object_indices": object_indices,
            "att_labels": att_labels,
            "spa_multi_hot": spa_multi_hot,
            "con_multi_hot": con_multi_hot,
            "pair_sources": pair_sources,
            "valid_raw_pair_indices": valid_raw_pair_indices,
            "K_valid": K_valid,
        }

    # ------------------------------------------------------------------
    # Full object inventory (inventory="full")
    # ------------------------------------------------------------------

    @staticmethod
    def _append_entries(ff: Dict[str, Any], new: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Copy of feature-frame ``ff`` with entries ``new`` appended (each: label, label_id,
        source, box3d or None). Every per-entry array grows by len(new) rows (zero appearance
        feature, zero 2-D box, score 0, unassigned); each new entry gets one person-object pair
        (class ids, as in ``pair_indices``) with a zero union feature."""
        if not new:
            return ff
        out = dict(ff)
        k = len(new)
        n_old = len(ff.get("labels", []))
        roi = np.asarray(ff.get("roi_features", np.zeros((0, 1024), np.float32)))
        d = roi.shape[1] if roi.ndim == 2 else 1024
        out["roi_features"] = np.concatenate([roi.reshape(-1, d), np.zeros((k, d), roi.dtype if roi.size else np.float32)])
        bx = np.asarray(ff.get("bboxes_xyxy", np.zeros((0, 4), np.float32))).reshape(-1, 4)
        out["bboxes_xyxy"] = np.concatenate([bx, np.zeros((k, 4), bx.dtype if bx.size else np.float32)])
        out["labels"] = list(ff.get("labels", [])) + [e["label"] for e in new]
        out["label_ids"] = [int(x) for x in ff.get("label_ids", [])] + [int(e["label_id"]) for e in new]
        default_src = "detector" if "assigned_labels" in ff else "gt"
        out["sources"] = list(ff.get("sources", [default_src] * n_old)) + [e["source"] for e in new]
        for key in ("scores", "gdino_scores", "assigned_labels"):
            if ff.get(key) is not None:
                out[key] = list(np.asarray(ff[key]).reshape(-1)) + [0.0] * k
        if ff.get("boxes_3d") is not None:
            b3 = np.asarray(ff["boxes_3d"], dtype=np.float32).reshape(-1, 8, 3)
            add = np.stack([np.asarray(e["box3d"], np.float32) if e.get("box3d") is not None
                            else np.zeros((8, 3), np.float32) for e in new])
            out["boxes_3d"] = np.concatenate([b3, add])
        out["pair_indices"] = [tuple(int(v) for v in p) for p in ff.get("pair_indices", [])] + \
                              [(PERSON_CLASS_ID, int(e["label_id"])) for e in new]
        if ff.get("union_features") is not None:
            uf = np.asarray(ff["union_features"])
            du = uf.shape[1] if uf.ndim == 2 else d
            out["union_features"] = np.concatenate([uf.reshape(-1, du), np.zeros((k, du), uf.dtype if uf.size else np.float32)])
        return out

    def _predcls_unobserved_rules(self, feat_frames, common_frames, annot_frames, annot_frame_to_key):
        """inventory="full", PredCls (user rules, Nov 10): an unobserved object carries no box-derived evidence.
          * every split: appearance feature, 2-D box and the union features of its pairs are zero (natural mask);
          * training split only: its 3-D box is not the annotation's (training videos have no 3-D box for
            unobserved objects). It is the annotated box of the same object in the nearest frame where it is
            visible (ties: the earlier frame); an object never visible in the video has no box and no slot.
        Test-split 3-D boxes stay the annotated ones (PredCls supplies every object's box).
        Returns {frame_file: frame dict copy} with ``_corners_override`` {entry: corners} and ``_drop_entries``."""
        train = self._phase == "train"
        st = self._inventory_stats
        # visible annotated boxes per class over the video (training split)
        vis_box = {}                                 # short label -> [(frame index, corners)]
        frames_annot = []
        for fi, f in enumerate(common_frames):
            af = annot_frames.get(annot_frame_to_key.get(f, f), {})
            first = {}
            for obj in af.get("object_info_list", []):
                first.setdefault(obj.get("label", _to_short(obj.get("class", ""))), obj)
            frames_annot.append(first)
            if train:
                for lab, obj in first.items():
                    c = obj.get("corners_final")
                    if obj.get("visible", True) and c is not None and np.asarray(c).shape == (8, 3):
                        vis_box.setdefault(lab, []).append((fi, np.asarray(c, dtype=np.float32)))
        out = {}
        for fi, f in enumerate(common_frames):
            ff = dict(feat_frames[f])
            labels = [_to_short(str(l)) for l in ff.get("labels", [])]
            lids = [int(x) for x in ff.get("label_ids", [])]
            srcs = list(ff.get("sources", ["gt"] * len(labels)))
            first_pos = {}
            for i, lid in enumerate(lids):
                first_pos.setdefault(lid, i)
            unobs = []
            for i, lab in enumerate(labels):
                obj = frames_annot[fi].get(lab)
                hidden = (srcs[i] if i < len(srcs) else "gt") in UNOBSERVED_SOURCES or (
                    self._visibility_from_annotation and obj is not None and not obj.get("visible", True))
                if hidden and lab != "person":
                    unobs.append(i)
            if unobs:
                roi = np.array(ff.get("roi_features"), dtype=np.float32, copy=True)
                bx = np.array(ff.get("bboxes_xyxy"), dtype=np.float32, copy=True).reshape(-1, 4)
                roi[unobs] = 0.0
                bx[unobs] = 0.0
                ff["roi_features"], ff["bboxes_xyxy"] = roi, bx
                if ff.get("union_features") is not None:
                    uf = np.array(ff["union_features"], dtype=np.float32, copy=True)
                    hid_lids = {lids[i] for i in unobs if first_pos.get(lids[i]) == i}
                    for k, p in enumerate(ff.get("pair_indices", [])):
                        if k < uf.shape[0] and int(p[1]) in hid_lids:
                            uf[k] = 0.0
                    ff["union_features"] = uf
                st["unobserved_entries_zeroed"] += len(unobs)
            if train:
                over, drop = {}, set()
                for i in unobs:
                    occ = vis_box.get(labels[i])
                    if not occ:
                        drop.add(i)
                        continue
                    over[i] = min(occ, key=lambda x: (abs(x[0] - fi), x[0]))[1]
                ff["_corners_override"], ff["_drop_entries"] = over, drop
                st["train_unobserved_held_box"] += len(over)
                st["train_unobserved_never_visible_dropped"] += len(drop)
            out[f] = ff
        return out

    @staticmethod
    def _pose_map(annot_data):
        """{frame file basename: (4, 4) camera->world pose} from the annotation PKL (empty if absent)."""
        poses = annot_data.get("camera_poses", None)
        keys = annot_data.get("camera_frame_keys", [])
        if poses is None or len(keys) == 0:
            return {}
        poses = np.asarray(poses, dtype=np.float32)
        return {(k.split("/")[-1] if "/" in k else k): poses[i] for i, k in enumerate(keys) if i < len(poses)}

    def _expand_inventory(self, feat_frames, common_frames, annot_frames, annot_frame_to_key, poses=None):
        """inventory="full": per common frame, the feature entry dict with the missing objects
        appended (see ``INVENTORIES``). Returns {frame_file: frame dict}; frames are copies."""
        out = {}
        st = self._inventory_stats
        if self._mode == "predcls":
            for f in common_frames:
                ff = feat_frames[f]
                af = annot_frames.get(annot_frame_to_key.get(f, f), {})
                present = {_to_short(str(l)) for l in ff.get("labels", [])}
                new = []
                for obj in af.get("object_info_list", []):
                    short = obj.get("label", _to_short(obj.get("class", "")))
                    if short in present or short == "person" or short not in NAME_TO_IDX:
                        continue
                    present.add(short)
                    new.append({"label": short, "label_id": NAME_TO_IDX[short],
                                "source": INVENTORY_SOURCE, "box3d": None})
                    st["appended_visible" if obj.get("visible", True) else "appended_unobserved"] += 1
                st["frames"] += 1
                out[f] = self._append_entries(ff, new)
            return out

        # SGDet: held identities, derived from the detections only (no annotation). An object
        # class is established in the video once one of its detections scores >= held_min_score;
        # an established class is held in every frame where it is not detected.
        det = []                                   # per frame: {class id: rank-0 entry box3d}
        established = set()
        for f in common_frames:
            ff = feat_frames[f]
            b3 = ff.get("boxes_3d")
            b3 = None if b3 is None else np.asarray(b3, dtype=np.float32).reshape(-1, 8, 3)
            sc = ff.get("scores")
            sc = None if sc is None else np.asarray(sc, dtype=np.float32).reshape(-1)
            m = {}
            for i, lid in enumerate(ff.get("label_ids", [])):
                lid = int(lid)
                if lid <= PERSON_CLASS_ID:
                    continue
                if sc is None or (i < sc.shape[0] and sc[i] >= self._held_min_score):
                    established.add(lid)
                if lid in m:
                    continue
                m[lid] = b3[i] if b3 is not None and i < b3.shape[0] else None
            det.append(m)
        frames_of = {}
        for fi, m in enumerate(det):
            for lid in m:
                if lid in established:
                    frames_of.setdefault(lid, []).append(fi)
        poses = poses or {}
        for fi, f in enumerate(common_frames):
            new = []
            for lid in sorted(frames_of):
                if lid in det[fi]:
                    continue
                fs = frames_of[lid]
                # nearest detected frame, ties -> the earlier one
                src_f = min(fs, key=lambda x: (abs(x - fi), x))
                box = det[src_f][lid]
                # SGDet boxes_3d live in each frame's camera frame: move the held box from the source frame's
                # camera to this frame's camera through the world frame (fix, Nov 10; round-2 runs before it
                # copied the source-frame camera coordinates unchanged)
                ps = poses.get(common_frames[src_f].split("/")[-1])
                pt = poses.get(f.split("/")[-1])
                if box is not None and ps is not None and pt is not None:
                    w = np.asarray(box, np.float32) @ ps[:3, :3].T + ps[:3, 3]
                    box = (w - pt[:3, 3]) @ pt[:3, :3]
                    st["held_box_reframed"] += 1
                elif box is not None:
                    st["held_box_no_pose"] += 1
                new.append({"label": OBJECT_CLASSES[lid], "label_id": lid, "source": HELD_SOURCE,
                            "box3d": box})
            st["appended_held"] += len(new)
            st["frames"] += 1
            out[f] = self._append_entries(feat_frames[f], new)
        return out

    # ------------------------------------------------------------------
    # Per-video slot alignment (slot_alignment="per_video_class")
    # ------------------------------------------------------------------

    def _per_video_slot_maps(self, feat_frame_list: List[Dict[str, Any]]):
        """One fixed slot per object identity for a whole video.

        Identity of entry i of a frame = (class id, rank), rank 0 = the first
        entry of that class in the frame (``_build_frame_tensors`` attaches the
        class's pairs to it), ranks 1.. = the other same-class entries by
        descending detector score (SGDet duplicates; PredCls has one entry per
        class). Slot order: person (1, 0) first, then rank-0 identities by first
        appearance (frame, entry), then duplicates by (rank, -max score, first
        appearance). Identities beyond ``max_objects`` are dropped.

        Returns (per-frame lists entry -> slot or -1, number of slots, stats).
        """
        info = {}            # key -> [first_frame, first_entry, max_score]
        frame_keys = []
        for fi, ff in enumerate(feat_frame_list):
            n = len(ff.get("labels", []))
            lids_raw = ff.get("label_ids", [])
            lids = [int(lids_raw[i]) if i < len(lids_raw) else 0 for i in range(n)]
            sc_raw = ff.get("scores", None)
            sc = [float(sc_raw[i]) if sc_raw is not None and i < len(sc_raw) else 0.0
                  for i in range(n)]
            first = {}
            for i, l in enumerate(lids):
                first.setdefault(l, i)
            rank = [0] * n
            for l, i0 in first.items():
                rest = sorted((i for i in range(n) if lids[i] == l and i != i0),
                              key=lambda i: (-sc[i], i))
                for r, i in enumerate(rest, start=1):
                    rank[i] = r
            keys = [(lids[i], rank[i]) for i in range(n)]
            for i, k in enumerate(keys):
                cur = info.get(k)
                if cur is None:
                    info[k] = [fi, i, sc[i]]
                elif sc[i] > cur[2]:
                    cur[2] = sc[i]
            frame_keys.append(keys)

        def _order(k):
            fi, i, ms = info[k]
            return (0 if k == (PERSON_CLASS_ID, 0) else 1, k[1],
                    0.0 if k[1] == 0 else -ms, fi, i, k[0])

        ordered = sorted(info, key=_order)
        kept = ordered[:self._max_objects]
        slot_of = {k: s for s, k in enumerate(kept)}
        maps = [[slot_of.get(k, -1) for k in keys] for keys in frame_keys]
        n_dropped_entries = sum(1 for m in maps for s in m if s < 0)
        stats = {
            "n_identities": len(ordered),
            "n_slots": len(kept),
            "dropped_identities": len(ordered) - len(kept),
            "dropped_entries": n_dropped_entries,
            "person_slot0": bool(kept and kept[0] == (PERSON_CLASS_ID, 0)),
        }
        return maps, max(len(kept), 1), stats

    @staticmethod
    def _remap_frame_tensors(ft: Dict[str, Any], entry_to_slot: List[int], n_slots: int):
        """Move one frame's entries (built in entry order) to their video slots.

        Node tensors are scattered to the slot rows (unused slots stay zero /
        invalid); pairs keep their order and point to the new slot indices; a
        pair whose endpoint has no slot is dropped together with its union-
        feature index."""
        out = {}
        src = [i for i, s in enumerate(entry_to_slot) if s >= 0]
        dst = [entry_to_slot[i] for i in src]
        src_t = torch.tensor(src, dtype=torch.long)
        dst_t = torch.tensor(dst, dtype=torch.long)
        for key in _NODE_KEYS:
            v = ft[key]
            o = torch.zeros((n_slots,) + tuple(v.shape[1:]), dtype=v.dtype)
            if src:
                o[dst_t] = v[src_t]
            out[key] = o
        edges = {key: [] for key in _EDGE_KEYS}
        for k in range(ft["K_valid"]):
            p_slot = entry_to_slot[ft["person_indices"][k]]
            o_slot = entry_to_slot[ft["object_indices"][k]]
            if p_slot < 0 or o_slot < 0:
                continue
            for key in _EDGE_KEYS:
                edges[key].append(ft[key][k])
            edges["person_indices"][-1] = p_slot
            edges["object_indices"][-1] = o_slot
        out.update(edges)
        out["K_valid"] = len(edges["person_indices"])
        return out

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_list[index]

        # Load both PKLs
        feat_data = self._load_feature_pkl(video_id)
        annot_data = self._load_annotation_pkl(video_id)

        # Align frames
        common_frames, annot_frame_to_key = self._align_frames(
            feat_data, annot_data
        )

        assert len(common_frames) >= 2, (
            f"Video {video_id} has {len(common_frames)} common frames; "
            f"should have been filtered during _build_video_list"
        )

        T = len(common_frames)

        # First pass: determine N_max and K_max across all frames
        feat_frames = feat_data["frames"]
        annot_frames = annot_data["frames"]
        if self._inventory == "full":
            feat_frames = self._expand_inventory(feat_frames, common_frames, annot_frames,
                                                 annot_frame_to_key, poses=self._pose_map(annot_data))
            if self._mode == "predcls":
                feat_frames = self._predcls_unobserved_rules(feat_frames, common_frames, annot_frames,
                                                             annot_frame_to_key)

        n_objects_per_frame = []
        k_pairs_per_frame = []
        D = 1024  # default feature dim

        for frame_file in common_frames:
            ff = feat_frames[frame_file]
            roi = ff.get("roi_features", np.zeros((0, 1024)))
            n_obj = roi.shape[0] if isinstance(roi, np.ndarray) else len(roi)
            n_objects_per_frame.append(n_obj)
            pairs = ff.get("pair_indices", [])
            k_pairs_per_frame.append(len(pairs))
            if isinstance(roi, np.ndarray) and roi.ndim == 2 and roi.shape[0] > 0:
                D = roi.shape[1]

        N_max = min(max(n_objects_per_frame) if n_objects_per_frame else 1,
                    self._max_objects)
        K_max = max(k_pairs_per_frame) if k_pairs_per_frame else 1

        # Per-video slot alignment: one fixed slot per object identity
        slot_maps, slot_stats = None, None
        if self._slot_alignment == "per_video_class":
            slot_maps, N_max, slot_stats = self._per_video_slot_maps(
                [feat_frames[f] for f in common_frames])

        # Second pass: build per-frame tensors
        all_visual = []
        all_corners = []
        all_gt_corners = []
        all_bboxes = []
        all_gt_bboxes = []
        all_valid = []
        all_vis = []
        all_classes = []
        all_gtvalid = []

        all_pidx = []
        all_oidx = []
        all_pair_valid = []
        all_att = []
        all_spa = []
        all_con = []
        all_psrc = []
        all_valid_raw_pair_indices = []  # for union feature alignment

        # For evaluator compatibility
        gt_annotations = []

        for fi, frame_file in enumerate(common_frames):
            ff = feat_frames[frame_file]
            af_key = annot_frame_to_key.get(frame_file, frame_file)
            af = annot_frames.get(af_key, {})

            if slot_maps is None:
                frame_tensors = self._build_frame_tensors(
                    ff, af, N_max,
                    bbox_scale=self._bbox_scale(video_id, frame_file, ff),
                )
            else:
                # Build in the frame's own entry order (no truncation), then
                # move every entry to its video slot.
                n_feat = len(ff.get("labels", []))
                frame_tensors = self._remap_frame_tensors(
                    self._build_frame_tensors(
                        ff, af, max(n_feat, 1),
                        bbox_scale=self._bbox_scale(video_id, frame_file, ff),
                    ),
                    slot_maps[fi], N_max,
                )

            # Node tensors (already N_max padded)
            all_visual.append(frame_tensors["visual_features"])
            all_corners.append(frame_tensors["corners"])
            all_gt_corners.append(frame_tensors["gt_corners"])
            all_bboxes.append(frame_tensors["bboxes_2d"])
            all_gt_bboxes.append(frame_tensors["gt_bboxes_2d"])
            all_valid.append(frame_tensors["valid_mask"])
            all_vis.append(frame_tensors["visibility_mask"])
            all_classes.append(frame_tensors["object_classes"])
            all_gtvalid.append(frame_tensors["gt_box_valid"])

            # Edge tensors: pad to K_max
            K_v = frame_tensors["K_valid"]
            pidx = torch.zeros(K_max, dtype=torch.long)
            oidx = torch.zeros(K_max, dtype=torch.long)
            pair_valid = torch.zeros(K_max, dtype=torch.bool)
            att = torch.zeros(K_max, dtype=torch.long)
            spa = torch.zeros(K_max, NUM_SPATIAL)
            con = torch.zeros(K_max, NUM_CONTACTING)
            psrc = torch.zeros(K_max, dtype=torch.long)

            if K_v > 0:
                pidx[:K_v] = torch.tensor(
                    frame_tensors["person_indices"], dtype=torch.long
                )
                oidx[:K_v] = torch.tensor(
                    frame_tensors["object_indices"], dtype=torch.long
                )
                pair_valid[:K_v] = True
                att[:K_v] = torch.tensor(
                    frame_tensors["att_labels"], dtype=torch.long
                )
                spa[:K_v] = torch.stack(frame_tensors["spa_multi_hot"])
                con[:K_v] = torch.stack(frame_tensors["con_multi_hot"])
                psrc[:K_v] = torch.tensor(
                    frame_tensors["pair_sources"], dtype=torch.long
                )

            all_pidx.append(pidx)
            all_oidx.append(oidx)
            all_pair_valid.append(pair_valid)
            all_att.append(att)
            all_spa.append(spa)
            all_con.append(con)
            all_psrc.append(psrc)
            all_valid_raw_pair_indices.append(frame_tensors["valid_raw_pair_indices"])

            # Raw annotation for evaluator
            gt_annotations.append(af)

        # Stack all frames → (T, ...)
        result = {
            "video_id": video_id,
            "T": T,
            "N_max": N_max,
            "K_max": K_max,
            "frame_names": common_frames,

            # Per-node: (T, N_max, ...)
            "visual_features": torch.stack(all_visual),         # (T, N_max, D)
            "corners": torch.stack(all_corners),                 # (T, N_max, 8, 3)
            "gt_corners": torch.stack(all_gt_corners),             # (T, N_max, 8, 3)
            "bboxes_2d": torch.stack(all_bboxes),               # (T, N_max, 4)
            "gt_bboxes_2d": torch.stack(all_gt_bboxes),         # (T, N_max, 4)
            "valid_mask": torch.stack(all_valid),                 # (T, N_max)
            "visibility_mask": torch.stack(all_vis),             # (T, N_max)
            "object_classes": torch.stack(all_classes),           # (T, N_max)
            "gt_box_valid": torch.stack(all_gtvalid),             # (T, N_max) bool

            # Per-edge: (T, K_max, ...)
            "person_idx": torch.stack(all_pidx),                 # (T, K_max)
            "object_idx": torch.stack(all_oidx),                 # (T, K_max)
            "pair_valid": torch.stack(all_pair_valid),           # (T, K_max)

            # GT labels (pre-encoded)
            "gt_attention": torch.stack(all_att),                 # (T, K_max)
            "gt_spatial": torch.stack(all_spa),                   # (T, K_max, 6)
            "gt_contacting": torch.stack(all_con),               # (T, K_max, 17)

            # Edge metadata
            "pair_source": torch.stack(all_psrc),                # (T, K_max)

            # Raw annotations for evaluator
            "gt_annotations": gt_annotations,
        }
        if slot_stats is not None:
            result["slot_stats"] = slot_stats
        if getattr(self, "_record_target_size", False):
            # (T, 2) feature-PKL box space per frame, as stored (W, H); used by
            # WorldAGGrid(rescale_boxes_to_grid=True)
            result["feat_target_size"] = torch.tensor(
                [self._feat_target_size(video_id, f, feat_frames[f]) for f in common_frames],
                dtype=torch.float32)

        # Camera poses (from annotation PKL)
        camera_poses = annot_data.get("camera_poses", None)
        camera_frame_keys = annot_data.get("camera_frame_keys", [])
        if camera_poses is not None and len(camera_frame_keys) > 0:
            # Filter to common frames only
            cam_key_to_idx = {
                k.split("/")[-1] if "/" in k else k: i
                for i, k in enumerate(camera_frame_keys)
            }
            cam_indices = [
                cam_key_to_idx[f] for f in common_frames
                if f in cam_key_to_idx
            ]
            if len(cam_indices) == T:
                cam = np.asarray(camera_poses, dtype=np.float32)
                result["camera_poses"] = torch.from_numpy(cam[cam_indices])
            else:
                result["camera_poses"] = None
        else:
            result["camera_poses"] = None

        # Union features (from feature PKL, per-pair)
        # Must be reordered to match the valid-pair subset from _build_frame_tensors.
        has_union = any(
            "union_features" in feat_frames[f] for f in common_frames
        )
        if has_union:
            union_all = []
            for fi, frame_file in enumerate(common_frames):
                ff = feat_frames[frame_file]
                uf = ff.get("union_features", None)
                union_padded = torch.zeros(K_max, D)
                if uf is not None:
                    uf_t = torch.from_numpy(
                        np.asarray(uf, dtype=np.float32)
                    )
                    # Use cached valid_raw_pair_indices to pick only the
                    # union features for valid pairs, in the correct order.
                    for k_new, raw_k in enumerate(all_valid_raw_pair_indices[fi]):
                        if k_new >= K_max:
                            break
                        if raw_k < uf_t.shape[0]:
                            union_padded[k_new] = uf_t[raw_k]
                union_all.append(union_padded)
            result["union_features"] = torch.stack(union_all)  # (T, K_max, D)
        else:
            result["union_features"] = None

        return result




def world_collate_fn(batch):
    """Simple collate that returns the first (and only) item in the batch."""
    return batch[0]
