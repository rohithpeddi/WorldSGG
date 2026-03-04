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
import os
import pickle
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


def _to_short(label: str) -> str:
    """Full AG name → short form. E.g. 'closet/cabinet' → 'closet'."""
    return LABEL_NORMALIZE_MAP.get(label, label)


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
    ):
        """
        Args:
            phase: "train" or "test"
            data_path: Root directory of Action Genome dataset
            mode: "predcls" or "sgdet"
            feature_model: Feature model directory name (e.g. "dinov2b")
            include_invisible: If True, include RAG-predicted objects
            max_objects: Maximum number of objects per frame (N_max cap)
        """
        super().__init__()

        self._phase = phase
        self._data_path = Path(data_path)
        self._mode = mode
        self._feature_model = feature_model
        self._include_invisible = include_invisible
        self._max_objects = max_objects

        # Directories
        self._feat_dir = (
            self._data_path / "features" / "roi_features"
            / mode / feature_model / phase
        )
        self._annot_dir = (
            self._data_path / "world4d_rel_annotations" / phase
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
            f"{len(self.video_list)} videos"
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

    def _build_frame_tensors(
        self,
        feat_frame: Dict[str, Any],
        annot_frame: Dict[str, Any],
        max_N: int,
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
        bboxes_2d = torch.zeros(max_N, 4)
        gt_bboxes_2d = torch.zeros(max_N, 4)  # real GT annotation boxes
        valid_mask = torch.zeros(max_N, dtype=torch.bool)
        visibility_mask = torch.zeros(max_N, dtype=torch.bool)
        object_classes = torch.zeros(max_N, dtype=torch.long)

        for i in range(N):
            visual_features[i] = roi_features[i]
            if i < bboxes_xyxy.shape[0]:
                bboxes_2d[i] = bboxes_xyxy[i]
            valid_mask[i] = True

            # Object class label (GT for predcls, detector for sgdet —
            # both stored in feat_label_ids by their extraction scripts)
            class_idx = feat_label_ids[i] if i < len(feat_label_ids) else 0
            object_classes[i] = class_idx

            # Source determines visibility
            src = feat_sources[i] if i < len(feat_sources) else "gt"
            is_visible = src not in ("rag", "gdino", "correction")
            visibility_mask[i] = is_visible

            # Skip invisible if requested
            if not is_visible and not self._include_invisible:
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
                gt_box = annot_obj.get("bbox", None)
                if gt_box is not None:
                    gt_bboxes_2d[i] = torch.tensor(
                        np.asarray(gt_box, dtype=np.float32)
                    )

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
                if annot_obj is not None:
                    c = annot_obj.get("corners_final", None)
                    if c is not None:
                        c = np.asarray(c, dtype=np.float32)
                        if c.shape == (8, 3):
                            corners[i] = torch.from_numpy(c)

            # Update visibility from annotation if available (applies to both modes)
            if annot_obj is not None and not annot_obj.get("visible", True):
                visibility_mask[i] = False

        # --- Person GT bbox (slot 0) ---
        person_bbox = person_info.get("person_bbox", None)
        if person_bbox is not None:
            gt_bboxes_2d[0] = torch.tensor(
                np.asarray(person_bbox, dtype=np.float32)
            )

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

            person_indices.append(p_pos)
            object_indices.append(o_pos)
            valid_raw_pair_indices.append(raw_k)

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
            "bboxes_2d": bboxes_2d,
            "gt_bboxes_2d": gt_bboxes_2d,
            "valid_mask": valid_mask,
            "visibility_mask": visibility_mask,
            "object_classes": object_classes,
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

        # Second pass: build per-frame tensors
        all_visual = []
        all_corners = []
        all_bboxes = []
        all_gt_bboxes = []
        all_valid = []
        all_vis = []
        all_classes = []

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

        for frame_file in common_frames:
            ff = feat_frames[frame_file]
            af_key = annot_frame_to_key.get(frame_file, frame_file)
            af = annot_frames.get(af_key, {})

            frame_tensors = self._build_frame_tensors(ff, af, N_max)

            # Node tensors (already N_max padded)
            all_visual.append(frame_tensors["visual_features"])
            all_corners.append(frame_tensors["corners"])
            all_bboxes.append(frame_tensors["bboxes_2d"])
            all_gt_bboxes.append(frame_tensors["gt_bboxes_2d"])
            all_valid.append(frame_tensors["valid_mask"])
            all_vis.append(frame_tensors["visibility_mask"])
            all_classes.append(frame_tensors["object_classes"])

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
            "bboxes_2d": torch.stack(all_bboxes),               # (T, N_max, 4)
            "gt_bboxes_2d": torch.stack(all_gt_bboxes),         # (T, N_max, 4)
            "valid_mask": torch.stack(all_valid),                 # (T, N_max)
            "visibility_mask": torch.stack(all_vis),             # (T, N_max)
            "object_classes": torch.stack(all_classes),           # (T, N_max)

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
