#!/usr/bin/env python3
"""
Corrected World BBox Generator
===============================

Generates world-frame 3D bounding boxes (OBB) using manually
corrected floor transforms downloaded from Firebase, instead of the
auto-computed floor alignment from SMPL-scene correspondences.

Transformation priority:
    1. If final_alignment.combined_transform exists → use the 4×4 matrix directly
       as the world→canonical transform.
    2. Else compose T_XY ∘ T_delta ∘ T_auto per the documented transformation order.

Output: bbox_annotations_3d_obb_corrected/<video>.pkl

Usage:
    python corrected_world_bbox_generator.py                              # Process all
    python corrected_world_bbox_generator.py --video 001YG.mp4            # Single video
    python corrected_world_bbox_generator.py --split 04                   # Specific split
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as SciRot
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from annotation_utils import (
    get_video_belongs_to_split,
    _faces_u32,
    _xywh_to_xyxy,
    _resize_bbox_to,
    _mask_from_bbox,
    _resize_mask_to,
    _finite_and_nonzero,
    _load_pkl_if_exists,
    _npz_open,
    _as_np,
)

from datasets.preprocess.annotations.raw.bb3D_base import BBox3DBase


# =====================================================================
# LABEL NORMALIZATION
# =====================================================================

# GT compound names → short form
LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}

# GDino detection scripts expand compound GT classes into individual words
# for prompting (e.g., "closet/cabinet" → "closet", "cabinet").
# After detection, the PKLs can contain these expanded labels.
# This reverse mapping normalizes them back to GT short forms.
GDINO_LABEL_TO_GT_LABEL = {
    "cabinet": "closet",
    "glass": "cup",
    "bottle": "cup",
    "notebook": "paper",
    "couch": "sofa",
    "camera": "phone",
}


def _normalize_gdino_label(label: str) -> str:
    """Normalize a GDino label to match GT label space."""
    label = label.lower().strip()
    # Step 1: handle compound GT forms
    label = LABEL_NORMALIZE_MAP.get(label, label)
    # Step 2: handle expanded GDino forms
    label = GDINO_LABEL_TO_GT_LABEL.get(label, label)
    return label


# =====================================================================
# OBB HELPERS (reused from raw/bb3D_generator_gt_obb.py)
# =====================================================================

def _compute_obb_pca(points: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Compute arbitrary OBB using PCA.
    Returns: {
        "center": [x, y, z],
        "extent": [w, h, d],  (full lengths)
        "rotation_matrix": [[r00, r01, r02], ...],
        "corners_world": [[x,y,z], ...] (8 corners)
    }
    """
    if points.shape[0] < 4:
        return None

    center = np.mean(points, axis=0)
    centered_points = points - center

    cov = np.cov(centered_points, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)

    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1

    projected = centered_points @ evecs
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)

    extent = maxs - mins
    center_offset = (mins + maxs) / 2.0
    obb_center = center + (evecs @ center_offset)

    corners_local_xyz = []
    for cx in [mins[0], maxs[0]]:
        for cy in [mins[1], maxs[1]]:
            for cz in [mins[2], maxs[2]]:
                corners_local_xyz.append([cx, cy, cz])
    corners_local_xyz = np.array(corners_local_xyz)
    corners_world = corners_local_xyz @ evecs.T + center[None, :]

    return {
        "center": obb_center.tolist(),
        "extent": extent.tolist(),
        "rotation_matrix": evecs.tolist(),
        "corners_world": corners_world.tolist(),
    }


def _compute_obb_floor_parallel(
    points_floor: np.ndarray,
    floor_to_world_fn,
) -> Dict[str, Any]:
    """
    Compute OBB parallel to the floor plane.
    Uses cv2.minAreaRect on XZ projection (Y is up/normal in floor frame).

    Args:
        points_floor: (N, 3) points already in floor/canonical frame.
        floor_to_world_fn: callable that maps (N, 3) floor coords → (N, 3) world coords.
    """
    pts_2d = points_floor[:, [0, 2]].astype(np.float32)
    rect = cv2.minAreaRect(pts_2d[:, None, :])
    (center_2d, size_2d, angle) = rect
    box_2d = cv2.boxPoints(rect)  # (4, 2) corners in XZ

    y_vals = points_floor[:, 1]
    y_min = y_vals.min()
    y_max = y_vals.max()

    # Build 8 corners: bottom face (y_min) then top face (y_max)
    corners_floor = []
    for i in range(4):
        x, z = box_2d[i]
        corners_floor.append([x, y_min, z])
    for i in range(4):
        x, z = box_2d[i]
        corners_floor.append([x, y_max, z])
    corners_floor = np.array(corners_floor, dtype=np.float32)

    # Transform back to world using the proper inverse
    corners_world = floor_to_world_fn(corners_floor)

    return {
        "rect_2d": {"center": center_2d, "size": size_2d, "angle": angle},
        "y_range": [float(y_min), float(y_max)],
        "corners_world": corners_world.tolist(),
    }





# =====================================================================
# CORRECTED FLOOR TRANSFORM
# =====================================================================

def build_corrected_transform(
    manual_correction: Dict[str, Any],
    original_global_floor_sim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the corrected world → canonical transform from manual corrections.

    Priority:
        1. final_alignment.combined_transform (4×4) if available
        2. Compose T_XY ∘ T_delta ∘ T_auto

    Returns:
        {
            "combined_transform_4x4": (4, 4) ndarray,
            "R": (3, 3) ndarray,   # rotation part
            "t": (3,) ndarray,     # translation part
            "source": str,         # "final_alignment" or "composed"
        }
    """
    final_alignment = manual_correction.get("final_alignment", None)

    # ------------------------------------------------------------------
    # Priority 1: Use pre-computed combined_transform from final_alignment
    # ------------------------------------------------------------------
    if final_alignment is not None:
        combined = final_alignment.get("combined_transform", None)
        if combined is not None:
            # combined_transform can be:
            #   (a) a dict with {rotation_matrix, scale, translation}
            #   (b) a flat/nested 4×4 array
            if isinstance(combined, dict):
                R_ct = np.asarray(combined["rotation_matrix"], dtype=np.float64)  # (3,3)
                s_ct = np.asarray(combined.get("scale", [1, 1, 1]), dtype=np.float64)  # (3,)
                t_ct = np.asarray(combined["translation"], dtype=np.float64)  # (3,)

                # Build 4×4: upper-left = diag(scale) @ R, last col = translation
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = np.diag(s_ct) @ R_ct
                T[:3, 3] = t_ct
            else:
                T = np.array(combined, dtype=np.float64).reshape(4, 4)

            R = T[:3, :3].astype(np.float32)
            t = T[:3, 3].astype(np.float32)
            return {
                "combined_transform_4x4": T.astype(np.float32),
                "R": R,
                "t": t,
                "source": "final_alignment",
            }

    # ------------------------------------------------------------------
    # Priority 2: Compose T_XY ∘ T_delta ∘ T_auto
    # ------------------------------------------------------------------
    # Step 1: T_auto from original_global_floor_sim
    if original_global_floor_sim is not None:
        s_g = float(original_global_floor_sim["s"])
        R_g = np.asarray(original_global_floor_sim["R"], dtype=np.float64)
        t_g = np.asarray(original_global_floor_sim["t"], dtype=np.float64)

        # Build T_auto: floor-aligned frame
        t1 = R_g[:, 0]
        t2 = R_g[:, 2]
        n = R_g[:, 1]
        F = np.stack([t1, t2, n], axis=1)
        R_align = F.T
        M_mirror = np.diag([-1.0, 1.0, 1.0])
        R_auto = M_mirror @ R_align
        t_auto = -R_auto @ t_g

        T_auto = np.eye(4, dtype=np.float64)
        T_auto[:3, :3] = R_auto
        T_auto[:3, 3] = t_auto
    else:
        T_auto = np.eye(4, dtype=np.float64)

    # Step 2: T_delta from floor_correction
    T_delta = np.eye(4, dtype=np.float64)
    floor_correction = manual_correction.get("floor_correction", None)
    if floor_correction is not None:
        delta = floor_correction.get("delta_transform", None)
        if delta is not None:
            rx = float(delta.get("rx", 0))
            ry = float(delta.get("ry", 0))
            rz = float(delta.get("rz", 0))
            tx = float(delta.get("tx", 0))
            ty = float(delta.get("ty", 0))
            tz = float(delta.get("tz", 0))
            sx = float(delta.get("sx", 1))
            sy = float(delta.get("sy", 1))
            sz = float(delta.get("sz", 1))

            # Rotation: intrinsic XYZ Euler angles → R_delta = Rx @ Ry @ Rz
            R_delta = SciRot.from_euler("XYZ", [rx, ry, rz]).as_matrix()

            # Scale matrix
            S_delta = np.diag([sx, sy, sz])

            T_delta[:3, :3] = R_delta @ S_delta
            T_delta[:3, 3] = [tx, ty, tz]

    # Step 3: T_XY from xy_alignment
    T_xy = np.eye(4, dtype=np.float64)
    xy_alignment = manual_correction.get("xy_alignment", None)
    if xy_alignment is not None:
        align = xy_alignment.get("alignment_transform", None)
        if align is not None:
            rx = float(align.get("rx", 0))
            ry = float(align.get("ry", 0))
            rz = float(align.get("rz", 0))
            tx = float(align.get("tx", 0))
            ty = float(align.get("ty", 0))
            tz = float(align.get("tz", 0))

            R_xy = SciRot.from_euler("XYZ", [rx, ry, rz]).as_matrix()
            T_xy[:3, :3] = R_xy
            T_xy[:3, 3] = [tx, ty, tz]

    # Compose: T_combined = T_XY @ T_delta @ T_auto
    T_combined = T_xy @ T_delta @ T_auto
    R = T_combined[:3, :3].astype(np.float32)
    t = T_combined[:3, 3].astype(np.float32)

    return {
        "combined_transform_4x4": T_combined.astype(np.float32),
        "R": R,
        "t": t,
        "source": "composed",
    }


# =====================================================================
# CORRECTED WORLD BBOX GENERATOR
# =====================================================================

class CorrectedWorldBBoxGenerator(BBox3DBase):
    """
    Generates corrected world-frame 3D bounding boxes (AABB + OBB) using
    manually corrected floor transforms from Firebase.
    """

    def __init__(
        self,
        dynamic_scene_dir_path: Optional[str] = None,
        ag_root_directory: Optional[str] = None,
        manual_corrections_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            dynamic_scene_dir_path=dynamic_scene_dir_path,
            ag_root_directory=ag_root_directory,
        )

        # Manual corrections PKL directory
        if manual_corrections_dir:
            self.manual_corrections_dir = Path(manual_corrections_dir)
        else:
            self.manual_corrections_dir = Path("/data/rohith/ag/world_annotations/manual_corrections")

        # Output directories for corrected annotations
        self.bbox_3d_obb_corrected_root_dir = (
            self.world_annotations_root_dir / "bbox_annotations_3d_obb_corrected"
        )
        os.makedirs(self.bbox_3d_obb_corrected_root_dir, exist_ok=True)

        # Multiscale erosion config (same as original)
        self.erosion_kernel_sizes = [0, 3, 5, 7, 9]
        self.min_points_per_scale = 50

        # GDino score threshold for 2D→3D lift
        self.gdino_score_threshold = 0.3

    # ------------------------------------------------------------------
    # Manual correction loading
    # ------------------------------------------------------------------

    def load_manual_correction(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load the per-video manual correction PKL."""
        video_key = video_id.replace(".mp4", "").replace(".", "_")
        pkl_path = self.manual_corrections_dir / f"{video_key}.pkl"
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Point loading (from dynamic predictions)
    # ------------------------------------------------------------------

    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"

        with _npz_open(video_dynamic_3d_scene_path) as video_dynamic_predictions:
            points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
            imgs_f32 = video_dynamic_predictions["images"]
            confidence = video_dynamic_predictions["conf"]
            camera_poses = video_dynamic_predictions["camera_poses"]
            colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

            conf = None
            if "conf" in video_dynamic_predictions:
                conf = video_dynamic_predictions["conf"]
                if conf.ndim == 4 and conf.shape[-1] == 1:
                    conf = conf[..., 0]

            S, H, W, _ = points.shape

            (frame_idx_frame_path_map, sample_idx, video_sampled_frame_id_list,
             annotated_frame_id_list, annotated_frame_idx_in_sample_idx) = self.idx_to_frame_idx_path(video_id)
            assert S == len(sample_idx), "dynamic predictions length must match annotated sampled range"

            sampled_idx_frame_name_map = {}
            frame_name_sampled_idx_map = {}
            for idx_in_s, frame_idx in enumerate(sample_idx):
                frame_name = f"{video_sampled_frame_id_list[frame_idx]:06d}.png"
                sampled_idx_frame_name_map[idx_in_s] = frame_name
                frame_name_sampled_idx_map[frame_name] = idx_in_s

            annotated_idx_in_sampled_idx = []
            for frame_name in annotated_frame_id_list:
                if frame_name in frame_name_sampled_idx_map:
                    annotated_idx_in_sampled_idx.append(frame_name_sampled_idx_map[frame_name])

            points_sub = points[annotated_idx_in_sampled_idx]
            conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
            stems_sub = [sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx]
            colors_sub = colors[annotated_idx_in_sampled_idx]
            camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx]

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub
        }

    # ------------------------------------------------------------------
    # Load original world bbox PKL (for global_floor_sim fallback)
    # ------------------------------------------------------------------

    def _load_original_world_bbox(self, video_id: str) -> Optional[Dict[str, Any]]:
        """Load the original (uncorrected) world bbox PKL if it exists."""
        pkl_path = self.bbox_3d_obb_root_dir / f"{video_id[:-4]}.pkl"
        return _load_pkl_if_exists(pkl_path)

    # ------------------------------------------------------------------
    # Core: Build corrected bboxes for a video
    # ------------------------------------------------------------------

    def _lift_2d_bbox_to_3d(
        self,
        *,
        label: str,
        bbox_xyxy: List[float],
        pts_hw3: np.ndarray,
        conf_hw: Optional[np.ndarray],
        conf_thr: float,
        frame_non_zero_pts: np.ndarray,
        H: int,
        W: int,
        floor_align_fn,
        floor_to_world_fn,
        source_tag: str = "gt",
    ) -> Optional[Dict[str, Any]]:
        """
        Lift a 2D bounding box to 3D using point cloud selection + multiscale erosion.

        This is the shared core for both GT and GDino 2D→3D lifting.
        Uses the 2D bbox as a pixel mask to select 3D points, then computes
        OBB variants.

        Returns an object dict or None if insufficient points.
        """
        frame_label_mask = _mask_from_bbox(H, W, bbox_xyxy)

        multi_scale_candidates = []
        for ksz in self.erosion_kernel_sizes:
            if ksz == 0:
                sel_mask = frame_label_mask.astype(bool)
            else:
                mask_u8 = frame_label_mask.astype(np.uint8)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                eroded = cv2.erode(mask_u8, kernel, iterations=1)
                sel_mask = eroded.astype(bool)

            sel = sel_mask & frame_non_zero_pts
            if conf_hw is not None:
                sel &= (conf_hw > conf_thr)
            num_sel = int(sel.sum())
            if num_sel < self.min_points_per_scale:
                continue

            obj_pts_world = pts_hw3[sel].reshape(-1, 3).astype(np.float32)
            pts_floor = floor_align_fn(obj_pts_world)

            # OBB (floor-parallel)
            obb_floor_res = _compute_obb_floor_parallel(
                pts_floor, floor_to_world_fn=floor_to_world_fn,
            )

            # OBB (arbitrary, PCA)
            obb_res = _compute_obb_pca(obj_pts_world)

            # Volume from floor-aligned min/max (for candidate ranking)
            mins = pts_floor.min(axis=0)
            maxs = pts_floor.max(axis=0)
            size = (maxs - mins).clip(1e-6)
            volume = float(size[0] * size[1] * size[2])

            multi_scale_candidates.append({
                "kernel_size": int(ksz),
                "num_points": num_sel,
                "volume": volume,
                "obb_floor_parallel": obb_floor_res,
                "obb_arbitrary": obb_res,
            })

        if not multi_scale_candidates:
            return {
                "label": label,
                "bbox_xyxy": bbox_xyxy,
                "obb_floor_parallel": None,
                "obb_arbitrary": None,
                "candidates": [],
                "source": source_tag,
            }

        # Pick best candidate (smallest volume with sufficient points)
        multi_scale_candidates.sort(key=lambda x: x.get("volume", float("inf")))
        best = multi_scale_candidates[0]

        return {
            "label": label,
            "bbox_xyxy": bbox_xyxy,
            "obb_floor_parallel": best.get("obb_floor_parallel"),
            "obb_arbitrary": best.get("obb_arbitrary"),
            "candidates": multi_scale_candidates,
            "source": source_tag,
        }

    def _build_corrected_bboxes_for_video(
        self,
        *,
        video_id: str,
        video_gt_annotations: List[Any],
        points_S: np.ndarray,
        conf_S: Optional[np.ndarray],
        stems_S: List[str],
        orig_size: Tuple[int, int],
        stem_to_idx: Dict[str, int],
        video_to_frame_to_label_mask: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        corrected_transform: Dict[str, Any],
        gdino_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
        gdino_score_threshold: float = 0.3,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build per-frame OBB bounding boxes using the corrected
        floor transform.

        For each annotated frame:
          1) Process GT annotations (as before)
          2) If gdino_predictions is provided, also process GDino detections
             for labels NOT already covered by GT in this frame.
             Only GDino detections with score >= gdino_score_threshold are used.
        """
        out_frames: Dict[str, Dict[str, Any]] = {}
        S, H, W, _ = points_S.shape
        orig_W, orig_H = orig_size

        # Extract corrected floor transform components
        T_4x4 = corrected_transform["combined_transform_4x4"]
        R_floor = T_4x4[:3, :3].astype(np.float32)
        t_floor = T_4x4[:3, 3].astype(np.float32)

        def _corrected_floor_align_points(points_world: np.ndarray) -> np.ndarray:
            """Transform world points to corrected canonical frame."""
            return (points_world @ R_floor.T) + t_floor[None, :]

        def _corrected_floor_to_world(points_canonical: np.ndarray) -> np.ndarray:
            """Transform canonical frame points back to world."""
            R_inv = np.linalg.inv(R_floor)
            return (points_canonical - t_floor[None, :]) @ R_inv.T

        # ----------------------------------------------------------
        # Pre-compute video-level GT label set for GDino filtering.
        # Only GDino detections whose normalized label belongs to this
        # set will be lifted to 3D.
        # ----------------------------------------------------------
        video_gt_labels: set = set()
        for frame_items in video_gt_annotations:
            for item in frame_items:
                if "person_bbox" in item:
                    video_gt_labels.add("person")
                else:
                    cid = item.get("class", None)
                    lbl = self.catid_to_name_map.get(cid, None)
                    if lbl:
                        lbl = LABEL_NORMALIZE_MAP.get(lbl, lbl)
                        video_gt_labels.add(lbl)

        print(
            f"[corrected-bbox][{video_id}] video GT labels "
            f"({len(video_gt_labels)}): {sorted(video_gt_labels)}"
        )

        for frame_items in video_gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
            stem = Path(frame_name).stem
            if stem not in stem_to_idx:
                continue

            sidx = stem_to_idx[stem]
            pts_hw3 = points_S[sidx]
            conf_hw = conf_S[sidx] if conf_S is not None else None
            frame_non_zero_pts = _finite_and_nonzero(pts_hw3)

            frame_rec = {"objects": []}

            # Adaptive confidence threshold
            conf_thr = 0.05
            if conf_hw is not None:
                cfs_flat = conf_hw.reshape(-1)
                mask_valid = np.isfinite(cfs_flat)
                cfs_valid = cfs_flat[mask_valid]
                if cfs_valid.size > 0:
                    p5 = np.percentile(cfs_valid, 5)
                    conf_thr = float(max(1e-3, p5))

            # Track which labels are covered by GT in this frame
            gt_labels_in_frame = set()

            for item in frame_items:
                # Resolve label + original bbox
                if "person_bbox" in item:
                    label = "person"
                    gt_xyxy = _xywh_to_xyxy(item["person_bbox"][0])
                else:
                    cid = item["class"]
                    label = self.catid_to_name_map.get(cid, None)
                    if not label:
                        continue
                    # Normalize label
                    if label == "closet/cabinet":
                        label = "closet"
                    elif label == "cup/glass/bottle":
                        label = "cup"
                    elif label == "paper/notebook":
                        label = "paper"
                    elif label == "sofa/couch":
                        label = "sofa"
                    elif label == "phone/camera":
                        label = "phone"
                    gt_xyxy = [float(v) for v in item["bbox"]]

                gt_labels_in_frame.add(label)

                # Resize bbox to dynamic prediction resolution
                gt_xyxy = _resize_bbox_to(gt_xyxy, (orig_W, orig_H), (W, H))

                # Get/resize/intersect segmentation mask
                frame_label_mask = video_to_frame_to_label_mask[video_id][stem].get(label, None)
                if frame_label_mask is None:
                    frame_label_mask = _mask_from_bbox(H, W, gt_xyxy)
                else:
                    frame_label_mask = _resize_mask_to(frame_label_mask, (H, W))
                    x1, y1, x2, y2 = map(int, gt_xyxy)
                    bbox_mask = np.zeros_like(frame_label_mask, dtype=bool)
                    bbox_mask[y1:y2, x1:x2] = True
                    frame_label_mask = frame_label_mask & bbox_mask

                # Multiscale erosion
                multi_scale_candidates = []
                for ksz in self.erosion_kernel_sizes:
                    if ksz == 0:
                        sel_mask = frame_label_mask.astype(bool)
                    else:
                        mask_u8 = frame_label_mask.astype(np.uint8)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                        eroded = cv2.erode(mask_u8, kernel, iterations=1)
                        sel_mask = eroded.astype(bool)

                    sel = sel_mask & frame_non_zero_pts
                    if conf_hw is not None:
                        sel &= (conf_hw > conf_thr)
                    num_sel = int(sel.sum())
                    if num_sel < self.min_points_per_scale:
                        continue

                    obj_pts_world = pts_hw3[sel].reshape(-1, 3).astype(np.float32)

                    # Transform to corrected canonical frame
                    pts_floor = _corrected_floor_align_points(obj_pts_world)

                    # OBB (floor-parallel)
                    obb_floor_res = _compute_obb_floor_parallel(
                        pts_floor, floor_to_world_fn=_corrected_floor_to_world,
                    )

                    # OBB (arbitrary, PCA)
                    obb_res = _compute_obb_pca(obj_pts_world)

                    # Volume from floor-aligned min/max (for candidate ranking)
                    mins = pts_floor.min(axis=0)
                    maxs = pts_floor.max(axis=0)
                    size = (maxs - mins).clip(1e-6)
                    volume = float(size[0] * size[1] * size[2])

                    multi_scale_candidates.append({
                        "kernel_size": int(ksz),
                        "num_points": num_sel,
                        "volume": volume,
                        "obb_floor_parallel": obb_floor_res,
                        "obb_arbitrary": obb_res,
                    })

                if not multi_scale_candidates:
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "obb_floor_parallel": None,
                        "obb_arbitrary": None,
                        "candidates": [],
                        "source": "gt",
                    })
                else:
                    # Pick best candidate (smallest volume with sufficient points)
                    multi_scale_candidates.sort(key=lambda x: x.get("volume", float("inf")))
                    best = multi_scale_candidates[0]

                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "obb_floor_parallel": best.get("obb_floor_parallel"),
                        "obb_arbitrary": best.get("obb_arbitrary"),
                        "candidates": multi_scale_candidates,
                        "source": "gt",
                    })

            # ----------------------------------------------------------
            # GDino 2D→3D lift: process detections for labels NOT in GT
            # ----------------------------------------------------------
            if gdino_predictions is not None:
                gdino_frame = gdino_predictions.get(frame_name, {})
                gd_boxes = gdino_frame.get("boxes", [])
                gd_labels = gdino_frame.get("labels", [])
                gd_scores = gdino_frame.get("scores", [])

                # Group GDino detections by label, keep highest-score per label
                gdino_best_per_label: Dict[str, Tuple[List[float], float]] = {}
                for gd_box, gd_label, gd_score in zip(gd_boxes, gd_labels, gd_scores):
                    gd_score = float(gd_score)
                    if gd_score < gdino_score_threshold:
                        continue
                    # Normalize GDino label to match GT label conventions
                    # (handles both compound forms and expanded forms)
                    gd_label_norm = _normalize_gdino_label(gd_label)

                    # Skip labels already covered by GT in this frame
                    if gd_label_norm in gt_labels_in_frame:
                        continue

                    # Skip labels NOT present anywhere in this video's GT
                    if gd_label_norm not in video_gt_labels:
                        continue

                    # Keep highest-score detection per label
                    if (gd_label_norm not in gdino_best_per_label
                            or gd_score > gdino_best_per_label[gd_label_norm][1]):
                        gd_box_list = [float(v) for v in gd_box]
                        gdino_best_per_label[gd_label_norm] = (gd_box_list, gd_score)

                # Lift each GDino detection to 3D
                for gd_label, (gd_box_xyxy, gd_score) in gdino_best_per_label.items():
                    # Resize GDino bbox to point cloud resolution
                    gd_box_resized = _resize_bbox_to(
                        gd_box_xyxy, (orig_W, orig_H), (W, H)
                    )

                    obj_result = self._lift_2d_bbox_to_3d(
                        label=gd_label,
                        bbox_xyxy=gd_box_resized,
                        pts_hw3=pts_hw3,
                        conf_hw=conf_hw,
                        conf_thr=conf_thr,
                        frame_non_zero_pts=frame_non_zero_pts,
                        H=H,
                        W=W,
                        floor_align_fn=_corrected_floor_align_points,
                        floor_to_world_fn=_corrected_floor_to_world,
                        source_tag="gdino",
                    )

                    if obj_result is not None:
                        obj_result["gdino_bbox_xyxy"] = gd_box_xyxy
                        obj_result["gdino_score"] = gd_score
                        frame_rec["objects"].append(obj_result)

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        return out_frames

    # ------------------------------------------------------------------
    # Main entry point: generate corrected bboxes for a single video
    # ------------------------------------------------------------------

    def generate_corrected_video_bb_annotations(
        self,
        video_id: str,
        video_gt_annotations: List[Any],
        *,
        overwrite: bool = False,
    ) -> bool:
        """
        Generate corrected world bboxes for a single video.

        Returns True if successful.
        """
        out_path = self.bbox_3d_obb_corrected_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[corrected-bbox][{video_id}] already exists, skipping.")
            return True

        # 1) Load manual corrections
        manual_correction = self.load_manual_correction(video_id)
        if manual_correction is None:
            print(f"[corrected-bbox][{video_id}] no manual corrections found, skipping.")
            return False

        # 2) Load original world bbox PKL for global_floor_sim fallback
        original_bbox_data = self._load_original_world_bbox(video_id)
        original_gfs = None
        gv, gf, gc = None, None, None
        if original_bbox_data is not None:
            original_gfs = original_bbox_data.get("global_floor_sim", None)
            gv = original_bbox_data.get("gv", None)
            gf = original_bbox_data.get("gf", None)
            gc = original_bbox_data.get("gc", None)

        # 3) Build corrected transform
        corrected_transform = build_corrected_transform(
            manual_correction=manual_correction,
            original_global_floor_sim=original_gfs,
        )
        print(
            f"[corrected-bbox][{video_id}] using corrected floor transform "
            f"(source={corrected_transform['source']})"
        )

        # 4) Load dynamic points for annotated frames
        P = self._load_points_for_video(video_id)

        points_S = P["points"]
        conf_S = P["conf"]
        stems_S = P["frame_stems"]
        S, H, W, _ = points_S.shape

        # Original image size
        sample_image_frame = self.frame_annotated_dir_path / video_id / f"{stems_S[0]}.png"
        if not sample_image_frame.exists():
            print(f"[corrected-bbox][{video_id}] annotated frame not found: {sample_image_frame}")
            return False
        orig_img = cv2.imread(str(sample_image_frame))
        orig_H, orig_W = orig_img.shape[:2]

        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # 5) Build segmentation masks
        video_to_frame_to_label_mask, _, _ = self.create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_gt_annotations,
        )

        # 6) Load GDino detections for this video
        gdino_predictions = self.get_video_gdino_annotations(video_id)
        if gdino_predictions is not None:
            print(f"[corrected-bbox][{video_id}] loaded GDino detections for {len(gdino_predictions)} frames")
        else:
            print(f"[corrected-bbox][{video_id}] no GDino detections available")

        # 7) Build corrected per-frame bboxes (GT + GDino fill)
        print(f"[corrected-bbox][{video_id}] building corrected multiscale 3D bboxes...")
        out_frames = self._build_corrected_bboxes_for_video(
            video_id=video_id,
            video_gt_annotations=video_gt_annotations,
            points_S=points_S,
            conf_S=conf_S,
            stems_S=stems_S,
            orig_size=(orig_W, orig_H),
            stem_to_idx=stem_to_idx,
            video_to_frame_to_label_mask=video_to_frame_to_label_mask,
            corrected_transform=corrected_transform,
            gdino_predictions=gdino_predictions,
            gdino_score_threshold=self.gdino_score_threshold,
        )

        # 7) Save results
        results_dictionary = {
            "video_id": video_id,
            "frames": out_frames,
            "corrected_floor_transform": {
                "combined_transform_4x4": corrected_transform["combined_transform_4x4"],
                "R": corrected_transform["R"],
                "t": corrected_transform["t"],
                "source": corrected_transform["source"],
                "original_global_floor_sim": original_gfs,
            },
            "gv": gv,
            "gf": gf,
            "gc": gc,
        }
        with open(out_path, "wb") as f:
            pickle.dump(results_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[corrected-bbox][{video_id}] saved corrected bboxes to {out_path}")
        return True

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def generate_all(
        self,
        dataloader,
        split: str,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """Process all videos in the dataloader for the given split."""
        results = {}
        for data in tqdm(dataloader, desc=f"Corrected BBox [split={split}]"):
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue
            vid_gt, full_gt = self.get_video_gt_annotations(video_id)
            success = self.generate_corrected_video_bb_annotations(
                video_id=video_id,
                video_gt_annotations=full_gt,
                overwrite=overwrite,
            )
            results[video_id] = success
        return results

    def generate_from_corrections_only(
        self,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """
        Process only videos that have manual corrections.
        Does not require a dataloader — discovers videos from the
        manual_corrections directory.
        """
        results = {}
        if not self.manual_corrections_dir.exists():
            print(f"[corrected-bbox] manual corrections dir not found: {self.manual_corrections_dir}")
            return results

        correction_files = sorted(self.manual_corrections_dir.glob("*.pkl"))
        print(f"[corrected-bbox] found {len(correction_files)} correction files")

        for pkl_path in tqdm(correction_files, desc="Corrected BBox"):
            video_key = pkl_path.stem
            video_id = video_key.replace("_", ".") + ".mp4"
            vid_gt, full_gt = self.get_video_gt_annotations(video_id)
            success = self.generate_corrected_video_bb_annotations(
                video_id=video_id,
                video_gt_annotations=full_gt,
                overwrite=overwrite,
            )
            results[video_id] = success
        return results

    # ------------------------------------------------------------------
    # Visualization (from saved PKLs, no recomputation)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_corners(obj: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract 8×3 corners from an object dict (OBB only)."""
        for key in ("obb_floor_parallel",):
            sub = obj.get(key)
            if isinstance(sub, dict):
                for ck in ("corners_world", "corners_final"):
                    c = sub.get(ck)
                    if c is not None:
                        c = np.asarray(c, dtype=np.float32)
                        if c.shape == (8, 3):
                            return c
        for ck in ("corners_world", "corners_final"):
            c = obj.get(ck)
            if c is not None:
                c = np.asarray(c, dtype=np.float32)
                if c.shape == (8, 3):
                    return c
        return None

    @staticmethod
    def _corners_to_box_params(
        corners: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Derive (center, half_sizes, quaternion_xyzw) from 8×3 corners.

        Uses SVD to find the three principal axes of the cuboid, which works
        regardless of the corner ordering convention.
        """
        center = corners.mean(axis=0)
        centred = corners - center  # (8, 3)
        # SVD gives principal axes as rows of Vt
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
        # Project corners onto each axis → half-extent = max projection
        projections = centred @ Vt.T  # (8, 3)
        half_sizes = np.abs(projections).max(axis=0)  # (3,)
        # Build rotation matrix (rows of Vt = axes)
        R = Vt.T  # (3, 3) — columns are the box axes
        # Ensure proper rotation (det = +1)
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        quat = SciRot.from_matrix(R).as_quat().astype(np.float32)  # xyzw
        return center.astype(np.float32), half_sizes.astype(np.float32), quat

    def visualize_from_saved(
        self,
        video_id: str,
        *,
        app_id: str = "Corrected-WorldBBox",
    ) -> None:
        """Launch rerun visualization of saved corrected world bboxes."""
        import rerun as rr

        pkl_path = self.bbox_3d_obb_corrected_root_dir / f"{video_id[:-4]}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(f"[vis] Missing corrected bbox PKL: {pkl_path}")

        # 1) Load saved PKL
        with open(pkl_path, "rb") as f:
            saved = pickle.load(f)

        frames_map = saved.get("frames", {})
        cft = saved.get("corrected_floor_transform", {})
        original_gfs = cft.get("original_global_floor_sim") if isinstance(cft, dict) else None

        # 2) Load dynamic predictions (points, colors, cameras)
        P = self._load_points_for_video(video_id)
        points_S = P["points"]      # (N, H, W, 3)
        colors_S = P["colors"]      # (N, H, W, 3) uint8
        cam_S = P["camera_poses"]   # (N, 4, 4)
        stems = P["frame_stems"]    # list of str
        conf_S = P["conf"]          # (N, H, W) or None
        N = len(stems)

        # 3) Corrected floor transform (4×4 combined_transform)
        T_4x4 = None
        if isinstance(cft, dict) and "combined_transform_4x4" in cft:
            T_4x4 = np.asarray(cft["combined_transform_4x4"], dtype=np.float64)
            if T_4x4.shape != (4, 4):
                T_4x4 = T_4x4.reshape(4, 4)

        # 4) Floor mesh — keep in raw world frame (same as points & bboxes)
        gv = saved.get("gv")
        gf = saved.get("gf")
        gc = saved.get("gc")
        floor_verts = None
        floor_faces = None
        if gv is not None and gf is not None:
            gv0 = np.asarray(gv, dtype=np.float32)
            gf0 = _faces_u32(np.asarray(gf))
            # gv is already in world frame — no transform needed.
            # Points, bboxes (corners_world), and floor mesh all stay in world frame.
            floor_verts = gv0
            floor_faces = gf0

        # 5) Init rerun with a beautiful layout
        import rerun.blueprint as rrb

        rr.init(app_id, spawn=True)
        BASE = "world"
        rr.log(BASE, rr.ViewCoordinates.RUB, static=True)

        # Blueprint: large 3D panel left (75%), image panel right (25%)
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=f"{BASE}",
                    name="3D Scene",
                ),
                rrb.Spatial2DView(
                    origin=f"{BASE}/cam/pinhole",
                    name="Camera View",
                ),
                column_shares=[3, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

        # 6) World coordinate frame (static)
        axis_len = 1.0
        rr.log(f"{BASE}/origin/x", rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=[[axis_len, 0, 0]], colors=[[255, 0, 0]],
        ), static=True)
        rr.log(f"{BASE}/origin/y", rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=[[0, axis_len, 0]], colors=[[0, 255, 0]],
        ), static=True)
        rr.log(f"{BASE}/origin/z", rr.Arrows3D(
            origins=[[0, 0, 0]], vectors=[[0, 0, axis_len]], colors=[[0, 0, 255]],
        ), static=True)

        # 7) Frame loop
        for t_idx in range(N):
            frame_name = f"{stems[t_idx]}.png"
            rr.set_time_sequence("frame", t_idx)

            # Floor mesh (static, logged once per frame to keep timeline)
            if floor_verts is not None:
                fkw = {}
                if gc is not None:
                    fkw["vertex_colors"] = np.asarray(gc, dtype=np.uint8)
                rr.log(f"{BASE}/floor", rr.Mesh3D(
                    vertex_positions=floor_verts, triangle_indices=floor_faces, **fkw,
                ))

            # Point cloud (raw dynamic points — NOT corrected)
            pts = points_S[t_idx].reshape(-1, 3)
            cols = colors_S[t_idx].reshape(-1, 3)
            keep = np.isfinite(pts).all(axis=1)
            if conf_S is not None:
                cfs = conf_S[t_idx].reshape(-1)
                keep &= cfs > np.percentile(cfs[np.isfinite(cfs)], 5) if np.isfinite(cfs).any() else keep
            if keep.sum() > 0:
                rr.log(f"{BASE}/points", rr.Points3D(pts[keep], colors=cols[keep]))

            # Bbox cuboids — clear previous frame's entities first
            rr.log(f"{BASE}/bbox", rr.Clear(recursive=True))

            if frame_name in frames_map:
                objs = frames_map[frame_name].get("objects", [])

                # Separate GT and GDino objects for batch logging
                gt_centers, gt_halfs, gt_quats, gt_colors, gt_labels = [], [], [], [], []
                gd_centers, gd_halfs, gd_quats, gd_colors, gd_labels = [], [], [], [], []

                for oi, obj in enumerate(objs):
                    corners = self._extract_corners(obj)
                    if corners is None:
                        continue
                    label = obj.get("label", f"obj_{oi}")
                    src = obj.get("source", "gt")
                    center, half_sizes, q = self._corners_to_box_params(corners)

                    if src == "gdino":
                        gd_centers.append(center)
                        gd_halfs.append(half_sizes)
                        gd_quats.append(rr.Quaternion(xyzw=q))
                        gd_colors.append([0, 180, 255])
                        gd_labels.append(f"{label} (gdino)")
                    else:
                        gt_centers.append(center)
                        gt_halfs.append(half_sizes)
                        gt_quats.append(rr.Quaternion(xyzw=q))
                        gt_colors.append([255, 180, 0])
                        gt_labels.append(label)

                if gt_centers:
                    rr.log(f"{BASE}/bbox/gt", rr.Boxes3D(
                        centers=gt_centers, half_sizes=gt_halfs,
                        quaternions=gt_quats, colors=gt_colors, labels=gt_labels,
                    ))
                if gd_centers:
                    rr.log(f"{BASE}/bbox/gdino", rr.Boxes3D(
                        centers=gd_centers, half_sizes=gd_halfs,
                        quaternions=gd_quats, colors=gd_colors, labels=gd_labels,
                    ))

            # Camera frustum (raw world frame — same as points & bboxes)
            if cam_S is not None and t_idx < cam_S.shape[0]:
                cam = cam_S[t_idx].astype(np.float64)
                R_wc, t_wc = cam[:3, :3].astype(np.float32), cam[:3, 3].astype(np.float32)
                img = colors_S[t_idx]
                H_img, W_img = img.shape[:2]
                fov_y = 0.96
                fy = float(0.5 * H_img / np.tan(0.5 * fov_y))
                fx = fy
                cx, cy = W_img / 2.0, H_img / 2.0
                quat = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)
                rr.log(f"{BASE}/cam", rr.Transform3D(
                    translation=t_wc, rotation=rr.Quaternion(xyzw=quat),
                ))
                rr.log(f"{BASE}/cam/pinhole", rr.Pinhole(
                    focal_length=(fx, fy), principal_point=(cx, cy),
                    resolution=(W_img, H_img),
                ))
                rr.log(f"{BASE}/cam/pinhole/image", rr.Image(img))

        print(f"[vis] Rerun running for {video_id}. Scrub the 'frame' timeline.")


# =====================================================================
# CLI
# =====================================================================

def load_dataset(ag_root_directory: str):
    from dataloader.ag_dataset import StandardAG
    from torch.utils.data import DataLoader

    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    dataloader_train = DataLoader(
        train_dataset, shuffle=True, collate_fn=lambda b: b[0],
        pin_memory=False, num_workers=0,
    )
    dataloader_test = DataLoader(
        test_dataset, shuffle=False, collate_fn=lambda b: b[0], pin_memory=False,
    )
    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate corrected world 3D bboxes using manual floor corrections."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path", type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument(
        "--manual_corrections_dir", type=str,
        default="/data/rohith/ag/world_annotations/manual_corrections",
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--video", type=str, default=None, help="Process single video")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--corrections-only", action="store_true",
        help="Only process videos that have manual corrections (no dataloader needed)",
    )
    parser.add_argument(
        "--gdino-score-threshold", type=float, default=0.3,
        help="Min GDino detection score for 2D→3D lift (default: 0.3)",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize saved corrected bboxes with rerun (requires --video)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generator = CorrectedWorldBBoxGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        manual_corrections_dir=args.manual_corrections_dir,
    )
    generator.gdino_score_threshold = args.gdino_score_threshold

    if args.visualize:
        if not args.video:
            print("ERROR: --visualize requires --video")
            return
        generator.visualize_from_saved(args.video)
        return

    if args.video:
        # Single video mode
        vid_gt, full_gt = generator.get_video_gt_annotations(args.video)
        success = generator.generate_corrected_video_bb_annotations(
            video_id=args.video,
            video_gt_annotations=full_gt,
            overwrite=args.overwrite,
        )
        print(f"Result: {'SUCCESS' if success else 'FAILED'}")
        return

    # Default: process only videos that have manual corrections
    # (skip-if-exists is handled inside generate_corrected_video_bb_annotations)
    results = generator.generate_from_corrections_only(overwrite=args.overwrite)
    success = sum(1 for v in results.values() if v)
    print(f"\n[Summary] {success}/{len(results)} videos processed successfully")


def main_sample():
    """Process a single sample video, then launch rerun visualization."""
    args = parse_args()
    video_id = args.video or "015XE.mp4"

    generator = CorrectedWorldBBoxGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        manual_corrections_dir=args.manual_corrections_dir,
    )
    generator.gdino_score_threshold = args.gdino_score_threshold

    # Generate (skips if already exists unless --overwrite)
    vid_gt, full_gt = generator.get_video_gt_annotations(video_id)
    success = generator.generate_corrected_video_bb_annotations(
        video_id=video_id,
        video_gt_annotations=full_gt,
        overwrite=args.overwrite,
    )
    print(f"[main_sample] Generation result for {video_id}: {'SUCCESS' if success else 'FAILED'}")

    if success:
        generator.visualize_from_saved(video_id)


if __name__ == "__main__":
    # main()
    main_sample()

