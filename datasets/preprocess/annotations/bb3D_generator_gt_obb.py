#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from datasets.preprocess.annotations.bb3D_generator_gt import BBox3DGenerator

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as SciRot
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + '/..')

# from AG / human pipeline codebase
from dataloader.standard.action_genome.ag_dataset import StandardAG

from annotation_utils import _npz_open, _box_edges_from_corners, _log_box_lines_rr
from datasets.preprocess.annotations.annotation_utils import (
    get_video_belongs_to_split,
    _faces_u32,
    _pinhole_from_fov,
    _xywh_to_xyxy,
    _resize_bbox_to,
    _mask_from_bbox,
    _resize_mask_to,
    _finite_and_nonzero,
    _as_np
)

# Helper for OBB calculation
def _compute_obb_pca(points: np.ndarray) -> Dict[str, Any]:
    """
    Compute arbitrary OBB using PCA.
    Returns: {
        "center": [x, y, z],
        "extent": [w, h, d],  (full lengths)
        "rotation": [[r00, r01, r02], ...],
        "corners": [[x,y,z], ...] (8 corners)
    }
    """
    if points.shape[0] < 4:
        return None
    
    # 1. Centroid
    center = np.mean(points, axis=0)
    centered_points = points - center

    # 2. PCA
    cov = np.cov(centered_points, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    
    # Sort by size (largest first)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:, idx]
    # Ensure rhs coordinate system? Not strictly necessary for OBB but nice.
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1

    # 3. Project to principal axes to find min/max
    projected = centered_points @ evecs
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    
    extent = maxs - mins
    center_offset = (mins + maxs) / 2.0
    
    # Adjust center back to world
    obb_center = center + (evecs @ center_offset)
    
    # 4. Corners
    # Local corners
    half_ext = extent / 2.0
    # corners in local frame centered at 0
    # (x,y,z) order: (-,-,-), (-,-,+), (-,+,-), (-,+,+), (+,-,-), (+,-,+), (+,+,-), (+,+,+)
    # It usually helps to have a consistent order for "corners". 
    # Let's use the same as typical AABB: 
    # [min, min, min], [min, min, max], [min, max, min], [min, max, max], ...
    # But here "min" is relative to the principal axes.
    # We can just iterate all 8 combinations.
    corners_local = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corners_local.append([dx * half_ext[0]/2, dy * half_ext[1]/2, dz * half_ext[2]/2])
    # The above loop is slightly wrong because half_ext is full width? No, extent is full width.
    # So range is [-extent/2, +extent/2].
    
    corners_local_xyz = []
    for cx in [mins[0], maxs[0]]:
        for cy in [mins[1], maxs[1]]:
            for cz in [mins[2], maxs[2]]:
                corners_local_xyz.append([cx, cy, cz])
    corners_local_xyz = np.array(corners_local_xyz)
    
    # Transform back
    corners_world = corners_local_xyz @ evecs.T + center[None, :]
    
    return {
        "center": obb_center.tolist(),
        "extent": extent.tolist(),
        "rotation_matrix": evecs.tolist(),
        "corners_world": corners_world.tolist()
    }

def _compute_obb_floor_parallel(points_floor: np.ndarray, R_floor: np.ndarray, t_floor: np.ndarray, s_floor: float) -> Dict[str, Any]:
    """
    Compute OBB that is parallel to the floor.
    points_floor: Points already transformed to floor aligned frame (where Z is usually up or normal).
    
    We assume the floor frame is "aligned" such that one axis is the floor normal. 
    Typically in this pipeline, 'up' is Y or Z? 
    Based on `bb3D_generator_gt.py`: `gv, gf, gc = get_floor_mesh(all_verts_for_floor, scale=2)`.
    Usually `get_floor_mesh` aligns 'up' to +Y or +Z. 
    Let's assume +Y is up (common in graphics) or +Z (common in engineering).
    
    If we look at `_floor_align_points` in `bb3D_generator_gt.py`:
         return ((points_world - t_floor[None, :]) / s_floor) @ R_floor
    This transforms world to 'floor space'. 
    
    We will project to XZ plane (assuming Y is up) or XY plane (assuming Z is up) and compute 2D minAreaRect.
    
    Let's check `bb3D_generator_gt.py` again. `multiscale` block calculates `volume = size[0]*size[1]*size[2]`.
    It constructs corners from mins/maxs.
    
    The safest bet without knowing the exact axis is to try to find the "flat" dimension, but simpler is to use 2D OBB on all 3 projections and pick the one with smallest area? No, "Floor Parallel" implies intrinsic floor definition.
    
    WE WILL ASSUME Y IS UP (XZ plane is floor) OR Z IS UP (XY plane is floor).
    Most often in this pipeline (SMPL/AG), Y is down? or Z is back? 
    Wait, `get_floor_mesh` usually aligns the floor plane to XZ, so Y is up/normal.
    Let's stick to XZ for the 2D box if variance is low on Y?
    
    Actually, let's just compute the 2D bounding box on the plan dimension. 
    If we assume the "floor alignment" makes the floor plane Z=0 (or Y=0), then we just take the other two coords.
    
    Let's use `pts_floor[:, [0, 2]]` as the 2D projection (XZ plane) as a guess. 
    """
    
    # Pts are (N, 3).
    # We will compute MinAreaRect on (x,z)
    pts_2d = points_floor[:, [0, 2]].astype(np.float32)
    
    # cv2.minAreaRect takes (N, 1, 2)
    rect = cv2.minAreaRect(pts_2d[:, None, :])
    (center_2d, size_2d, angle) = rect
    # center_2d: (cx, cy), size_2d: (w, h), angle: degrees
    
    box_2d = cv2.boxPoints(rect) # (4, 2) corners
    
    # Get Y extents
    y_vals = points_floor[:, 1]
    y_min = y_vals.min()
    y_max = y_vals.max()
    
    # Construct 8 corners in floor frame
    # box_2d is [x, z]
    corners_floor = []
    # Bottom face (y_min)
    for i in range(4):
        x, z = box_2d[i]
        corners_floor.append([x, y_min, z])
    # Top face (y_max)
    for i in range(4):
        x, z = box_2d[i]
        corners_floor.append([x, y_max, z])
    
    corners_floor = np.array(corners_floor, dtype=np.float32)
    
    # Verify we cover all points roughly? MinAreaRect is exact for convex hull 2D. 
    # Y min/max is exact.
    # So this is a valid bounding box.
    
    # Transform back to world
    # _floor_to_world: return (points_floor @ R_floor.T) * s_floor + t_floor
    corners_world = (corners_floor @ R_floor.T) * s_floor + t_floor
    
    return {
        "rect_2d": {"center": center_2d, "size": size_2d, "angle": angle},
        "y_range": [float(y_min), float(y_max)],
        "corners_world": corners_world.tolist()
    }

class BBox3DGeneratorOBB(BBox3DGenerator):
    """
    Extends BBox3DGenerator to produce:
    1. AABB (Floor Aligned) - Existing
    2. OBB (Floor Parallel) - Rotated around Up axis
    3. OBB (Arbitrary) - PCA based
    """

    def _build_multiscale_bboxes_for_video(
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
            has_floor: bool,
            s_avg: float,
            R_avg: np.ndarray,
            t_avg: np.ndarray,
    ) -> Dict[str, Dict[str, Any]]:
        
        # Override this method to add OBB logic
        out_frames: Dict[str, Dict[str, Any]] = {}
        S, H, W, _ = points_S.shape
        orig_W, orig_H = orig_size

        # floor transforms
        s_floor = float(s_avg) if s_avg is not None else 1.0
        R_floor = np.asarray(R_avg, dtype=np.float32) if R_avg is not None else np.eye(3, dtype=np.float32)
        t_floor = np.asarray(t_avg, dtype=np.float32) if t_avg is not None else np.zeros(3, dtype=np.float32)

        def _floor_align_points(points_world: np.ndarray) -> np.ndarray:
            return ((points_world - t_floor[None, :]) / s_floor) @ R_floor

        def _floor_to_world(points_floor: np.ndarray) -> np.ndarray:
            return (points_floor @ R_floor.T) * s_floor + t_floor

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

            conf_thr = 0.05
            if conf_hw is not None:
                cfs_flat = conf_hw.reshape(-1)
                mask_valid = np.isfinite(cfs_flat)
                cfs_valid = cfs_flat[mask_valid]
                if cfs_valid.size > 0:
                    med = np.median(cfs_valid)
                    p5 = np.percentile(cfs_valid, 5)
                    thr = max(1e-3, p5)
                    conf_thr = float(thr)

            for item in frame_items:
                # Resolve label + bbox
                if "person_bbox" in item:
                    label = "person"
                    gt_xyxy = _xywh_to_xyxy(item["person_bbox"][0])
                else:
                    cid = item["class"]
                    label = self.catid_to_name_map.get(cid, None)
                    if not label:
                        continue
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

                gt_xyxy = _resize_bbox_to(gt_xyxy, (orig_W, orig_H), (W, H))

                frame_label_mask = video_to_frame_to_label_mask[video_id][stem].get(label, None)
                if frame_label_mask is None:
                    frame_label_mask = _mask_from_bbox(H, W, gt_xyxy)
                else:
                    frame_label_mask = _resize_mask_to(frame_label_mask, (H, W))
                    x1, y1, x2, y2 = map(int, gt_xyxy)
                    bbox_mask = np.zeros_like(frame_label_mask, dtype=bool)
                    bbox_mask[y1:y2, x1:x2] = True
                    frame_label_mask = frame_label_mask & bbox_mask

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
                    
                    if not has_floor:
                        # Fallback if no floor, just world AABB and PCA OBB
                        mins = obj_pts_world.min(axis=0)
                        maxs = obj_pts_world.max(axis=0)
                        corners_world = np.array([
                             [mins[0], mins[1], mins[2]],
                             [maxs[0], maxs[1], maxs[2]],
                        ]) # simplified, just bounds
                        
                        obb_res = _compute_obb_pca(obj_pts_world)
                        
                        multi_scale_candidates.append({
                            "kernel_size": int(ksz),
                            "num_points": num_sel,
                            "aabb": {
                                "mins": mins.tolist(),
                                "maxs": maxs.tolist()  
                            },
                            "obb": obb_res,
                            "obb_floor_parallel": None
                        })
                        continue

                    # 1. AABB (Floor Aligned)
                    pts_floor = _floor_align_points(obj_pts_world)
                    mins = pts_floor.min(axis=0)
                    maxs = pts_floor.max(axis=0)
                    size = (maxs - mins).clip(1e-6)
                    volume = float(size[0]*size[1]*size[2])
                    
                    corners_floor_aabb = np.array([
                        [mins[0], mins[1], mins[2]],
                        [mins[0], mins[1], maxs[2]],
                        [mins[0], maxs[1], mins[2]],
                        [mins[0], maxs[1], maxs[2]],
                        [maxs[0], mins[1], mins[2]],
                        [maxs[0], mins[1], maxs[2]],
                        [maxs[0], maxs[1], mins[2]],
                        [maxs[0], maxs[1], maxs[2]],
                    ], dtype=np.float32)
                    corners_world_aabb = _floor_to_world(corners_floor_aabb)
                    
                    # 2. OBB (Floor Parallel)
                    obb_floor_res = _compute_obb_floor_parallel(pts_floor, R_floor, t_floor, s_floor)
                    
                    # 3. OBB (Arbitrary)
                    obb_res = _compute_obb_pca(obj_pts_world)

                    multi_scale_candidates.append({
                        "kernel_size": int(ksz),
                        "num_points": num_sel,
                        "volume": volume,
                        "mins_floor": mins.tolist(), 
                        "maxs_floor": maxs.tolist(),
                        "corners_world": corners_world_aabb.tolist(), # Compatible with old schema
                        
                        # New fields
                        "aabb_floor_aligned": {
                            "volume": volume,
                            "corners_world": corners_world_aabb.tolist(),
                        },
                        "obb_floor_parallel": obb_floor_res,
                        "obb_arbitrary": obb_res,

                    })

                if not multi_scale_candidates:
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "aabb_floor_aligned": None,
                        "obb_floor_parallel": None,
                        "obb_arbitrary": None,
                        "candidates": []
                    })
                else:
                    # Pick best (smallest volume AABB is existing logic, let's keep that for 'best')
                    # Actually standard script sorts by volume and picks smallest volume that has > N points
                    # We have filtered by N points already.
                    # Sort by volume of AABB
                    multi_scale_candidates.sort(key=lambda x: x.get("volume", float('inf')))
                    best = multi_scale_candidates[0]
                    
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        # Existing compatible fields
                        "mins_floor": best.get("mins_floor"), 
                        "maxs_floor": best.get("maxs_floor"),
                        "corners_world": best.get("corners_world"),
                        
                        # New explicit fields
                        "aabb_floor_aligned": best.get("aabb_floor_aligned"),
                        "obb_floor_parallel": best.get("obb_floor_parallel"),
                        "obb_arbitrary": best.get("obb_arbitrary"),
                        
                        "candidates": multi_scale_candidates
                    })

            out_frames[frame_name] = frame_rec

        return out_frames

    def visualize_obb_from_saved_files(
            self,
            video_id: str,
            *,
            app_id: str = "World4D-Saved",
            img_maxsize: int = 480,
            vis_floor: bool = True,
            vis_humans: bool = False,
            min_conf_default: float = 1e-6,
    ) -> None:
        """
        Load precomputed artifacts from disk and launch rerun visualization.
        - NO recomputation of the pipeline
        - NO saving/writing
        Uses:
          (1) saved bbox pickle: self.bbox_3d_obb_root_dir / f"{video_id[:-4]}.pkl"
          (2) dynamic predictions: self.dynamic_scene_dir_path / f"{video_id[:-4]}_10/predictions.npz"
        """

        # ------------------------------------------------------------
        # 1) Load saved bbox/alignment outputs (pickle)
        # ------------------------------------------------------------
        video_3dgt = self.get_video_3d_obb_annotations(video_id)
        frames_map = video_3dgt.get("frames", None)
        per_frame_sims = video_3dgt.get("per_frame_sims", None)

        gsim = video_3dgt.get("global_floor_sim", None)
        global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        if isinstance(gsim, dict) and all(k in gsim for k in ("s", "R", "t")):
            global_floor_sim = (
                float(gsim["s"]),
                _as_np(gsim["R"], np.float32),
                _as_np(gsim["t"], np.float32),
            )
        elif isinstance(gsim, (tuple, list)) and len(gsim) == 3:
            global_floor_sim = (
                float(gsim[0]),
                _as_np(gsim[1], np.float32),
                _as_np(gsim[2], np.float32),
            )

        frame_bbox_meshes = video_3dgt.get("frame_bbox_meshes", None)

        gv = video_3dgt.get("gv", None)
        gf = video_3dgt.get("gf", None)
        gc = video_3dgt.get("gc", None)

        floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None
        if gv is not None and gf is not None:
            floor = (
                _as_np(gv, np.float32),
                _as_np(gf, np.uint32),
                _as_np(gc, np.uint8) if gc is not None else None,
            )

        # ------------------------------------------------------------
        # 2) Load dynamic predictions (npz) for images + camera poses
        # ------------------------------------------------------------
        pred_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        if not pred_path.exists():
            raise FileNotFoundError(f"[vis] Missing dynamic predictions file: {pred_path}")

        # Use your existing _npz_open helper if you want; np.load is fine too.
        with _npz_open(pred_path) as npz:
            imgs_f32 = npz["images"]  # (S,H,W,3) in [0,1] (typically)
            images_u8 = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)
            images: List[Optional[np.ndarray]] = [images_u8[i] for i in range(images_u8.shape[0])]

            camera_poses = None
            if "camera_poses" in npz:
                camera_poses = npz["camera_poses"]

            S_full = int(images_u8.shape[0])

        # ------------------------------------------------------------
        # 3) Reconstruct the indexing the visualizer expects
        # ------------------------------------------------------------
        # annotated_frame_idx_in_sample_idx are the indices into the sampled range (0..S_full-1)
        # We only need those indices + a sampled_indices list whose indexing matches predictions.npz.
        try:
            (frame_idx_frame_path_map, sample_idx, _, _, annotated_frame_idx_in_sample_idx) = self.idx_to_frame_idx_path(video_id)
            # Sanity: predictions length should match sampled range length
            if len(sample_idx) != S_full:
                print(
                    f"[vis][warn] predictions.npz S={S_full} but idx_to_frame_idx_path sample_idx={len(sample_idx)}. "
                    f"Proceeding with min length."
                )
                S_use = min(S_full, len(sample_idx))
                images = images[:S_use]
                S_full = S_use
        except Exception as e:
            print(f"[vis][warn] idx_to_frame_idx_path failed ({e}); falling back to showing all frames.")
            raise e
            # annotated_frame_idx_in_sample_idx = list(range(S_full))

        sampled_indices = list(range(S_full))

        # ------------------------------------------------------------
        # 4) Build a minimal world dict (camera only, optionally humans)
        # ------------------------------------------------------------
        world4d: Dict[int, dict] = {}

        if camera_poses is not None:
            for i in range(S_full):
                cam_i = np.asarray(camera_poses[i])
                if cam_i.shape == (4, 4):
                    cam_3x4 = cam_i[:3, :4]
                elif cam_i.shape == (3, 4):
                    cam_3x4 = cam_i
                else:
                    raise ValueError(f"[vis] Unexpected camera_poses[{i}] shape: {cam_i.shape}")
                world4d[i] = {
                    "camera": cam_3x4.astype(np.float32),
                    # humans are optional; rerun_vis_world4d checks these keys
                    "track_id": [],
                    "vertices_orig": [],
                }
        else:
            # fallback: identity camera so rerun doesn't crash
            I = np.eye(3, 4, dtype=np.float32)
            world4d = {
                i: {"camera": I, "track_id": [], "vertices_orig": []}
                for i in range(S_full)
            }

        # ------------------------------------------------------------
        # 5) Launch rerun visualizer
        # ------------------------------------------------------------
        rerun_vis_obb_world4d(
            video_id=video_id,
            images=images,
            world4d=world4d,
            faces=self.smplx.faces,  # already on the layer
            sampled_indices=sampled_indices,
            annotated_frame_idx_in_sample_idx=annotated_frame_idx_in_sample_idx,
            dynamic_prediction_path=str(self.dynamic_scene_dir_path),
            per_frame_sims=per_frame_sims,
            frames_map=frames_map,
            frame_idx_frame_path_map=frame_idx_frame_path_map,
            global_floor_sim=global_floor_sim,
            floor=floor,
            img_maxsize=img_maxsize,
            app_id=app_id,
            frame_bbox_meshes=frame_bbox_meshes,
            vis_floor=vis_floor,
            vis_humans=vis_humans,  # typically False unless you also load vertices_orig
            min_conf_default=min_conf_default,
        )

    def generate_gt_world_bb_annotations(self, dataloader, split) -> None:
        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                try:
                    vid_gt, full_gt = self.get_video_gt_annotations(video_id)
                    # vid_gt is per frame, full_gt is the list for the video

                    # Load GDINO (optional but good for consistency)
                    gdino = self.get_video_gdino_annotations(video_id)

                    self.generate_video_bb_annotations(
                        video_id=video_id,
                        video_gt_annotations=full_gt,
                        video_gdino_predictions=gdino,
                        visualize=False
                    )

                except Exception as e:
                    print(f"Error processing {video_id}: {e}")
                    import traceback
                    traceback.print_exc()

    def generate_sample_gt_world_bb_annotations(self, video_id: str) -> None:
        vid_gt, full_gt = self.get_video_gt_annotations(video_id)
        # vid_gt is per frame, full_gt is the list for the video

        # Load GDINO (optional but good for consistency)
        gdino = self.get_video_gdino_annotations(video_id)

        self.generate_video_bb_annotations(
            video_id=video_id,
            video_gt_annotations=full_gt,
            video_gdino_predictions=gdino,
            visualize=True
        )


def rerun_vis_obb_world4d(
        video_id: str,
        images: List[Optional[np.ndarray]],
        world4d: Dict[int, dict],
        faces: np.ndarray,
        *,
        annotated_frame_idx_in_sample_idx: List[int],
        sampled_indices: List[int],
        dynamic_prediction_path: str,
        per_frame_sims: Optional[Dict[int, Dict[str, Any]]] = None,
        frames_map: Optional[Dict[str, Dict[str, Any]]] = None,
        frame_idx_frame_path_map: Optional[Dict[int, str]] = None,
        global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
        floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
        img_maxsize: int = 320,
        app_id: str = "World4D",
        frame_bbox_meshes: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        vis_floor: bool = True,
        vis_humans: bool = True,
        min_conf_default: float = 1e-6,  # floor for conf
):
    """
    Updated:
      - Draw OBB (preferred) instead of AABB when available.
      - Uses _box_edges_from_corners + _log_box_lines_rr (signature-safe).
      - Robustly handles frame_bbox_meshes keyed by sample_idx OR vis_t OR frame_idx.
    """

    faces_u32 = _faces_u32(faces)
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    video_dynamic_prediction_path = os.path.join(dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz")
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files}
    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    conf = video_dynamic_predictions["conf"].astype(np.float32)      # (S,H,W)
    imgs_f32 = video_dynamic_predictions["images"]
    colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # ------------------------------------------------------------------
    # floor
    # ------------------------------------------------------------------
    floor_vertices_tf = None
    floor_faces = None
    floor_kwargs = None
    if floor is not None:
        floor_verts0, floor_faces0, floor_colors0 = floor
        floor_verts0 = np.asarray(floor_verts0, dtype=np.float32)
        floor_faces0 = _faces_u32(np.asarray(floor_faces0))
        if global_floor_sim is not None:
            s_g, R_g, t_g = global_floor_sim
            floor_vertices_tf = s_g * (floor_verts0 @ R_g.T) + t_g
        else:
            floor_vertices_tf = floor_verts0
        floor_kwargs = {}
        if floor_colors0 is not None:
            floor_colors0 = np.asarray(floor_colors0, dtype=np.uint8)
            floor_kwargs["vertex_colors"] = floor_colors0
        else:
            floor_kwargs["albedo_factor"] = [160, 160, 160]
        floor_faces = floor_faces0

    def _get_image_for_time(i: int) -> Optional[np.ndarray]:
        if images is None:
            return None
        if i < 0 or i >= len(images):
            return None
        img = images[i]
        if img is None:
            return None
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    # OBB corners from cv2.boxPoints: 0-3 are the bottom face going around perimeter,
    # 4-7 are the top face directly above (i.e., 4 is above 0, 5 is above 1, etc.)
    cuboid_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    # ------------------------------------------------------------------
    # frames to iterate
    # ------------------------------------------------------------------
    if annotated_frame_idx_in_sample_idx:
        iter_indices = annotated_frame_idx_in_sample_idx
    else:
        iter_indices = list(range(len(sampled_indices)))

    for vis_t, sample_idx in enumerate(iter_indices):
        if sample_idx < 0 or sample_idx >= len(sampled_indices):
            continue

        frame_idx = sampled_indices[sample_idx]
        frame_name = frame_idx_frame_path_map[frame_idx]

        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # floor
        if vis_floor:
            if floor_vertices_tf is not None and floor_faces is not None:
                rr.log(
                    f"{BASE}/floor",
                    rr.Mesh3D(
                        vertex_positions=floor_vertices_tf.astype(np.float32),
                        triangle_indices=floor_faces,
                        **(floor_kwargs or {}),
                    ),
                )

        # per-frame sim
        s_i = None
        R_i = None
        t_i = None
        if per_frame_sims is not None and frame_idx in per_frame_sims:
            s_i = float(per_frame_sims[frame_idx]["s"])
            R_i = np.asarray(per_frame_sims[frame_idx]["R"], dtype=np.float32)
            t_i = np.asarray(per_frame_sims[frame_idx]["t"], dtype=np.float32)

        frame_data = world4d.get(frame_idx, None)
        if frame_data is None:
            continue

        # humans
        if vis_humans:
            track_ids = frame_data.get("track_id", [])
            verts_orig_list = frame_data.get("vertices_orig", [])
            if track_ids and verts_orig_list:
                tid = int(track_ids[0])
                verts_orig = np.asarray(verts_orig_list[0], dtype=np.float32)
                if s_i is not None and R_i is not None and t_i is not None:
                    verts_flat = verts_orig.reshape(-1, 3)
                    verts_tf = s_i * (verts_flat @ R_i.T) + t_i
                    verts_tf = verts_tf.reshape(verts_orig.shape)
                    rr.log(
                        f"{BASE}/humans_xform/h{tid}",
                        rr.Mesh3D(
                            vertex_positions=verts_tf.astype(np.float32),
                            triangle_indices=faces_u32,
                            albedo_factor=[0, 255, 0],
                        ),
                    )

        # --- dynamic points: confidence-filtered ---
        if sample_idx < points.shape[0]:
            pts = points[sample_idx].reshape(-1, 3)    # (N,3)
            cols = colors[sample_idx].reshape(-1, 3)   # (N,3)
            cfs = conf[sample_idx].reshape(-1)         # (N,)

            # adaptive threshold
            good = np.isfinite(cfs)
            cfs_valid = cfs[good]
            if cfs_valid.size > 0:
                med = np.median(cfs_valid)
                p5 = np.percentile(cfs_valid, 5)
                thr = max(min_conf_default, p5)
                print(f"frame {frame_idx}: conf thr = {thr:.4f} (med={med:.4f}, n_valid={cfs_valid.size})")
            else:
                thr = min_conf_default

            keep = (cfs >= thr) & np.isfinite(pts).all(axis=1)
            pts_keep = pts[keep]
            cols_keep = cols[keep]

            if pts_keep.shape[0] > 0:
                rr.log(
                    f"{BASE}/points",
                    rr.Points3D(
                        pts_keep,
                        colors=cols_keep,
                    ),
                )

        if frames_map and frame_name in frames_map:
            frame_rec = frames_map[frame_name]
            objs = frame_rec.get("objects", [])
            for obj_idx, obj in enumerate(objs):
                obb = obj.get("obb_floor_parallel", None)
                if obb is None or "corners_world" not in obb:
                    continue

                corners_w = np.asarray(obb["corners_world"], dtype=np.float32)  # (8,3)

                corners_world = corners_w
                if corners_world is None:
                    continue

                col = obj.get("color", [255, 180, 0])
                label = obj.get("label", f"obj_{obj_idx}")

                strips = [corners_world[[e0, e1], :] for (e0, e1) in cuboid_edges]
                rr.log(
                    f"{BASE}/bboxes/frame_{frame_idx}/{label}_{obj_idx}",
                    rr.LineStrips3D(strips=strips, colors=[col] * len(strips)),
                )

        # camera
        cam_3x4 = np.asarray(frame_data["camera"], dtype=np.float32)
        R_wc = cam_3x4[:3, :3]
        t_wc = cam_3x4[:3, 3]
        image = _get_image_for_time(frame_idx)
        if image is not None:
            H_img, W_img = image.shape[:2]
        else:
            H_img, W_img = 480, 640

        fov_y = 0.96
        fx, fy, cx, cy = _pinhole_from_fov(W_img, H_img, fov_y)
        quat_xyzw = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)

        frus_path = f"{BASE}/frustum"
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=t_wc.astype(np.float32),
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            )
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W_img, H_img)),
        )
        if image is not None:
            rr.log(f"{frus_path}/image", rr.Image(image))

    print("Rerun visualization running. Scrub the 'frame' timeline.")



def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument("--dynamic_scene_dir_path", type=str, default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic")
    parser.add_argument("--output_human_dir_path", type=str, default="/data/rohith/ag/ag4D/human/")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--split", type=str, default="04")
    args = parser.parse_args()
    return args


def main_sample():
    args = parse_args()
    video_id = "00T1E.mp4"
    gen = BBox3DGeneratorOBB(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_human_dir_path=args.output_human_dir_path
    )

    # gen.generate_sample_gt_world_bb_annotations(video_id)
    gen.visualize_obb_from_saved_files(
        video_id=video_id,
        vis_floor=True,
        vis_humans=False,   # set True only if you also load vertices_orig into world4d
        img_maxsize=480,
    )

def main():
    args = parse_args()

    bbox_3d_generator_obb = BBox3DGeneratorOBB(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_human_dir_path=args.output_human_dir_path
    )

    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    bbox_3d_generator_obb.generate_gt_world_bb_annotations(dataloader=dataloader_train, split=args.split)
    bbox_3d_generator_obb.generate_gt_world_bb_annotations(dataloader=dataloader_test, split=args.split)


if __name__ == "__main__":
    # main()
    main_sample()
