#!/usr/bin/env python3
"""
bb3D_bridge_generator_obb.py  —  Bridge script

Reads the existing GT-only OBB PKL (bbox_annotations_3d_obb/),
lifts GDino 2D detections to 3D OBB using point clouds + floor transform,
applies the WORLD→FINAL transform, and saves the merged result to
bbox_annotations_3d_obb_final/.

This replaces the need to run frame_bbox_3D_gt_obb.py as a separate step.
Points are loaded from disk only ONCE.
"""
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
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets.preprocess.annotations.raw.frame_bbox_3D_base import (
    FrameToWorldAnnotationsBase,
)
from datasets.preprocess.annotations.annotation_utils import (
    _faces_u32,
    _mask_from_bbox,
    _resize_mask_to,
    _finite_and_nonzero,
    _npz_open,
    _is_empty_array,
    get_video_belongs_to_split,
)
from dataloader.ag_dataset import StandardAG
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Helpers (ported from bb3D_generator_gt_obb.py)
# ---------------------------------------------------------------------------


def _compute_obb_floor_parallel(
    points_floor: np.ndarray,
    R_floor: np.ndarray,
    t_floor: np.ndarray,
    s_floor: float,
) -> Dict[str, Any]:
    """
    Compute OBB that is parallel to the floor.
    Projects points onto the XZ plane (Y is up/normal in floor frame),
    computes cv2.minAreaRect, and extrudes to full Y extent.
    Returns corners in WORLD coordinates.
    """
    pts_2d = points_floor[:, [0, 2]].astype(np.float32)
    rect = cv2.minAreaRect(pts_2d[:, None, :])
    (center_2d, size_2d, angle) = rect
    box_2d = cv2.boxPoints(rect)  # (4, 2)

    y_vals = points_floor[:, 1]
    y_min = float(y_vals.min())
    y_max = float(y_vals.max())

    corners_floor = []
    for i in range(4):
        x, z = box_2d[i]
        corners_floor.append([x, y_min, z])
    for i in range(4):
        x, z = box_2d[i]
        corners_floor.append([x, y_max, z])
    corners_floor = np.array(corners_floor, dtype=np.float32)

    corners_world = (corners_floor @ R_floor.T) * s_floor + t_floor

    return {
        "rect_2d": {"center": tuple(center_2d), "size": tuple(size_2d), "angle": float(angle)},
        "y_range": [y_min, y_max],
        "corners_world": corners_world.tolist(),
    }


def _floor_align_points(
    pts_world: np.ndarray,
    s_floor: float,
    R_floor: np.ndarray,
    t_floor: np.ndarray,
) -> np.ndarray:
    return ((pts_world - t_floor[None, :]) / s_floor) @ R_floor


def _normalize_label(lbl: str) -> str:
    mapping = {
        "closet/cabinet": "closet",
        "cup/glass/bottle": "cup",
        "paper/notebook": "paper",
        "sofa/couch": "sofa",
        "phone/camera": "phone",
    }
    return mapping.get(lbl, lbl)


def _resize_bbox_to(
    bbox_xyxy: List[float],
    from_size: Tuple[int, int],
    to_size: Tuple[int, int],
) -> List[float]:
    """Resize bbox from (from_W, from_H) to (to_W, to_H)."""
    fw, fh = from_size
    tw, th = to_size
    sx, sy = tw / fw, th / fh
    return [bbox_xyxy[0] * sx, bbox_xyxy[1] * sy, bbox_xyxy[2] * sx, bbox_xyxy[3] * sy]


# ---------------------------------------------------------------------------
# Bridge class
# ---------------------------------------------------------------------------

EROSION_KERNEL_SIZES = [0, 3, 5, 7, 9]
MIN_POINTS_PER_SCALE = 50


class BBox3DBridgeOBB(FrameToWorldAnnotationsBase):
    """
    Bridge: loads existing GT-only OBB PKL, lifts GDino 2D→3D OBB,
    applies WORLD→FINAL, saves merged final PKL.
    """

    def bridge_video(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
        gdino_score_thr: float = 0.3,
    ) -> Optional[Path]:
        """
        Single-pass pipeline:
          1. Load GT OBB PKL + point clouds + GDino predictions
          2. Lift GDino 2D→3D OBB for labels missing from GT
          3. Tag all objects with source="gt"/"gdino"
          4. Apply WORLD→FINAL transform
          5. Save to bbox_annotations_3d_obb_final/
        """
        out_path = self.bbox_3d_obb_final_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[bridge][{video_id}] exists: {out_path} (overwrite=False). Skipping.")
            return out_path

        # ==================================================================
        # Phase 1 — Load everything
        # ==================================================================
        video_3dgt_obb = self.get_video_3d_obb_annotations(video_id)
        if video_3dgt_obb is None:
            print(f"[bridge][{video_id}] OBB PKL not found. Skipping.")
            return None

        frames_map = video_3dgt_obb.get("frames", None)
        if frames_map is None or not frames_map:
            print(f"[bridge][{video_id}] No 'frames' in OBB PKL. Skipping.")
            return None

        gfs = video_3dgt_obb.get("global_floor_sim", None)
        if gfs is None:
            print(f"[bridge][{video_id}] global_floor_sim missing. Skipping.")
            return None

        s_floor = float(gfs["s"])
        R_floor = np.asarray(gfs["R"], dtype=np.float32)
        t_floor = np.asarray(gfs["t"], dtype=np.float32)
        global_floor_sim = (s_floor, R_floor, t_floor)

        # World → FINAL transform
        Tinfo = self._compute_world_to_final(global_floor_sim=global_floor_sim)
        origin_world = Tinfo["origin_world"]
        A = Tinfo["A_world_to_final"]

        # Load point clouds (ONCE)
        print(f"[bridge][{video_id}] Loading point clouds …")
        P = self._load_original_points_for_video(video_id)
        points_S = P["points"]      # (S, H, W, 3)
        conf_S = P["conf"]          # (S, H, W) or None
        stems_S = P["frame_stems"]  # ["000010", ...]
        cams_world = P["camera_poses"]
        S, H, W, _ = points_S.shape

        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # Original image size (for resizing GDino bboxes)
        sample_image_path = self.frame_annotated_dir_path / video_id / f"{stems_S[0]}.png"
        orig_img = cv2.imread(str(sample_image_path))
        if orig_img is not None:
            orig_H, orig_W = orig_img.shape[:2]
        else:
            orig_H, orig_W = H, W

        # Load GDino predictions
        try:
            gdino_predictions = self.get_video_gdino_annotations(video_id)
        except ValueError:
            gdino_predictions = {}
            print(f"[bridge][{video_id}] No GDino predictions found (continuing with GT only).")

        # Collect video GT labels for filtering
        try:
            _, video_gt_annotations = self.get_video_gt_annotations(video_id)
            video_gt_labels = set()
            for frame_items in video_gt_annotations:
                for item in frame_items:
                    if "person_bbox" in item:
                        video_gt_labels.add("person")
                    elif "class" in item:
                        cid = item["class"]
                        lbl = self.catid_to_name_map.get(cid, None)
                        if lbl:
                            video_gt_labels.add(_normalize_label(lbl))
        except FileNotFoundError:
            video_gt_labels = None

        # ==================================================================
        # Phase 2 — Tag GT objects + lift GDino
        # ==================================================================
        stats = {"gt": 0, "gdino_added": 0, "gdino_skipped_label": 0,
                 "gdino_skipped_score": 0, "gdino_skipped_exists": 0,
                 "gdino_failed_lift": 0}

        for frame_name, frame_rec in frames_map.items():
            objs = frame_rec.get("objects", [])

            # Tag existing GT objects
            for obj in objs:
                if "source" not in obj:
                    obj["source"] = "gt"
                    stats["gt"] += 1

            # Determine which GT labels are already in this frame
            gt_labels_in_frame = set()
            for obj in objs:
                lbl = obj.get("label", None)
                if lbl:
                    gt_labels_in_frame.add(_normalize_label(lbl))

            # Find the point cloud index for this frame
            stem = Path(frame_name).stem
            if stem not in stem_to_idx:
                continue
            sidx = stem_to_idx[stem]
            pts_hw3 = points_S[sidx]       # (H, W, 3)
            conf_hw = conf_S[sidx] if conf_S is not None else None
            frame_non_zero_pts = _finite_and_nonzero(pts_hw3)

            # Confidence threshold (same logic as bb3D_generator_gt_obb.py)
            conf_thr = 0.05
            if conf_hw is not None:
                cfs_flat = conf_hw.reshape(-1)
                mask_valid = np.isfinite(cfs_flat)
                cfs_valid = cfs_flat[mask_valid]
                if cfs_valid.size > 0:
                    p5 = np.percentile(cfs_valid, 5)
                    conf_thr = float(max(1e-3, p5))

            # Get GDino detections for this frame
            gdino_frame = gdino_predictions.get(frame_name, {})
            gdino_boxes = gdino_frame.get("boxes", [])
            gdino_labels = gdino_frame.get("labels", [])
            gdino_scores = gdino_frame.get("scores", [])

            if _is_empty_array(gdino_boxes):
                continue

            for box, lbl_raw, score in zip(gdino_boxes, gdino_labels, gdino_scores):
                score = float(score)
                if score < gdino_score_thr:
                    stats["gdino_skipped_score"] += 1
                    continue

                lbl = _normalize_label(lbl_raw)

                # Skip if label not in video's GT label set
                if video_gt_labels is not None and lbl not in video_gt_labels:
                    stats["gdino_skipped_label"] += 1
                    continue

                # Skip if GT already has this label in this frame
                if lbl in gt_labels_in_frame:
                    stats["gdino_skipped_exists"] += 1
                    continue

                # Lift 2D → 3D
                gdino_xyxy = [float(v) for v in box]
                gdino_xyxy_resized = _resize_bbox_to(gdino_xyxy, (orig_W, orig_H), (W, H))

                # Create mask from bbox
                bbox_mask = _mask_from_bbox(H, W, gdino_xyxy_resized)

                # Multiscale erosion — pick tightest OBB
                best_candidate = None
                for ksz in EROSION_KERNEL_SIZES:
                    if ksz == 0:
                        sel_mask = bbox_mask.astype(bool)
                    else:
                        mask_u8 = bbox_mask.astype(np.uint8)
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                        eroded = cv2.erode(mask_u8, kernel, iterations=1)
                        sel_mask = eroded.astype(bool)

                    sel = sel_mask & frame_non_zero_pts
                    if conf_hw is not None:
                        sel &= (conf_hw > conf_thr)
                    num_sel = int(sel.sum())
                    if num_sel < MIN_POINTS_PER_SCALE:
                        continue

                    obj_pts_world = pts_hw3[sel].reshape(-1, 3).astype(np.float32)
                    pts_floor = _floor_align_points(obj_pts_world, s_floor, R_floor, t_floor)
                    mins = pts_floor.min(axis=0)
                    maxs = pts_floor.max(axis=0)
                    size = (maxs - mins).clip(1e-6)
                    volume = float(size[0] * size[1] * size[2])

                    # AABB corners in floor space → world
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
                    corners_world_aabb = (corners_floor_aabb @ R_floor.T) * s_floor + t_floor

                    # OBB floor parallel
                    obb_floor_res = _compute_obb_floor_parallel(pts_floor, R_floor, t_floor, s_floor)

                    candidate = {
                        "kernel_size": int(ksz),
                        "num_points": num_sel,
                        "volume": volume,
                        "aabb_floor_aligned": {
                            "volume": volume,
                            "corners_world": corners_world_aabb.tolist(),
                        },
                        "obb_floor_parallel": obb_floor_res,
                    }
                    if best_candidate is None or volume < best_candidate["volume"]:
                        best_candidate = candidate

                if best_candidate is None:
                    stats["gdino_failed_lift"] += 1
                    continue

                # Build object dict
                gdino_obj = {
                    "label": lbl,
                    "source": "gdino",
                    "gdino_score": score,
                    "gdino_bbox_xyxy": gdino_xyxy,
                    "aabb_floor_aligned": best_candidate["aabb_floor_aligned"],
                    "obb_floor_parallel": best_candidate["obb_floor_parallel"],
                    "candidates": [best_candidate],
                }
                frame_rec["objects"].append(gdino_obj)
                gt_labels_in_frame.add(lbl)  # prevent duplicates
                stats["gdino_added"] += 1

        print(
            f"[bridge][{video_id}] Stats: "
            f"gt={stats['gt']}, gdino_added={stats['gdino_added']}, "
            f"gdino_skipped(score={stats['gdino_skipped_score']}, "
            f"label={stats['gdino_skipped_label']}, "
            f"exists={stats['gdino_skipped_exists']}, "
            f"failed={stats['gdino_failed_lift']})"
        )

        # ==================================================================
        # Phase 3 — WORLD → FINAL transform (same as frame_bbox_3D_gt_obb.py)
        # ==================================================================

        # Points FINAL
        pts_flat = points_S.reshape(-1, 3)
        pts_final_flat = self._apply_world_to_final_points_row(
            pts_flat, origin_world=origin_world, A_world_to_final=A
        )
        # Don't store full points in PKL — too large. Only for vis.

        # Cameras FINAL
        cams_final = None
        if cams_world is not None:
            cams_final_list = []
            for i in range(min(S, cams_world.shape[0])):
                cams_final_list.append(
                    self._apply_world_to_final_camera_pose(
                        cams_world[i],
                        origin_world=origin_world,
                        A_world_to_final=A,
                    )
                )
            cams_final = np.stack(cams_final_list, axis=0).astype(np.float32)

        # BBoxes FINAL
        obb_bbox_frames_final: Dict[str, Any] = {}
        for frame_name, frame_rec in frames_map.items():
            objs = frame_rec.get("objects", [])
            if not objs:
                continue
            out_objs = []
            for obj in objs:
                bb = obj.get("obb_floor_parallel", None)
                if bb is None or "corners_world" not in bb:
                    continue
                corners_world = np.asarray(bb["corners_world"], dtype=np.float32)
                corners_final = self._apply_world_to_final_points_row(
                    corners_world, origin_world=origin_world, A_world_to_final=A
                ).astype(np.float32)

                out_obj = dict(obj)
                out_obj["corners_final"] = corners_final
                out_objs.append(out_obj)

            if out_objs:
                obb_bbox_frames_final[frame_name] = {"objects": out_objs}

        # Floor FINAL
        gv = video_3dgt_obb.get("gv", None)
        gf = video_3dgt_obb.get("gf", None)
        gc = video_3dgt_obb.get("gc", None)
        floor_final = None
        if gv is not None and gf is not None:
            gv0 = np.asarray(gv, dtype=np.float32)
            gf0 = _faces_u32(np.asarray(gf))
            floor_world = s_floor * (gv0 @ R_floor.T) + t_floor[None, :]
            floor_final_v = self._apply_world_to_final_points_row(
                floor_world, origin_world=origin_world, A_world_to_final=A
            ).astype(np.float32)
            floor_final = {"vertices": floor_final_v, "faces": gf0}
            if gc is not None:
                floor_final["colors"] = np.asarray(gc, dtype=np.uint8)

        # ==================================================================
        # Phase 4 — Save
        # ==================================================================
        video_3dgt_updated = dict(video_3dgt_obb)
        video_3dgt_updated["frames_final"] = {
            "frame_stems": stems_S,
            "camera_poses": cams_final,
            "obb_bbox_frames": obb_bbox_frames_final,
            "floor": floor_final,
        }
        video_3dgt_updated["world_to_final"] = {
            "origin_world": origin_world,
            "A_world_to_final": A,
        }

        saved_path = self.save_video_3d_obb_annotations_final(video_id, video_3dgt_updated)
        print(f"[bridge][{video_id}] Saved merged final PKL: {saved_path}")
        return saved_path

    # ======================================================================
    # Visualization (self-contained, adapted from frame_bbox_3D_base.py)
    # ======================================================================

    def visualize_bridge_result(
        self,
        video_id: str,
        *,
        app_id: str = "Bridge-FinalOnly",
        img_maxsize: int = 480,
        min_conf_default: float = 1e-6,
    ) -> None:
        """
        Load saved final PKL and visualize with rerun.
        Color-codes bboxes by source: green=GT, orange=GDino.
        """
        final_pkl = self.bbox_3d_obb_final_root_dir / f"{video_id[:-4]}.pkl"
        if not final_pkl.exists():
            raise FileNotFoundError(f"Final PKL not found: {final_pkl}")

        with open(final_pkl, "rb") as f:
            rec = pickle.load(f)

        frames_final = rec.get("frames_final", None)
        if frames_final is None:
            raise ValueError(f"[bridge-vis][{video_id}] frames_final missing in {final_pkl}")

        origin_world = rec["world_to_final"]["origin_world"]
        A = rec["world_to_final"]["A_world_to_final"]

        # Load points and transform to FINAL
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)
        S, H, W, _ = points_world.shape
        pts_flat = points_world.reshape(-1, 3)
        pts_final_flat = self._apply_world_to_final_points_row(
            pts_flat, origin_world=origin_world, A_world_to_final=A
        )
        points_final = pts_final_flat.reshape(S, H, W, 3).astype(np.float32)
        colors_S = P.get("colors", None)
        conf_S = P.get("conf", None)

        stems_S = frames_final["frame_stems"]
        camera_poses_S = frames_final["camera_poses"]
        obb_bbox_frames = frames_final["obb_bbox_frames"]
        floor = frames_final["floor"]

        # Source color map
        source_colors = {
            "gt": [0, 200, 0],       # green
            "gdino": [255, 160, 0],   # orange
        }

        # OBB cuboid edges
        cuboid_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        # ---- Rerun init ----
        rr.init(app_id, spawn=True)
        rr.log("/", rr.ViewCoordinates.RUB)
        BASE = "world_final"
        rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

        # Axes
        axis_len = 0.5
        rr.log(
            f"{BASE}/axes",
            rr.Arrows3D(
                origins=[[0, 0, 0]] * 3,
                vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                labels=["+X", "+Y", "+Z"],
            ),
            timeless=True,
        )

        # Floor
        if floor is not None:
            v = np.asarray(floor["vertices"], dtype=np.float32)
            f = _faces_u32(np.asarray(floor["faces"]))
            kwargs = {}
            if floor.get("colors", None) is not None:
                kwargs["vertex_colors"] = np.asarray(floor["colors"], dtype=np.uint8)
            else:
                kwargs["albedo_factor"] = [160, 160, 160]
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(vertex_positions=v, triangle_indices=f, **kwargs),
                timeless=True,
            )

        def _get_image_for_stem(stem: str) -> Optional[np.ndarray]:
            img_path = self.frame_annotated_dir_path / video_id / f"{stem}.png"
            if not img_path.exists():
                return None
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                return None
            ih, iw = img.shape[:2]
            if max(ih, iw) > img_maxsize:
                scale = float(img_maxsize) / float(max(ih, iw))
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            return img

        for vis_t in range(S):
            stem = stems_S[vis_t]
            rr.set_time_sequence("frame", vis_t)
            rr.log("/", rr.Clear(recursive=True))

            # Points
            pts = points_final[vis_t].reshape(-1, 3)
            cols = colors_S[vis_t].reshape(-1, 3) if colors_S is not None else None
            conf_flat = conf_S[vis_t].reshape(-1) if conf_S is not None else None

            if conf_flat is not None:
                good = np.isfinite(conf_flat)
                cfs_valid = conf_flat[good]
                thr = (
                    min_conf_default
                    if cfs_valid.size == 0
                    else max(min_conf_default, np.percentile(cfs_valid, 5))
                )
                keep = (conf_flat >= thr) & np.isfinite(pts).all(axis=1)
            else:
                keep = np.isfinite(pts).all(axis=1)

            pts_keep = pts[keep]
            kwargs_pts = {}
            if cols is not None:
                kwargs_pts["colors"] = cols[keep].astype(np.uint8)
            if pts_keep.shape[0] > 0:
                rr.log(f"{BASE}/points", rr.Points3D(pts_keep, **kwargs_pts))

            # Camera
            if camera_poses_S is not None and vis_t < camera_poses_S.shape[0]:
                T = np.asarray(camera_poses_S[vis_t], dtype=np.float32)
                if T.shape == (3, 4):
                    T4 = np.eye(4, dtype=np.float32)
                    T4[:3, :4] = T
                    T = T4
                cam_origin = T[:3, 3]
                R_cam = T[:3, :3]
                rr.log(
                    f"{BASE}/camera/frustum",
                    rr.Pinhole(
                        fov_y=0.7853982,
                        aspect_ratio=float(W) / float(H),
                        camera_xyz=rr.ViewCoordinates.RUB,
                        image_plane_distance=0.1,
                    ),
                    rr.Transform3D(translation=cam_origin.tolist(), mat3x3=R_cam),
                )

            # Bounding boxes (color-coded by source)
            frame_name = f"{stem}.png"
            frame_rec = obb_bbox_frames.get(frame_name, None)
            if frame_rec is not None:
                for bi, obj in enumerate(frame_rec.get("objects", [])):
                    cf = obj.get("corners_final", None)
                    if cf is None:
                        continue
                    corners_final = np.asarray(cf, dtype=np.float32)
                    if corners_final.shape != (8, 3):
                        continue

                    source = obj.get("source", "gt")
                    col = source_colors.get(source, [255, 255, 255])
                    lbl = obj.get("label", f"obj_{bi}")
                    lbl_display = f"{lbl} [{source}]"

                    strips = [corners_final[[e0, e1], :] for (e0, e1) in cuboid_edges]
                    rr.log(
                        f"{BASE}/bboxes/{lbl}_{bi}",
                        rr.LineStrips3D(strips=strips, colors=[col] * len(strips)),
                    )

            # Original RGB image
            img = _get_image_for_stem(stem)
            if img is not None:
                rr.log(f"{BASE}/image", rr.Image(img))

        print(f"[bridge-vis][{video_id}] Rerun running. Scrub the 'frame' timeline.")


# ---------------------------------------------------------------------------
# Dataset + CLI
# ---------------------------------------------------------------------------


def load_dataset(ag_root_directory: str):
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
        train_dataset, shuffle=True, collate_fn=lambda b: b[0], pin_memory=False, num_workers=0,
    )
    dataloader_test = DataLoader(
        test_dataset, shuffle=False, collate_fn=lambda b: b[0], pin_memory=False,
    )
    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge: merge GDino into OBB PKL + WORLD→FINAL")
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--video", type=str, default=None, help="Single video_id (e.g., 00T1E.mp4)")
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--gdino_score_thr", type=float, default=0.3)
    return parser.parse_args()


def main():
    args = parse_args()
    bridge = BBox3DBridgeOBB(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    if args.video:
        # Single video mode
        bridge.bridge_video(
            args.video,
            overwrite=args.overwrite,
            gdino_score_thr=args.gdino_score_thr,
        )
        if args.visualize:
            bridge.visualize_bridge_result(args.video)
    else:
        # Batch mode via dataloader
        _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
        for dataloader in [dataloader_train, dataloader_test]:
            for data in tqdm(dataloader, desc="Bridge"):
                video_id = data["video_id"]
                if get_video_belongs_to_split(video_id) == args.split:
                    try:
                        bridge.bridge_video(
                            video_id,
                            overwrite=args.overwrite,
                            gdino_score_thr=args.gdino_score_thr,
                        )
                    except Exception as e:
                        print(f"[bridge] Error processing {video_id}: {e}")
                        import traceback
                        traceback.print_exc()


if __name__ == "__main__":
    main()
