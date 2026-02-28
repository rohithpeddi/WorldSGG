#!/usr/bin/env python3
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + "/..")

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr

from datasets.preprocess.annotations.annotation_utils import (
    _load_pkl_if_exists,
    _npz_open,
    _is_empty_array,
)


# --------------------------------------------------------------------------------------
# Simple (no-transform) Rerun visualization
#   - All geometry must already be in the desired coordinates.
#   - This function only logs what it's given.
# --------------------------------------------------------------------------------------

def rerun_frame_vis_results(
        video_id: str,
        stems_S: List[str],
        frame_annotated_dir_path: Path,
        *,
        # BEFORE (world/original) data
        points_before: Optional[np.ndarray] = None,  # (S,H,W,3)
        conf_before: Optional[np.ndarray] = None,  # (S,H,W)
        colors_before: Optional[np.ndarray] = None,  # (S,H,W,3)
        cameras_before: Optional[np.ndarray] = None,  # (S,4,4) or (S,3,4)
        floor_vertices_before: Optional[np.ndarray] = None,
        floor_axes_before: Optional[np.ndarray] = None,  # (3,3) row-wise vectors
        floor_origin_before: Optional[np.ndarray] = None,  # (3,)
        frame_3dbb_before: Optional[Dict[Any, Dict[str, Any]]] = None,
        # AFTER (final, transformed) data
        points_after: Optional[np.ndarray] = None,  # (S,H,W,3)
        colors_after: Optional[np.ndarray] = None,  # (S,H,W,3)
        cameras_after: Optional[np.ndarray] = None,  # (S,4,4)
        floor_vertices_after: Optional[np.ndarray] = None,
        floor_axes_after: Optional[np.ndarray] = None,  # (3,3)
        floor_origin_after: Optional[np.ndarray] = None,  # (3,) usually [0,0,0]
        frame_3dbb_after: Optional[Dict[Any, Dict[str, Any]]] = None,
        # Floor mesh topology & style (shared)
        floor_faces: Optional[np.ndarray] = None,
        floor_kwargs: Optional[Dict[str, Any]] = None,
        img_maxsize: int = 320,
        app_id: str = "World4D-Original",
        min_conf_default: float = 1e-6,
        vis_mode: str = "both",  # "before", "after", or "both"
) -> None:
    """
    Pure visualization function.

    This function assumes:
      - All "before" arrays are in the original/world frame.
      - All "after" arrays are already transformed into the FINAL frame
        (e.g., floor-aligned + mirrored) by external code.

    It does *no* geometric transformations, only logging.
    """

    # -------------------------------------------------------------------------
    # Normalize vis_mode
    # -------------------------------------------------------------------------
    vis_mode = (vis_mode or "both").lower()
    if vis_mode not in {"before", "after", "both"}:
        print(f"[orig-pts][{video_id}] Unknown vis_mode={vis_mode!r}, defaulting to 'both'.")
        vis_mode = "both"

    show_before = vis_mode in {"before", "both"}
    show_after = vis_mode in {"after", "both"}

    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    BASE = "world"
    BASE_BEFORE = f"{BASE}/before"
    BASE_AFTER = f"{BASE}/after"

    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # Cuboid edge list
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Try to get grid size for pinhole aspect ratio
    H_grid = W_grid = 1
    if points_before is not None and points_before.ndim == 4:
        _, H_grid, W_grid, _ = points_before.shape
    elif points_after is not None and points_after.ndim == 4:
        _, H_grid, W_grid, _ = points_after.shape

    # -------------------------------------------------------------------------
    # Static world & XYZ axes
    # -------------------------------------------------------------------------
    axis_len_world = 0.5
    world_axes = rr.Arrows3D(
        origins=[[0.0, 0.0, 0.0]] * 3,
        vectors=[
            [axis_len_world, 0.0, 0.0],
            [0.0, axis_len_world, 0.0],
            [0.0, 0.0, axis_len_world],
        ],
        colors=[
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
        ],
        labels=["World +X", "World +Y", "World +Z"],
    )

    axis_len_xyz = 0.4
    xyz_axes = rr.Arrows3D(
        origins=[[0.0, 0.0, 0.0]] * 3,
        vectors=[
            [axis_len_xyz, 0.0, 0.0],
            [0.0, axis_len_xyz, 0.0],
            [0.0, 0.0, axis_len_xyz],
        ],
        colors=[
            [255, 128, 128],
            [128, 255, 128],
            [128, 128, 255],
        ],
        labels=["X", "Y", "Z"],
    )

    def _log_static_info() -> None:
        rr.log(f"{BASE}/world_axes", world_axes)
        rr.log(f"{BASE}/xyz_axes", xyz_axes)
        rr.log(
            f"{BASE}/info",
            rr.TextLog(
                f"[{video_id}] BEFORE = original/world frame. "
                f"AFTER = final transformed frame (e.g., floor-aligned + mirror). "
                f"vis_mode='{vis_mode}' (before/after/both)."
            ),
        )

    def _get_image_for_stem(stem: str) -> Optional[np.ndarray]:
        img_path = frame_annotated_dir_path / video_id / f"{stem}.png"
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        return img

    # Number of frames
    S = len(stems_S)

    for vis_t in range(S):
        stem = stems_S[vis_t]

        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        _log_static_info()

        # ---------------------------------------------------------------------
        # FLOOR MESH: BEFORE / AFTER
        # ---------------------------------------------------------------------
        if floor_faces is not None and floor_kwargs is None:
            floor_kwargs_local: Dict[str, Any] = {"albedo_factor": [160, 160, 160]}
        else:
            floor_kwargs_local = floor_kwargs or {}

        if show_before and floor_vertices_before is not None and floor_faces is not None:
            rr.log(
                f"{BASE_BEFORE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_before.astype(np.float32),
                    triangle_indices=floor_faces,
                    **floor_kwargs_local,
                ),
            )

        if show_after and floor_vertices_after is not None and floor_faces is not None:
            rr.log(
                f"{BASE_AFTER}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_after.astype(np.float32),
                    triangle_indices=floor_faces,
                    **floor_kwargs_local,
                ),
            )

        # ---------------------------------------------------------------------
        # FLOOR AXES: BEFORE / AFTER
        # ---------------------------------------------------------------------
        if show_before and floor_origin_before is not None and floor_axes_before is not None:
            origins_floor_before = np.repeat(
                floor_origin_before[None, :], 3, axis=0
            )
            rr.log(
                f"{BASE_BEFORE}/floor_frame",
                rr.Arrows3D(
                    origins=origins_floor_before,
                    vectors=floor_axes_before,
                    colors=[
                        [200, 200, 0],
                        [0, 200, 200],
                        [200, 0, 200],
                    ],
                    labels=["Floor(Before) +X", "Floor(Before) +Y", "Floor(Before) +Z"],
                ),
            )

        if show_after and floor_axes_after is not None:
            if floor_origin_after is None:
                floor_origin_after = np.zeros(3, dtype=np.float32)
            origins_floor_after = np.repeat(
                floor_origin_after[None, :], 3, axis=0
            )
            rr.log(
                f"{BASE_AFTER}/floor_frame",
                rr.Arrows3D(
                    origins=origins_floor_after,
                    vectors=floor_axes_after,
                    colors=[
                        [255, 200, 0],
                        [0, 255, 255],
                        [255, 0, 255],
                    ],
                    labels=["Floor(After) +X", "Floor(After) +Y", "Floor(After) +Z"],
                ),
            )

        # ---------------------------------------------------------------------
        # CAMERAS: BEFORE / AFTER
        # ---------------------------------------------------------------------
        def _log_camera(path_prefix: str, T: np.ndarray, prefix_label: str) -> None:
            if T.shape == (3, 4):
                T_full = np.eye(4, dtype=np.float32)
                T_full[:3, :4] = T
            else:
                T_full = T.astype(np.float32)

            cam_origin = T_full[:3, 3]
            R_cam = T_full[:3, :3]

            axis_len_cam = 0.4
            cam_axes_vec = np.stack(
                [
                    R_cam[:, 0] * axis_len_cam,
                    R_cam[:, 1] * axis_len_cam,
                    R_cam[:, 2] * axis_len_cam,
                ],
                axis=0,
            )
            origins_cam = np.repeat(cam_origin[None, :], 3, axis=0)

            rr.log(
                f"{path_prefix}/camera_axes",
                rr.Arrows3D(
                    origins=origins_cam,
                    vectors=cam_axes_vec,
                    colors=[
                        [180, 0, 0],
                        [0, 180, 0],
                        [0, 0, 180],
                    ],
                    labels=[
                        f"Cam({prefix_label}) +X",
                        f"Cam({prefix_label}) +Y",
                        f"Cam({prefix_label}) +Z",
                    ],
                ),
            )

            rr.log(
                f"{path_prefix}/camera/frustum",
                rr.Pinhole(
                    fov_y=0.7853982,
                    aspect_ratio=float(W_grid) / float(H_grid),
                    camera_xyz=rr.ViewCoordinates.RUB,
                    image_plane_distance=0.1,
                ),
                rr.Transform3D(
                    translation=cam_origin.tolist(),
                    mat3x3=R_cam,
                ),
            )

        if show_before and cameras_before is not None and vis_t < cameras_before.shape[0]:
            _log_camera(BASE_BEFORE, cameras_before[vis_t], "Before")

        if show_after and cameras_after is not None and vis_t < cameras_after.shape[0]:
            _log_camera(BASE_AFTER, cameras_after[vis_t], "After")

        # ---------------------------------------------------------------------
        # POINTS: BEFORE (dark) / AFTER (colorful)
        # ---------------------------------------------------------------------
        pts_b = None
        pts_a = None
        cols_a = None
        conf_flat = None

        if points_before is not None:
            pts_b = points_before[vis_t].reshape(-1, 3)
        if points_after is not None:
            pts_a = points_after[vis_t].reshape(-1, 3)
        if colors_after is not None:
            cols_a = colors_after[vis_t].reshape(-1, 3)

        if conf_before is not None:
            conf_flat = conf_before[vis_t].reshape(-1)

        if conf_flat is not None:
            good = np.isfinite(conf_flat)
            cfs_valid = conf_flat[good]
            if cfs_valid.size > 0:
                med = np.median(cfs_valid)
                p5 = np.percentile(cfs_valid, 5)
                thr = max(min_conf_default, p5)
                print(
                    f"[orig-pts][{video_id}] frame {stem}: "
                    f"conf thr = {thr:.4f} (med={med:.4f}, n_valid={cfs_valid.size})"
                )
            else:
                thr = min_conf_default
            keep_mask = (conf_flat >= thr)
        else:
            keep_mask = None

        if pts_b is not None:
            if keep_mask is None:
                keep_b = np.isfinite(pts_b).all(axis=1)
            else:
                keep_b = keep_mask & np.isfinite(pts_b).all(axis=1)

            if show_before:
                pts_b_keep = pts_b[keep_b]
                if pts_b_keep.shape[0] > 0:
                    rr.log(
                        f"{BASE_BEFORE}/points",
                        rr.Points3D(
                            pts_b_keep.astype(np.float32),
                            colors=np.array([[60, 60, 60]], dtype=np.uint8),
                        ),
                    )

        if pts_a is not None and show_after:
            if keep_mask is None:
                keep_a = np.isfinite(pts_a).all(axis=1)
            else:
                keep_a = keep_mask & np.isfinite(pts_a).all(axis=1)

            pts_a_keep = pts_a[keep_a]
            if cols_a is not None:
                cols_a_keep = cols_a[keep_a].astype(np.uint8)
            else:
                cols_a_keep = None

            if pts_a_keep.shape[0] > 0:
                kwargs_pts: Dict[str, Any] = {}
                if cols_a_keep is not None:
                    kwargs_pts["colors"] = cols_a_keep
                rr.log(
                    f"{BASE_AFTER}/points",
                    rr.Points3D(
                        pts_a_keep.astype(np.float32),
                        **kwargs_pts,
                    ),
                )

        # -----------------------------------------------------------------------------
        # 3D BOUNDING BOXES: BEFORE / AFTER + LABELS
        # -----------------------------------------------------------------------------
        def _log_bbox_with_label(
                base_path: str,
                frame_idx: int,
                bbox_index: int,
                corners: np.ndarray,
                color: List[int],
                label: Optional[str],
        ) -> None:
            """
            Log a wireframe 3D box (LineStrips3D) and, if available, a point label
            slightly above the box center.
            """
            # Wireframe edges
            strips = []
            for e0, e1 in cuboid_edges:
                strips.append(corners[[e0, e1], :])

            rr.log(
                f"{base_path}/bboxes/frame_{frame_idx}/bbox_{bbox_index}",
                rr.LineStrips3D(
                    strips=strips,
                    colors=[color] * len(strips),
                ),
            )

            # Optional label above the box
            if label is not None:
                # Center of the box in 3D
                center = corners.mean(axis=0)

                # Use ~5% of the diagonal length as vertical offset; fall back to small constant
                diag_len = np.linalg.norm(
                    corners.max(axis=0) - corners.min(axis=0)
                )
                offset = 0.05 * diag_len if diag_len > 0 else 0.05

                # Assume +Y is "up" in both BEFORE (world) and AFTER (final) frames
                label_pos = center + np.array([0.0, offset, 0.0], dtype=np.float32)

                rr.log(
                    f"{base_path}/bboxes/frame_{frame_idx}/bbox_{bbox_index}_label",
                    rr.Points3D(
                        positions=label_pos[None, :].astype(np.float32),
                        labels=[str(label)],
                        colors=[color],
                    ),
                )

        if show_before and frame_3dbb_before is not None:
            frame_name = f"{stem}.png"
            if frame_name in frame_3dbb_before:
                frame_objects = frame_3dbb_before[frame_name]["objects"]
                for bi, obj in enumerate(frame_objects):
                    obb_data = obj.get("obb_floor_parallel", None)
                    if not isinstance(obb_data, dict) or obb_data.get("corners_world") is None:
                        continue
                    corners_world = np.asarray(
                        obb_data["corners_world"], dtype=np.float32
                    )
                    if corners_world.size == 0:
                        continue

                    col = obj.get("color", [255, 180, 0])
                    label = obj.get("label", None)

                    _log_bbox_with_label(
                        base_path=BASE_BEFORE,
                        frame_idx=vis_t,
                        bbox_index=bi,
                        corners=corners_world,
                        color=col,
                        label=label,
                    )

        if show_after and frame_3dbb_after is not None:
            frame_name = f"{stem}.png"
            if frame_name in frame_3dbb_after:
                frame_objects = frame_3dbb_after[frame_name]["objects"]
                for bi, obj in enumerate(frame_objects):
                    obb_final = obj.get("obb_final", None)
                    if not isinstance(obb_final, dict) or obb_final.get("corners_final") is None:
                        continue
                    corners_final = np.asarray(
                        obb_final["corners_final"], dtype=np.float32
                    )
                    if corners_final.size == 0:
                        continue

                    col = obj.get("color_after", [255, 230, 80])
                    label = obj.get("label", None)

                    _log_bbox_with_label(
                        base_path=BASE_AFTER,
                        frame_idx=vis_t,
                        bbox_index=bi,
                        corners=corners_final,
                        color=col,
                        label=label,
                    )

        # ---------------------------------------------------------------------
        # ORIGINAL RGB FRAME (single copy, not transformed)
        # ---------------------------------------------------------------------
        img = _get_image_for_stem(stem)
        if img is not None:
            rr.log(f"{BASE}/image", rr.Image(img))

    print(
        "[orig-pts] visualization running for "
        f"{video_id}. BEFORE = original frame, AFTER = final transformed frame. "
        f"vis_mode='{vis_mode}'. Scrub the 'frame' timeline in Rerun and "
        "toggle entities in the UI."
    )


# --------------------------------------------------------------------------------------
# Base class: shared paths + loaders/utilities
# --------------------------------------------------------------------------------------


class FrameToWorldBase:

    def __init__(
            self,
            ag_root_directory: str,
            dynamic_scene_dir_path: str
    ):
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)

        self.dataset_classnames = [
            "__background__",
            "person",
            "bag",
            "bed",
            "blanket",
            "book",
            "box",
            "broom",
            "chair",
            "closet/cabinet",
            "clothes",
            "cup/glass/bottle",
            "dish",
            "door",
            "doorknob",
            "doorway",
            "floor",
            "food",
            "groceries",
            "laptop",
            "light",
            "medicine",
            "mirror",
            "paper/notebook",
            "phone/camera",
            "picture",
            "pillow",
            "refrigerator",
            "sandwich",
            "shelf",
            "shoe",
            "sofa/couch",
            "table",
            "television",
            "towel",
            "vacuum",
            "window",
        ]
        self.name_to_catid = {
            name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0
        }
        self.catid_to_name_map = {v: k for k, v in self.name_to_catid.items()}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        # Detection directories (same layout as BBox3DGenerator)
        self.dynamic_detections_root_path = (
                self.ag_root_directory / "detection" / "gdino_bboxes"
        )
        self.static_detections_root_path = (
                self.ag_root_directory / "detection" / "gdino_bboxes_static"
        )

        # Annotated frame and sampling info
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = (
                self.ag_root_directory / "sampled_frames_idx"
        )

        # GT annotations
        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        # Segmentation dirs (if you later want to build label-wise masks)
        self.dynamic_masked_frames_im_dir_path = (
                self.ag_root_directory
                / "segmentation"
                / "masked_frames"
                / "image_based"
        )

        # World annotation dirs (3D bboxes already generated by BBox3DGenerator)
        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        self.bbox_4d_root_dir = self.world_annotations_root_dir / "bbox_annotations_4d"
        self.bbox_3d_final_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_final"
        self.bbox_3d_gt_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_gt"
        self.bbox_3d_gt_obb_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_obb"
        self.bbox_3d_gt_obb_final_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_obb_final"

        os.makedirs(self.bbox_4d_root_dir, exist_ok=True)
        os.makedirs(self.bbox_3d_final_root_dir, exist_ok=True)
        os.makedirs(self.bbox_3d_gt_root_dir, exist_ok=True)

        self.active_objects_b_annotations_path = self.ag_root_directory / 'active_objects' / 'annotations'
        self.active_objects_b_reasoned_path = self.ag_root_directory / 'active_objects' / 'sampled_videos'

        self.video_id_active_objects_annotations_map = {}
        self.video_id_active_objects_b_reasoned_map = {}

    # ----------------------------------------------------------------------------------
    # GT + GDINO + 3D annotations loaders
    # ----------------------------------------------------------------------------------

    def fetch_stored_active_objects_in_video(self, video_id):
        video_id_object_reasoning_path = self.active_objects_b_reasoned_path / f"{video_id[:-4]}.txt"
        video_id_object_annotations_path = self.active_objects_b_annotations_path / f"{video_id[:-4]}.txt"
        if video_id_object_annotations_path.exists():
            with open(video_id_object_annotations_path, "r") as f:
                annotated_objects = [line.strip() for line in f if line.strip()]
            self.video_id_active_objects_annotations_map[video_id] = sorted(annotated_objects)
            if video_id_object_reasoning_path.exists():
                with open(video_id_object_reasoning_path, "r") as f:
                    video_reasoned_objects = [line.strip() for line in f if line.strip()]

                # Ensure presence of "person", as it's always active
                # If there is a television in annotated objects, add it to reasoned objects
                # If there is a mirror in annotated objects, add it to reasoned objects
                video_reasoned_objects = set(video_reasoned_objects)
                video_reasoned_objects.add("person")

                if "television" in annotated_objects:
                    video_reasoned_objects.add("television")
                if "mirror" in annotated_objects:
                    video_reasoned_objects.add("mirror")
                self.video_id_active_objects_b_reasoned_map[video_id] = sorted(list(video_reasoned_objects))
            else:
                print(f"Video {video_id} has no reasoned objects. Loading annotated objects instead.")
                self.video_id_active_objects_b_reasoned_map[video_id] = sorted(annotated_objects)

    def fetch_stored_active_objects_in_videos(self, dataloader):
        for data in dataloader:
            video_id = data['video_id']
            self.fetch_stored_active_objects_in_video(video_id)

    def get_video_3d_annotations(self, video_id: str):
        """
        Load the floor-aligned 3D bbox annotations created by BBox3DGenerator.
        """
        out_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
        if not out_path.exists():
            print(f"[world4d][{video_id}] 3D bbox annotations not found at {out_path}")
            return None

        with open(out_path, "rb") as f:
            video_3d_annotations = pickle.load(f)
        return video_3d_annotations

    def get_video_gt_annotations(self, video_id: str):
        video_gt_annotations_json_path = self.gt_annotations_root_dir / video_id / "gt_annotations.json"
        if not video_gt_annotations_json_path.exists():
            raise FileNotFoundError(f"GT annotations file not found: {video_gt_annotations_json_path}")

        with open(video_gt_annotations_json_path, "r") as f:
            video_gt_annotations = json.load(f)

        video_gt_bboxes: Dict[str, Dict[str, Any]] = {}
        for frame_idx, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]
            boxes = []
            labels = []
            for item in frame_items:
                if "person_bbox" in item:
                    boxes.append(item["person_bbox"][0])
                    labels.append("person")
                    continue

                category_id = item["class"]
                category_name = self.catid_to_name_map[category_id]

                if category_name:
                    # Normalize some label names
                    if category_name == "closet/cabinet":
                        category_name = "closet"
                    elif category_name == "cup/glass/bottle":
                        category_name = "cup"
                    elif category_name == "paper/notebook":
                        category_name = "paper"
                    elif category_name == "sofa/couch":
                        category_name = "sofa"
                    elif category_name == "phone/camera":
                        category_name = "phone"

                    boxes.append(item["bbox"])
                    labels.append(category_name)

            if boxes:
                video_gt_bboxes[frame_name] = {"boxes": boxes, "labels": labels}

        return video_gt_bboxes, video_gt_annotations

    def get_video_gdino_annotations(self, video_id: str):
        video_dynamic_gdino_prediction_file_path = self.dynamic_detections_root_path / f"{video_id}.pkl"
        video_dynamic_predictions = _load_pkl_if_exists(video_dynamic_gdino_prediction_file_path)

        video_static_gdino_prediction_file_path = self.static_detections_root_path / f"{video_id}.pkl"
        video_static_predictions = _load_pkl_if_exists(video_static_gdino_prediction_file_path)

        if video_dynamic_predictions is None:
            video_dynamic_predictions = {}
        if video_static_predictions is None:
            video_static_predictions = {}

        if not video_dynamic_predictions and not video_static_predictions:
            raise ValueError(f"No GDINO predictions found for video {video_id}")

        all_frame_names = set(video_dynamic_predictions.keys()) | set(
            video_static_predictions.keys()
        )
        combined_gdino_predictions: Dict[str, Dict[str, Any]] = {}

        for frame_name in all_frame_names:
            dyn_pred = video_dynamic_predictions.get(frame_name, None)
            stat_pred = video_static_predictions.get(frame_name, None)

            if dyn_pred is None:
                dyn_pred = {"boxes": [], "labels": [], "scores": []}
            if stat_pred is None:
                stat_pred = {"boxes": [], "labels": [], "scores": []}

            if _is_empty_array(dyn_pred["boxes"]) and _is_empty_array(
                    stat_pred["boxes"]
            ):
                combined_gdino_predictions[frame_name] = {
                    "boxes": [],
                    "labels": [],
                    "scores": [],
                }
                continue

            combined_boxes = []
            combined_labels = []
            combined_scores = []

            if not _is_empty_array(dyn_pred["boxes"]):
                combined_boxes += list(dyn_pred["boxes"])
                combined_labels += list(dyn_pred["labels"])
                combined_scores += list(dyn_pred["scores"])

            if not _is_empty_array(stat_pred["boxes"]):
                combined_boxes += list(stat_pred["boxes"])
                combined_labels += list(stat_pred["labels"])
                combined_scores += list(stat_pred["scores"])

            final_pred = {
                "boxes": combined_boxes,
                "labels": combined_labels,
                "scores": combined_scores,
            }
            combined_gdino_predictions[frame_name] = final_pred

        return combined_gdino_predictions

    # ----------------------------------------------------------------------------------
    # Indexing helper  mirror BBox3DGenerator.idx_to_frame_idx_path
    # ----------------------------------------------------------------------------------

    def _bbox_idx_to_frame_idx_path(self, video_id: str) -> Tuple[
        Dict[int, str], List[int], List[int], List[str], List[int]
    ]:
        """
        Replica of BBox3DGenerator.idx_to_frame_idx_path, so that
        the points we load here match exactly the subset used
        when constructing the bbox_3D .pkl files.
        """
        video_frames_annotated_dir_path = self.frame_annotated_dir_path / video_id
        annotated_frame_id_list = [
            f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith(".png")
        ]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))

        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        video_sampled_frames_npy_path = (
                self.sampled_frames_idx_root_dir / f"{video_id[:-4]}.npy"
        )
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(
            annotated_first_frame_id
        )
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(
            annotated_last_frame_id
        )
        sample_idx = list(
            range(
                an_first_id_in_vid_sam_frame_id_list,
                an_last_id_in_vid_sam_frame_id_list + 1,
            )
        )

        annotated_frame_idx_in_sample_idx: List[int] = []
        for frame_name in annotated_frame_id_list:
            frame_id = int(frame_name[:-4])
            if frame_id in video_sampled_frame_id_list:
                idx_in_sampled = video_sampled_frame_id_list.index(frame_id)
                annotated_frame_idx_in_sample_idx.append(sample_idx.index(idx_in_sampled))

        chosen_frames = [video_sampled_frame_id_list[i] for i in sample_idx]
        frame_idx_frame_path_map = {
            i: f"{frame_id:06d}.png" for i, frame_id in enumerate(chosen_frames)
        }

        return (
            frame_idx_frame_path_map,
            sample_idx,
            video_sampled_frame_id_list,
            annotated_frame_id_list,
            annotated_frame_idx_in_sample_idx,
        )

    # ----------------------------------------------------------------------------------
    # ORIGINAL POINTS LOADER  matches BBox3DGenerator._load_points_for_video
    # ----------------------------------------------------------------------------------

    def _load_original_points_for_video(self, video_id: str) -> Dict[str, Any]:
            """
            Load the *original* Pi3 points, confidences, colors, and camera poses
            for the **annotated frames only**, using exactly the same indexing
            and slicing logic as BBox3DGenerator._load_points_for_video.

            Returns:
                {
                    "points": (S,H,W,3) float32,
                    "conf":   (S,H,W) float32 or None,
                    "frame_stems": List[str],       # "000123", ...
                    "colors": (S,H,W,3) uint8,
                    "camera_poses": (S,4,4) or None
                }
            """
            video_dynamic_3d_scene_path = (
                    self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
            )
            if not video_dynamic_3d_scene_path.exists():
                raise FileNotFoundError(
                    f"[original-points] predictions.npz not found: {video_dynamic_3d_scene_path}"
                )

            with _npz_open(video_dynamic_3d_scene_path) as video_dynamic_predictions:
                points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
                imgs_f32 = video_dynamic_predictions["images"]
                colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

                conf = None
                if "conf" in video_dynamic_predictions:
                    conf = video_dynamic_predictions["conf"]
                    if conf.ndim == 4 and conf.shape[-1] == 1:
                        conf = conf[..., 0]
                    conf = conf.astype(np.float32)

                camera_poses = None
                if "camera_poses" in video_dynamic_predictions:
                    camera_poses = video_dynamic_predictions["camera_poses"].astype(np.float32)

                S, H, W, _ = points.shape

                (
                    frame_idx_frame_path_map,
                    sample_idx,
                    video_sampled_frame_id_list,
                    annotated_frame_id_list,
                    annotated_frame_idx_in_sample_idx,
                ) = self._bbox_idx_to_frame_idx_path(video_id)

                assert S == len(
                    sample_idx
                ), (
                    f"[original-points] points axis ({S}) != sample_idx range "
                    f"({len(sample_idx)}) for {video_id}"
                )

                sampled_idx_frame_name_map: Dict[int, str] = {}
                frame_name_sampled_idx_map: Dict[str, int] = {}
                for idx_in_s, frame_idx in enumerate(sample_idx):
                    frame_name = f"{video_sampled_frame_id_list[frame_idx]:06d}.png"
                    sampled_idx_frame_name_map[idx_in_s] = frame_name
                    frame_name_sampled_idx_map[frame_name] = idx_in_s

                annotated_idx_in_sampled_idx: List[int] = []
                for frame_name in annotated_frame_id_list:
                    if frame_name in frame_name_sampled_idx_map:
                        annotated_idx_in_sampled_idx.append(
                            frame_name_sampled_idx_map[frame_name]
                        )

                points_sub = points[annotated_idx_in_sampled_idx]
                colors_sub = colors[annotated_idx_in_sampled_idx]
                conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
                camera_poses_sub = (
                    camera_poses[annotated_idx_in_sampled_idx]
                    if camera_poses is not None
                    else None
                )
                stems_sub = [
                    sampled_idx_frame_name_map[idx][:-4]
                    for idx in annotated_idx_in_sampled_idx
                ]

            return {
                "points": points_sub,
                "conf": conf_sub,
                "frame_stems": stems_sub,
                "colors": colors_sub,
                "camera_poses": camera_poses_sub,
            }