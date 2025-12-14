#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import DataLoader
import cv2
from scipy.spatial.transform import Rotation as SciRot
import rerun as rr
import matplotlib.pyplot as plt

from dataloader.standard.action_genome.ag_dataset import StandardAG
from annotation_utils import (
    get_video_belongs_to_split,
    _load_pkl_if_exists,
    _npz_open,
    _torch_inference_ctx,
    _del_and_collect,
    _lift_2d_to_3d,
    _find_actor_index_in_frame,
    _choose_primary_actor,
    _build_frame_to_kps_map,
    _robust_similarity_ransac,
    _faces_u32,
    _resize_mask_to,
    _mask_from_bbox,
    _resize_bbox_to,
    _xywh_to_xyxy,
    _average_sims_robust,
    _finite_and_nonzero,
    _pinhole_from_fov,
    _is_empty_array,
)


def rerun_vis_world4d(
    video_id: str,
    points: np.ndarray,          # (S, H, W, 3)  -- Pi3 / dynamic coords
    colors: np.ndarray,          # (S, H, W, 3) uint8
    conf: Optional[np.ndarray],  # (S, H, W) or None
    camera_poses: np.ndarray,    # (S, 4, 4)     -- Pi3 camera->world (3x4 or 4x4)
    frame_bbox_meshes: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    *,
    global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
    floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
    img_maxsize: int = 320,
    app_id: str = "World4D",
    vis_floor: bool = True,
    min_conf_default: float = 1e-6,  # floor for conf
) -> None:
    """
    Visualize 4D world for a video with all geometry expressed in a common
    world coordinate frame tied to the XYZ axes.

    We assume `global_floor_sim = (s, R, t)` defines a similarity transform from
    the raw Pi3 coordinate frame into this world frame:

        p_world = s * (p_pi3 @ R^T) + t

    We apply this to:
      - dynamic 3D points
      - floor mesh vertices
      - 3D bbox mesh vertices (frame_bbox_meshes)
      - camera poses (approximately; see note below)

    NOTE:
      For cameras, we also apply the same transform to orientation and origin
      so the frustum lives in the same world frame. Exact similarity-composition
      of a scaled transform cannot be represented as a pure rigid transform, but
      this approximation is sufficient for visualization and keeps axes aligned.
    """

    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    S = points.shape[0]
    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # --------------------------------------------------------------------------
    # Unpack global floor similarity, if provided
    #   p_world = s_g * (p_pi3 @ R_g.T) + t_g
    # --------------------------------------------------------------------------
    s_g: Optional[float] = None
    R_g: Optional[np.ndarray] = None
    t_g: Optional[np.ndarray] = None
    if global_floor_sim is not None:
        s_g, R_g, t_g = global_floor_sim
        s_g = float(s_g)
        R_g = np.asarray(R_g, dtype=np.float32)
        t_g = np.asarray(t_g, dtype=np.float32)  # shape (3,)

    # --------------------------------------------------------------------------
    # Floor mesh (optionally transformed by global floor similarity)
    # --------------------------------------------------------------------------
    floor_vertices_tf = None
    floor_faces = None
    floor_kwargs = None
    if floor is not None:
        floor_verts0, floor_faces0, floor_colors0 = floor
        floor_verts0 = np.asarray(floor_verts0, dtype=np.float32)
        floor_faces0 = _faces_u32(np.asarray(floor_faces0))

        if s_g is not None:
            floor_vertices_tf = s_g * (floor_verts0 @ R_g.T) + t_g[None, :]
        else:
            floor_vertices_tf = floor_verts0

        floor_kwargs = {}
        if floor_colors0 is not None:
            floor_colors0 = np.asarray(floor_colors0, dtype=np.uint8)
            floor_kwargs["vertex_colors"] = floor_colors0
        else:
            floor_kwargs["albedo_factor"] = [160, 160, 160]
        floor_faces = floor_faces0

    # --------------------------------------------------------------------------
    # Utility: get 2D image for a given frame index (using colors)
    # --------------------------------------------------------------------------
    def _get_image_for_time(idx: int) -> Optional[np.ndarray]:
        if idx < 0 or idx >= S:
            return None
        img = colors[idx]
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    # edges for a cuboid (8 vertices)
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # --------------------------------------------------------------------------
    # Iterate through frames (S = number of annotated frames)
    # --------------------------------------------------------------------------
    for vis_t in range(S):
        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # ---------------------- world axis visualizer ----------------------
        # Draw small RGB axes at world origin:
        # X = red, Y = green, Z = blue
        axis_len = 0.5  # tweak as you like
        x_axis = np.array([[0.0, 0.0, 0.0],
                           [axis_len, 0.0, 0.0]], dtype=np.float32)
        y_axis = np.array([[0.0, 0.0, 0.0],
                           [0.0, axis_len, 0.0]], dtype=np.float32)
        z_axis = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, axis_len]], dtype=np.float32)

        rr.log(
            f"{BASE}/world_axes",
            rr.LineStrips3D(
                strips=[x_axis, y_axis, z_axis],
                colors=[
                    [255, 0, 0],   # X (red)
                    [0, 255, 0],   # Y (green)
                    [0, 0, 255],   # Z (blue)
                ],
            ),
        )

        # ---------------------- floor ----------------------
        if vis_floor and (floor_vertices_tf is not None) and (floor_faces is not None):
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_tf.astype(np.float32),
                    triangle_indices=floor_faces,
                    **(floor_kwargs or {}),
                ),
            )

        # ---------------------- dynamic points ----------------------
        pts = points[vis_t].reshape(-1, 3)          # (N,3) in Pi3 coords
        cols = colors[vis_t].reshape(-1, 3)         # (N,3)

        # Map points to world-aligned XYZ frame if we have global sim
        if s_g is not None:
            pts = s_g * (pts @ R_g.T) + t_g[None, :]

        if conf is not None:
            cfs = conf[vis_t].reshape(-1)           # (N,)
            good = np.isfinite(cfs)
            cfs_valid = cfs[good]
            if cfs_valid.size > 0:
                med = np.median(cfs_valid)
                p5 = np.percentile(cfs_valid, 5)
                # conservative threshold: keep at least ~95% of valid points
                thr = max(min_conf_default, p5)
                print(
                    f"[{video_id}] frame {vis_t}: conf thr = {thr:.4f} "
                    f"(med={med:.4f}, n_valid={cfs_valid.size})"
                )
            else:
                thr = min_conf_default

            keep = (cfs >= thr) & np.isfinite(pts).all(axis=1)
        else:
            # no confidence provided, just keep finite points
            keep = np.isfinite(pts).all(axis=1)

        pts_keep = pts[keep]
        cols_keep = cols[keep]

        if pts_keep.shape[0] > 0:
            rr.log(
                f"{BASE}/points",
                rr.Points3D(
                    pts_keep.astype(np.float32),
                    colors=cols_keep.astype(np.uint8),
                ),
            )

        # ---------------------- per-frame cuboid bboxes ----------------------
        # Transform bbox vertices by the same global similarity as points.
        if frame_bbox_meshes is not None and vis_t in frame_bbox_meshes:
            for bi, bbox_m in enumerate(frame_bbox_meshes[vis_t]):
                verts_world = np.asarray(bbox_m["verts"], dtype=np.float32)  # (8,3) in Pi3 coords
                if s_g is not None:
                    verts_world = s_g * (verts_world @ R_g.T) + t_g[None, :]
                col = bbox_m.get("color", [255, 180, 0])

                strips = []
                for e0, e1 in cuboid_edges:
                    strips.append(verts_world[[e0, e1], :])

                rr.log(
                    f"{BASE}/bboxes/frame_{vis_t}/bbox_{bi}",
                    rr.LineStrips3D(
                        strips=strips,
                        colors=[col] * len(strips),
                    ),
                )

        # ---------------------- camera frustum + image ----------------------
        cam_4x4 = np.asarray(camera_poses[vis_t], dtype=np.float32)  # (4,4) or (3,4) padded
        if cam_4x4.shape == (3, 4):
            cam_4x4 = np.vstack(
                [cam_4x4, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)]
            )

        R_wc = cam_4x4[:3, :3]   # Pi3 world-from-camera rotation
        t_wc = cam_4x4[:3, 3]    # Pi3 camera origin in world coords

        # Map camera pose into world-aligned frame (approximate)
        if s_g is not None:
            # Rotate orientation by R_g (compose global->Pi3)
            # Using column-vector convention: x' = R_g * x
            R_wc = R_wc @ R_g.T
            # Transform origin the same way we transform points
            t_wc = s_g * (t_wc @ R_g.T) + t_g

        image = _get_image_for_time(vis_t)
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
            ),
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(
                focal_length=(fx, fy),
                principal_point=(cx, cy),
                resolution=(W_img, H_img),
            ),
        )
        if image is not None:
            rr.log(f"{frus_path}/image", rr.Image(image))

    print(f"[{video_id}] Rerun visualization running. Scrub the 'frame' timeline.")


class FrameToWorldAnnotations:
    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str) -> None:
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

        self.dynamic_detections_root_path = (
            self.ag_root_directory / "detection" / "gdino_bboxes"
        )
        self.static_detections_root_path = (
            self.ag_root_directory / "detection" / "gdino_bboxes_static"
        )
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = (
            self.ag_root_directory / "sampled_frames_idx"
        )

        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        self.bbox_4d_root_dir = self.world_annotations_root_dir / "bbox_annotations_4d"
        os.makedirs(self.bbox_4d_root_dir, exist_ok=True)

        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        self.dynamic_masked_frames_im_dir_path = (
            self.ag_root_directory
            / "segmentation"
            / "masked_frames"
            / "image_based"
        )
        self.dynamic_masked_frames_vid_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_frames" / "video_based"
        )
        self.dynamic_masked_frames_combined_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_frames" / "combined"
        )
        self.dynamic_masked_videos_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_videos"
        )

        self.dynamic_masks_im_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "image_based"
        )
        self.dynamic_masks_vid_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "video_based"
        )
        self.dynamic_masks_combined_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "combined"
        )

        self.static_masks_im_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        )
        self.static_masks_vid_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        )
        self.static_masks_combined_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "combined"
        )

    # ----------------------------------------------------------------------
    # GT + GDINO + 3D annotations loaders
    # ----------------------------------------------------------------------
    def get_video_gt_annotations(self, video_id: str):
        video_gt_annotations_json_path = (
            self.gt_annotations_root_dir / video_id / "gt_annotations.json"
        )
        if not video_gt_annotations_json_path.exists():
            raise FileNotFoundError(
                f"GT annotations file not found: {video_gt_annotations_json_path}"
            )

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
        video_dynamic_gdino_prediction_file_path = (
            self.dynamic_detections_root_path / f"{video_id}.pkl"
        )
        video_dynamic_predictions = _load_pkl_if_exists(
            video_dynamic_gdino_prediction_file_path
        )

        video_static_gdino_prediction_file_path = (
            self.static_detections_root_path / f"{video_id}.pkl"
        )
        video_static_predictions = _load_pkl_if_exists(
            video_static_gdino_prediction_file_path
        )

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

    def get_video_3d_annotations(self, video_id: str):
        """
        Load pre-computed 3D bbox annotations for a video, if present.
        """
        out_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
        if not out_path.exists():
            return None

        with open(out_path, "rb") as f:
            video_3d_annotations = pickle.load(f)
        return video_3d_annotations

    def get_video_dynamic_predictions(self, video_id: str):
        video_dynamic_3d_scene_path = (
            self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        )
        if not video_dynamic_3d_scene_path.exists():
            return None
        video_dynamic_predictions = _npz_open(video_dynamic_3d_scene_path)
        return video_dynamic_predictions

    # ----------------------------------------------------------------------
    # World 4D bbox annotations
    # ----------------------------------------------------------------------
    def _create_label_wise_masks_map(
        self, video_id: str, gt_annotations: List[Any]
    ):
        """
        PLACEHOLDER: use your existing implementation here.

        Should return:
            video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels
        """
        raise NotImplementedError(
            "_create_label_wise_masks_map should be replaced with your existing implementation."
        )

    def generate_video_bb_annotations(
        self,
        video_id: str,
        video_id_gt_annotations,
        video_id_gdino_annotations,
        video_id_3d_bbox_predictions,
        visualize: bool = False,
    ) -> None:
        """
        Generate world 4D annotations and frame-wise statistics
        for a single video.

        Parameters
        ----------
        video_id:
            Video id (e.g. "0DJ6R.mp4").
        video_id_gt_annotations:
            Raw GT annotations list (as loaded from get_video_gt_annotations).
        video_id_gdino_annotations:
            Combined GDINO predictions (dynamic + static).
        video_id_3d_bbox_predictions:
            Dict with precomputed 3D bbox results:

                {
                    "video_id": video_id,
                    "frames": out_frames,
                    "per_frame_sims": per_frame_sims,
                    "global_floor_sim": {
                        "s": float(s_avg),
                        "R": R_avg,
                        "t": t_avg,
                    },
                    "primary_track_id_0": primary_track_id_0,
                    "frame_bbox_meshes": frame_bbox_meshes,
                    "gv": gv,
                    "gf": gf,
                    "gc": gc
                }

            If None, this method will try to reload it from disk.
        visualize:
            If True, run Rerun-based visualization for this video.
        """

        print(f"[{video_id}] Generating world SGG annotations")

        # ------------------------------------------------------------------
        # 1) Load 3D points for the video from dynamic scene predictions
        # ------------------------------------------------------------------
        try:
            P = self._load_points_for_video(video_id)
            points_S = P["points"]          # (S, H, W, 3) for annotated frames
            conf_S = P["conf"]              # (S, H, W) or None
            stems_S = P["frame_stems"]      # list of frame stems
            colors_S = P["colors"]          # (S, H, W, 3) uint8
            camera_poses_S = P["camera_poses"]  # (S, 4, 4)
            S, H, W, _ = points_S.shape
        except Exception as e:
            print(f"[{video_id}] Failed to load 3D points: {e}")
            return

        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # ------------------------------------------------------------------
        # 2) Load the 3D bbox annotations for this video (from arg or disk)
        # ------------------------------------------------------------------
        video_3dgt = video_id_3d_bbox_predictions
        video_3dgt_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"

        if video_3dgt is None:
            if not video_3dgt_path.exists():
                print(f"[{video_id}] 3D bbox annotations not found at {video_3dgt_path}")
                return
            with open(video_3dgt_path, "rb") as f:
                video_3dgt = pickle.load(f)

        frame_3dbb_map = video_3dgt["frames"]
        per_frame_sims = video_3dgt.get("per_frame_sims", None)
        global_floor_sim_dict = video_3dgt.get("global_floor_sim", None)
        primary_track_id_0 = video_3dgt.get("primary_track_id_0", None)
        frame_bbox_meshes = video_3dgt.get("frame_bbox_meshes", {})
        gv = video_3dgt.get("gv", None)
        gf = video_3dgt.get("gf", None)
        gc = video_3dgt.get("gc", None)

        # Convert global_floor_sim to a tuple (s, R, t) if present
        global_floor_sim = None
        if global_floor_sim_dict is not None:
            s_avg = float(global_floor_sim_dict["s"])
            R_avg = np.asarray(global_floor_sim_dict["R"], dtype=np.float32)
            t_avg = np.asarray(global_floor_sim_dict["t"], dtype=np.float32)
            global_floor_sim = (s_avg, R_avg, t_avg)

        # ------------------------------------------------------------------
        # 3) Visualization: points + 3D bboxes + camera frustum (per frame)
        # ------------------------------------------------------------------
        if visualize:
            rerun_vis_world4d(
                video_id=video_id,
                points=points_S,
                colors=colors_S,
                conf=conf_S,
                camera_poses=camera_poses_S,
                frame_bbox_meshes=frame_bbox_meshes,
                global_floor_sim=global_floor_sim,
                floor=(gv, gf, gc) if gv is not None else None,
                img_maxsize=480,
                app_id="World4D-Combined",
                vis_floor=False,  # toggle as you like
            )

            print("Visualization running. Press Ctrl+C to stop.")
            # Keep process alive so the Rerun viewer can stay connected
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Visualization stopped by user.")

        # ------------------------------------------------------------------
        # 4) Frame-wise statistics about missing objects
        # ------------------------------------------------------------------
        # 4.1 All unique object labels in this video
        all_labels: set = set()
        for frame_name, objects in frame_3dbb_map.items():
            for obj in objects:
                label = obj.get("label", None)
                if label is None:
                    continue
                # Allow label to be a list/tuple; take first element
                if isinstance(label, (list, tuple, set)):
                    if not label:
                        continue
                    label = next(iter(label))
                if label is None:
                    continue
                all_labels.add(str(label))

        all_labels_sorted = sorted(all_labels)
        if not all_labels_sorted:
            print(f"[{video_id}] No object labels found in 3D bbox map.")
            if not visualize:
                print(f"[{video_id}] Finished (no labels).")
            return

        # 4.2 Per-frame stats: which labels are present / missing
        frame_names_sorted = sorted(
            frame_3dbb_map.keys(),
            key=lambda x: int(Path(x).stem),
        )

        per_frame_stats: List[Dict[str, Any]] = []
        per_object_missing_counts: Dict[str, int] = {
            lbl: 0 for lbl in all_labels_sorted
        }

        for frame_name in frame_names_sorted:
            objects = frame_3dbb_map.get(frame_name, [])
            present_labels: set = set()

            for obj in objects:
                label = obj.get("label", None)
                if label is None:
                    continue
                if isinstance(label, (list, tuple, set)):
                    if not label:
                        continue
                    label = next(iter(label))
                if label is None:
                    continue
                present_labels.add(str(label))

            missing_labels = sorted(list(all_labels - present_labels))
            for lbl in missing_labels:
                per_object_missing_counts[lbl] += 1

            per_frame_stats.append(
                {
                    "frame_name": frame_name,
                    "num_present": len(present_labels),
                    "num_missing": len(missing_labels),
                    "present_labels": sorted(list(present_labels)),
                    "missing_labels": missing_labels,
                }
            )

        # 4.3 Save stats (JSON + plots) under world_annotations/stats/<video_id>/
        stats_root = self.world_annotations_root_dir / "stats"
        stats_root.mkdir(parents=True, exist_ok=True)
        stats_dir = stats_root / video_id[:-4]
        stats_dir.mkdir(parents=True, exist_ok=True)

        stats = {
            "video_id": video_id,
            "num_frames": len(frame_names_sorted),
            "all_object_labels": all_labels_sorted,
            "per_frame_stats": per_frame_stats,
            "per_object_missing_counts": per_object_missing_counts,
        }

        stats_json_path = stats_dir / "missing_object_stats.json"
        with open(stats_json_path, "w") as f:
            json.dump(stats, f, indent=2)

        # 4.4 Plot: missing objects per frame
        frame_indices = [int(Path(s["frame_name"]).stem) for s in per_frame_stats]
        missing_counts = [s["num_missing"] for s in per_frame_stats]

        if frame_indices:
            plt.figure()
            plt.plot(frame_indices, missing_counts, marker="o")
            plt.xlabel("Frame index")
            plt.ylabel("# missing objects")
            plt.title(f"{video_id}: missing objects per frame")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(stats_dir / "missing_objects_per_frame.png")
            plt.close()

        # 4.5 Plot: number of frames where each object is missing
        labels_sorted_by_missing = sorted(
            per_object_missing_counts.keys(),
            key=lambda lbl: per_object_missing_counts[lbl],
            reverse=True,
        )
        counts_sorted_by_missing = [
            per_object_missing_counts[lbl] for lbl in labels_sorted_by_missing
        ]

        if labels_sorted_by_missing:
            plt.figure(figsize=(max(6, 0.4 * len(labels_sorted_by_missing)), 4))
            plt.bar(range(len(labels_sorted_by_missing)), counts_sorted_by_missing)
            plt.xticks(
                range(len(labels_sorted_by_missing)),
                labels_sorted_by_missing,
                rotation=45,
                ha="right",
            )
            plt.xlabel("Object label")
            plt.ylabel("# frames where label is missing")
            plt.title(f"{video_id}: missing-frames count per object")
            plt.tight_layout()
            plt.savefig(stats_dir / "missing_frames_per_object.png")
            plt.close()

        print(f"[{video_id}] Saved stats to {stats_json_path}")
        print(f"[{video_id}] Finished world 4D annotations/statistics generation.")

    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        """Load 3D points from dynamic scene predictions (sampled to annotated frames)."""
        video_dynamic_3d_scene_path = (
            self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        )
        video_dynamic_predictions = np.load(
            video_dynamic_3d_scene_path, allow_pickle=True
        )
        video_dynamic_predictions = {
            k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files
        }

        points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
        imgs_f32 = video_dynamic_predictions["images"]
        camera_poses = video_dynamic_predictions["camera_poses"]
        colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

        conf = None
        if "conf" in video_dynamic_predictions:
            conf = video_dynamic_predictions["conf"]
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]

        S, H, W, _ = points.shape

        # Get frame mapping
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

        assert S == len(sample_idx)

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
        conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
        stems_sub = [
            sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx
        ]
        colors_sub = colors[annotated_idx_in_sampled_idx]
        camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx]

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub,
        }

    def generate_sample_gt_world_4D_annotations(self, video_id: str) -> None:
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(
            video_id
        )
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)
        self.generate_video_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            video_id_3d_bbox_predictions,
            visualize=True,
        )

    def generate_gt_world_bb_annotations(
        self, dataloader: DataLoader, split: str
    ) -> None:
        """
        PLACEHOLDER: your existing implementation that iterates over the dataloader
        and calls generate_video_bb_annotations for videos in the given AG split.
        """
        raise NotImplementedError(
            "generate_gt_world_bb_annotations should be replaced with your existing implementation."
        )


# ----------------------------------------------------------------------
# Dataset + CLI
# ----------------------------------------------------------------------
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
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combined: "
            "(a) floor-aligned 3D bbox generator + "
            "(b) SMPL↔PI3 human mesh aligner (sampled frames only)."
        )
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument(
        "--output_world_annotations",
        type=str,
        default="/data/rohith/ag/ag4D/human/",
    )
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument(
        "--include_dense",
        action="store_true",
        help="use dense correspondences for human aligner",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(
        args.ag_root_directory
    )
    frame_to_world_generator.generate_gt_world_bb_annotations(
        dataloader=dataloader_train, split=args.split
    )
    frame_to_world_generator.generate_gt_world_bb_annotations(
        dataloader=dataloader_test, split=args.split
    )


def main_sample():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    video_id = "0DJ6R.mp4"
    frame_to_world_generator.generate_sample_gt_world_4D_annotations(video_id=video_id)


if __name__ == "__main__":
    # main()
    main_sample()
