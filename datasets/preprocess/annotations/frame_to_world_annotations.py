#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
from scipy.spatial.transform import Rotation as SciRot
from torch.utils.data import DataLoader

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


# ======================================================================================
# Rerun visualization with floor aligned to XY plane
# ======================================================================================
def rerun_vis_world4d(
        video_id: str,
        images: List[Optional[np.ndarray]],
        world4d: Dict[int, dict],
        faces: np.ndarray,
        *,
        annotated_frame_idx_in_sample_idx: List[int],
        sampled_indices: List[int],
        dynamic_prediction_path: str,
        per_frame_sims: Optional[Dict[int, Dict[str, Any]]] = None,
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
    Visualize world4d with both:
      (a) original meshes (after global_floor_sim, before floor-plane alignment), and
      (b) transformed meshes aligned so the floor lies in the XY plane (z=0, normal=+Z).

    Meshes shown with original+aligned:
      - floor mesh
      - human meshes (if vis_humans=True and faces is provided)
      - 3D bbox cuboids

    Points and camera frustum are shown only in the aligned world to keep
    visualization lighter.
    """

    faces_u32 = _faces_u32(faces) if faces is not None else None

    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    # ----------------------------------------------------------------------
    # Load dynamic scene predictions
    # ----------------------------------------------------------------------
    video_dynamic_prediction_path = os.path.join(
        dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz"
    )
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {
        k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files
    }
    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    conf = video_dynamic_predictions.get("conf", None)
    if conf is not None:
        conf = conf.astype(np.float32)
    imgs_f32 = video_dynamic_predictions["images"]
    colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    S = points.shape[0]

    # ----------------------------------------------------------------------
    # Global similarity (Pi3 -> base) from global_floor_sim, if provided
    # ----------------------------------------------------------------------
    s_g: Optional[float] = None
    R_g: Optional[np.ndarray] = None
    t_g: Optional[np.ndarray] = None
    if global_floor_sim is not None:
        s_g, R_g, t_g = global_floor_sim
        s_g = float(s_g)
        R_g = np.asarray(R_g, dtype=np.float32)
        t_g = np.asarray(t_g, dtype=np.float32)  # (3,)

    # ----------------------------------------------------------------------
    # Floor: bring into base frame (via global_floor_sim), then compute
    # a plane-alignment transform so the floor lies in the XY plane (z=0),
    # with its normal pointing along +Z.
    # ----------------------------------------------------------------------
    floor_world_base = None          # floor in "original" base frame (after global_floor_sim)
    floor_vertices_final = None      # floor aligned to XY plane
    floor_faces = None
    floor_kwargs = None

    # Alignment transform: base -> final world
    R_align = np.eye(3, dtype=np.float32)
    center = np.zeros(3, dtype=np.float32)  # point on the floor plane (base frame)

    if floor is not None:
        floor_verts0, floor_faces0, floor_colors0 = floor
        floor_verts0 = np.asarray(floor_verts0, dtype=np.float32)
        floor_faces0 = _faces_u32(np.asarray(floor_faces0))

        # Step 1: Pi3 -> base using global_floor_sim (if available)
        if s_g is not None:
            floor_world_base = s_g * (floor_verts0 @ R_g.T) + t_g[None, :]
        else:
            floor_world_base = floor_verts0

        # Step 2: fit plane & compute rotation to XY
        if floor_world_base.shape[0] >= 3:
            center = floor_world_base.mean(axis=0)
            centered = floor_world_base - center[None, :]

            # Plane normal via SVD (smallest singular vector)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            normal = Vt[-1]
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            else:
                normal = normal / norm

            # Make normal point "up" (positive z)
            if normal[2] < 0:
                normal = -normal

            z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if np.allclose(normal, z_axis, atol=1e-4):
                R_align = np.eye(3, dtype=np.float32)
            elif np.allclose(normal, -z_axis, atol=1e-4):
                # 180-degree flip around X axis
                R_align = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
            else:
                # Rodrigues' formula: rotation taking 'normal' -> z_axis
                v = np.cross(normal, z_axis)
                s = np.linalg.norm(v)
                c = float(np.dot(normal, z_axis))
                vx = np.array(
                    [
                        [0.0, -v[2], v[1]],
                        [v[2], 0.0, -v[0]],
                        [-v[1], v[0], 0.0],
                    ],
                    dtype=np.float32,
                )
                R_align = (
                    np.eye(3, dtype=np.float32)
                    + vx
                    + vx @ vx * ((1.0 - c) / (s**2 + 1e-8))
                )

            # Step 3: floor vertices in final world frame (z=0 plane)
            floor_vertices_final = (floor_world_base - center[None, :]) @ R_align.T
        else:
            # Not enough verts to fit a plane; just keep them as-is in base frame
            floor_vertices_final = floor_world_base

        # Floor material / colors
        floor_kwargs = {}
        if floor_colors0 is not None:
            floor_colors0 = np.asarray(floor_colors0, dtype=np.uint8)
            floor_kwargs["vertex_colors"] = floor_colors0
        else:
            floor_kwargs["albedo_factor"] = [160, 160, 160]
        floor_faces = floor_faces0

    # ----------------------------------------------------------------------
    # Helpers: Pi3 -> base, then base -> final (aligned)
    # ----------------------------------------------------------------------
    def _to_base(pi3_pts: np.ndarray) -> np.ndarray:
        """Pi3 -> base (apply only global_floor_sim)."""
        pts = np.asarray(pi3_pts, dtype=np.float32)
        if s_g is not None:
            pts = s_g * (pts @ R_g.T) + t_g[None, :]
        return pts

    def _to_final_from_base(base_pts: np.ndarray) -> np.ndarray:
        """base -> final (apply only floor-plane alignment)."""
        pts = np.asarray(base_pts, dtype=np.float32)
        if floor is not None and floor_vertices_final is not None:
            pts = (pts - center[None, :]) @ R_align.T
        return pts

    def _transform_points(pi3_pts: np.ndarray) -> np.ndarray:
        """Full transform: Pi3 -> base -> aligned."""
        return _to_final_from_base(_to_base(pi3_pts))

    def _transform_bbox_verts(verts_pi3: np.ndarray) -> np.ndarray:
        return _transform_points(verts_pi3)

    def _transform_camera(R_wc_pi3: np.ndarray, t_wc_pi3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Camera pose in Pi3 coords -> final world coords.

        For orientation, we compose rotations (no scaling).
        For translation, we treat the camera origin as a point and use _transform_points.
        """
        R_wc_pi3 = np.asarray(R_wc_pi3, dtype=np.float32)
        t_wc_pi3 = np.asarray(t_wc_pi3, dtype=np.float32)

        # Camera origin: transform as a point
        t_wc_final = _transform_points(t_wc_pi3[None, :])[0]

        # Orientation:
        # Pi3 -> base rotation
        if R_g is not None:
            R_pi3_to_base = R_g
        else:
            R_pi3_to_base = np.eye(3, dtype=np.float32)

        # base -> final rotation
        if floor is not None and floor_vertices_final is not None:
            R_base_to_final = R_align
        else:
            R_base_to_final = np.eye(3, dtype=np.float32)

        R_pi3_to_final = R_base_to_final @ R_pi3_to_base
        R_wc_final = R_pi3_to_final @ R_wc_pi3

        return R_wc_final, t_wc_final

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

    # edges for a cuboid (8 vertices)
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # frames to iterate
    if annotated_frame_idx_in_sample_idx:
        iter_indices = annotated_frame_idx_in_sample_idx
    else:
        iter_indices = list(range(len(sampled_indices)))

    for vis_t, sample_idx in enumerate(iter_indices):
        # sample_idx is index into points axis and sampled_indices
        if sample_idx < 0 or sample_idx >= len(sampled_indices):
            continue
        if sample_idx >= points.shape[0]:
            continue

        frame_idx = int(sampled_indices[sample_idx])

        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # ---------------------- world axis gizmo (aligned frame) ----------------------
        axis_len = 0.5
        x_axis = np.array([[0.0, 0.0, 0.0],
                           [axis_len, 0.0, 0.0]], dtype=np.float32)
        y_axis = np.array([[0.0, 0.0, 0.0],
                           [0.0, axis_len, 0.0]], dtype=np.float32)
        z_axis = np.array([[0.0, 0.0, 0.0],
                           [0.0, 0.0, axis_len]], dtype=np.float32)
        rr.log(
            f"{BASE}/aligned/world_axes",
            rr.LineStrips3D(
                strips=[x_axis, y_axis, z_axis],
                colors=[
                    [255,   0,   0],  # X = red
                    [  0, 255,   0],  # Y = green
                    [  0,   0, 255],  # Z = blue
                ],
            ),
        )

        # ---------------------- floor: original vs aligned ----------------------
        if vis_floor and (floor_world_base is not None) and (floor_faces is not None):
            # Original (base)
            rr.log(
                f"{BASE}/orig/floor",
                rr.Mesh3D(
                    vertex_positions=floor_world_base.astype(np.float32),
                    triangle_indices=floor_faces,
                    **(floor_kwargs or {}),
                ),
            )
        if vis_floor and (floor_vertices_final is not None) and (floor_faces is not None):
            # Aligned
            rr.log(
                f"{BASE}/aligned/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_final.astype(np.float32),
                    triangle_indices=floor_faces,
                    **(floor_kwargs or {}),
                ),
            )

        # ---------------------- frame data (for humans + camera) ----------------------
        frame_data = world4d.get(frame_idx, None)
        if frame_data is None:
            frame_data = {}

        # ---------------------- humans (SMPL meshes): original vs aligned ----------------------
        if vis_humans and faces_u32 is not None:
            track_ids = frame_data.get("track_id", [])
            verts_orig_list = frame_data.get("vertices_orig", [])
            if track_ids and verts_orig_list:
                tid = int(track_ids[0])
                verts_orig = np.asarray(verts_orig_list[0], dtype=np.float32)
                verts_world_pi3 = verts_orig

                # SMPL -> world (Pi3) via per-frame similarity (if provided)
                if per_frame_sims is not None and frame_idx in per_frame_sims:
                    s_i = float(per_frame_sims[frame_idx]["s"])
                    R_i = np.asarray(per_frame_sims[frame_idx]["R"], dtype=np.float32)
                    t_i = np.asarray(per_frame_sims[frame_idx]["t"], dtype=np.float32)
                    verts_flat = verts_orig.reshape(-1, 3)
                    verts_tf = s_i * (verts_flat @ R_i.T) + t_i
                    verts_world_pi3 = verts_tf.reshape(verts_orig.shape)

                # Original mesh: in base frame (only global_floor_sim)
                verts_base = _to_base(verts_world_pi3.reshape(-1, 3)).reshape(
                    verts_world_pi3.shape
                )
                rr.log(
                    f"{BASE}/orig/humans/h{tid}",
                    rr.Mesh3D(
                        vertex_positions=verts_base.astype(np.float32),
                        triangle_indices=faces_u32,
                        albedo_factor=[0, 200, 0],
                    ),
                )

                # Aligned mesh: base -> final
                verts_aligned = _to_final_from_base(
                    verts_base.reshape(-1, 3)
                ).reshape(verts_base.shape)
                rr.log(
                    f"{BASE}/aligned/humans/h{tid}",
                    rr.Mesh3D(
                        vertex_positions=verts_aligned.astype(np.float32),
                        triangle_indices=faces_u32,
                        albedo_factor=[0, 255, 0],
                    ),
                )

        # ---------------------- dynamic points: aligned only ----------------------
        pts_pi3 = points[sample_idx].reshape(-1, 3)    # (N,3)
        cols = colors[sample_idx].reshape(-1, 3)       # (N,3)

        if conf is not None:
            cfs = conf[sample_idx].reshape(-1)
            good = np.isfinite(cfs)
            cfs_valid = cfs[good]
            if cfs_valid.size > 0:
                med = np.median(cfs_valid)
                p5 = np.percentile(cfs_valid, 5)
                thr = max(min_conf_default, p5)
                print(
                    f"[{video_id}] frame {frame_idx}: conf thr = {thr:.4f} "
                    f"(med={med:.4f}, n_valid={cfs_valid.size})"
                )
            else:
                thr = min_conf_default

            keep = (cfs >= thr) & np.isfinite(pts_pi3).all(axis=1)
        else:
            keep = np.isfinite(pts_pi3).all(axis=1)

        pts_keep_pi3 = pts_pi3[keep]
        cols_keep = cols[keep]

        if pts_keep_pi3.shape[0] > 0:
            pts_keep_world = _transform_points(pts_keep_pi3)
            rr.log(
                f"{BASE}/aligned/points",
                rr.Points3D(
                    pts_keep_world.astype(np.float32),
                    colors=cols_keep.astype(np.uint8),
                ),
            )

        # ---------------------- per-frame cuboid bboxes: original vs aligned ----------------------
        if frame_bbox_meshes is not None and vis_t in frame_bbox_meshes:
            for bi, bbox_m in enumerate(frame_bbox_meshes[vis_t]):
                verts_pi3 = bbox_m["verts"].astype(np.float32)

                # Original: base frame
                verts_base = _to_base(verts_pi3)
                strips_orig = [
                    verts_base[[e0, e1], :] for (e0, e1) in cuboid_edges
                ]
                col = bbox_m.get("color", [255, 180, 0])
                rr.log(
                    f"{BASE}/orig/bboxes/frame_{vis_t}/bbox_{bi}",
                    rr.LineStrips3D(
                        strips=strips_orig,
                        colors=[col] * len(strips_orig),
                    ),
                )

                # Aligned: final world frame
                verts_world = _to_final_from_base(verts_base)
                strips_aligned = [
                    verts_world[[e0, e1], :] for (e0, e1) in cuboid_edges
                ]
                rr.log(
                    f"{BASE}/aligned/bboxes/frame_{vis_t}/bbox_{bi}",
                    rr.LineStrips3D(
                        strips=strips_aligned,
                        colors=[col] * len(strips_aligned),
                    ),
                )

        # ---------------------- camera: aligned only ----------------------
        cam_3x4 = frame_data.get("camera", None)
        if cam_3x4 is not None:
            cam_3x4 = np.asarray(cam_3x4, dtype=np.float32)
            if cam_3x4.shape == (4, 4):
                cam_3x4 = cam_3x4[:3, :4]

            R_wc_pi3 = cam_3x4[:3, :3]
            t_wc_pi3 = cam_3x4[:3, 3]

            R_wc_world, t_wc_world = _transform_camera(R_wc_pi3, t_wc_pi3)
        else:
            R_wc_world = None
            t_wc_world = None

        image = _get_image_for_time(frame_idx)

        if R_wc_world is not None and t_wc_world is not None:
            if image is not None:
                H_img, W_img = image.shape[:2]
            else:
                H_img, W_img = 480, 640
            fov_y = 0.96
            fx, fy, cx, cy = _pinhole_from_fov(W_img, H_img, fov_y)
            quat_xyzw = SciRot.from_matrix(R_wc_world).as_quat().astype(np.float32)

            frus_path = f"{BASE}/aligned/frustum"
            rr.log(
                frus_path,
                rr.Transform3D(
                    translation=t_wc_world.astype(np.float32),
                    rotation=rr.Quaternion(xyzw=quat_xyzw),
                )
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

    print("Rerun visualization running. Scrub the 'frame' timeline.")

# ======================================================================================
# FrameToWorldAnnotations
# ======================================================================================
class FrameToWorldAnnotations:

    def __init__(
        self,
        ag_root_directory: str,
        dynamic_scene_dir_path: str,
        smplx_faces: Optional[np.ndarray] = None,
    ) -> None:
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)
        self.smplx_faces = smplx_faces  # pass actual SMPL-X faces from outside if you want human meshes

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

    # ----------------------------------------------------------------------------------
    # GT + GDINO + 3D annotations loaders
    # ----------------------------------------------------------------------------------
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

    # ----------------------------------------------------------------------------------
    # Helper to compute sampling mappings: points index -> frame index, etc.
    # ----------------------------------------------------------------------------------
    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        """
        Compute mapping between dynamic predictions index and original frame indices,
        plus annotated frame indices.

        Returns:
            {
                "sampled_indices": List[int],                    # len = S (points axis)
                "annotated_frame_idx_in_sample_idx": List[int],  # subset of [0..S-1]
                "annotated_frame_id_list": List[str],            # ['000000.png', ...]
                "camera_poses": np.ndarray of shape (S, 4, 4),   # from predictions.npz
            }
        """
        video_dynamic_3d_scene_path = (
            self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        )
        video_dynamic_predictions = np.load(
            video_dynamic_3d_scene_path, allow_pickle=True
        )

        points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
        camera_poses = video_dynamic_predictions["camera_poses"].astype(np.float32)
        S = points.shape[0]
        del points  # we don't need them here

        # Get annotated frames list
        video_frames_annotated_dir_path = self.frame_annotated_dir_path / video_id
        annotated_frame_id_list = [
            f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith(".png")
        ]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        # Sampled frames index list (global indices for whole video)
        video_sampled_frames_npy_path = (
            self.sampled_frames_idx_root_dir / f"{video_id[:-4]}.npy"
        )
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_idx = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_idx = video_sampled_frame_id_list.index(annotated_last_frame_id)

        sample_idx_range = list(range(an_first_idx, an_last_idx + 1))
        assert S == len(
            sample_idx_range
        ), f"points axis ({S}) != sample range ({len(sample_idx_range)}) for {video_id}"

        # For each points index i in [0..S-1], we map to global frame idx:
        sampled_indices: List[int] = [
            int(video_sampled_frame_id_list[j]) for j in sample_idx_range
        ]

        # Map frame_name -> points index (0..S-1)
        frame_name_to_points_idx: Dict[str, int] = {}
        for i, vid_idx in enumerate(sample_idx_range):
            frame_id = video_sampled_frame_id_list[vid_idx]
            frame_name = f"{frame_id:06d}.png"
            frame_name_to_points_idx[frame_name] = i

        annotated_frame_idx_in_sample_idx: List[int] = []
        for frame_name in annotated_frame_id_list:
            if frame_name in frame_name_to_points_idx:
                annotated_frame_idx_in_sample_idx.append(
                    frame_name_to_points_idx[frame_name]
                )

        return {
            "sampled_indices": sampled_indices,
            "annotated_frame_idx_in_sample_idx": annotated_frame_idx_in_sample_idx,
            "annotated_frame_id_list": annotated_frame_id_list,
            "camera_poses": camera_poses,
        }

    # ----------------------------------------------------------------------------------
    # World 4D bbox annotations + stats
    # ----------------------------------------------------------------------------------
    def generate_video_bb_annotations(
        self,
        video_id: str,
        video_id_gt_annotations,
        video_id_gdino_annotations,
        video_id_3d_bbox_predictions,
        visualize: bool = False,
    ) -> None:
        """
        video_id_3D_bbox_predictions format (expected):

        {
            "video_id": video_id,
            "frames": frame_3dbb_map,
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
            "gc": gc,
            # optionally: "world4d": { frame_idx -> {...} }
        }

        This function:
          1. Runs Rerun visualization (if visualize=True) with floor-aligned world frame.
          2. Computes frame-wise statistics of missing objects and saves them to disk.
        """

        print(f"[{video_id}] Generating world SGG annotations")

        # ------------------------------------------------------------------
        # 1) Load point/frame mappings & camera poses from dynamic predictions
        # ------------------------------------------------------------------
        try:
            mapping = self._load_points_for_video(video_id)
            sampled_indices: List[int] = mapping["sampled_indices"]
            annotated_frame_idx_in_sample_idx: List[int] = mapping[
                "annotated_frame_idx_in_sample_idx"
            ]
            annotated_frame_id_list: List[str] = mapping["annotated_frame_id_list"]
            camera_poses: np.ndarray = mapping["camera_poses"]
        except Exception as e:
            print(f"[{video_id}] Failed to load dynamic mappings: {e}")
            return

        # ------------------------------------------------------------------
        # 2) Load 3D bbox annotations for this video
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

        # Optional world4d (if you already store it)
        world4d = video_3dgt.get("world4d", {})

        # If world4d is missing, build a minimal one from camera_poses
        if not world4d:
            world4d = {}
            S = camera_poses.shape[0]
            for i in range(S):
                frame_idx = int(sampled_indices[i])
                cam = camera_poses[i]
                if cam.shape == (4, 4):
                    cam_3x4 = cam[:3, :4]
                else:
                    cam_3x4 = cam
                world4d[frame_idx] = {"camera": cam_3x4}

        # Convert global_floor_sim to a tuple (s, R, t) if present
        global_floor_sim = None
        if global_floor_sim_dict is not None:
            s_avg = float(global_floor_sim_dict["s"])
            R_avg = np.asarray(global_floor_sim_dict["R"], dtype=np.float32)
            t_avg = np.asarray(global_floor_sim_dict["t"], dtype=np.float32)
            global_floor_sim = (s_avg, R_avg, t_avg)

        # ------------------------------------------------------------------
        # 3) Prepare images list indexed by global frame index
        # ------------------------------------------------------------------
        images: Optional[List[Optional[np.ndarray]]] = None
        try:
            if sampled_indices:
                max_frame_idx = max(sampled_indices)
                images = [None] * (max_frame_idx + 1)

                video_frames_annotated_dir_path = self.frame_annotated_dir_path / video_id
                for fname in annotated_frame_id_list:
                    frame_id = int(Path(fname).stem)
                    img_path = video_frames_annotated_dir_path / fname
                    if not img_path.exists():
                        continue
                    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    images[frame_id] = img
        except Exception as e:
            print(f"[{video_id}] Failed to load annotated images: {e}")
            images = None

        # ------------------------------------------------------------------
        # 4) Visualization: points + 3D bboxes + camera frustum (per frame)
        #     with floor plane aligned to the XY plane.
        # ------------------------------------------------------------------
        if visualize:
            faces_for_vis = (
                self.smplx_faces
                if self.smplx_faces is not None
                else np.zeros((0, 3), dtype=np.int32)
            )

            rerun_vis_world4d(
                video_id=video_id,
                images=images,
                world4d=world4d,
                faces=faces_for_vis,
                annotated_frame_idx_in_sample_idx=annotated_frame_idx_in_sample_idx,
                sampled_indices=sampled_indices,
                dynamic_prediction_path=str(self.dynamic_scene_dir_path),
                per_frame_sims=per_frame_sims,
                global_floor_sim=global_floor_sim,
                floor=(gv, gf, gc) if gv is not None else None,
                img_maxsize=480,
                app_id="World4D-Combined",
                frame_bbox_meshes=frame_bbox_meshes,
                vis_floor=True,
                vis_humans=False,  # flip to True if your world4d contains SMPL verts
            )

        # ------------------------------------------------------------------
        # 5) Frame-wise statistics about missing objects
        # ------------------------------------------------------------------
        # 5.1 All unique object labels in this video
        all_labels: set = set()
        for frame_name, frame_objects in frame_3dbb_map.items():
            for obj in frame_objects["objects"]:
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
            return

        # 5.2 Per-frame stats: which labels are present / missing
        frame_names_sorted = sorted(
            frame_3dbb_map.keys(),
            key=lambda x: int(Path(x).stem),
        )

        per_frame_stats: List[Dict[str, Any]] = []
        per_object_missing_counts: Dict[str, int] = {
            lbl: 0 for lbl in all_labels_sorted
        }

        for frame_name in frame_names_sorted:
            frame_objects = frame_3dbb_map.get(frame_name, [])
            objects = frame_objects.get("objects", [])
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

        # 5.3 Save stats (JSON + plots) under world_annotations/stats/<video_id>/
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

        # 5.4 Plot: missing objects per frame
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

        # 5.5 Plot: number of frames where each object is missing
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

    # ----------------------------------------------------------------------------------
    # Sample wrapper to run for a single video
    # ----------------------------------------------------------------------------------
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


# ======================================================================================
# Dataset + CLI
# ======================================================================================
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
            "World4D: floor-aligned 3D bbox visualizer + "
            "frame-wise missing-object stats generator."
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
        help="(placeholder flag, not used here)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        smplx_faces=None,  # replace with actual SMPL-X faces if you want human meshes
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
        smplx_faces=None,
    )
    video_id = "0DJ6R.mp4"
    frame_to_world_generator.generate_sample_gt_world_4D_annotations(video_id=video_id)


if __name__ == "__main__":
    # main()
    main_sample()
