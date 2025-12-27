#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from torch.utils.data import DataLoader

from dataloader.standard.action_genome.ag_dataset import StandardAG
from annotation_utils import (
    get_video_belongs_to_split,
    _load_pkl_if_exists,
    _npz_open,
    _faces_u32,
    _resize_mask_to,
    _mask_from_bbox,
    _resize_bbox_to,
    _xywh_to_xyxy,
    _finite_and_nonzero,
    _pinhole_from_fov,
    _is_empty_array,
)



# --------------------------------------------------------------------------------------
# Rerun visualization for ORIGINAL (Pi3) results
#   - Shows original point clouds for annotated frames.
#   - Now also shows:
#       * floor mesh (transformed to original/world coords).
#       * world coordinate frame (World +X/+Y/+Z).
#       * XYZ coordinate frame (X/Y/Z).
#       * camera frustum + camera axes (Cam +X/+Y/+Z).
#       * original 3D bounding boxes (BEFORE).
#       * transformed 3D bounding boxes (AFTER floor-alignment).
# --------------------------------------------------------------------------------------

def rerun_frame_vis_results(
    video_id: str,
    points_S: np.ndarray,
    conf_S: Optional[np.ndarray],
    stems_S: List[str],
    colors_S: Optional[np.ndarray],
    camera_poses_S: Optional[np.ndarray],
    frame_annotated_dir_path: Path,
    *,
    floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
    global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
    frame_3dbb_map: Optional[Dict[Any, List[Dict[str, Any]]]] = None,
    frame_bbox_meshes:  Optional[Dict[int, List[Dict[str, Any]]]] = None,
    img_maxsize: int = 320,
    app_id: str = "World4D-Original",
    min_conf_default: float = 1e-6,
) -> None:
    """
    Visualize ORIGINAL Pi3 outputs with BEFORE/AFTER comparison.

    BEFORE:
      - Floor mesh, floor axes, camera poses, points, and 3D bounding boxes in
        original/world coords.
      - Points are shown as dark gray.
      - 3D boxes are shown as white line segments.

    AFTER:
      - Floor mesh, floor axes, camera poses, points, and 3D bounding boxes
        transformed into a frame where the floor lies in the XY plane and
        Z is the floor normal.
      - Points are shown with their original RGB colors.
      - 3D boxes are shown as yellow line segments.

    AFTER_MIRROR:
      - Mirror image (about the ZY plane, i.e., x -> -x) of the AFTER points,
        3D bounding boxes, and cameras, logged under `world/after_mirror/...`.

    vis_mode:
        "before" -> only show original/world frame
        "after"  -> only show floor-aligned frame (plus its mirror branch)
        "both"   -> show both BEFORE and AFTER branches (plus mirror for AFTER)

    Args:
        video_id: video identifier (e.g., "0DJ6R.mp4")
        points_S: (S,H,W,3) Pi3/world points
        conf_S: (S,H,W) confidence (optional)
        stems_S: list of "000123" frame stems
        colors_S: (S,H,W,3) uint8 colors (optional)
        camera_poses_S: (S,4,4) or (S,3,4) camera extrinsics (optional)
        frame_annotated_dir_path: root dir for annotated PNG frames
        floor: (gv, gf, gc) floor mesh vertices/faces/colors (optional)
        global_floor_sim: (s,R,t) scaling+rotation+translation for floor
        frame_3dbb_map: dict mapping frame name -> 3D bbox annotations
        frame_bbox_meshes: dict mapping frame idx -> list of bbox meshes
        img_maxsize: max image size for RGB frames
        app_id: Rerun app id
        min_conf_default: fallback confidence threshold
        vis_mode: visualization mode ("before", "after", or "both")
    """

    # -------------------------------------------------------------------------
    # Normalize vis_mode + convenience flags
    # -------------------------------------------------------------------------
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    BASE = "world"
    BASE_AFTER_MIRROR = f"{BASE}/after_mirror"

    # Ensure both branches share the same coordinate convention
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # Cuboid edge list (same as rerun_vis_world4d)
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    # Mirror transform about ZY plane (x -> -x) in the aligned frame
    M_mirror = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

    # -------------------------------------------------------------------------
    # Static world & XYZ axes (common; drawn at origin of aligned coords)
    # -------------------------------------------------------------------------
    axis_len_world = 0.5
    world_axes = rr.Arrows3D(
        origins=[[0.0, 0.0, 0.0]] * 3,
        vectors=[
            [axis_len_world, 0.0, 0.0],  # +X
            [0.0, axis_len_world, 0.0],  # +Y
            [0.0, 0.0, axis_len_world],  # +Z
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

    # -------------------------------------------------------------------------
    # Floor mesh and floor-frame axes: BEFORE & AFTER
    # (mirror not requested for floor, so left as-is)
    # -------------------------------------------------------------------------
    floor_vertices = None
    floor_faces = None
    floor_kwargs: Optional[Dict[str, Any]] = None
    floor_origin_world = None
    floor_axes = None      # (3,3)

    R_align: Optional[np.ndarray] = None  # world -> aligned

    if floor is not None:
        floor_verts0, floor_faces0, floor_colors0 = floor
        floor_verts0 = np.asarray(floor_verts0, dtype=np.float32)
        floor_faces0 = _faces_u32(np.asarray(floor_faces0))

        if global_floor_sim is not None:
            s_g, R_g, t_g = global_floor_sim  # s: float, R: (3,3), t: (3,)
            R_g = np.asarray(R_g, dtype=np.float32)
            t_g = np.asarray(t_g, dtype=np.float32)

            # BEFORE: floor in original/world coords
            floor_vertices = s_g * (floor_verts0 @ R_g.T) + t_g

            # Canonical floor frame: XZ plane, Y normal
            t1 = R_g[:, 0]  # in-plane
            t2 = R_g[:, 2]  # in-plane
            n  = R_g[:, 1]  # normal

            floor_origin_world = t_g.astype(np.float32)
            axis_len_floor = float(s_g) * 0.5 if s_g is not None else 0.5

            floor_axes = np.stack(
                [
                    t1 * axis_len_floor,
                    t2 * axis_len_floor,
                    n  * axis_len_floor,
                ],
                axis=0,
            )  # each row = vector

            # Build floor-frame basis F (columns), and rotation to aligned frame.
            F = np.stack([t1, t2, n], axis=1)   # (3,3)
            R_align = F.T.astype(np.float32)    # world -> aligned


        floor_kwargs = {}
        if floor_colors0 is not None:
            floor_colors0 = np.asarray(floor_colors0, dtype=np.uint8)
            floor_kwargs["vertex_colors"] = floor_colors0
        else:
            floor_kwargs["albedo_factor"] = [160, 160, 160]

        floor_faces = floor_faces0

    # -------------------------------------------------------------------------
    # Points & cameras: BEFORE & AFTER (use R_align if available)
    # -------------------------------------------------------------------------
    S_pts, H_grid, W_grid, _ = points_S.shape

    points = points_S.astype(np.float32)
    camera = camera_poses_S

    # -------------------------------------------------------------------------
    # Helpers for static logging and images
    # -------------------------------------------------------------------------
    def _log_static_frames() -> None:
        rr.log(f"{BASE}/world_axes", world_axes)
        rr.log(f"{BASE}/xyz_axes", xyz_axes)
        rr.log(
            f"{BASE}/info",
            rr.TextLog(
                f"[{video_id}] BEFORE: original Pi3/world frame. "
                f"AFTER: frame where floor is in XY plane and Z is floor normal "
                f"(when global_floor_sim is available). "
                f"AFTER_MIRROR: AFTER mirrored about ZY plane (x -> -x). "
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

    # -------------------------------------------------------------------------
    # Main per-frame loop
    # -------------------------------------------------------------------------
    S = points.shape[0]

    for vis_t in range(S):
        stem = stems_S[vis_t]

        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        _log_static_frames()

        # ---------------------------------------------------------------------
        # FLOOR MESH: BEFORE / AFTER
        # ---------------------------------------------------------------------
        if floor_vertices is not None and floor_faces is not None:
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices.astype(np.float32),
                    triangle_indices=floor_faces,
                    **(floor_kwargs or {}),
                ),
            )

        # ---------------------------------------------------------------------
        # FLOOR AXES: BEFORE / AFTER
        # ---------------------------------------------------------------------
        if floor_origin_world is not None and floor_axes is not None:
            origins_floor_before = np.repeat(floor_origin_world[None, :], 3, axis=0)
            rr.log(
                f"{BASE}/floor_frame",
                rr.Arrows3D(
                    origins=origins_floor_before,
                    vectors=floor_axes,
                    colors=[
                        [200, 200, 0],
                        [0, 200, 200],
                        [200, 0, 200],
                    ],
                    labels=["Floor(Before) +X", "Floor(Before) +Y", "Floor(Before) +Z"],
                ),
            )

        # ---------------------------------------------------------------------
        # CAMERAS: BEFORE / AFTER + MIRROR
        # ---------------------------------------------------------------------
        if camera is not None and vis_t < camera.shape[0]:
            cam_pose_b = camera[vis_t]
            if cam_pose_b.shape == (3, 4):
                T_b = np.eye(4, dtype=np.float32)
                T_b[:3, :4] = cam_pose_b
            elif cam_pose_b.shape == (4, 4):
                T_b = cam_pose_b.astype(np.float32)
            else:
                T_b = np.eye(4, dtype=np.float32)

            cam_origin_b = T_b[:3, 3]
            R_cam_b = T_b[:3, :3]

            axis_len_cam = 0.4
            cam_axes_vec_b = np.stack(
                [
                    R_cam_b[:, 0] * axis_len_cam,
                    R_cam_b[:, 1] * axis_len_cam,
                    R_cam_b[:, 2] * axis_len_cam,
                ],
                axis=0,
            )
            origins_cam_b = np.repeat(cam_origin_b[None, :], 3, axis=0)

            rr.log(
                f"{BASE}/camera_axes",
                rr.Arrows3D(
                    origins=origins_cam_b,
                    vectors=cam_axes_vec_b,
                    colors=[
                        [180, 0, 0],
                        [0, 180, 0],
                        [0, 0, 180],
                    ],
                    labels=["Cam(Before) +X", "Cam(Before) +Y", "Cam(Before) +Z"],
                ),
            )

            rr.log(
                f"{BASE}/camera/frustum",
                rr.Pinhole(
                    fov_y=0.7853982,
                    aspect_ratio=float(W_grid) / float(H_grid),
                    camera_xyz=rr.ViewCoordinates.RUB,
                    image_plane_distance=0.1,
                ),
                rr.Transform3D(
                    translation=cam_origin_b.tolist(),
                    mat3x3=R_cam_b,
                ),
            )

        # ---------------------------------------------------------------------
        # POINTS: BEFORE (dark) / AFTER (colorful) / AFTER_MIRROR
        # ---------------------------------------------------------------------
        pts = points[vis_t].reshape(-1, 3)  # (N,3)
        cols = (
            colors_S[vis_t].reshape(-1, 3) if colors_S is not None else None
        )
        conf_flat = (
            conf_S[vis_t].reshape(-1) if conf_S is not None else None
        )

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
            keep = (conf_flat >= thr) & np.isfinite(pts).all(axis=1)
        else:
            thr = None
            keep = np.isfinite(pts).all(axis=1)

        # AFTER: colorful points
        pts_a_keep = pts[keep]
        if cols is not None:
            cols_a_keep = cols[keep].astype(np.uint8)
        else:
            cols_a_keep = None

        if pts_a_keep.shape[0] > 0:
            kwargs_pts: Dict[str, Any] = {}
            if cols_a_keep is not None:
                kwargs_pts["colors"] = cols_a_keep
            rr.log(
                f"{BASE}/points",
                rr.Points3D(
                    pts_a_keep,
                    **kwargs_pts,  # same colors as AFTER
                ),
            )

        # -----------------------------------------------------------------------------
        # 3D BOUNDING BOXES: BEFORE / AFTER / AFTER_MIRROR
        # -----------------------------------------------------------------------------
        if frame_3dbb_map is not None:
            frame_name = f"{stem}.png"
            if frame_name in frame_3dbb_map:
                frame_objects = frame_3dbb_map[frame_name]["objects"]
                for bi, obj in enumerate(frame_objects):
                    label = obj["label"]
                    bbox_3d = obj["aabb_floor_aligned"]
                    corners_world = np.asarray(
                        bbox_3d["corners_world"], dtype=np.float32
                    )  # (8,3)

                    col = obj.get("color", [255, 180, 0])

                    # BEFORE: world coords
                    strips = []
                    for e0, e1 in cuboid_edges:
                        strips.append(corners_world[[e0, e1], :])

                    rr.log(
                        f"{BASE}/bboxes/frame_{vis_t}/bbox_{bi}",
                        rr.LineStrips3D(
                            strips=strips,
                            colors=[col] * len(strips),
                        ),
                    )
        # ---------------------------------------------------------------------
        # ORIGINAL RGB FRAME (single copy, not transformed)
        # ---------------------------------------------------------------------
        img = _get_image_for_stem(stem)
        if img is not None:
            rr.log(f"{BASE}/image", rr.Image(img))

    print(
        "[orig-pts] BEFORE/AFTER visualization running for "
        f"{video_id}. BEFORE = original frame, AFTER = floor-aligned frame, "
        "AFTER_MIRROR = mirror of AFTER about ZY plane (x -> -x). . Scrub the 'frame' timeline in Rerun and "
        "toggle entities in the UI."
    )

# --------------------------------------------------------------------------------------
# FrameToWorldAnnotations
#   - loads 3D bbox annotations (.pkl produced by BBox3DGenerator)
#   - can visualize ORIGINAL Pi3 points + floor mesh + 3D boxes for annotated frames
# --------------------------------------------------------------------------------------


class FrameToWorldAnnotations:
    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str):
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

        # World annotation dirs (3D bboxes already generated by BBox3DGenerator)
        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        self.bbox_4d_root_dir = self.world_annotations_root_dir / "bbox_annotations_4d"
        os.makedirs(self.bbox_4d_root_dir, exist_ok=True)

        # GT annotations
        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        # Segmentation dirs (if you later want to build label-wise masks)
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

    def get_video_dynamic_predictions(self, video_id: str):
        """
        If you ever need full dynamic scene predictions (not restricted to annotated frames).
        Not used in the original-points visualization, which uses a more precise slicing.
        """
        video_dynamic_3d_scene_path = (
            self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        )
        if not video_dynamic_3d_scene_path.exists():
            return None
        video_dynamic_predictions = _npz_open(video_dynamic_3d_scene_path)
        return video_dynamic_predictions

    # ----------------------------------------------------------------------------------
    # Indexing helper — mirror BBox3DGenerator.idx_to_frame_idx_path
    # ----------------------------------------------------------------------------------

    def _bbox_idx_to_frame_idx_path(
        self, video_id: str
    ) -> Tuple[
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
    # ORIGINAL POINTS LOADER – matches BBox3DGenerator._load_points_for_video
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
                camera_poses = video_dynamic_predictions["camera_poses"].astype(
                    np.float32
                )

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

    # ----------------------------------------------------------------------------------
    # World 4D bbox annotations (skeleton) + ORIGINAL-results visualization
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
        Placeholder skeleton for world 4D bbox annotations.

        Right now this:
          - loads the 3D bbox predictions (if not already passed);
          - prints some basic statistics over the 3D objects.

        Visualization of ORIGINAL results (points/bboxes/cameras/floor) is handled
        by `visualize_original_results`, and currently draws:
          - original point clouds
          - floor mesh
          - world + XYZ coordinate frames
          - camera frustum + camera axes
          - original & transformed 3D bounding boxes
        """
        print(f"[world4d][{video_id}] Generating world SGG annotations (skeleton)")

        # Load 3D bboxes from disk if not provided
        if video_id_3d_bbox_predictions is not None:
            video_3dgt = video_id_3d_bbox_predictions
        else:
            video_3dgt = self.get_video_3d_annotations(video_id)

        if video_3dgt is None:
            print(f"[world4d][{video_id}] No 3D bbox annotations available, skipping.")
            return

        frame_3dbb_map = video_3dgt["frames"]

        # Basic dataset-level stats
        all_labels = set()
        num_frames_with_objects = 0
        num_total_objects = 0

        for frame_name, frame_rec in frame_3dbb_map.items():
            objects = frame_rec.get("objects", [])
            if not objects:
                continue
            num_frames_with_objects += 1
            num_total_objects += len(objects)
            for obj in objects:
                lbl = obj.get("label", None)
                if lbl:
                    all_labels.add(lbl)

        print(
            f"[world4d][{video_id}] frames_with_objects={num_frames_with_objects}, "
            f"total_objects={num_total_objects}, "
            f"unique_labels={sorted(all_labels)}"
        )

        if visualize:
            print(
                f"[world4d][{video_id}] visualize=True requested, "
                "but world-4D bbox visualization is not yet implemented here."
            )

    def generate_video_world_3D_annotations(self, video_id: str) -> None:
        """
        Convenience wrapper:
          1) Loads original Pi3 points using the same slicing logic as bbox_3D construction.
          2) Loads floor mesh + global_floor_sim from the 3D bbox .pkl (if present).
          3) Launches a Rerun viewer that shows:
             - original point clouds
             - floor mesh
             - world & XYZ coordinate frames
             - camera frustum + camera axes
             - original & transformed 3D bounding boxes.
        """
        P = self._load_original_points_for_video(video_id)

        # Optional floor mesh + similarity transform from bbox_3d pkl
        floor = None
        global_floor_sim = None
        frame_3dbb_map = None
        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is not None:
            gv = video_3dgt.get("gv", None)
            gf = video_3dgt.get("gf", None)
            gc = video_3dgt.get("gc", None)
            if gv is not None and gf is not None:
                floor = (gv, gf, gc)

            gfs = video_3dgt.get("global_floor_sim", None)
            if gfs is not None:
                s_g = float(gfs["s"])
                R_g = np.asarray(gfs["R"], dtype=np.float32)
                t_g = np.asarray(gfs["t"], dtype=np.float32)
                global_floor_sim = (s_g, R_g, t_g)

            # 3D bboxes per frame
            frame_3dbb_map = video_3dgt.get("frames", None)
        else:
            print(
                f"[world4d][{video_id}] No 3D bbox annotations found; "
                "skipping floor and bbox visualization."
            )

        points_S = P["points"]  # (S,H,W,3)
        conf_S = P["conf"]      # (S,H,W) or None
        stems_S = P["frame_stems"]  # List[str]
        colors_S = P["colors"]  # (S,H,W,3) uint8
        camera_poses_S = P["camera_poses"]  # (S,4,4) or None

        # Cuboid edge list (same as rerun_vis_world4d)
        cuboid_edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),
            (4, 5), (5, 7), (7, 6), (6, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        # Mirror transform about ZY plane (x -> -x) in the aligned frame
        M_mirror = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

        # -------------------------------------------------------------------------
        # Floor mesh and floor-frame axes: BEFORE & AFTER
        # (mirror not requested for floor, so left as-is)
        # -------------------------------------------------------------------------
        floor_vertices_before = None
        floor_vertices_after = None
        floor_faces = None
        floor_kwargs: Optional[Dict[str, Any]] = None

        floor_origin_world = None
        floor_axes_before = None  # (3,3)
        floor_axes_after = None  # (3,3) in aligned frame

        R_align: Optional[np.ndarray] = None  # world -> aligned

        if floor is not None:
            floor_verts0, floor_faces0, floor_colors0 = floor
            floor_verts0 = np.asarray(floor_verts0, dtype=np.float32)
            floor_faces0 = _faces_u32(np.asarray(floor_faces0))

            if global_floor_sim is not None:
                s_g, R_g, t_g = global_floor_sim  # s: float, R: (3,3), t: (3,)
                R_g = np.asarray(R_g, dtype=np.float32)
                t_g = np.asarray(t_g, dtype=np.float32)

                # BEFORE: floor in original/world coords
                floor_vertices_before = s_g * (floor_verts0 @ R_g.T) + t_g

                # Canonical floor frame: XZ plane, Y normal
                t1 = R_g[:, 0]  # in-plane
                t2 = R_g[:, 2]  # in-plane
                n = R_g[:, 1]  # normal

                floor_origin_world = t_g.astype(np.float32)
                axis_len_floor = float(s_g) * 0.5 if s_g is not None else 0.5

                floor_axes_before = np.stack(
                    [
                        t1 * axis_len_floor,
                        t2 * axis_len_floor,
                        n * axis_len_floor,
                    ],
                    axis=0,
                )  # each row = vector

                # Build floor-frame basis F (columns), and rotation to aligned frame.
                F = np.stack([t1, t2, n], axis=1)  # (3,3)
                R_align = F.T.astype(np.float32)  # world -> aligned

                # AFTER: floor in aligned coords (XY plane, Z normal)
                v_centered = floor_vertices_before - floor_origin_world[None, :]
                floor_vertices_after = v_centered @ R_align.T

                floor_axes_after = np.array(
                    [
                        [axis_len_floor, 0.0, 0.0],
                        [0.0, axis_len_floor, 0.0],
                        [0.0, 0.0, axis_len_floor],
                    ],
                    dtype=np.float32,
                )
            else:
                # No global_floor_sim: we can only show "before", no aligned view
                floor_vertices_before = floor_verts0

            floor_kwargs = {}
            if floor_colors0 is not None:
                floor_colors0 = np.asarray(floor_colors0, dtype=np.uint8)
                floor_kwargs["vertex_colors"] = floor_colors0
            else:
                floor_kwargs["albedo_factor"] = [160, 160, 160]

            floor_faces = floor_faces0

            # -------------------------------------------------------------------------
            # Points & cameras: BEFORE & AFTER (use R_align if available)
            # -------------------------------------------------------------------------
            S_pts, H_grid, W_grid, _ = points_S.shape

            points_before = points_S.astype(np.float32)
            points_after = None
            camera_before = camera_poses_S
            camera_after = None

            if R_align is not None and floor_origin_world is not None:
                # Points AFTER
                pts_flat = points_before.reshape(-1, 3)
                v_centered_pts = pts_flat - floor_origin_world[None, :]
                pts_flat_after = v_centered_pts @ R_align.T
                points_after = pts_flat_after.reshape(S_pts, H_grid, W_grid, 3)
                points_after = (points_after @ M_mirror.T).astype(np.float32)

                # Cameras AFTER
                if camera_before is not None:
                    cam_aligned = []
                    for cam_pose in camera_before:
                        if cam_pose.shape == (3, 4):
                            T = np.eye(4, dtype=np.float32)
                            T[:3, :4] = cam_pose
                        elif cam_pose.shape == (4, 4):
                            T = cam_pose.astype(np.float32)
                        else:
                            T = np.eye(4, dtype=np.float32)

                        cam_origin_world = T[:3, 3]
                        R_cam_world = T[:3, :3]

                        cam_origin_centered = cam_origin_world - floor_origin_world
                        cam_origin_aligned = R_align @ cam_origin_centered
                        R_cam_aligned = R_align @ R_cam_world

                        T_new = np.eye(4, dtype=np.float32)
                        T_new[:3, :3] = R_cam_aligned
                        T_new[:3, 3] = cam_origin_aligned
                        cam_aligned.append(T_new)

                    camera_after = np.stack(cam_aligned, axis=0)

            S = points_after.shape[0]
            for vis_t in range(S):
                stem = stems_S[vis_t]
                if frame_3dbb_map is not None:
                    frame_name = f"{stem}.png"
                    if frame_name in frame_3dbb_map:
                        frame_objects = frame_3dbb_map[frame_name]["objects"]
                        for bi, obj in enumerate(frame_objects):
                            label = obj["label"]
                            bbox_3d = obj["aabb_floor_aligned"]
                            corners_world = np.asarray(
                                bbox_3d["corners_world"], dtype=np.float32
                            )  # (8,3)

                            col = obj.get("color", [255, 180, 0])
                            verts_centered = corners_world - floor_origin_world[None, :]
                            verts_after = verts_centered @ R_align.T  # world -> aligned

                            # AFTER_MIRROR: mirror about ZY plane (x -> -x)
                            verts_after = (verts_after @ M_mirror.T).astype(np.float32)



            # Create a pkl file that stores all the information together
            # Include only the transformation files needed for visualization
            # (no need to bloat the pkl with original points, colors, etc.)
            # TODO: Construct 3DBB Map with floor-aligned boxes directly

            # Update video_3dgt with final 3D bbox annotations information that includes things after transformation.
            out_dict = {
            }

    def generate_gt_world_3D_bb_annotations(
        self, dataloader: DataLoader, split: str
    ) -> None:
        """
        Iterate over an AG dataloader and call generate_video_bb_annotations
        for videos in the given AG split.

        This is useful if/when you implement full world-4D GT generation
        for all videos.
        """
        for data in dataloader:
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue

            try:
                print(f"[world4d] processing video {video_id}...")
                video_id_gt_bboxes, video_id_gt_annotations = (
                    self.get_video_gt_annotations(video_id)
                )
                video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
                video_id_3d_bbox_predictions = self.get_video_3d_annotations(video_id)

                self.generate_video_bb_annotations(
                    video_id,
                    video_id_gt_annotations,
                    video_id_gdino_annotations,
                    video_id_3d_bbox_predictions,
                    visualize=False,
                )
            except Exception as e:
                print(f"[world4d] failed to process video {video_id}: {e}")


# --------------------------------------------------------------------------------------
# Dataset + CLI
# --------------------------------------------------------------------------------------


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
            "World4D GT helper: "
            "(a) inspect 3D bbox annotations, "
            "(b) visualize original Pi3 outputs (points + floor + frames + camera + 3D boxes) "
            "for annotated frames."
        )
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--split", type=str, default="04")
    return parser.parse_args()


def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    # Example: run over one split
    frame_to_world_generator.generate_gt_world_3D_bb_annotations(
        dataloader=dataloader_train, split=args.split
    )
    frame_to_world_generator.generate_gt_world_3D_bb_annotations(
        dataloader=dataloader_test, split=args.split
    )


def main_sample():
    """
    Simple entry point to visualize original Pi3 point clouds + floor mesh
    + coordinate frames + camera frustum + 3D bounding boxes for a single video.
    Adjust `video_id` as needed.
    """
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    video_id = "0DJ6R.mp4"
    frame_to_world_generator.generate_sample_gt_world_3D_annotations(video_id=video_id)


if __name__ == "__main__":
    # main()
    main_sample()
