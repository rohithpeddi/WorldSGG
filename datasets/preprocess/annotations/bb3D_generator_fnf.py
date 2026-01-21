#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import torch
from scipy.spatial.transform import Rotation as SciRot
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + '/..')

# from AG / human pipeline codebase
from dataloader.standard.action_genome.ag_dataset import StandardAG

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline
from datasets.preprocess.human.data_config import SMPLX_PATH
from datasets.preprocess.human.prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from datasets.preprocess.human.prompt_hmr.vis.traj import (
    get_floor_mesh,
)
from datasets.preprocess.human.pipeline.kp_utils import (
    get_openpose_joint_names,
    get_smpl_joint_names,
)


# =====================================================================
from datasets.preprocess.annotations.bb3D_base import BBox3DBase
from datasets.preprocess.annotations.annotation_utils import (
    _load_pkl_if_exists,
    _is_empty_array,
    get_video_belongs_to_split,
    _faces_u32,
    _pinhole_from_fov,
    _xywh_to_xyxy,
    _resize_bbox_to,
    _area_xyxy,
    _iou_xyxy,
    _mask_from_bbox,
    _resize_mask_to,
    _finite_and_nonzero,
    transform_pts_R_offset,
    inv_transform_pts_R_offset,
    _box_edges_from_corners,
    _lift_2d_to_3d,
    _build_frame_to_kps_map,
    _log_box_lines_rr,
    _match_gdino_to_gt,
    _choose_primary_actor,
    _find_actor_index_in_frame,
    _similarity_umeyama,
    _robust_similarity_ransac,
    _mad_based_mask,
    _average_sims_robust,
)



# =====================================================================
# BOUNDING BOX GENERATOR (AABB is floor-aligned)
# =====================================================================
class BBox3DGenerator(BBox3DBase):

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
            output_human_dir_path: Optional[str] = None,
    ) -> None:
        super().__init__(dynamic_scene_dir_path, ag_root_directory)

        self.pipeline = AgPipeline(static_cam=False, dynamic_scene_dir_path=self.dynamic_scene_dir_path)
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()

        self.output_human_dir_path = Path(output_human_dir_path)



    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        video_dynamic_predictions = np.load(video_dynamic_3d_scene_path, allow_pickle=True)

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
    # lifting and partial pointcloud
    # ------------------------------------------------------------------
    def _get_partial_pointcloud(self, video_id: str, frame_idx: int, frame_idx_frame_path_map,
                                label: str = "person") -> np.ndarray:
        pts_hw3 = self.pipeline.points[frame_idx]
        H, W, _ = pts_hw3.shape
        stem = frame_idx_frame_path_map[frame_idx][:-4]
        mask = self.get_union_mask(video_id, stem, label, is_static=False)
        if mask is not None:
            mask = mask.astype(bool)
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            pts = pts_hw3[mask]
        else:
            pts = pts_hw3.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        nonzero = ~(np.abs(pts).sum(axis=1) < 1e-6)
        pts = pts[nonzero]
        return pts

    def _collect_kp_corr_for_frame(self, frame_idx: int, frame_data: dict, frame_to_kps: dict,
                                   primary_track_id_0: Optional[int]):
        if frame_idx not in frame_to_kps:
            return None, None
        actor_idx = _find_actor_index_in_frame(frame_data, primary_track_id_0)
        if actor_idx is None:
            return None, None
        kps_2d = frame_to_kps[frame_idx]
        rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]

        smpl_out = self.smplx(
            global_orient=rotmat_actor[:, :1].cuda(),
            body_pose=rotmat_actor[:, 1:22].cuda(),
            betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
            transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
        )
        joints = smpl_out.joints.cpu().numpy()
        joints = joints[:, :24, :]
        smpl_joints_actor = joints[0]

        frame_points_hw3 = self.pipeline.points[frame_idx]
        smpl_joint_names = get_smpl_joint_names()
        openpose_joint_names = get_openpose_joint_names()
        OPENPOSE_TO_SMPL = {
            "OP Nose": "head",
            "OP Neck": "neck",
            "OP MidHip": "hips",
            "OP LHip": "leftUpLeg",
            "OP RHip": "rightUpLeg",
            "OP LKnee": "leftLeg",
            "OP RKnee": "rightLeg",
            "OP LAnkle": "leftFoot",
            "OP RAnkle": "rightFoot",
            "OP LShoulder": "leftShoulder",
            "OP RShoulder": "rightShoulder",
            "OP LElbow": "leftForeArm",
            "OP RElbow": "rightForeArm",
            "OP LWrist": "leftHand",
            "OP RWrist": "rightHand",
            "OP LBigToe": "leftToeBase",
            "OP RBigToe": "rightToeBase",
        }
        if not smpl_joint_names or len(smpl_joint_names) < smpl_joints_actor.shape[0]:
            return None, None
        smpl_name_to_pt = {smpl_joint_names[j]: smpl_joints_actor[j] for j in range(smpl_joints_actor.shape[0])}

        smpl_list = []
        scene_list = []
        for op_name, kp in kps_2d.items():
            smpl_name = OPENPOSE_TO_SMPL.get(op_name, None)
            if smpl_name is None:
                continue
            if smpl_name not in smpl_name_to_pt:
                continue
            u = float(kp[0]);
            v = float(kp[1])
            scene_p = _lift_2d_to_3d(frame_points_hw3, u, v)
            if scene_p is None:
                continue
            smpl_list.append(smpl_name_to_pt[smpl_name])
            scene_list.append(scene_p)

        if len(smpl_list) == 0:
            return None, None
        return np.stack(smpl_list, axis=0), np.stack(scene_list, axis=0)

    def _collect_dense_corr_for_frame(
            self,
            video_id: str,
            frame_idx: int,
            frame_data: dict,
            frame_idx_frame_path_map: dict,
            primary_track_id_0: Optional[int],
            num_smpl_samples_per_person: int = 400,
            num_scene_subsample: int = 800,
    ):
        actor_idx = _find_actor_index_in_frame(frame_data, primary_track_id_0)
        if actor_idx is None:
            return None, None
        rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]
        smpl_out = self.smplx(
            global_orient=rotmat_actor[:, :1].cuda(),
            body_pose=rotmat_actor[:, 1:22].cuda(),
            betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
            transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
        )
        verts = smpl_out.vertices.cpu().numpy()
        smpl_verts = verts[0]

        scene_pts = self._get_partial_pointcloud(video_id, frame_idx, frame_idx_frame_path_map, label="person")
        if scene_pts.shape[0] == 0:
            return None, None

        if scene_pts.shape[0] > num_scene_subsample:
            choice = np.random.choice(scene_pts.shape[0], num_scene_subsample, replace=False)
            scene_pts = scene_pts[choice]

        if smpl_verts.shape[0] > num_smpl_samples_per_person:
            idx = np.random.choice(smpl_verts.shape[0], num_smpl_samples_per_person, replace=False)
            smpl_sampled = smpl_verts[idx]
        else:
            smpl_sampled = smpl_verts

        matched_scene = []
        for sp in smpl_sampled:
            dists = np.sum((scene_pts - sp[None, :]) ** 2, axis=1)
            nn_idx = np.argmin(dists)
            matched_scene.append(scene_pts[nn_idx])
        matched_scene = np.asarray(matched_scene, dtype=np.float64)
        return smpl_sampled, matched_scene

    def process_video(self, video_id: str, include_dense: bool = False, use_consistent_transformation: bool = False):
        # 0) run human/scene pipeline
        self.pipeline.__call__(video_id, save_only_essential=False)
        self.pipeline.estimate_2d_keypoints()
        results = self.pipeline.results
        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        # make frame indices 0..N-1
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        # sampled / annotated indices
        (frame_idx_frame_path_map,
         sample_idx,
         _,
         _,
         annotated_frame_idx_in_sample_idx) = self.idx_to_frame_idx_path(video_id)
        sampled_frame_indices = sorted(frame_idx_frame_path_map.keys())

        # choose primary actor
        primary_person_id_1, primary_track_id_0 = _choose_primary_actor(results, world4d)
        print(f"[align] Using primary actor -> results={primary_person_id_1}, world4d_track={primary_track_id_0}")

        # map frame -> keypoints for that actor
        frame_to_kps = _build_frame_to_kps_map(results, primary_person_id_1)

        # containers
        frame_kp_corr: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        per_frame_sims: Dict[int, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # 1) estimate per-frame similarity *for sampled frames*
        # ------------------------------------------------------------------
        for frame_idx in sampled_frame_indices:
            frame_data = world4d.get(frame_idx, None)
            if frame_data is None:
                continue

            smpl_s, scene_s = self._collect_kp_corr_for_frame(
                frame_idx, frame_data, frame_to_kps, primary_track_id_0
            )

            smpl_all = []
            scene_all = []

            if smpl_s is not None and scene_s is not None and smpl_s.shape[0] > 0:
                frame_kp_corr[frame_idx] = (smpl_s, scene_s)
                smpl_all.append(smpl_s)
                scene_all.append(scene_s)

            if include_dense:
                smpl_d, scene_d = self._collect_dense_corr_for_frame(
                    video_id, frame_idx, frame_data, frame_idx_frame_path_map, primary_track_id_0
                )
                if smpl_d is not None and scene_d is not None:
                    smpl_all.append(smpl_d)
                    scene_all.append(scene_d)

            if len(smpl_all) == 0:
                print(f"[align][{video_id}] no corr for sampled frame {frame_idx}")
                continue

            smpl_all = np.concatenate(smpl_all, axis=0)
            scene_all = np.concatenate(scene_all, axis=0)

            if smpl_all.shape[0] < 3:
                print(f"[align][{video_id}] insufficient corr for frame {frame_idx}")
                continue

            # solve per-frame sim
            s_f, R_f, t_f = _robust_similarity_ransac(
                smpl_all, scene_all,
                max_iters=800,
                inlier_thresh=0.03,
                min_inliers=4,
                scale_bounds=(0.4, 3.0),
            )

            per_frame_sims[frame_idx] = {
                "s": float(s_f),
                "R": R_f,
                "t": t_f,
                "w": float(smpl_all.shape[0]),
            }

        # ------------------------------------------------------------------
        # 2) compute robust average over sampled frames
        # ------------------------------------------------------------------
        sampled_per_frame_sims = {k: v for k, v in per_frame_sims.items() if k in sampled_frame_indices}
        s_avg, R_avg, t_avg = _average_sims_robust(sampled_per_frame_sims)

        if use_consistent_transformation:
            per_frame_sims = {
                fidx: {
                    "s": float(s_avg),
                    "R": R_avg,
                    "t": t_avg,
                    "w": 1.0,
                }
                for fidx in sampled_frame_indices
            }

        # ------------------------------------------------------------------
        # 3) build verts per sampled frame and apply either per-frame or avg sim
        # ------------------------------------------------------------------
        all_verts_for_floor = []
        for frame_idx in sampled_frame_indices:
            frame_data = world4d.get(frame_idx, None)
            if frame_data is None:
                continue
            actor_idx = _find_actor_index_in_frame(frame_data, primary_track_id_0)
            if actor_idx is None:
                frame_data['track_id'] = []
                continue

            rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
            rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]
            smpl_out = self.smplx(
                global_orient=rotmat_actor[:, :1].cuda(),
                body_pose=rotmat_actor[:, 1:22].cuda(),
                betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
                transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
            )
            verts = smpl_out.vertices.cpu().numpy()

            frame_data['track_id'] = [primary_track_id_0]
            frame_data['vertices_orig'] = [verts[0].copy()]

            all_verts_for_floor.append(torch.tensor(verts, dtype=torch.bfloat16))

            if use_consistent_transformation and sampled_per_frame_sims:
                # use the robust average for EVERY frame
                s_use, R_use, t_use = s_avg, R_avg, t_avg
                verts_flat = verts.reshape(-1, 3)
                verts_tf = s_use * (verts_flat @ R_use.T) + t_use
                verts_tf = verts_tf.reshape(verts.shape)
            else:
                # fall back to per-frame sim if we have it
                if frame_idx in per_frame_sims:
                    s_f = per_frame_sims[frame_idx]["s"]
                    R_f = per_frame_sims[frame_idx]["R"]
                    t_f = per_frame_sims[frame_idx]["t"]
                    verts_flat = verts.reshape(-1, 3)
                    verts_tf = s_f * (verts_flat @ R_f.T) + t_f
                    verts_tf = verts_tf.reshape(verts.shape)
                else:
                    verts_tf = verts
            frame_data['vertices'] = [verts_tf[0]]

        # ------------------------------------------------------------------
        # 4) floor mesh from all transformed verts' originals
        # ------------------------------------------------------------------
        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            gv, gf, gc = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf, gc = None, None, None

        return (
            images,
            world4d,
            sampled_frame_indices,
            per_frame_sims,
            s_avg,
            R_avg,
            t_avg,
            gv,
            gf,
            gc,
            annotated_frame_idx_in_sample_idx,
            primary_track_id_0
        )

    def generate_video_bb_annotations(
            self,
            video_id: str,
            video_gt_annotations: List[Any],
            video_gdino_predictions: Dict[str, Any],
            *,
            min_points: int = 50,
            iou_thr: float = 0.3,
            visualize: bool = False,
            use_consistent_transformation: bool = False,
    ) -> None:
        # 1) load dynamic points (already sub-sampled to annotated frames)
        P = self._load_points_for_video(video_id)
        points_S = P["points"]  # (S, H, W, 3)
        conf_S = P["conf"]  # (S, H, W) or None
        stems_S = P["frame_stems"]  # list[str], len S
        colors = P["colors"]
        S, H, W, _ = points_S.shape

        # Original image height and width of images corresponding to the video
        # We will use them to re-size the bounding boxes and masks to the size points - (H, W)
        sample_image_frame = self.frame_annotated_dir_path / video_id / f"{stems_S[0]}.png"
        orig_img = cv2.imread(str(sample_image_frame))
        orig_H, orig_W = orig_img.shape[:2]

        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # 2) build label-wise masks (static + dynamic) for this video
        video_to_frame_to_label_mask, _, _ = self.create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_gt_annotations
        )

        # 3) run the human/scene pipeline & get the global floor sim
        (
            images,
            world4d,
            sampled_frame_indices,
            per_frame_sims,
            s_avg,
            R_avg,
            t_avg,
            gv,
            gf,
            gc,
            annotated_frame_idx_in_sampled_idx,
            primary_track_id_0
        ) = self.process_video(video_id, use_consistent_transformation)

        # we will collect bboxes for visualization here
        frame_bbox_meshes: Dict[int, List[Dict[str, Any]]] = {}

        # ----- helper to make a box mesh from 8 world corners -----
        def _make_box_mesh(corners_world: np.ndarray):
            faces = np.array([
                [0, 1, 2], [1, 3, 2],  # min-x side
                [4, 6, 5], [5, 6, 7],  # max-x side
                [0, 4, 1], [1, 4, 5],  # min-y side
                [2, 3, 6], [3, 7, 6],  # max-y side
                [0, 2, 4], [2, 6, 4],  # min-z side
                [1, 5, 3], [3, 5, 7],  # max-z side
            ], dtype=np.uint32)
            return corners_world.astype(np.float32), faces

        # ----- floor transform -----
        has_floor = gv is not None and gf is not None
        s_floor = float(s_avg) if s_avg is not None else 1.0
        R_floor = np.asarray(R_avg, dtype=np.float32) if R_avg is not None else np.eye(3, dtype=np.float32)
        t_floor = np.asarray(t_avg, dtype=np.float32) if t_avg is not None else np.zeros(3, dtype=np.float32)

        # ----- helpers specific to humans -----
        def _get_human_verts_world(world4d: Dict[int, dict], frame_idx: int, track_id: int) -> Optional[np.ndarray]:
            """
            Try a few common keys to get the human mesh vertices for this frame.
            Adjust this if your world4d uses a different key.
            """
            f = world4d.get(frame_idx, None)
            if f is None:
                return None
            # common patterns from your earlier scripts
            if "vertices" in f:
                pos_id_track_id = f["track_id"].index(track_id)
                return np.asarray(f["vertices"][pos_id_track_id], dtype=np.float32)
            else:
                print(f"[bbox][{video_id}][frame {frame_idx}] no 'vertices' key in world4d frame data")
            return None

        def _floor_align_points(points_world: np.ndarray) -> np.ndarray:
            # world -> floor-local
            return ((points_world - t_floor[None, :]) / s_floor) @ R_floor

        def _floor_to_world(points_floor: np.ndarray) -> np.ndarray:
            # floor-local -> world
            return (points_floor @ R_floor.T) * s_floor + t_floor

        def _corners_from_mins_maxs(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
            return np.array([
                [mins[0], mins[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [mins[0], maxs[1], mins[2]],
                [mins[0], maxs[1], maxs[2]],
                [maxs[0], mins[1], mins[2]],
                [maxs[0], mins[1], maxs[2]],
                [maxs[0], maxs[1], mins[2]],
                [maxs[0], maxs[1], maxs[2]],
            ], dtype=np.float32)

        def _corners_from_center_dims(center_floor: np.ndarray, dims: np.ndarray) -> np.ndarray:
            """Make floor-aligned cuboid using center + (dx,dy,dz)."""
            half = 0.5 * dims
            mins = center_floor - half
            maxs = center_floor + half
            return _corners_from_mins_maxs(mins, maxs)

        out_frames: Dict[str, Dict[str, Any]] = {}

        for frame_idx_anno, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]
            stem = Path(frame_name).stem

            if stem not in stem_to_idx:
                continue

            sidx = stem_to_idx[stem]
            pts_hw3 = points_S[sidx]
            colors_hw3 = colors[sidx]
            conf_hw = conf_S[sidx] if conf_S is not None else None

            ann_frame_id_in_sampled = annotated_frame_idx_in_sampled_idx[sidx]
            frame_non_zero_pts = _finite_and_nonzero(pts_hw3)

            # # gdino per-frame predictions
            # gd = video_gdino_predictions.get(frame_name, None)
            # if gd is None:
            #     gd_boxes, gd_labels, gd_scores = [], [], []
            # else:
            #     gd_boxes = [list(map(float, b)) for b in gd["boxes"]]
            #     gd_labels = gd["labels"]
            #     gd_scores = [float(s) for s in gd["scores"]]

            frame_rec = {"objects": []}

            # try to read per-frame human mesh (we'll reuse for all "person" in this frame)
            human_mesh_floor_aabb = None
            human_mesh_dims = None
            human_mesh_volume = None
            human_mesh_available = False
            if has_floor:
                human_verts_world = _get_human_verts_world(world4d, ann_frame_id_in_sampled, primary_track_id_0)
                if human_verts_world is not None and human_verts_world.size > 0:
                    human_verts_floor = _floor_align_points(human_verts_world)
                    hmins = human_verts_floor.min(axis=0)
                    hmaxs = human_verts_floor.max(axis=0)
                    human_mesh_dims = (hmaxs - hmins).astype(np.float32)
                    # very small guard to avoid 0 volume
                    human_mesh_volume = float(np.prod(np.maximum(human_mesh_dims, 1e-4)))
                    human_mesh_floor_aabb = (hmins, hmaxs)
                    human_mesh_available = True
                    print(f"[bbox][{video_id}][{frame_name}] human mesh floor-aabb dims {human_mesh_dims}, volume {human_mesh_volume:.4f}")
            else:
                print(f"[bbox][{video_id}][{frame_name}] no floor mesh available; skipping human mesh bbox")

            # iterate over GT objects in this frame
            for item in frame_items:
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

                # match GDINO
                # chosen_gd_xyxy = _match_gdino_to_gt(
                #     label,
                #     gt_xyxy,
                #     gd_boxes,
                #     gd_labels,
                #     gd_scores,
                #     iou_thr=iou_thr
                # )
                chosen_gd_xyxy = gt_xyxy
                # Resize gt_xyxy to (H, W) from (orig_H, orig_W)
                gt_xyxy = _resize_bbox_to(gt_xyxy, (orig_W, orig_H), (W, H))
                if chosen_gd_xyxy is not None:
                    chosen_gd_xyxy = _resize_bbox_to(chosen_gd_xyxy, (orig_W, orig_H), (W, H))

                frame_label_mask = video_to_frame_to_label_mask[video_id][stem].get(label, None)
                if frame_label_mask is None:
                    print(f"[bbox][{video_id}][{frame_name}] no mask for label '{label}', Creating from bbox")
                    box = chosen_gd_xyxy if chosen_gd_xyxy is not None else gt_xyxy
                    frame_label_mask = _mask_from_bbox(H, W, box)
                else:
                    mask_h, mask_w = frame_label_mask.shape
                    assert  mask_h == orig_H and mask_w == orig_W

                    frame_label_mask = _resize_mask_to(frame_label_mask, (H, W))

                    mask_h, mask_w = frame_label_mask.shape
                    assert  mask_h == H and mask_w == W

                    # Change the mask to include only those that are inside the chosen_gd_xyxy bbox
                    box = chosen_gd_xyxy if chosen_gd_xyxy is not None else gt_xyxy
                    x1, y1, x2, y2 = map(int, box)
                    bbox_mask = np.zeros_like(frame_label_mask, dtype=bool)
                    bbox_mask[y1:y2, x1:x2] = True
                    frame_label_mask = frame_label_mask & bbox_mask

                sel = frame_label_mask
                if sel.sum() > min_points:
                    # Erode the mask to avoid boundary points
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    sel = cv2.erode(frame_label_mask, kernel, iterations=1)

                sel = sel & frame_non_zero_pts
                if conf_hw is not None:
                    sel &= (conf_hw > 1e-3)

                if sel.sum() < min_points:
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "gdino_bbox_xyxy": chosen_gd_xyxy,
                        "num_points": int(sel.sum()),
                        "aabb_floor_aligned": None
                    })
                    continue

                # actual 3D points for this object (world space)
                label_non_zero_pts = pts_hw3[sel].reshape(-1, 3).astype(np.float32)

                if has_floor:
                    # world → floor-local
                    pts_floor = _floor_align_points(label_non_zero_pts)
                    mins = pts_floor.min(axis=0)
                    maxs = pts_floor.max(axis=0)

                    # default corners from point cloud
                    corners_floor = _corners_from_mins_maxs(mins, maxs)
                    corners_world = _floor_to_world(corners_floor)

                    # ---------------- PERSON SPECIAL CASE ----------------
                    if label == "person" and human_mesh_available:
                        # volume from sparse PC box
                        pc_dims = (maxs - mins)
                        pc_volume = float(np.prod(np.maximum(pc_dims, 1e-4)))

                        # allow some slack over mesh volume
                        volume_scale = 1.5
                        use_mesh_like_box = (pc_volume > volume_scale * human_mesh_volume)

                        # center of observed points (in floor coords)
                        pc_center_floor = pts_floor.mean(axis=0)

                        if use_mesh_like_box:
                            print(f"[bbox][{video_id}][{frame_name}] using mesh-shaped box for person "
                                  f"(pc volume {pc_volume:.4f} > {volume_scale} x mesh volume {human_mesh_volume:.4f})")
                            corners_floor = _corners_from_center_dims(pc_center_floor, human_mesh_dims)
                            corners_world = _floor_to_world(corners_floor)
                            frame_rec["objects"].append({
                                "label": label,
                                "gt_bbox_xyxy": gt_xyxy,
                                "gdino_bbox_xyxy": chosen_gd_xyxy,
                                "num_points": int(label_non_zero_pts.shape[0]),
                                "aabb_floor_aligned": {
                                    "mins_floor": (pc_center_floor - 0.5 * human_mesh_dims).tolist(),
                                    "maxs_floor": (pc_center_floor + 0.5 * human_mesh_dims).tolist(),
                                    "corners_world": corners_world.tolist(),
                                    "source": "mesh-shaped-from-volume",
                                    "mesh_volume": human_mesh_volume,
                                    "pc_volume": pc_volume,
                                },
                            })

                            verts_box, faces_box = _make_box_mesh(corners_world)
                            frame_bbox_meshes.setdefault(sidx, []).append({
                                "verts": verts_box,
                                "faces": faces_box,
                                "color": [0, 255, 0],
                                "label": label,
                            })
                            continue

                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "gdino_bbox_xyxy": chosen_gd_xyxy,
                        "num_points": int(label_non_zero_pts.shape[0]),
                        "aabb_floor_aligned": {
                            "mins_floor": mins.tolist(),
                            "maxs_floor": maxs.tolist(),
                            "corners_world": corners_world.tolist(),
                            "source": "pc-aabb",
                        },
                    })

                    verts_box, faces_box = _make_box_mesh(corners_world)
                    frame_bbox_meshes.setdefault(sidx, []).append({
                        "verts": verts_box,
                        "faces": faces_box,
                        "color": [255, 180, 0] if label != "person" else [0, 255, 0],
                        "label": label,
                    })
                else:
                    # fallback: just world AABB
                    mins = label_non_zero_pts.min(axis=0)
                    maxs = label_non_zero_pts.max(axis=0)
                    corners_world = _corners_from_mins_maxs(mins, maxs)
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "gdino_bbox_xyxy": chosen_gd_xyxy,
                        "num_points": int(label_non_zero_pts.shape[0]),
                        "aabb_floor_aligned": {
                            "mins_world": mins.tolist(),
                            "maxs_world": maxs.tolist(),
                            "corners_world": corners_world.tolist(),
                            "source": "world-aabb",
                        },
                    })
                    verts_box, faces_box = _make_box_mesh(corners_world)
                    frame_bbox_meshes.setdefault(sidx, []).append({
                        "verts": verts_box,
                        "faces": faces_box,
                        "color": [255, 0, 0],
                        "label": label,
                    })

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        # --------------- visualization ---------------
        if visualize:
            rerun_vis_world4d(
                video_id=video_id,
                images=images,
                world4d=world4d,
                faces=self.smplx.faces,
                sampled_indices=sampled_frame_indices,
                annotated_frame_idx_in_sample_idx=annotated_frame_idx_in_sampled_idx,
                dynamic_prediction_path=str(self.dynamic_scene_dir_path),
                per_frame_sims=per_frame_sims,
                global_floor_sim=(s_avg, R_avg, t_avg),
                floor=(gv, gf, gc) if gv is not None else None,
                img_maxsize=480,
                app_id="World4D-Combined",
                frame_bbox_meshes=frame_bbox_meshes,
            )

            print("Visualization running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

        # --------------- dump to disk ---------------
        out_path = self.bbox_3d_root_dir / f"{video_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "video_id": video_id,
                "frames": out_frames
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[bbox] saved floor-aligned 3D bboxes to {out_path}")

    def generate_gt_world_bb_annotations(self, dataloader, split) -> None:
        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
                video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
                self.generate_video_bb_annotations(
                    video_id,
                    video_id_gt_annotations,
                    video_id_gdino_annotations,
                    visualize=True
                )

    def generate_sample_gt_world_bb_annotations(self, video_id: str) -> None:
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        self.generate_video_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            visualize=True
        )

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
):
    faces_u32 = _faces_u32(faces)
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    video_dynamic_prediction_path = os.path.join(dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz")
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files}
    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    imgs_f32 = video_dynamic_predictions["images"]
    colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # floor
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

    # edges for a cuboid (8 vertices) — indices match the order we stored earlier
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # bottom rectangle
        (4, 5), (5, 7), (7, 6), (6, 4),  # top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    # ------------------------------------------------------------------
    # use only the annotated frames (indices into sampled_indices)
    # ------------------------------------------------------------------
    # if list is empty, we can fall back to showing all sampled_indices
    if annotated_frame_idx_in_sample_idx:
        iter_indices = annotated_frame_idx_in_sample_idx
    else:
        iter_indices = list(range(len(sampled_indices)))

    for vis_t, sample_idx in enumerate(iter_indices):
        # sample_idx is an index into sampled_indices
        if sample_idx < 0 or sample_idx >= len(sampled_indices):
            continue

        frame_idx = sampled_indices[sample_idx]

        # set timeline to a dense 0..N-1 sequence of annotated frames
        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # floor (constant per frame)
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

        # human meshes (orig is already stored in world4d; we only show transformed)
        track_ids = frame_data.get("track_id", [])
        verts_orig_list = frame_data.get("vertices_orig", [])
        if track_ids and verts_orig_list:
            tid = int(track_ids[0])
            verts_orig = np.asarray(verts_orig_list[0], dtype=np.float32)

            if s_i is not None:
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

        # --- dynamic points: NOTE we index by sample_idx, not vis_t ---
        if sample_idx < points.shape[0]:
            rr.log(
                f"{BASE}/points",
                rr.Points3D(
                    points[sample_idx].reshape(-1, 3),
                    colors=colors[sample_idx].reshape(-1, 3),
                ),
            )

        # --- per-frame cuboid bboxes ---
        if frame_bbox_meshes is not None and vis_t in frame_bbox_meshes:
            for bi, bbox_m in enumerate(frame_bbox_meshes[vis_t]):
                verts_world = bbox_m["verts"].astype(np.float32)  # (8,3)
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
    parser = argparse.ArgumentParser(
        description="Combined: (a) floor-aligned 3D bbox generator + (b) SMPL↔PI3 human mesh aligner (sampled frames only)."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument("--dynamic_scene_dir_path", type=str,
                        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic")
    parser.add_argument("--output_human_dir_path", type=str, default="/data/rohith/ag/ag4D/human/")
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument("--include_dense", action="store_true",
                        help="use dense correspondences for human aligner")
    return parser.parse_args()

def main():
    args = parse_args()

    bbox_3d_generator = BBox3DGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_human_dir_path=args.output_human_dir_path,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    bbox_3d_generator.generate_gt_world_bb_annotations(dataloader=dataloader_train, split=args.split)
    bbox_3d_generator.generate_gt_world_bb_annotations(dataloader=dataloader_test, split=args.split)

def main_sample():
    args = parse_args()

    bbox_3d_generator = BBox3DGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_human_dir_path=args.output_human_dir_path,
    )
    video_id = "L1O0N.mp4"
    bbox_3d_generator.generate_sample_gt_world_bb_annotations(video_id=video_id)


if __name__ == "__main__":
    main_sample()
