#!/usr/bin/env python3
import argparse
import contextlib
import gc
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

from annotation_utils import get_video_belongs_to_split, _load_pkl_if_exists, _npz_open, _torch_inference_ctx
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
    _safe_empty_cuda_cache,
    _torch_inference_ctx,
    _del_and_collect,
    _as_np
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

        self.erosion_kernel_sizes = list(range(0, 11))
        self.min_points_per_scale = 50


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

    def process_video(
            self,
            video_id: str,
            include_dense: bool = False,
            use_consistent_transformation: bool = False,
            return_visualization_payload: bool = True,
    ):
        # 0) run human/scene pipeline
        with _torch_inference_ctx():
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
            print(f"[align] Estimating per-frame similarities for {len(sampled_frame_indices)} sampled frames")
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
                    _del_and_collect(smpl_d, scene_d)

                if len(smpl_all) == 0:
                    print(f"[align][{video_id}] no corr for sampled frame {frame_idx}")
                    _del_and_collect(smpl_s, scene_s)
                    continue

                smpl_all = np.concatenate(smpl_all, axis=0)
                scene_all = np.concatenate(scene_all, axis=0)

                if smpl_all.shape[0] < 3:
                    print(f"[align][{video_id}] insufficient corr for frame {frame_idx}")
                    _del_and_collect(smpl_s, scene_s, smpl_all, scene_all)
                    continue

                # solve per-frame sim
                s_f, R_f, t_f = _robust_similarity_ransac(
                    smpl_all, scene_all,
                    max_iters=500,
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
            print(f"[align] Computing robust average similarity over {len(per_frame_sims)} sampled frames")
            sampled_per_frame_sims = {k: v for k, v in per_frame_sims.items() if k in sampled_frame_indices}

            # If the non-zero/non-empty per-frame sims are less than 10% of the sampled frames raise Exception
            if len(sampled_per_frame_sims) < max(3, 0.2 * len(sampled_frame_indices)):
                raise ValueError(
                    f"Insufficient per-frame similarities computed ({len(sampled_per_frame_sims)}/"
                    f"{len(sampled_frame_indices)}) for video {video_id} to compute robust average."
                )

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
            print(f"[align] Generating transformed verts for {len(sampled_frame_indices)} sampled frames")
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
        print(f"[align] Generating floor mesh from {len(all_verts_for_floor)} frames' verts")
        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            gv, gf, gc = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf, gc = None, None, None

        _del_and_collect(all_verts_for_floor)
        _del_and_collect(verts, verts_tf)
        _del_and_collect(smpl_out, rotmat_all, rotmat_actor)

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

    def _clear_pipeline_state(self):
        # Drop large per-video arrays stored on the AgPipeline instance
        print(f"[cleanup] clearing pipeline state to free memory")
        for attr in [
            "images",
            "points",
            "results",
            "world4d",
            "world4d_dict",
            "kp2d",
            "kp2d_map",
            "frames",
        ]:
            if hasattr(self.pipeline, attr):
                try:
                    setattr(self.pipeline, attr, None)
                except Exception:
                    pass
        print(f"[cleanup] pipeline state cleared, check: {torch.cuda.memory_summary()}")

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
            label_colors: Optional[Dict[str, List[int]]] = None,  # NEW
    ) -> None:
        # load dynamic points (annotated frames)
        try:
            # out_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
            out_path = self.bbox_3d_obb_root_dir / f"{video_id[:-4]}.pkl"
            if out_path.exists():
                print(f"[bbox] floor-aligned 3D bboxes already exist for video {video_id}, skipping...")
                return
            P = self._load_points_for_video(video_id)
            points_S = P["points"]  # (S,H,W,3)
            conf_S = P["conf"]  # (S,H,W) or None
            stems_S = P["frame_stems"]  # ["000123", ...]
            S, H, W, _ = points_S.shape

            # original image size (for resizing bboxes/masks)
            sample_image_frame = self.frame_annotated_dir_path / video_id / f"{stems_S[0]}.png"
            orig_img = cv2.imread(str(sample_image_frame))
            orig_H, orig_W = orig_img.shape[:2]

            stem_to_idx = {stems_S[i]: i for i in range(S)}

            # segmentation maps for the video
            video_to_frame_to_label_mask, _, _ = self.create_label_wise_masks_map(
                video_id=video_id,
                gt_annotations=video_gt_annotations
            )

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

            # 1) build per-frame multiscale boxes
            print(f"[bbox] Building per-frame multiscale 3D bboxes for video {video_id}...")
            out_frames = self._build_multiscale_bboxes_for_video(
                video_id=video_id,
                video_gt_annotations=video_gt_annotations,
                points_S=points_S,
                conf_S=conf_S,
                stems_S=stems_S,
                orig_size=(orig_W, orig_H),
                stem_to_idx=stem_to_idx,
                video_to_frame_to_label_mask=video_to_frame_to_label_mask,
                has_floor=(gv is not None and gf is not None),
                s_avg=s_avg,
                R_avg=R_avg,
                t_avg=t_avg,
            )

            # 2) temporal smoothing (multiscale fuse + forward KF + RTS), return meshes for vis
            print(f"[bbox] Temporal smoothing of 3D bboxes for video {video_id}...")
            frame_bbox_meshes = self._temporal_smooth_bboxes_for_video(
                video_id=video_id,
                out_frames=out_frames,
                stem_to_idx=stem_to_idx,
                s_avg=s_avg,
                R_avg=R_avg,
                t_avg=t_avg,
                gv=gv,
                gf=gf,
                gc=gc,
                label_colors=label_colors,
                enable_temporal_smoothing=False
            )

            # 3) visualize if asked
            if visualize:
                print(f"[bbox] Launching World4D visualization for video {video_id}...")
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
                    vis_floor=False,
                    vis_humans=False
                )

                print("Visualization running. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)

            # 4) save to disk
            print(f"[bbox] Saving floor-aligned 3D bboxes for video {video_id}...")
            results_dictionary = {
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
            with open(out_path, "wb") as f:
                pickle.dump(results_dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"[bbox] saved floor-aligned 3D bboxes (multiscale + fused KF + RTS) to {out_path}")
        finally:
            _del_and_collect(
                results_dictionary if 'results_dictionary' in locals() else None,
                out_frames if 'out_frames' in locals() else None,
                points_S if 'points_S' in locals() else None,
                conf_S if 'conf_S' in locals() else None,
                images if 'images' in locals() else None,
                world4d if 'world4d' in locals() else None,
                gv if 'gv' in locals() else None,
                gf if 'gf' in locals() else None,
                gc if 'gc' in locals() else None,
                video_gdino_predictions if 'video_gdino_predictions' in locals() else None,
                video_gt_annotations if 'video_gt_annotations' in locals() else None,
            )
            if hasattr(self, "pipeline"):
                try:
                    self._clear_pipeline_state()
                except Exception:
                    pass
            print(f"[bbox] Cleared all files")

    def visualize_from_saved_files(
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
        pkl_path = self.bbox_3d_obb_root_dir / f"{video_id[:-4]}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"[vis] Missing saved bbox file: {pkl_path}\n"
                f"Run generation once (generate_video_bb_annotations) before visualization."
            )

        with open(pkl_path, "rb") as f:
            saved: Dict[str, Any] = pickle.load(f)

        per_frame_sims = saved.get("per_frame_sims", None)

        gsim = saved.get("global_floor_sim", None)
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

        frame_bbox_meshes = saved.get("frame_bbox_meshes", None)

        gv = saved.get("gv", None)
        gf = saved.get("gf", None)
        gc = saved.get("gc", None)

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
            (_, sample_idx, _, _, annotated_frame_idx_in_sample_idx) = self.idx_to_frame_idx_path(video_id)
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
            annotated_frame_idx_in_sample_idx = list(range(S_full))

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
        rerun_vis_world4d(
            video_id=video_id,
            images=images,
            world4d=world4d,
            faces=self.smplx.faces,  # already on the layer
            sampled_indices=sampled_indices,
            annotated_frame_idx_in_sample_idx=annotated_frame_idx_in_sample_idx,
            dynamic_prediction_path=str(self.dynamic_scene_dir_path),
            per_frame_sims=per_frame_sims,
            global_floor_sim=global_floor_sim,
            floor=floor,
            img_maxsize=img_maxsize,
            app_id=app_id,
            frame_bbox_meshes=frame_bbox_meshes,
            vis_floor=vis_floor,
            vis_humans=vis_humans,  # typically False unless you also load vertices_orig
            min_conf_default=min_conf_default,
        )


    # ------------------------------------------------------------------
    # 1) MULTISCALE BLOCK (per-frame, per-label, keep all scales)
    # ------------------------------------------------------------------
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

            conf_thr = None
            if conf_hw is not None:
                cfs_flat = conf_hw.reshape(-1)
                mask_valid = np.isfinite(cfs_flat)
                cfs_valid = cfs_flat[mask_valid]
                if cfs_valid.size > 0:
                    med = np.median(cfs_valid)
                    p5 = np.percentile(cfs_valid, 5)
                    # base rule
                    # thr = max(1e-3, 0.5 * med)
                    thr = max(1e-3, p5)
                    conf_thr = float(thr)
                    print(f"[bbox][{video_id}][{stem}] conf thr set to {conf_thr:.4f} (med={med:.4f})")
                else:
                    conf_thr = 0.05  # fallback

            for item in frame_items:
                # resolve label + original bbox
                if "person_bbox" in item:
                    label = "person"
                    gt_xyxy = _xywh_to_xyxy(item["person_bbox"][0])
                else:
                    cid = item["class"]
                    label = self.catid_to_name_map.get(cid, None)
                    if not label:
                        continue
                    # normalize label
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

                # resize bbox to (W,H) of dynamic prediction
                gt_xyxy = _resize_bbox_to(gt_xyxy, (orig_W, orig_H), (W, H))

                # get/resize/intersect mask
                frame_label_mask = video_to_frame_to_label_mask[video_id][stem].get(label, None)
                if frame_label_mask is None:
                    frame_label_mask = _mask_from_bbox(H, W, gt_xyxy)
                else:
                    frame_label_mask = _resize_mask_to(frame_label_mask, (H, W))
                    x1, y1, x2, y2 = map(int, gt_xyxy)
                    bbox_mask = np.zeros_like(frame_label_mask, dtype=bool)
                    bbox_mask[y1:y2, x1:x2] = True
                    frame_label_mask = frame_label_mask & bbox_mask

                # run multiscale erosion 0..10 px
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
                        continue

                    pts_floor = _floor_align_points(obj_pts_world)
                    mins = pts_floor.min(axis=0)
                    maxs = pts_floor.max(axis=0)
                    size = (maxs - mins).clip(1e-6)
                    volume = float(size[0] * size[1] * size[2])

                    corners_floor = np.array([
                        [mins[0], mins[1], mins[2]],
                        [mins[0], mins[1], maxs[2]],
                        [mins[0], maxs[1], mins[2]],
                        [mins[0], maxs[1], maxs[2]],
                        [maxs[0], mins[1], mins[2]],
                        [maxs[0], mins[1], maxs[2]],
                        [maxs[0], maxs[1], mins[2]],
                        [maxs[0], maxs[1], maxs[2]],
                    ], dtype=np.float32)
                    corners_world = _floor_to_world(corners_floor)

                    multi_scale_candidates.append({
                        "kernel_size": int(ksz),
                        "num_points": num_sel,
                        "mins_floor": mins.tolist(),
                        "maxs_floor": maxs.tolist(),
                        "corners_world": corners_world.tolist(),
                        "volume": volume,
                    })

                # if nothing valid, store empty
                if not multi_scale_candidates:
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "aabb_floor_aligned": None,
                        "multi_scale_candidates": [],
                    })
                    continue

                # pick main candidate (your original logic)
                multi_scale_candidates.sort(key=lambda c: c["kernel_size"])
                base_candidate = next((c for c in multi_scale_candidates if c["kernel_size"] == 0), None)
                if base_candidate is not None:
                    base_vol = base_candidate["volume"]
                    chosen = None
                    for c in multi_scale_candidates:
                        if c["kernel_size"] == 0:
                            continue
                        if c["volume"] <= 0.5 * base_vol:
                            chosen = c
                            break
                    if chosen is None:
                        chosen = next(
                            (c for c in multi_scale_candidates if c["kernel_size"] == 3),
                            base_candidate
                        )
                else:
                    chosen = next(
                        (c for c in multi_scale_candidates if c["kernel_size"] == 3),
                        min(multi_scale_candidates, key=lambda c: c["volume"])
                    )

                frame_rec["objects"].append({
                    "label": label,
                    "gt_bbox_xyxy": gt_xyxy,
                    "aabb_floor_aligned": {
                        "mins_floor": chosen["mins_floor"],
                        "maxs_floor": chosen["maxs_floor"],
                        "corners_world": chosen["corners_world"],
                        "source": "pc-aabb-multiscale",
                        "kernel_size": chosen["kernel_size"],
                        "volume": float(chosen["volume"]),
                    },
                    "multi_scale_candidates": multi_scale_candidates,
                })

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        return out_frames

    # ------------------------------------------------------------------
    # 2) TEMPORAL SMOOTHING BLOCK (fuse across scales + KF + RTS)
    # ------------------------------------------------------------------
    def _temporal_smooth_bboxes_for_video(
            self,
            *,
            video_id: str,
            out_frames: Dict[str, Dict[str, Any]],
            stem_to_idx: Dict[str, int],
            s_avg: float,
            R_avg: np.ndarray,
            t_avg: np.ndarray,
            gv: Optional[np.ndarray],
            gf: Optional[np.ndarray],
            gc: Optional[np.ndarray],
            label_colors: Optional[Dict[str, List[int]]] = None,
            enable_temporal_smoothing: bool = True,
    ) -> Dict[int, List[Dict[str, Any]]]:
        # floor transforms (used in both branches)
        s_floor = float(s_avg) if s_avg is not None else 1.0
        R_floor = np.asarray(R_avg, dtype=np.float32) if R_avg is not None else np.eye(3, dtype=np.float32)
        t_floor = np.asarray(t_avg, dtype=np.float32) if t_avg is not None else np.zeros(3, dtype=np.float32)

        def _floor_to_world(points_floor: np.ndarray) -> np.ndarray:
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

        # FAST PATH: no temporal smoothing, just rebuild frame_bbox_meshes
        if not enable_temporal_smoothing:
            frame_bbox_meshes: Dict[int, List[Dict[str, Any]]] = {}
            for frame_name, frame_rec in out_frames.items():
                stem = Path(frame_name).stem
                if stem not in stem_to_idx:
                    continue
                sidx = stem_to_idx[stem]
                for obj in frame_rec["objects"]:
                    aabb = obj.get("aabb_floor_aligned", None)
                    if not aabb:
                        continue
                    corners_world = np.array(aabb["corners_world"], dtype=np.float32)
                    verts_box, faces_box = self._make_box_mesh_from_corners(corners_world)

                    label = obj["label"]
                    if label_colors is not None and label in label_colors:
                        color = label_colors[label]
                    else:
                        color = [0, 255, 0] if label == "person" else [255, 180, 0]

                    frame_bbox_meshes.setdefault(sidx, []).append({
                        "verts": verts_box,
                        "faces": faces_box,
                        "color": color,
                        "label": label,
                    })
            return frame_bbox_meshes

        # ------------------------------------------------------------------
        # TEMPORAL SMOOTHING PATH
        # ------------------------------------------------------------------
        center_delta_thresh = 0.15
        center_consistency_thresh = 0.12

        # make volume gate stricter than before
        # (smaller ratio -> less tolerance to sudden volume jumps)
        volume_consistency_ratio = 1.3

        # label -> frame_order -> objects
        label_to_frameobjs: Dict[str, Dict[int, List[dict]]] = {}
        for frame_name, frame_rec in out_frames.items():
            frame_order = int(Path(frame_name).stem)
            for obj_idx, obj in enumerate(frame_rec["objects"]):
                label = obj["label"]
                if label in ("floor", "doorway"):
                    continue
                label_to_frameobjs.setdefault(label, {}).setdefault(frame_order, []).append({
                    "frame_name": frame_name,
                    "obj_idx": obj_idx,
                    "obj": obj,
                })

        for label, frame_dict in label_to_frameobjs.items():
            last_centers_per_scale: Dict[int, np.ndarray] = {}
            fused_meas = []
            frame_orders = sorted(frame_dict.keys())

            for fo in frame_orders:
                obj_refs = frame_dict[fo]

                # collect all scale candidates (per label, this frame)
                scale_to_entries: Dict[int, List[dict]] = {}
                for ref in obj_refs:
                    obj = ref["obj"]
                    msc = obj.get("multi_scale_candidates", [])
                    for cand in msc:
                        ksz = int(cand["kernel_size"])
                        mins = np.asarray(cand["mins_floor"], dtype=np.float32)
                        maxs = np.asarray(cand["maxs_floor"], dtype=np.float32)
                        center = 0.5 * (mins + maxs)
                        vol = float(cand["volume"])
                        npts = int(cand["num_points"])
                        scale_to_entries.setdefault(ksz, []).append({
                            "center": center,
                            "volume": vol,
                            "mins": mins,
                            "maxs": maxs,
                            "num_points": npts,
                        })

                # aggregate per scale
                scale_meas = {}
                for ksz, entries in scale_to_entries.items():
                    centers = np.stack([e["center"] for e in entries], axis=0)
                    vols = np.array([e["volume"] for e in entries], dtype=np.float32)
                    mins_arr = np.stack([e["mins"] for e in entries], axis=0)
                    maxs_arr = np.stack([e["maxs"] for e in entries], axis=0)
                    num_points = sum(e["num_points"] for e in entries)
                    scale_meas[ksz] = {
                        "center": centers.mean(axis=0),
                        "volume": float(vols.mean()),
                        "mins": mins_arr.mean(axis=0),
                        "maxs": maxs_arr.mean(axis=0),
                        "num_points": num_points,
                    }

                # temporal gating on centers
                valid_scales = []
                for ksz, m in scale_meas.items():
                    c = m["center"]
                    if ksz in last_centers_per_scale:
                        if np.linalg.norm(c - last_centers_per_scale[ksz]) > center_delta_thresh:
                            continue
                    valid_scales.append((ksz, m))
                if not valid_scales:
                    valid_scales = list(scale_meas.items())

                # cross-scale consistency
                centers_all = np.stack([m["center"] for _, m in valid_scales], axis=0)
                vols_all = np.array([m["volume"] for _, m in valid_scales], dtype=np.float32)
                center_med = np.median(centers_all, axis=0)
                vol_med = float(np.median(vols_all))

                inlier_scales = []
                for ksz, m in valid_scales:
                    err = np.linalg.norm(m["center"] - center_med)
                    vol_ratio = max(m["volume"], vol_med) / max(min(m["volume"], vol_med), 1e-6)
                    if err <= center_consistency_thresh and vol_ratio <= volume_consistency_ratio:
                        inlier_scales.append((ksz, m))

                if not inlier_scales:
                    # fall back: pick best-supported scale
                    ksz_best, m_best = max(valid_scales, key=lambda km: km[1]["num_points"])
                    inlier_scales = [(ksz_best, m_best)]

                # fuse across remaining scales
                weights = np.array([m["num_points"] for _, m in inlier_scales], dtype=np.float32)
                weights = weights / (weights.sum() + 1e-6)
                centers_stack = np.stack([m["center"] for _, m in inlier_scales], axis=0)
                vols_stack = np.array([m["volume"] for _, m in inlier_scales], dtype=np.float32)
                mins_stack = np.stack([m["mins"] for _, m in inlier_scales], axis=0)
                maxs_stack = np.stack([m["maxs"] for _, m in inlier_scales], axis=0)

                fused_center = (weights[:, None] * centers_stack).sum(axis=0)
                fused_volume = float((weights * vols_stack).sum())
                fused_mins = (weights[:, None] * mins_stack).sum(axis=0)
                fused_maxs = (weights[:, None] * maxs_stack).sum(axis=0)

                # NEW: clamp fused_volume to be near the median for this frame
                # lets say we allow only 0.6x .. 1.4x of the frame's median volume
                vol_low = 0.6 * vol_med
                vol_high = 1.4 * vol_med
                fused_volume = float(np.clip(fused_volume, vol_low, vol_high))

                fused_meas.append({
                    "frame_order": fo,
                    "obj_refs": obj_refs,
                    "center": fused_center,
                    "volume": fused_volume,
                    "mins": fused_mins,
                    "maxs": fused_maxs,
                })

                # update last centers
                for ksz, m in inlier_scales:
                    last_centers_per_scale[ksz] = m["center"]

            if not fused_meas:
                continue

            fused_meas.sort(key=lambda x: x["frame_order"])

            # KF setup (discourage volume jumps)
            F = np.eye(4, dtype=np.float32)
            H = np.eye(4, dtype=np.float32)

            # process noise: allow motion in center, very low on volume
            Q = np.diag([1e-4, 1e-4, 1e-4, 1e-5]).astype(np.float32)

            # measurement noise: trust centers, be more skeptical about volume
            Rm = np.diag([1e-2, 1e-2, 1e-2, 5e-1]).astype(np.float32)

            x_fwd = []
            P_fwd = []
            x_pred_list = []
            P_pred_list = []

            first = fused_meas[0]
            x = np.array([
                first["center"][0],
                first["center"][1],
                first["center"][2],
                first["volume"]
            ], dtype=np.float32)
            P = np.eye(4, dtype=np.float32)

            for fm in fused_meas:
                z = np.array([
                    fm["center"][0],
                    fm["center"][1],
                    fm["center"][2],
                    fm["volume"]
                ], dtype=np.float32)

                # predict
                x_pred = F @ x
                P_pred = F @ P @ F.T + Q

                # update
                y = z - (H @ x_pred)
                S_mat = H @ P_pred @ H.T + Rm
                K = P_pred @ H.T @ np.linalg.inv(S_mat)

                x = x_pred + K @ y
                P = (np.eye(4, dtype=np.float32) - K @ H) @ P_pred

                x_fwd.append(x.copy())
                P_fwd.append(P.copy())
                x_pred_list.append(x_pred.copy())
                P_pred_list.append(P_pred.copy())

            # RTS smoothing
            Tn = len(fused_meas)
            x_smooth = [None] * Tn
            P_smooth = [None] * Tn
            x_smooth[-1] = x_fwd[-1]
            P_smooth[-1] = P_fwd[-1]

            for k in range(Tn - 2, -1, -1):
                Pk = P_fwd[k]
                Pk_pred_next = P_pred_list[k + 1]
                Ck = Pk @ F.T @ np.linalg.inv(Pk_pred_next)
                x_smooth[k] = x_fwd[k] + Ck @ (x_smooth[k + 1] - x_pred_list[k + 1])
                P_smooth[k] = Pk + Ck @ (P_smooth[k + 1] - Pk_pred_next) @ Ck.T

            # write back smoothed bboxes
            for fm, xs in zip(fused_meas, x_smooth):
                cx, cy, cz, v_smooth = xs.tolist()
                for ref in fm["obj_refs"]:
                    frame_name = ref["frame_name"]
                    obj_idx = ref["obj_idx"]
                    obj = out_frames[frame_name]["objects"][obj_idx]
                    aabb = obj.get("aabb_floor_aligned", None)
                    if aabb is None:
                        continue

                    mins0 = fm["mins"]
                    maxs0 = fm["maxs"]
                    size0 = maxs0 - mins0
                    vol0 = float(size0[0] * size0[1] * size0[2])
                    if vol0 <= 0.0:
                        continue

                    # adjust size by the smoothed volume, but gently
                    scale = (v_smooth / vol0) ** (1.0 / 3.0) if v_smooth > 0 else 1.0
                    new_center = np.array([cx, cy, cz], dtype=np.float32)
                    new_half_size = 0.5 * size0 * scale
                    new_mins = new_center - new_half_size
                    new_maxs = new_center + new_half_size

                    corners_floor = _corners_from_mins_maxs(new_mins, new_maxs)
                    corners_world = _floor_to_world(corners_floor)

                    aabb["mins_floor"] = new_mins.tolist()
                    aabb["maxs_floor"] = new_maxs.tolist()
                    aabb["corners_world"] = corners_world.tolist()
                    aabb["volume"] = float(max(v_smooth, 1e-6))
                    aabb["source"] = aabb.get("source", "pc-aabb-multiscale") + "+kf-rts"

        # rebuild meshes
        frame_bbox_meshes: Dict[int, List[Dict[str, Any]]] = {}
        for frame_name, frame_rec in out_frames.items():
            stem = Path(frame_name).stem
            if stem not in stem_to_idx:
                continue
            sidx = stem_to_idx[stem]
            for obj in frame_rec["objects"]:
                aabb = obj.get("aabb_floor_aligned", None)
                if not aabb:
                    continue
                corners_world = np.array(aabb["corners_world"], dtype=np.float32)
                verts_box, faces_box = self._make_box_mesh_from_corners(corners_world)

                label = obj["label"]
                if label_colors is not None and label in label_colors:
                    color = label_colors[label]
                else:
                    color = [0, 255, 0] if label == "person" else [255, 180, 0]

                frame_bbox_meshes.setdefault(sidx, []).append({
                    "verts": verts_box,
                    "faces": faces_box,
                    "color": color,
                    "label": label,
                })

        return frame_bbox_meshes

    # small helper to keep mesh creation outside
    def _make_box_mesh_from_corners(self, corners_world: np.ndarray):
        faces = np.array([
            [0, 1, 2], [1, 3, 2],
            [4, 6, 5], [5, 6, 7],
            [0, 4, 1], [1, 4, 5],
            [2, 3, 6], [3, 7, 6],
            [0, 2, 4], [2, 6, 4],
            [1, 5, 3], [3, 5, 7],
        ], dtype=np.uint32)
        return corners_world.astype(np.float32), faces

    def generate_gt_world_bb_annotations(self, dataloader, split) -> None:
        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                out_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
                if out_path.exists():
                    print(f"[bbox] floor-aligned 3D bboxes already exist for video {video_id}, skipping...")
                    continue
                video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
                video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
                try:
                    print(f"[bbox] processing video {video_id}...")
                    self.generate_video_bb_annotations(
                        video_id,
                        video_id_gt_annotations,
                        video_id_gdino_annotations,
                        visualize=False
                    )
                except Exception as e:
                    print(f"[bbox] failed to process video {video_id}: {e}")
            else:
                print(f"[bbox] video {video_id} does not belong to split {split}, skipping...")

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
        vis_floor: bool = True,
        vis_humans: bool = True,
        min_conf_default: float = 1e-6,  # floor for conf
):
    faces_u32 = _faces_u32(faces)
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    video_dynamic_prediction_path = os.path.join(dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz")
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files}
    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    conf = video_dynamic_predictions["conf"].astype(np.float32)  # (S,H,W)
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
        if sample_idx < 0 or sample_idx >= len(sampled_indices):
            continue

        frame_idx = sampled_indices[sample_idx]

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

        # --- dynamic points: confidence-filtered ---
        if sample_idx < points.shape[0]:
            pts = points[sample_idx].reshape(-1, 3)  # (N,3)
            cols = colors[sample_idx].reshape(-1, 3)  # (N,3)
            cfs = conf[sample_idx].reshape(-1)  # (N,)

            # adaptive threshold
            good = np.isfinite(cfs)
            cfs_valid = cfs[good]
            if cfs_valid.size > 0:
                med = np.median(cfs_valid)
                p5 = np.percentile(cfs_valid, 5)
                # base threshold from median
                # thr = max(min_conf_default, 0.5 * med)
                # don't let it exceed the 75th percentile (keeps enough points)
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

        # --- per-frame cuboid bboxes ---
        if frame_bbox_meshes is not None and vis_t in frame_bbox_meshes:
            for bi, bbox_m in enumerate(frame_bbox_meshes[vis_t]):
                verts_world = bbox_m["verts"].astype(np.float32)
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
    video_id = "0DJ6R.mp4"
    bbox_3d_generator.generate_sample_gt_world_bb_annotations(video_id=video_id)


if __name__ == "__main__":
    # main_sample()
    main()
