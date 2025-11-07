import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm
import rerun as rr
from scipy.spatial.transform import Rotation as SciRot

# make local imports work like in your original file
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline
from datasets.preprocess.human.data_config import SMPLX_PATH
from datasets.preprocess.human.prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from datasets.preprocess.human.prompt_hmr.vis.traj import get_floor_mesh
from datasets.preprocess.human.pipeline.kp_utils import (
    get_openpose_joint_names,
    get_smpl_joint_names,
)


# ------------------------------------------------------------
# helper funcs for visualization
# ------------------------------------------------------------
def _faces_u32(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces)
    if faces.dtype != np.uint32:
        faces = faces.astype(np.uint32)
    return faces


def _pinhole_from_fov(w: int, h: int, fov_y: float):
    fy = 0.5 * h / np.tan(0.5 * fov_y)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy


# ------------------------------------------------------------
# updated visualization
# ------------------------------------------------------------
def rerun_vis_world4d(
        video_id: str,
        images: List[Optional[np.ndarray]],
        world4d: Dict[int, dict],
        results: dict,
        pipeline: AgPipeline,
        faces: np.ndarray,
        init_fps: float = 25.0,
        floor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        img_maxsize: int = 320,
        app_id: str = "World4D",
        *,
        image_frame_map: Optional[Dict[int, int]] = None,
        image_fn: Optional[Callable[[int, Optional[np.ndarray]], Optional[np.ndarray]]] = None,
        reuse_paths: bool = True,
        dynamic_prediction_path: Optional[str] = None,
        frame_kp_corr: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        per_frame_sims: Optional[Dict[int, Dict[str, Any]]] = None,
):
    """
    Visualization now supports per-frame similarity transforms.
    For each frame:
      - take world4d[i]['vertices_orig'] (list per person)
      - if per_frame_sims[i] exists -> apply that (s,R,t) and show in green
      - always show original in red
    """
    faces_u32 = _faces_u32(faces)

    rr.init(app_id, spawn=True)
    try:
        rr.log("/", rr.ViewCoordinates.RUB)
    except Exception:
        pass

    rr.set_time_seconds("frame_fps", 1.0 / max(1e-6, float(init_fps)))
    num_frames = len(world4d)

    # load dynamic PI3 predictions
    video_dynamic_prediction_path = os.path.join(dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz")
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files}
    print(f"[rerun] Loaded dynamic predictions from {video_dynamic_prediction_path}")

    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    imgs_f32 = video_dynamic_predictions["images"]  # float32 [0,1]
    colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    def _get_image_for_time(i: int) -> Optional[np.ndarray]:
        src_idx = i
        if image_frame_map and i in image_frame_map:
            src_idx = image_frame_map[i]
        base_img = None
        if images is not None and 0 <= src_idx < len(images):
            base_img = images[src_idx]
        if image_fn is not None:
            base_img = image_fn(i, base_img)
        if base_img is None:
            return None
        img = base_img
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    def _human_path(tid: int, i: int, kind: str) -> str:
        base = f"{BASE}/humans_{kind}/h{tid}"
        if reuse_paths:
            return base
        return f"{BASE}/frames/t{i}/{kind}_human_{tid}"

    def _frustum_path(i: int) -> str:
        return f"{BASE}/frustum" if reuse_paths else f"{BASE}/frames/t{i}/frustum"

    for i in range(num_frames):
        rr.set_time_sequence("frame", i)
        rr.log("/", rr.Clear(recursive=True))

        # humans (original + transformed)
        track_ids = world4d[i].get("track_id", [])
        verts_list_orig = world4d[i].get("vertices_orig", [])

        # if we have per-frame s,R,t for this frame, use it
        sft = None
        if per_frame_sims is not None and i in per_frame_sims:
            sft = per_frame_sims[i]
            s_i = float(sft["s"])
            R_i = np.asarray(sft["R"], dtype=np.float32)
            t_i = np.asarray(sft["t"], dtype=np.float32)
        else:
            s_i, R_i, t_i = None, None, None

        if len(track_ids) > 0 and len(verts_list_orig) > 0:
            for idx, tid in enumerate(track_ids):
                # original in red
                if idx < len(verts_list_orig):
                    verts_orig = np.asarray(verts_list_orig[idx], dtype=np.float32)
                    # rr.log(
                    #     _human_path(int(tid), i, "orig"),
                    #     rr.Mesh3D(
                    #         vertex_positions=verts_orig,
                    #         triangle_indices=faces_u32,
                    #         albedo_factor=[255, 0, 0],
                    #     ),
                    # )

                    # transformed in green (apply per-frame sim)
                    if s_i is not None:
                        verts_flat = verts_orig.reshape(-1, 3)
                        verts_tf = s_i * (verts_flat @ R_i.T) + t_i
                        verts_tf = verts_tf.reshape(verts_orig.shape)
                        rr.log(
                            _human_path(int(tid), i, "xform"),
                            rr.Mesh3D(
                                vertex_positions=verts_tf.astype(np.float32),
                                triangle_indices=faces_u32,
                                albedo_factor=[0, 255, 0],
                            ),
                        )
                # else: nothing to render for that idx

        # # floor
        # if floor is not None:
        #     fv, ff = floor
        #     fv = np.asarray(fv, dtype=np.float32)
        #     ff = _faces_u32(np.asarray(ff))
        #     rr.log(
        #         f"{BASE}/floor",
        #         rr.Mesh3D(vertex_positions=fv, triangle_indices=ff),
        #     )

        # dynamic points
        rr.log(
            f"{BASE}/points",
            rr.Points3D(
                points[i].reshape(-1, 3),
                colors=colors[i].reshape(-1, 3),
            ),
        )

        # # sparse correspondences
        # if frame_kp_corr is not None and i in frame_kp_corr:
        #     smpl_pts, scene_pts = frame_kp_corr[i]
        #     smpl_pts = np.asarray(smpl_pts, dtype=np.float32)
        #     scene_pts = np.asarray(scene_pts, dtype=np.float32)
        #
        #     rr.log(
        #         f"{BASE}/corr/frame_{i}/smpl_kps",
        #         rr.Points3D(
        #             positions=smpl_pts,
        #             colors=np.full((smpl_pts.shape[0], 3), [255, 0, 0], dtype=np.uint8),
        #             radii=0.015,
        #         ),
        #     )
        #     rr.log(
        #         f"{BASE}/corr/frame_{i}/scene_kps",
        #         rr.Points3D(
        #             positions=scene_pts,
        #             colors=np.full((scene_pts.shape[0], 3), [0, 255, 0], dtype=np.uint8),
        #             radii=0.017,
        #         ),
        #     )
        #
        #     if smpl_pts.shape[0] == scene_pts.shape[0]:
        #         rr.log(
        #             f"{BASE}/corr/frame_{i}/arrows",
        #             rr.Arrows3D(
        #                 origins=smpl_pts,
        #                 vectors=(scene_pts - smpl_pts),
        #                 colors=np.full((smpl_pts.shape[0], 3), [0, 0, 255], dtype=np.uint8),
        #             ),
        #         )

        # camera
        cam_3x4 = np.asarray(world4d[i]["camera"], dtype=np.float32)
        R_wc = cam_3x4[:3, :3]
        t_wc = cam_3x4[:3, 3]

        image = _get_image_for_time(i)
        if image is not None:
            H, W = image.shape[:2]
            aspect = W / float(H)
        else:
            aspect = 16.0 / 9.0
            H, W = img_maxsize, int(img_maxsize * aspect)

        fov_y = 0.96
        fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)
        quat_xyzw = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)

        frus_path = _frustum_path(i)
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=t_wc.astype(np.float32),
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            )
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W, H)),
        )
        if image is not None:
            rr.log(f"{frus_path}/image", rr.Image(image))

        axes_len = 0.3
        rr.log(
            f"{frus_path}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.asarray(
                    [[axes_len, 0, 0], [0, axes_len, 0], [0, 0, axes_len]],
                    dtype=np.float32,
                ),
                colors=np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            ),
        )

    print("Rerun visualization started. Scrub the 'frame' timeline to compare original (red) vs per-frame transformed (green).")


# ------------------------------------------------------------
# split logic
# ------------------------------------------------------------
def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"
    return None


# ------------------------------------------------------------
# main alignment class
# ------------------------------------------------------------
class AlignHMRPi3:

    def __init__(
            self,
            output_root,
            ag_root_directory,
            dynamic_scene_dir_path,
    ):
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)

        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = self.ag_root_directory / "sampled_frames_idx"
        self.videos_directory = self.ag_root_directory / "videos"

        # segmentation dirs
        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"

        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"

        self.pipeline = AgPipeline(static_cam=False, dynamic_scene_dir_path=self.dynamic_scene_dir_path)
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    # --------------------------------------------------------
    # similarity estimators
    # --------------------------------------------------------
    def _similarity_umeyama(self, src: np.ndarray, dst: np.ndarray):
        """
        Standard Umeyama for similarity. Assumes src, dst: (N, 3)
        """
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        assert src.shape == dst.shape
        n = src.shape[0]

        mu_src = src.mean(axis=0)
        mu_dst = dst.mean(axis=0)

        src_c = src - mu_src
        dst_c = dst - mu_dst

        cov = (dst_c.T @ src_c) / n
        U, S, Vt = np.linalg.svd(cov)
        R = U @ Vt
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = U @ Vt

        var_src = (src_c ** 2).sum() / n
        s = (S * np.array([1, 1, np.linalg.det(U @ Vt)])).sum() / var_src
        t = mu_dst - s * (R @ mu_src)
        return s, R, t

    def _robust_similarity_ransac(
            self,
            src: np.ndarray,
            dst: np.ndarray,
            *,
            max_iters: int = 800,
            inlier_thresh: float = 0.03,
            min_inliers: int = 4,
            scale_bounds: Tuple[float, float] = (0.4, 3.0),
    ):
        """
        Robust similarity using RANSAC over Umeyama.
        src, dst: (N, 3)
        Returns: s, R, t
        """
        src = np.asarray(src, dtype=np.float64)
        dst = np.asarray(dst, dtype=np.float64)
        assert src.shape == dst.shape
        N = src.shape[0]

        if N < 3:
            # not enough to RANSAC, fallback
            return self._similarity_umeyama(src, dst)

        best_num = -1
        best_model = None

        # adaptive max iters based on N
        iters = min(max_iters, 100 + 30 * N)

        for _ in range(iters):
            idx = np.random.choice(N, 3, replace=False)
            s_cand, R_cand, t_cand = self._similarity_umeyama(src[idx], dst[idx])

            s_cand = float(np.clip(s_cand, scale_bounds[0], scale_bounds[1]))

            src_tf = s_cand * (src @ R_cand.T) + t_cand
            err = np.linalg.norm(dst - src_tf, axis=1)

            inliers = err < inlier_thresh
            num_inl = int(inliers.sum())

            if num_inl > best_num and num_inl >= min_inliers:
                s_ref, R_ref, t_ref = self._similarity_umeyama(src[inliers], dst[inliers])
                s_ref = float(np.clip(s_ref, scale_bounds[0], scale_bounds[1]))
                best_num = num_inl
                best_model = (s_ref, R_ref, t_ref)

        if best_model is None:
            return self._similarity_umeyama(src, dst)

        return best_model

    # --------------------------------------------------------
    # lifting, masks, etc.
    # --------------------------------------------------------
    def _lift_2d_to_3d(self, frame_points_hw3: np.ndarray, u: float, v: float):
        H, W, _ = frame_points_hw3.shape
        ui = int(round(u))
        vi = int(round(v))
        if ui < 0 or ui >= W or vi < 0 or vi >= H:
            return None
        p3d = frame_points_hw3[vi, ui]
        if not np.isfinite(p3d).all() or np.abs(p3d).sum() < 1e-6:
            return None
        return p3d

    def _build_frame_to_kps_map(self, results: dict):
        frame_to_kps = {}
        people_results = results.get("people", {})

        for _, pdata in people_results.items():
            frames = pdata.get("frames", None)
            if frames is None:
                continue

            kp_maps = pdata.get("keypoints_2d_map", None)

            for i, fidx in enumerate(frames):
                if kp_maps is not None:
                    kp_this = kp_maps[i]
                else:
                    continue

                frame_to_kps.setdefault(fidx, []).append(kp_this)

        return frame_to_kps

    def _get_partial_pointcloud(self, video_id: str, frame_idx: int, frame_idx_frame_path_map,
                                label: str = "person") -> np.ndarray:
        pts_hw3 = self.pipeline.points[frame_idx]  # (H, W, 3)
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

    def get_union_mask(self, video_id: str, stem: str, label: str, is_static: bool):
        if is_static:
            return None
        else:
            im_p = self.dynamic_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.dynamic_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
        m_im = cv2.imread(str(im_p), cv2.IMREAD_GRAYSCALE) if im_p.exists() else None
        m_vd = cv2.imread(str(vd_p), cv2.IMREAD_GRAYSCALE) if vd_p.exists() else None
        if m_im is None and m_vd is None:
            return None
        if m_im is None:
            m = (m_vd > 127)
        elif m_vd is None:
            m = (m_im > 127)
        else:
            m = (m_im > 127) | (m_vd > 127)
        return m.astype(bool)

    def idx_to_frame_idx_path(self, video_id: str):
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)

        annotated_frame_id_list = os.listdir(video_frames_annotated_dir_path)
        annotated_frame_id_list = [f for f in annotated_frame_id_list if f.endswith('.png')]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))

        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        chosen_frames = [video_sampled_frame_id_list[i] for i in sample_idx]
        frame_idx_frame_path_map = {i: f"{frame_id:06d}.png" for i, frame_id in enumerate(chosen_frames)}
        return frame_idx_frame_path_map

    def _collect_kp_corr_for_frame(self, frame_idx: int, frame_data: dict, frame_to_kps: dict):
        if frame_idx not in frame_to_kps:
            return None, None
        if len(frame_data['track_id']) == 0:
            return None, None

        rotmat = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        smpl_out = self.smplx(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=frame_data['shape'].cuda(),
            transl=frame_data['trans'].cuda()
        )
        joints = smpl_out.joints.cpu().numpy()  # (P, J, 3)
        joints = joints[:, :24, :]

        frame_points_hw3 = self.pipeline.points[frame_idx]

        smpl_list = []
        scene_list = []

        kps_per_person = frame_to_kps[frame_idx]

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

        for person_idx, kps_2d in enumerate(kps_per_person):
            if person_idx >= joints.shape[0]:
                break

            smpl_joints_person = joints[person_idx]

            if smpl_joint_names and len(smpl_joint_names) >= smpl_joints_person.shape[0]:
                smpl_name_to_pt = {
                    smpl_joint_names[j]: smpl_joints_person[j]
                    for j in range(smpl_joints_person.shape[0])
                }
            else:
                raise ValueError("SMPL joint names list is missing or too short.")

            for op_name, kp in kps_2d.items():
                smpl_name = OPENPOSE_TO_SMPL.get(op_name, None)
                if smpl_name is None:
                    continue

                if smpl_name not in smpl_name_to_pt:
                    continue

                u = float(kp[0])
                v = float(kp[1])
                scene_p = self._lift_2d_to_3d(frame_points_hw3, u, v)
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
            num_smpl_samples_per_person: int = 400,
            num_scene_subsample: int = 800,
    ):
        if len(frame_data['track_id']) == 0:
            return None, None

        rotmat = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        smpl_out = self.smplx(
            global_orient=rotmat[:, :1].cuda(),
            body_pose=rotmat[:, 1:22].cuda(),
            betas=frame_data['shape'].cuda(),
            transl=frame_data['trans'].cuda()
        )
        verts = smpl_out.vertices.cpu().numpy()  # (P,V,3)

        scene_pts = self._get_partial_pointcloud(
            video_id,
            frame_idx,
            frame_idx_frame_path_map,
            label="person",
        )
        if scene_pts.shape[0] == 0:
            return None, None

        if scene_pts.shape[0] > num_scene_subsample:
            choice = np.random.choice(scene_pts.shape[0], num_scene_subsample, replace=False)
            scene_pts = scene_pts[choice]

        smpl_all = []
        scene_all = []

        for p in range(verts.shape[0]):
            smpl_verts = verts[p]
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

            smpl_all.append(smpl_sampled)
            scene_all.append(matched_scene)

        if len(smpl_all) == 0:
            return None, None

        return np.concatenate(smpl_all, axis=0), np.concatenate(scene_all, axis=0)

    def process_video(self, video_id: str, include_dense: bool = False):
        # run pipeline
        self.pipeline.__call__(video_id, save_only_essential=False)
        self.pipeline.estimate_2d_keypoints()
        results = self.pipeline.results  # pipeline stores into self.results

        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        frame_idx_frame_path_map = self.idx_to_frame_idx_path(video_id)
        frame_to_kps = self._build_frame_to_kps_map(results)

        max_frames = 60

        frame_kp_corr: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        per_frame_sims: Dict[int, Dict[str, Any]] = {}

        # ---------- 1) per-frame robust sim (NO averaging) ----------
        for frame_idx, frame_data in list(world4d.items())[:max_frames]:
            smpl_s, scene_s = self._collect_kp_corr_for_frame(frame_idx, frame_data, frame_to_kps)

            if smpl_s is not None and scene_s is not None and smpl_s.shape[0] > 0:
                frame_kp_corr[frame_idx] = (smpl_s, scene_s)

            smpl_all = []
            scene_all = []
            if smpl_s is not None:
                smpl_all.append(smpl_s)
                scene_all.append(scene_s)

            if include_dense:
                smpl_d, scene_d = self._collect_dense_corr_for_frame(
                    video_id, frame_idx, frame_data, frame_idx_frame_path_map
                )
                if smpl_d is not None and scene_d is not None:
                    smpl_all.append(smpl_d)
                    scene_all.append(scene_d)

            if len(smpl_all) == 0:
                continue

            smpl_all = np.concatenate(smpl_all, axis=0)
            scene_all = np.concatenate(scene_all, axis=0)
            if smpl_all.shape[0] < 3:
                continue

            s_f, R_f, t_f = self._robust_similarity_ransac(
                smpl_all,
                scene_all,
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

        # ---------- 2) build final verts per frame using per-frame sim ----------
        all_verts_for_floor = []
        for frame_idx, frame_data in world4d.items():
            if len(frame_data['track_id']) == 0:
                continue

            rotmat = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
            smpl_out = self.smplx(
                global_orient=rotmat[:, :1].cuda(),
                body_pose=rotmat[:, 1:22].cuda(),
                betas=frame_data['shape'].cuda(),
                transl=frame_data['trans'].cuda()
            )
            verts = smpl_out.vertices.cpu().numpy()  # (P, V, 3)
            frame_data['vertices_orig'] = verts.copy()

            # if we have per-frame sim, apply it now to collect for floor
            if frame_idx in per_frame_sims:
                s_f = per_frame_sims[frame_idx]["s"]
                R_f = per_frame_sims[frame_idx]["R"]
                t_f = per_frame_sims[frame_idx]["t"]
                verts_flat = verts.reshape(-1, 3)
                verts_tf = s_f * (verts_flat @ R_f.T) + t_f
                verts_tf = verts_tf.reshape(verts.shape)
            else:
                verts_tf = verts

            # store transformed too (not strictly needed by vis now, but handy)
            frame_data['vertices'] = verts_tf

            all_verts_for_floor.append(torch.tensor(verts_tf, dtype=torch.bfloat16))

        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            [gv, gf, gc] = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf = None, None

        # ---------- 3) visualize (now with per-frame sims) ----------
        rerun_vis_world4d(
            video_id=video_id,
            images=images,
            world4d=world4d,
            results=results,
            pipeline=self.pipeline,
            faces=self.smplx.faces,
            floor=(gv, gf) if gv is not None else None,
            init_fps=10,
            img_maxsize=480,
            dynamic_prediction_path=str(self.dynamic_scene_dir_path),
            frame_kp_corr=frame_kp_corr,
            per_frame_sims=per_frame_sims,
        )

        print('Rerun visualization running. Press Ctrl+C to terminate.')
        while True:
            time.sleep(1)

    def infer_all_videos(self, split: str):
        video_id_list = ["0DJ6R.mp4"]
        for video_id in tqdm(video_id_list, desc=f"Processing videos in split {split}", unit="video"):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            self.process_video(video_id)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Align SMPL to PI3 dynamic scene and visualize with rerun (per-frame sim).")
    parser.add_argument(
        "--output_dir_path", type=str, default="/data/rohith/ag/ag4D/human/",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    parser.add_argument(
        "--ag_root_directory", type=str, default="/data/rohith/ag/",
        help="Path to directory containing input videos."
    )
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument(
        "--split", default="04",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}."
    )
    parser.add_argument(
        "--include_dense", action="store_true",
        help="Whether to add dense point correspondences per frame for more stable per-frame similarity."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AlignHMRPi3(
        output_root=args.output_dir_path,
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path
    )
    processor.infer_all_videos(split=args.split)


if __name__ == '__main__':
    main()
