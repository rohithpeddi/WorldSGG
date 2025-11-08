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
# visualization
# ------------------------------------------------------------
def rerun_vis_world4d(
        video_id: str,
        images: List[Optional[np.ndarray]],
        world4d: Dict[int, dict],
        results: dict,
        pipeline: AgPipeline,
        faces: np.ndarray,
        init_fps: float = 25.0,
        floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
        img_maxsize: int = 320,
        app_id: str = "World4D",
        *,
        image_frame_map: Optional[Dict[int, int]] = None,
        image_fn: Optional[Callable[[int, Optional[np.ndarray]], Optional[np.ndarray]]] = None,
        reuse_paths: bool = True,
        dynamic_prediction_path: Optional[str] = None,
        frame_kp_corr: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        per_frame_sims: Optional[Dict[int, Dict[str, Any]]] = None,
        global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
):
    """
    Visualize:
      - dynamic point cloud from PI3
      - per-frame transformed SMPL (green) for a single actor
      - fixed floor mesh transformed by an averaged similarity
      - camera frustum + image

    Floor is kept fixed across frames using the averaged similarity.
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
    if dynamic_prediction_path is None:
        raise ValueError("dynamic_prediction_path is required for visualization.")
    video_dynamic_prediction_path = os.path.join(dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz")
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files}
    print(f"[rerun] Loaded dynamic predictions from {video_dynamic_prediction_path}")

    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    imgs_f32 = video_dynamic_predictions["images"]  # float32 [0,1]
    colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # precompute floor transformed by global sim
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
        # clear current frame
        rr.log("/", rr.Clear(recursive=True))

        # log fixed floor every frame (so clear won't remove it)
        if floor_vertices_tf is not None and floor_faces is not None:
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_tf.astype(np.float32),
                    triangle_indices=floor_faces,
                    **(floor_kwargs or {}),
                ),
            )

        # per-frame sim for humans
        if per_frame_sims is not None and i in per_frame_sims:
            s_i = float(per_frame_sims[i]["s"])
            R_i = np.asarray(per_frame_sims[i]["R"], dtype=np.float32)
            t_i = np.asarray(per_frame_sims[i]["t"], dtype=np.float32)
        else:
            s_i, R_i, t_i = None, None, None

        # humans (now only 1 actor per frame)
        track_ids = world4d[i].get("track_id", [])
        verts_list_orig = world4d[i].get("vertices_orig", [])
        if len(track_ids) > 0 and len(verts_list_orig) > 0:
            # there should be exactly 1 now
            tid = int(track_ids[0])
            verts_orig = np.asarray(verts_list_orig[0], dtype=np.float32)
            if s_i is not None:
                verts_flat = verts_orig.reshape(-1, 3)
                verts_tf = s_i * (verts_flat @ R_i.T) + t_i
                verts_tf = verts_tf.reshape(verts_orig.shape)
                rr.log(
                    _human_path(tid, i, "xform"),
                    rr.Mesh3D(
                        vertex_positions=verts_tf.astype(np.float32),
                        triangle_indices=faces_u32,
                        albedo_factor=[0, 255, 0],
                    ),
                )
        else:
            print(f"[{video_id}] Track id not found for frame {i}, skipping human visualization.")


        # dynamic points
        rr.log(
            f"{BASE}/points",
            rr.Points3D(
                points[i].reshape(-1, 3),
                colors=colors[i].reshape(-1, 3),
            ),
        )

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

    print("Rerun visualization started. Scrub the 'frame' timeline to compare per-frame transformed humans vs fixed floor.")


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
    # utility for single actor
    # --------------------------------------------------------
    def _choose_primary_actor(self, results: dict, world4d: Dict[int, dict]) -> Optional[str]:
        """
        Try to pick one stable actor:
          1. prefer the person in results['people'] with the most frames
          2. otherwise fall back to the first track_id found in world4d
        Returns actor id as string (since results keys are often strings).
        """
        people = results.get("people", {})
        if people:
            counts = {}
            for pid, pdata in people.items():
                frames = pdata.get("frames", [])
                counts[pid] = len(frames)
            primary = max(counts.items(), key=lambda x: x[1])[0]
            return primary

        # fallback
        for _, frame_data in world4d.items():
            tids = frame_data.get("track_id", [])
            if tids:
                return str(tids[0])
        return None

    def _find_actor_index_in_frame(self, frame_data: dict, primary_actor_id: Optional[str]) -> Optional[int]:
        """
        Given frame_data and the chosen actor id, return the index into
        frame_data['pose']/['shape']/['trans'] for that actor.
        """
        track_ids = frame_data.get("track_id", [])
        if len(track_ids) == 0:
            print("No track ids in frame data.")
            return None

        if primary_actor_id is None:
            print(f"No primary actor id specified, defaulting to first person.")
            return 0  # fallback: first person

        # normalize to string for comparison
        # Here track_ids is a tensor
        for idx, tid in enumerate(track_ids):
            tid_str = str(tid.item()) if isinstance(tid, torch.Tensor) else str(tid)
            if tid_str == str(primary_actor_id):
                return idx

        # actor not present in this frame
        print("Primary actor id not present in this frame.")
        return None

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

    def _average_sims(self, per_frame_sims: Dict[int, Dict[str, Any]]):
        """
        per_frame_sims[i] = {"s": float, "R": (3,3), "t": (3,), "w": weight}
        We compute a weighted avg scale, weighted avg translation, and
        a weighted avg rotation via quaternion averaging.
        """
        if len(per_frame_sims) == 0:
            return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

        ws = []
        scales = []
        trans = []
        quats = []

        for _, d in per_frame_sims.items():
            w = float(d.get("w", 1.0))
            s = float(d["s"])
            R = np.asarray(d["R"], dtype=np.float64)
            t = np.asarray(d["t"], dtype=np.float64)

            quat = SciRot.from_matrix(R).as_quat()  # (4,)

            ws.append(w)
            scales.append(s)
            trans.append(t)
            quats.append(quat)

        ws = np.asarray(ws, dtype=np.float64)
        ws = ws / (ws.sum() + 1e-8)

        scales = np.asarray(scales, dtype=np.float64)
        trans = np.asarray(trans, dtype=np.float64)
        s_avg = float((ws * scales).sum())
        t_avg = (ws[:, None] * trans).sum(axis=0)

        quats = np.asarray(quats, dtype=np.float64)  # (N,4)
        q_avg = (ws[:, None] * quats).sum(axis=0)
        q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-8)
        R_avg = SciRot.from_quat(q_avg).as_matrix().astype(np.float32)

        return s_avg, R_avg, t_avg.astype(np.float32)

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

    def _build_frame_to_kps_map(self, results: dict, primary_actor_id: Optional[str]):
        """
        Build {frame_idx -> keypoints_dict} but ONLY for the chosen actor.
        Assumes results['people'][actor]['keypoints_2d_map'] exists.
        """
        frame_to_kps = {}
        people_results = results.get("people", {})
        if not people_results:
            return frame_to_kps

        if primary_actor_id is None:
            # fallback to any person
            primary_actor_id = list(people_results.keys())[0]

        pdata = people_results.get(primary_actor_id, None)
        if pdata is None:
            return frame_to_kps

        frames = pdata.get("frames", None)
        kp_maps = pdata.get("keypoints_2d_map", None)

        if frames is None or kp_maps is None:
            return frame_to_kps

        for i, fidx in enumerate(frames):
            frame_to_kps[fidx] = kp_maps[i]

        return frame_to_kps

    def _get_partial_pointcloud(self, video_id: str, frame_idx: int, frame_idx_frame_path_map,
                                label: str = "person") -> np.ndarray:
        pts_hw3 = self.pipeline.points[frame_idx]  # (H, W, 3)
        H, W, _ = pts_hw3.shape

        stem = frame_idx_frame_path_map[frame_idx][:-4]
        # we still use union person mask; per-track instance mask is not available here
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

    def _collect_kp_corr_for_frame(self, frame_idx: int, frame_data: dict, frame_to_kps: dict,
                                   primary_actor_id: Optional[str]):
        """
        Collect sparse 2D->3D correspondences for ONLY the chosen actor.
        """
        if frame_idx not in frame_to_kps:
            print(f"[Frame {frame_idx}] No 2D keypoints for primary actor {primary_actor_id}")
            return None, None

        actor_idx = self._find_actor_index_in_frame(frame_data, primary_actor_id)
        if actor_idx is None:
            print(f"[Frame {frame_idx}] Primary actor {primary_actor_id} not present in frame.")
            return None, None

        kps_2d = frame_to_kps[frame_idx]

        # build smpl joints for that actor only
        rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]

        smpl_out = self.smplx(
            global_orient=rotmat_actor[:, :1].cuda(),
            body_pose=rotmat_actor[:, 1:22].cuda(),
            betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
            transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
        )
        joints = smpl_out.joints.cpu().numpy()  # (1, J, 3)
        joints = joints[:, :24, :]
        smpl_joints_actor = joints[0]

        frame_points_hw3 = self.pipeline.points[frame_idx]

        smpl_list = []
        scene_list = []

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

        if smpl_joint_names and len(smpl_joint_names) >= smpl_joints_actor.shape[0]:
            smpl_name_to_pt = {
                smpl_joint_names[j]: smpl_joints_actor[j]
                for j in range(smpl_joints_actor.shape[0])
            }
        else:
            raise ValueError("SMPL joint names list is missing or too short.")

        # kps_2d is assumed to be a dict: {op_name: (u,v,...) }
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
            primary_actor_id: Optional[str],
            num_smpl_samples_per_person: int = 400,
            num_scene_subsample: int = 800,
    ):
        """
        Dense-ish correspondences but for a single actor.
        """
        actor_idx = self._find_actor_index_in_frame(frame_data, primary_actor_id)
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
        verts = smpl_out.vertices.cpu().numpy()  # (1,V,3)
        smpl_verts = verts[0]

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

        # subsample smpl verts
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

    def process_video(self, video_id: str, include_dense: bool = False):
        # run pipeline
        self.pipeline.__call__(video_id, save_only_essential=False)
        self.pipeline.estimate_2d_keypoints()
        results = self.pipeline.results  # pipeline stores into self.results

        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        # pick one actor for the whole video
        primary_actor_id = self._choose_primary_actor(results, world4d)
        print(f"[align] Using primary actor: {primary_actor_id}")

        frame_idx_frame_path_map = self.idx_to_frame_idx_path(video_id)
        frame_to_kps = self._build_frame_to_kps_map(results, primary_actor_id)

        frame_kp_corr: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        per_frame_sims: Dict[int, Dict[str, Any]] = {}

        # ---------- 1) per-frame robust sim ----------
        for frame_idx, frame_data in list(world4d.items()):
            smpl_s, scene_s = self._collect_kp_corr_for_frame(
                frame_idx, frame_data, frame_to_kps, primary_actor_id
            )

            if smpl_s is not None and scene_s is not None and smpl_s.shape[0] > 0:
                frame_kp_corr[frame_idx] = (smpl_s, scene_s)
            else:
                print(f"[{video_id}][align] Skipping frame {frame_idx} as no sparse kp corr found.")

            smpl_all = []
            scene_all = []
            if smpl_s is not None:
                smpl_all.append(smpl_s)
                scene_all.append(scene_s)

            if include_dense:
                smpl_d, scene_d = self._collect_dense_corr_for_frame(
                    video_id, frame_idx, frame_data, frame_idx_frame_path_map, primary_actor_id
                )
                if smpl_d is not None and scene_d is not None:
                    smpl_all.append(smpl_d)
                    scene_all.append(scene_d)

            if len(smpl_all) == 0:
                continue

            smpl_all = np.concatenate(smpl_all, axis=0)
            scene_all = np.concatenate(scene_all, axis=0)
            if smpl_all.shape[0] < 3:
                print(f"[{video_id}][align] Skipping frame {frame_idx} as insufficient total corr ({smpl_all.shape[0]} pts).")
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

        # ---------- 2) build verts per frame + collect untransformed verts for floor ----------
        all_verts_for_floor = []
        for frame_idx, frame_data in world4d.items():
            actor_idx = self._find_actor_index_in_frame(frame_data, primary_actor_id)
            if actor_idx is None:
                # ensure it's empty so visualization won't try to draw random actors
                frame_data['track_id'] = []
                print("[align] Primary actor not present in frame", frame_idx)
                continue

            rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
            rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]

            smpl_out = self.smplx(
                global_orient=rotmat_actor[:, :1].cuda(),
                body_pose=rotmat_actor[:, 1:22].cuda(),
                betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
                transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
            )
            verts = smpl_out.vertices.cpu().numpy()  # (1, V, 3)

            # force only this actor's track_id for this frame
            frame_data['track_id'] = [primary_actor_id]
            frame_data['vertices_orig'] = [verts[0].copy()]

            # store UNTRANSFORMED verts to derive canonical floor
            all_verts_for_floor.append(torch.tensor(verts, dtype=torch.bfloat16))

            # also store transformed verts per frame if available
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

        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            gv, gf, gc = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf, gc = None, None, None

        # ---------- 3) average sim for fixed floor ----------
        s_avg, R_avg, t_avg = self._average_sims(per_frame_sims)

        # ---------- 4) visualize ----------
        rerun_vis_world4d(
            video_id=video_id,
            images=images,
            world4d=world4d,
            results=results,
            pipeline=self.pipeline,
            faces=self.smplx.faces,
            floor=(gv, gf, gc) if gv is not None else None,
            init_fps=10,
            img_maxsize=480,
            dynamic_prediction_path=str(self.dynamic_scene_dir_path),
            frame_kp_corr=frame_kp_corr,
            per_frame_sims=per_frame_sims,
            global_floor_sim=(s_avg, R_avg, t_avg),
        )

        print('Rerun visualization running. Press Ctrl+C to terminate.')
        while True:
            time.sleep(1)

    def infer_all_videos(self, split: str):
        video_id_list = ["0A8CF.mp4"]
        for video_id in tqdm(video_id_list, desc=f"Processing videos in split {split}", unit="video"):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            self.process_video(video_id)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Align SMPL to PI3 dynamic scene and visualize with rerun (per-frame sim + fixed floor) for a single actor.")
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
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
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
