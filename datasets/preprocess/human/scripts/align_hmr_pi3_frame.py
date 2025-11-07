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
from scipy.spatial.transform import Rotation

# make local imports work like in your original file
sys.path.insert(0, os.path.dirname(__file__) + '/..')

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline
from datasets.preprocess.human.data_config import SMPLX_PATH
from datasets.preprocess.human.prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from datasets.preprocess.human.prompt_hmr.vis.traj import get_floor_mesh


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


def _color_from_id(idx: int) -> Tuple[int, int, int]:
    base = np.array([123, 200, 124], dtype=np.uint8)
    rand = np.array(
        [((idx * 37) % 255), ((idx * 57) % 255), ((idx * 97) % 255)],
        dtype=np.uint8,
    )
    return tuple(((base + rand) % 255).tolist())


# ------------------------------------------------------------
# updated visualization: now also logs per-frame correspondences
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
):
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

    # your viewer-space rotation
    scene_R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    scene_R = scene_R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    points = points @ scene_R.T

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # (optional) static floor logged once - rotate too
    if floor is not None:
        fv, ff = floor
        fv = np.asarray(fv, dtype=np.float32)
        ff = _faces_u32(np.asarray(ff))
        fv = fv @ scene_R.T
        rr.log(
            "floor",
            rr.Mesh3D(
                vertex_positions=fv,
                triangle_indices=ff,
            ),
        )

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

    def _human_path(tid: int, i: int) -> str:
        return f"{BASE}/humans/h{tid}" if reuse_paths else f"{BASE}/frames/t{i}/human_{tid}"

    def _frustum_path(i: int) -> str:
        return f"{BASE}/frustum" if reuse_paths else f"{BASE}/frames/t{i}/frustum"

    for i in range(num_frames):
        rr.set_time_sequence("frame", i)
        rr.log("/", rr.Clear(recursive=True))

        # humans (already globally aligned & stored in world4d[i]['vertices'])
        track_ids = world4d[i].get("track_id", [])
        verts_list = world4d[i].get("vertices", [])
        if len(track_ids) > 0 and len(verts_list) == len(track_ids):
            for tid, verts in zip(track_ids, verts_list):
                verts = np.asarray(verts, dtype=np.float32)
                # rotate into the same display frame as the point cloud
                verts = verts @ scene_R.T
                rr.log(
                    _human_path(int(tid), i),
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=faces_u32,
                        albedo_factor=_color_from_id(int(tid)),
                    ),
                )

        # per-frame floor (if you want it every frame)
        if floor is not None:
            fv, ff = floor
            fv = np.asarray(fv, dtype=np.float32)
            ff = _faces_u32(np.asarray(ff))
            fv = fv @ scene_R.T
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(vertex_positions=fv, triangle_indices=ff),
            )

        # dynamic points
        rr.log(
            f"{BASE}/points",
            rr.Points3D(
                points[i].reshape(-1, 3),
                colors=colors[i].reshape(-1, 3),
            ),
        )

        # --- NEW: per-frame correspondences (SMPL kps vs scene kps) ---
        if frame_kp_corr is not None and i in frame_kp_corr:
            smpl_pts, scene_pts = frame_kp_corr[i]
            smpl_pts = np.asarray(smpl_pts, dtype=np.float32)
            scene_pts = np.asarray(scene_pts, dtype=np.float32)

            # rotate them to match display frame
            smpl_pts_disp = smpl_pts @ scene_R.T
            scene_pts_disp = scene_pts @ scene_R.T

            rr.log(
                f"{BASE}/corr/frame_{i}/smpl_kps",
                rr.Points3D(
                    positions=smpl_pts_disp,
                    colors=np.full((smpl_pts_disp.shape[0], 3), [255, 0, 0], dtype=np.uint8),
                    radii=0.015,
                ),
            )
            rr.log(
                f"{BASE}/corr/frame_{i}/scene_kps",
                rr.Points3D(
                    positions=scene_pts_disp,
                    colors=np.full((scene_pts_disp.shape[0], 3), [0, 255, 0], dtype=np.uint8),
                    radii=0.017,
                ),
            )

            # arrows from SMPL -> scene
            if smpl_pts_disp.shape[0] == scene_pts_disp.shape[0]:
                rr.log(
                    f"{BASE}/corr/frame_{i}/arrows",
                    rr.Arrows3D(
                        origins=smpl_pts_disp,
                        vectors=(scene_pts_disp - smpl_pts_disp),
                        colors=np.full((smpl_pts_disp.shape[0], 3), [0, 0, 255], dtype=np.uint8),
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

    print("Rerun visualization started. Scrub the 'frame' timeline to see per-frame correspondences.")


# ------------------------------------------------------------
# your original split logic
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
        self.dynamic_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.static_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'

        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = self.ag_root_directory / "sampled_frames_idx"
        self.videos_directory = self.ag_root_directory / "videos"

        # dynamic / static masks
        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        self.pipeline = AgPipeline(static_cam=False, dynamic_scene_dir_path=self.dynamic_scene_dir_path)
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    # --- alignment math ---
    def _umeyama_similarity(self, src: np.ndarray, dst: np.ndarray):
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
            kps_2d = pdata.get("keypoints_2d", None)
            frames = pdata.get("frames", None)
            if kps_2d is None or frames is None:
                continue
            for i, fidx in enumerate(frames):
                frame_to_kps.setdefault(fidx, []).append(kps_2d[i])
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
            im_p = self.static_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.static_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
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

    # ---- per-frame sparse correspondences ----
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

        frame_points_hw3 = self.pipeline.points[frame_idx]

        smpl_list = []
        scene_list = []

        kps_2d_list = frame_to_kps[frame_idx]
        for person_idx, kps_2d in enumerate(kps_2d_list):
            if person_idx >= joints.shape[0]:
                break
            smpl_joints_person = joints[person_idx]
            for j in range(min(len(kps_2d), smpl_joints_person.shape[0])):
                u = float(kps_2d[j][0])
                v = float(kps_2d[j][1])
                scene_p = self._lift_2d_to_3d(frame_points_hw3, u, v)
                if scene_p is None:
                    continue
                smpl_list.append(smpl_joints_person[j])
                scene_list.append(scene_p)

        if len(smpl_list) == 0:
            return None, None

        return np.stack(smpl_list, axis=0), np.stack(scene_list, axis=0)

    # ---- per-frame dense correspondences ----
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

    # ---- average many similarities into one global ----
    def _average_similarities(self, sims: List[dict]):
        if len(sims) == 0:
            return 1.0, np.eye(3), np.zeros(3)

        total_w = sum(s["w"] for s in sims)
        if total_w <= 0:
            total_w = 1.0

        s_acc = 0.0
        t_acc = np.zeros(3, dtype=np.float64)
        R_acc = np.zeros((3, 3), dtype=np.float64)

        for s in sims:
            w = s["w"]
            s_acc += w * s["s"]
            t_acc += w * s["t"]
            R_acc += w * s["R"]

        s_g = s_acc / total_w
        t_g = t_acc / total_w

        U, _, Vt = np.linalg.svd(R_acc)
        R_g = U @ Vt
        if np.linalg.det(R_g) < 0:
            Vt[-1, :] *= -1
            R_g = U @ Vt

        s_g = float(np.clip(s_g, 0.5, 2.5))
        return s_g, R_g, t_g

    # ---- main per-video pipeline ----
    def process_video(self, video_id: str):
        # run pipeline
        self.pipeline.__call__(video_id, save_only_essential=False)
        self.pipeline.estimate_2d_keypoints()
        results = self.pipeline.results  # pipeline stores into self.results

        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        frame_idx_frame_path_map = self.idx_to_frame_idx_path(video_id)
        frame_to_kps = self._build_frame_to_kps_map(results)

        per_frame_sims = []
        max_frames = 60

        # NEW: store per-frame sparse correspondences for visualization
        frame_kp_corr: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

        for frame_idx, frame_data in list(world4d.items())[:max_frames]:
            # sparse
            smpl_s, scene_s = self._collect_kp_corr_for_frame(frame_idx, frame_data, frame_to_kps)
            # dense
            smpl_d, scene_d = self._collect_dense_corr_for_frame(
                video_id,
                frame_idx,
                frame_data,
                frame_idx_frame_path_map,
                num_smpl_samples_per_person=400,
                num_scene_subsample=800,
            )

            # record per-frame sparse for viz (even if dense is None)
            if smpl_s is not None and scene_s is not None and smpl_s.shape[0] > 0:
                frame_kp_corr[frame_idx] = (smpl_s, scene_s)

            smpl_all = []
            scene_all = []
            if smpl_s is not None:
                smpl_all.append(smpl_s)
                scene_all.append(scene_s)
            if smpl_d is not None:
                smpl_all.append(smpl_d)
                scene_all.append(scene_d)

            if len(smpl_all) == 0:
                continue

            smpl_all = np.concatenate(smpl_all, axis=0)
            scene_all = np.concatenate(scene_all, axis=0)

            if smpl_all.shape[0] < 3:
                continue

            s_f, R_f, t_f = self._umeyama_similarity(smpl_all, scene_all)
            per_frame_sims.append({
                "s": float(s_f),
                "R": R_f,
                "t": t_f,
                "w": float(smpl_all.shape[0]),
            })

        if len(per_frame_sims) == 0:
            s_g, R_g, t_g = 1.0, np.eye(3), np.zeros(3)
        else:
            s_g, R_g, t_g = self._average_similarities(per_frame_sims)

        # apply global transform to all SMPL verts
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

            verts_flat = verts.reshape(-1, 3)
            verts_tf = s_g * (verts_flat @ R_g.T) + t_g
            verts_tf = verts_tf.reshape(verts.shape)

            frame_data['vertices'] = verts_tf
            all_verts_for_floor.append(torch.tensor(verts_tf, dtype=torch.bfloat16))

        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            [gv, gf, gc] = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf = None, None

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
            frame_kp_corr=frame_kp_corr,  # <-- NEW
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
def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(f"Invalid split '{s}'. Choose one of: {sorted(valid)}")
    return val


def parse_args():
    parser = argparse.ArgumentParser(description="Align SMPL to PI3 dynamic scene and visualize with rerun.")
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
