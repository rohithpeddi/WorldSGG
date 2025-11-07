import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, List

import cv2
import numpy as np
import torch
from tqdm import tqdm

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from datasets.preprocess.human.data_config import SMPLX_PATH
from datasets.preprocess.human.prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from datasets.preprocess.human.prompt_hmr.vis.traj import get_floor_mesh
from datasets.preprocess.human.prompt_hmr.vis import rerun_vis as rrvis


# ---------------------------
# Split logic (yours)
# ---------------------------

def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """
    Get the split that the video belongs to based on its ID.
    Accepts either a bare ID (e.g., '0DJ6R') or a filename (e.g., '0DJ6R.mp4').
    """
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

        # Internal (per-object) mask stores
        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        # Internal (per-object) mask stores
        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        self.pipeline = AgPipeline(static_cam=False, dynamic_scene_dir_path=self.dynamic_scene_dir_path, )
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()

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
        # trace-style scale
        s = (S * np.array([1, 1, np.linalg.det(U @ Vt)])).sum() / var_src

        t = mu_dst - s * (R @ mu_src)
        return s, R, t

    # ------------------------------------------------------------
    # Lift a single 2D keypoint (u,v) to 3D from per-frame (H,W,3) points
    # ------------------------------------------------------------
    def _lift_2d_to_3d(self, frame_points_hw3: np.ndarray, u: float, v: float):
        H, W, _ = frame_points_hw3.shape
        ui = int(round(u))
        vi = int(round(v))
        if ui < 0 or ui >= W or vi < 0 or vi >= H:
            return None
        p3d = frame_points_hw3[vi, ui]  # (3,)
        if not np.isfinite(p3d).all() or np.abs(p3d).sum() < 1e-6:
            return None
        return p3d

    # ------------------------------------------------------------
    # Collect all (SMPL_joint, scene_keypoint_3d) pairs over the video
    # ------------------------------------------------------------
    def _collect_kp_correspondences(self, video_id: str, world4d: dict, results: dict):
        """
        Returns:
            smpl_pts_all: (N,3)
            scene_pts_all: (N,3)
        Uses:
            - SMPLX forward per frame to get joints
            - results[...] keypoints_2d to know where the person was in 2D
            - pipeline.points[frame_idx] to lift to 3D
        """
        smpl_pts_all = []
        scene_pts_all = []

        # results['people'][person_id]['keypoints_2d'] is assumed from your previous code
        people_results = results.get("people", {})

        # Build a per-frame map: frame_idx -> list of (person_id, keypoints_2d_for_that_frame)
        # Your structure may differ slightly, but we'll assume each person stores per-frame keypoints.
        frame_to_kps = {}
        for pid, pdata in people_results.items():
            kps_2d = pdata.get("keypoints_2d", None)
            frames = pdata.get("frames", None)
            if kps_2d is None or frames is None:
                continue

            # kps_2d: (num_frames_for_person, K, 3) or (num_frames_for_person, K, 2)
            for i, fidx in enumerate(frames):
                if fidx not in frame_to_kps:
                    frame_to_kps[fidx] = []
                frame_to_kps[fidx].append(kps_2d[i])

        for frame_idx, frame_data in world4d.items():
            # 1) run SMPLX to get per-person meshes AND joints for this frame
            if len(frame_data['track_id']) == 0:
                continue

            rotmat = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
            smpl_out = self.smplx(
                global_orient=rotmat[:, :1].cuda(),
                body_pose=rotmat[:, 1:22].cuda(),
                betas=frame_data['shape'].cuda(),
                transl=frame_data['trans'].cuda()
            )
            verts = smpl_out.vertices.cpu().numpy()         # (P, V, 3)
            joints = smpl_out.joints.cpu().numpy()          # (P, J, 3)  <-- we will match to 2D kps

            # 2) get scene points for this frame
            frame_points_hw3 = self.pipeline.points[frame_idx]  # (H, W, 3)

            # 3) get 2D keypoints for this frame (could be multiple people)
            if frame_idx not in frame_to_kps:
                continue

            kps_2d_list = frame_to_kps[frame_idx]  # list of (K,2 or 3)

            # We'll match by index: kp j <-> joint j (best effort)
            for person_idx, kps_2d in enumerate(kps_2d_list):
                if person_idx >= joints.shape[0]:
                    break  # more 2D people than SMPL people

                smpl_joints_person = joints[person_idx]  # (J, 3)
                # kps_2d might be (K,3) with scores; take first 2
                for j in range(min(len(kps_2d), smpl_joints_person.shape[0])):
                    u = float(kps_2d[j][0])
                    v = float(kps_2d[j][1])
                    scene_p = self._lift_2d_to_3d(frame_points_hw3, u, v)
                    if scene_p is None:
                        continue
                    smpl_p = smpl_joints_person[j]
                    smpl_pts_all.append(smpl_p)
                    scene_pts_all.append(scene_p)

        if len(smpl_pts_all) == 0:
            return None, None

        smpl_pts_all = np.stack(smpl_pts_all, axis=0)
        scene_pts_all = np.stack(scene_pts_all, axis=0)
        return smpl_pts_all, scene_pts_all

    def _get_partial_pointcloud(self, video_id: str, frame_idx: int, frame_idx_frame_path_map,
                                label: str = "person") -> np.ndarray:
        """
        Pulls the point cloud for this frame from the pipeline and masks it
        using your stored masks (if present). Falls back to 'all points' for
        that frame if the mask is missing.
        """
        # pipeline already loaded npz with per-frame points: (S, H, W, 3)
        pts_hw3 = self.pipeline.points[frame_idx]  # (H, W, 3)
        H, W, _ = pts_hw3.shape

        stem = frame_idx_frame_path_map[frame_idx][:-4]
        mask = self.get_union_mask(video_id, stem, label, is_static=False)

        if mask is not None:
            # mask is H x W, boolean
            mask = mask.astype(bool)
            # safety in case mask shape != points shape
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            pts = pts_hw3[mask]
        else:
            pts = pts_hw3.reshape(-1, 3)

        # filter out NaNs / zeros
        pts = pts[np.isfinite(pts).all(axis=1)]
        # keep only where not all zeros
        nonzero = ~(np.abs(pts).sum(axis=1) < 1e-6)
        pts = pts[nonzero]

        return pts

    def _global_similarity_icp(self,
                               per_frame_smpl: list,
                               per_frame_scene: list,
                               iters: int = 5,
                               max_smpl_per_frame: int = 2500,
                               max_scene_per_frame: int = 15000):
        """
        per_frame_smpl: list of (V,3) or list of list-of-people [(P, V,3)]? we’ll flatten per frame
        per_frame_scene: list of (Nf,3) partial clouds
        returns single s, R, t that best fits ALL frames together
        """
        s_global = 1.0
        R_global = np.eye(3)
        t_global = np.zeros(3)

        num_frames = len(per_frame_smpl)

        for _ in range(iters):
            all_src_orig = []
            all_dst_match = []

            for f in range(num_frames):
                smpl_f = per_frame_smpl[f]
                scene_f = per_frame_scene[f]
                if smpl_f is None or scene_f is None:
                    continue
                if scene_f.shape[0] == 0:
                    continue

                # flatten people if needed
                if smpl_f.ndim == 3:
                    # (P,V,3) -> (P*V, 3)
                    smpl_flat = smpl_f.reshape(-1, 3)
                else:
                    smpl_flat = smpl_f  # (V,3)

                # optional subsample SMPL for speed
                if smpl_flat.shape[0] > max_smpl_per_frame:
                    idx_smpl = np.random.choice(smpl_flat.shape[0], max_smpl_per_frame, replace=False)
                    smpl_flat = smpl_flat[idx_smpl]

                # transform with current global estimate
                smpl_tf = (s_global * (R_global @ smpl_flat.T).T) + t_global  # (Ns,3)

                # subsample scene cloud once per frame
                if scene_f.shape[0] > max_scene_per_frame:
                    idx_scene = np.random.choice(scene_f.shape[0], max_scene_per_frame, replace=False)
                    scene_used = scene_f[idx_scene]
                else:
                    scene_used = scene_f

                # brute-force NN
                dists = np.linalg.norm(smpl_tf[:, None, :] - scene_used[None, :, :], axis=2)  # (Ns, Nd)
                nn_idx = dists.argmin(axis=1)
                matched_scene = scene_used[nn_idx]

                # store PAIRS: (original smpl BEFORE transform, matched scene)
                all_src_orig.append(smpl_flat)
                all_dst_match.append(matched_scene)

            if len(all_src_orig) == 0:
                break

            src_cat = np.concatenate(all_src_orig, axis=0)
            dst_cat = np.concatenate(all_dst_match, axis=0)

            # run one Umeyama on ALL pairs → gives a delta similarity
            s_delta, R_delta, t_delta = self._umeyama_similarity(src_cat, dst_cat)

            # compose: new = s_d * R_d * old + t_d
            s_global = s_delta * s_global
            R_global = R_delta @ R_global
            t_global = R_delta @ t_global + t_delta

        return s_global, R_global, t_global

    def get_union_mask(self, video_id: str, stem: str, label: str, is_static) -> Optional[np.ndarray]:
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

        # Sort the list where the frame ids are of the format '000123.png'
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))

        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        # Get the mapping for sampled_frame_id and the actual frame id
        # Now start from the sampled frame which corresponds to the first annotated frame and keep the rest of the sampled frames
        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()  # Numbers only

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        chosen_frames = [video_sampled_frame_id_list[i] for i in sample_idx]
        frame_idx_frame_path_map = {i: f"{frame_id:06d}.png" for i, frame_id in enumerate(chosen_frames)}
        return frame_idx_frame_path_map

    def process_video(self, video_id):
        # run pipeline as before
        self.pipeline.__call__(video_id, save_only_essential=False)
        results = self.pipeline.estimate_2d_keypoints()

        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        # 1) collect correspondences using keypoints (this is fast)
        smpl_kps, scene_kps = self._collect_kp_correspondences(video_id, world4d, results)

        if smpl_kps is not None and scene_kps is not None and smpl_kps.shape[0] >= 4:
            s_g, R_g, t_g = self._umeyama_similarity(smpl_kps, scene_kps)
        else:
            # fallback: no alignment
            s_g, R_g, t_g = 1.0, np.eye(3), np.zeros(3)

        # 2) now run through frames again, generate verts, apply SAME transform
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

            # apply global transform
            verts_tf = (s_g * (R_g @ verts.reshape(-1, 3).T).T) + t_g
            verts_tf = verts_tf.reshape(verts.shape)

            frame_data['vertices'] = verts_tf
            all_verts_for_floor.append(torch.tensor(verts_tf, dtype=torch.bfloat16))

        # floor (unchanged)
        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            [gv, gf, gc] = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf = None, None

        rrvis.rerun_vis_world4d(
            video_id=video_id,
            images=images,
            world4d=world4d,
            results=results,
            pipeline=self.pipeline,
            faces=self.smplx.faces,
            floor=(gv, gf) if gv is not None else None,
            init_fps=10,
            img_maxsize=480,
        )

        print('Rerun visualization running. Press Ctrl+C to terminate.')
        while True:
            time.sleep(1)

    def infer_all_videos(self, split):
        # video_id_list = os.listdir(self.videos_directory)
        video_id_list = ["0DJ6R.mp4"]
        for video_id in tqdm(video_id_list, desc=f"Processing videos in split {split}", unit="video"):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            # self.process_video(video_id)
            self.process_video(video_id)
            # try:
            #     self.process_video_intermediate_steps(video_id)
            # except Exception as e:
            #     print(f"[ERROR] Error processing video {video_id}: {e}")


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(f"Invalid split '{s}'. Choose one of: {sorted(valid)}")
    return val


def parse_args():
    parser = argparse.ArgumentParser(description="Sample frames from videos based on homography-overlap filtering.")
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
             "If omitted, processes all videos."
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
