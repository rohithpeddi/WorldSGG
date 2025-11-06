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

        # Segmentation masks paths
        self.dynamic_masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.dynamic_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.dynamic_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.dynamic_masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        # Internal (per-object) mask stores
        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.static_masked_frames_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'image_based'
        self.static_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'video_based'
        self.static_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'combined'
        self.static_masked_videos_dir_path = self.ag_root_directory / "segmentation_static" / "masked_videos"

        # Internal (per-object) mask stores
        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        self.pipeline = AgPipeline(static_cam=False, dynamic_scene_dir_path=self.dynamic_scene_dir_path,)
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()

    # ------------------------------------------------------------------
    # NEW: basic Umeyama similarity (scale + rot + trans)
    # ------------------------------------------------------------------
    def _umeyama_similarity(self, src: np.ndarray, dst: np.ndarray):
        """
        src: (N,3) SMPL points
        dst: (N,3) scene points (already masked to human)
        return s, R, t  such that  dst ≈ s * R @ src + t
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
        # diag term to handle reflection fix
        D = np.eye(3)
        D[2, 2] = np.linalg.det(U @ Vt)
        s = (S @ np.diag(D)) / var_src
        # S is 1D; simpler to just do trace:
        s = (S * np.diag(D)).sum() / var_src

        t = mu_dst - s * (R @ mu_src)
        return s, R, t

    # ------------------------------------------------------------------
    # NEW: one-liner to get partial human cloud for a frame
    # ------------------------------------------------------------------
    def _get_partial_pointcloud(self, video_id: str, frame_idx: int, label: str = "person") -> np.ndarray:
        """
        Pulls the point cloud for this frame from the pipeline and masks it
        using your stored masks (if present). Falls back to 'all points' for
        that frame if mask is missing.
        """
        # pipeline already loaded npz with per-frame points: (S, H, W, 3)
        pts_hw3 = self.pipeline.points[frame_idx]  # (H, W, 3)
        H, W, _ = pts_hw3.shape

        stem = f"{frame_idx:06d}"
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

    # ------------------------------------------------------------------
    # NEW: tiny similarity-ICP (few iters, NN via brute force)
    # ------------------------------------------------------------------
    def _similarity_icp(self, src: np.ndarray, dst: np.ndarray,
                        iters: int = 5,
                        max_src: int = 5000,
                        max_dst: int = 20000):
        """
        src: (Ns,3) SMPL vertices
        dst: (Nd,3) scene partial cloud
        returns cumulative s, R, t
        """
        src_curr = src.copy()
        # cumulative
        s_total = 1.0
        R_total = np.eye(3)
        t_total = np.zeros(3)

        # maybe subsample dst once
        if dst.shape[0] > max_dst:
            idx = np.random.choice(dst.shape[0], max_dst, replace=False)
            dst_used = dst[idx]
        else:
            dst_used = dst

        for _ in range(iters):
            # subsample src
            if src_curr.shape[0] > max_src:
                idx_src = np.random.choice(src_curr.shape[0], max_src, replace=False)
                src_used = src_curr[idx_src]
            else:
                src_used = src_curr

            # NN: brute force
            # dst_used: (Nd,3), src_used: (Ns,3)
            dists = np.linalg.norm(src_used[:, None, :] - dst_used[None, :, :], axis=2)  # (Ns, Nd)
            nn_idx = dists.argmin(axis=1)
            matched_dst = dst_used[nn_idx]

            s, R, t = self._umeyama_similarity(src_used, matched_dst)

            # apply to ALL current src
            src_curr = (s * (R @ src_curr.T).T) + t

            # update cumulative (compose similarity)
            # new = s * R * old + t
            s_total = s_total * s
            R_total = R @ R_total
            t_total = R @ t_total + t

        return s_total, R_total, t_total, src_curr

    def labels_for_frame(self, video_id: str, stem: str, is_static: bool) -> List[str]:
        lbls = set()
        if is_static:
            image_root_dir_list = [self.static_masks_im_dir_path, self.static_masks_vid_dir_path]
        else:
            image_root_dir_list = [self.dynamic_masks_im_dir_path, self.dynamic_masks_vid_dir_path]
        for root in image_root_dir_list:
            vdir = root / video_id
            if not vdir.exists():
                continue
            for fn in os.listdir(vdir):
                if not fn.endswith(".png"):
                    continue
                if "__" in fn:
                    st, lbl = fn.split("__", 1)
                    lbl = lbl.rsplit(".png", 1)[0]
                    if st == stem:
                        lbls.add(lbl)
        return sorted(lbls)

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

    def update_frame_map(
            self,
            frame_stems,
            video_id,
            frame_map: Dict[str, Dict[str, np.ndarray]],
            is_static
    ):
        all_labels = set()
        for stem in frame_stems:
            lbls = self.labels_for_frame(video_id, stem, is_static)
            if not lbls:
                continue
            all_labels.update(lbls)
            if stem not in frame_map:
                frame_map[stem] = {}
            for lbl in lbls:
                m = self.get_union_mask(video_id, stem, lbl, is_static)
                if m is not None:
                    frame_map[stem][lbl] = m
        return frame_map, all_labels

    def process_video(self, video_id):
        self.pipeline.__call__(video_id, save_only_essential=False)
        results = self.pipeline.estimate_2d_keypoints()

        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        # Get vertices (now: already aligned to the scene partial cloud)
        all_verts = []
        for frame_idx, frame_data in world4d.items():
            if len(frame_data['track_id']) == 0:  # no people
                continue

            rotmat = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
            verts = self.smplx(
                global_orient=rotmat[:, :1].cuda(),
                body_pose=rotmat[:, 1:22].cuda(),
                betas=frame_data['shape'].cuda(),
                transl=frame_data['trans'].cuda()
            ).vertices.cpu().numpy()  # (N_people, V, 3) in current pipeline world

            # NEW: get partial human cloud for this frame and align
            scene_pts = self._get_partial_pointcloud(video_id, frame_idx, label="person")
            aligned_verts_per_person = []
            for pi in range(verts.shape[0]):
                smpl_pi = verts[pi]  # (V,3)
                if scene_pts.shape[0] < 50:
                    # not enough scene points, keep as-is
                    aligned_verts_per_person.append(smpl_pi)
                    continue

                s, R, t, smpl_aligned = self._similarity_icp(smpl_pi, scene_pts, iters=4)
                aligned_verts_per_person.append(smpl_aligned)

            aligned_verts_per_person = np.stack(aligned_verts_per_person, axis=0)
            # overwrite in world4d so visualizer uses aligned mesh
            frame_data['vertices'] = aligned_verts_per_person
            all_verts.append(torch.tensor(aligned_verts_per_person, dtype=torch.bfloat16))

        if len(all_verts) > 0:
            all_verts = torch.cat(all_verts)
            [gv, gf, gc] = get_floor_mesh(all_verts, scale=2)
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

        print('Rerun visualization running. Please open the Rerun app to view the results.')
        print('Press Ctrl+C to terminate.')
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
