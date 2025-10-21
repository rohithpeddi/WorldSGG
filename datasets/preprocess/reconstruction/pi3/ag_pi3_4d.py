import argparse
import gc
import os
from pathlib import Path
from typing import Optional

import numpy as np
import rerun as rr
import torch
from tqdm import tqdm

from pi3.utils.basic import load_images_as_tensor


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

def _ensure_nhwc(images: np.ndarray) -> np.ndarray:
    """Ensure images are NHWC in [0,1]."""
    if images.ndim != 4:
        raise ValueError("images must be 4D")
    if images.shape[1] == 3:  # NCHW -> NHWC
        images = np.transpose(images, (0, 2, 3, 1))
    return np.clip(images, 0.0, 1.0)


def _flatten_points_colors_frames(points_wh: np.ndarray,
                                  colors_wh: np.ndarray,
                                  conf_wh: np.ndarray,
                                  conf_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten (S,H,W,3) points/colors and (S,H,W[,1]) conf to arrays:
      P: (N,3) points, C: (N,3) uint8 colors, F: (N,) frame idx
    Filter by conf >= conf_min and conf > 1e-5; drop NaNs/Infs.
    """
    S = points_wh.shape[0]
    if conf_wh.ndim == 4 and conf_wh.shape[-1] == 1:
        conf_wh = conf_wh[..., 0]
    mask = (conf_wh >= conf_min) & (conf_wh > 1e-5)

    P = points_wh[mask]                            # (N,3)
    C = (colors_wh[mask] * 255.0).astype(np.uint8) # (N,3)
    F = np.repeat(np.arange(S), repeats=points_wh.shape[1]*points_wh.shape[2])[mask.ravel()]

    good = np.isfinite(P).all(axis=1)
    return P[good], C[good], F[good]


def _voxel_ids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Integer voxel coordinates for each point."""
    return np.floor(points / voxel_size).astype(np.int64)


def build_static_background(predictions: dict,
                            conf_min: float = 0.5,
                            voxel_size: float = 0.03,
                            min_frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a 'sparse' static background:
      - keep high-confidence points from all frames
      - aggregate by voxel
      - mark voxels 'static' if seen in >= min_frames unique frames
      - output voxel-averaged xyz + averaged color
    """
    images = _ensure_nhwc(predictions["images"])        # (S,H,W,3) in [0,1]
    points = predictions["points"]                       # (S,H,W,3)
    conf   = predictions["conf"]                         # (S,H,W[,1])

    P, C, F = _flatten_points_colors_frames(points, images, conf, conf_min)  # N, N, N
    if P.size == 0:
        return np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.uint8)

    vox = _voxel_ids(P, voxel_size)                     # (N,3)
    # group id per point
    # pack voxel triplets to a single key for unique/grouping
    vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
    uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
    G = uniq_keys.shape[0]  # number of voxels

    # count unique frames per voxel (staticness)
    # do unique over (group, frame)
    gf = np.stack([inv, F], axis=1)
    gf_keys = np.ascontiguousarray(gf).view([('', gf.dtype)] * 2).ravel()
    uniq_gf, _ = np.unique(gf_keys, return_counts=False, return_index=False, return_inverse=False, axis=None), None
    # Faster: recompute with np.unique along rows
    uniq_gf_rows = np.unique(gf, axis=0)
    frames_per_group = np.zeros(G, dtype=np.int32)
    np.add.at(frames_per_group, uniq_gf_rows[:, 0], 1)

    static_mask_groups = frames_per_group >= int(min_frames)
    if not np.any(static_mask_groups):
        # fallback: keep densest 1% groups to avoid empty scene
        top = max(1, G // 100)
        # density ~ counts per group
        counts = np.bincount(inv, minlength=G)
        keep_idx = np.argpartition(-counts, top-1)[:top]
        static_mask_groups = np.zeros(G, dtype=bool)
        static_mask_groups[keep_idx] = True

    # voxel-wise mean position & color (only for static groups)
    counts = np.bincount(inv, minlength=G).astype(np.float32)
    sum_x = np.bincount(inv, weights=P[:, 0], minlength=G)
    sum_y = np.bincount(inv, weights=P[:, 1], minlength=G)
    sum_z = np.bincount(inv, weights=P[:, 2], minlength=G)

    sum_r = np.bincount(inv, weights=C[:, 0].astype(np.float32), minlength=G)
    sum_g = np.bincount(inv, weights=C[:, 1].astype(np.float32), minlength=G)
    sum_b = np.bincount(inv, weights=C[:, 2].astype(np.float32), minlength=G)

    # safe division
    counts[counts == 0] = 1.0
    mean_xyz = np.stack([sum_x/counts, sum_y/counts, sum_z/counts], axis=1)
    mean_rgb = np.stack([sum_r/counts, sum_g/counts, sum_b/counts], axis=1).astype(np.uint8)

    static_points = mean_xyz[static_mask_groups]
    static_colors = mean_rgb[static_mask_groups]
    return static_points.astype(np.float32), static_colors.astype(np.uint8)


def merge_static_with_frame(predictions: dict,
                            static_points: np.ndarray,
                            static_colors: np.ndarray,
                            frame_idx: int,
                            conf_min: float = 0.3,
                            dedup_voxel: Optional[float] = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge per-frame points with static background. Optionally de-duplicate with a small voxel size.
    """
    images = _ensure_nhwc(predictions["images"])     # (S,H,W,3)
    points = predictions["points"]                   # (S,H,W,3)
    conf   = predictions["conf"]                     # (S,H,W[,1])

    # slice this frame
    pts_f  = points[frame_idx]                       # (H,W,3)
    img_f  = images[frame_idx]                       # (H,W,3)
    conf_f = conf[frame_idx]                         # (H,W[,1])

    P, C, _ = _flatten_points_colors_frames(pts_f[None], img_f[None], conf_f[None], conf_min)

    if P.size == 0:
        merged_P = static_points
        merged_C = static_colors
    else:
        merged_P = np.concatenate([static_points, P], axis=0)
        merged_C = np.concatenate([static_colors, C], axis=0)

    if dedup_voxel is None or merged_P.size == 0:
        return merged_P.astype(np.float32), merged_C.astype(np.uint8)

    # light voxel de-dup to reduce overlap
    vox = _voxel_ids(merged_P, dedup_voxel)
    vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
    uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
    G = uniq_keys.shape[0]

    counts = np.bincount(inv, minlength=G).astype(np.float32)
    sum_x = np.bincount(inv, weights=merged_P[:, 0], minlength=G)
    sum_y = np.bincount(inv, weights=merged_P[:, 1], minlength=G)
    sum_z = np.bincount(inv, weights=merged_P[:, 2], minlength=G)

    sum_r = np.bincount(inv, weights=merged_C[:, 0].astype(np.float32), minlength=G)
    sum_g = np.bincount(inv, weights=merged_C[:, 1].astype(np.float32), minlength=G)
    sum_b = np.bincount(inv, weights=merged_C[:, 2].astype(np.float32), minlength=G)

    counts[counts == 0] = 1.0
    mean_xyz = np.stack([sum_x/counts, sum_y/counts, sum_z/counts], axis=1).astype(np.float32)
    mean_rgb = np.stack([sum_r/counts, sum_g/counts, sum_b/counts], axis=1).astype(np.uint8)
    return mean_xyz, mean_rgb


def _camera_R_t_from_4x4(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract rotation (3x3) and translation (3,) from a 4x4 transform."""
    if T.shape == (3, 4):
        R, t = T[:, :3], T[:, 3]
    elif T.shape == (4, 4):
        R, t = T[:3, :3], T[:3, 3]
    else:
        raise ValueError("camera pose must be (3,4) or (4,4)")
    return R.astype(np.float32), t.astype(np.float32)


def visualize_with_rerun(predictions: dict,
                         static_points: np.ndarray,
                         static_colors: np.ndarray,
                         per_frame_points: list[np.ndarray],
                         per_frame_colors: list[np.ndarray],
                         log_cameras: bool = True,
                         spawn: bool = True):
    rr.init("AG-Pi3 Scenes", spawn=spawn)
    rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

    # Log static background timelessly
    if static_points.size > 0:
        rr.log("world/static",
               rr.Points3D(positions=static_points.astype(np.float32),
                           colors=static_colors.astype(np.uint8)),
               timeless=True)

    # Cameras (optional): log transforms for each frame
    if log_cameras and "camera_poses" in predictions:
        cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)
        for i in range(len(cam_poses)):
            R, t = _camera_R_t_from_4x4(cam_poses[i])
            # Rerun expects a transform; we log per-frame transform timeless (pose-only)
            rr.log(f"world/cameras/cam_{i}",
                   rr.Transform3D(rotation=rr.Rotation3D(mat3x3=R), translation=t), timeless=True)

    # Animate per-frame merged points
    for i, (P, C) in enumerate(zip(per_frame_points, per_frame_colors)):
        rr.set_time_sequence("frame", i)
        if P.size == 0:
            continue
        rr.log("world/frame/merged_points",
               rr.Points3D(positions=P.astype(np.float32), colors=C.astype(np.uint8)))



class AgPi3:

    def __init__(
            self,
            root_dir_path, # data/rohith/ag/ag4D/static_frames
            output_dir_path=None,
            frame_annotated_dir_path=None,
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.output_dir_path = output_dir_path if output_dir_path is not None else root_dir_path
        os.makedirs(self.output_dir_path, exist_ok=True)

        self.frame_annotated_dir_path = frame_annotated_dir_path
        self.sampled_frames_idx_root_dir = "/data/rohith/ag/sampled_frames_idx"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    def load_model(self):
        pass

    def preprocess_image_list(self, data_path, is_video=False, video_id=None, sample_idx=None):
        interval = 10 if is_video else 1
        print(f'Sampling interval: {interval}')
        imgs = load_images_as_tensor(
            data_path,
            interval=interval,
            video_id=video_id,
            sample_idx=sample_idx
        ).to(self.device)  # (N, 3, H, W)
        return imgs

    def infer_video(self, video_id, conf_thres=10.0):
        data_path = f'{self.root_dir_path}/{video_id}'
        video_save_dir = os.path.join(self.output_dir_path, f"{video_id[:-4]}_{int(conf_thres)}")
        os.makedirs(video_save_dir, exist_ok=True)

        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        annotated_frame_id_list = os.listdir(video_frames_annotated_dir_path)
        annotated_frame_id_list = [f for f in annotated_frame_id_list if f.endswith('.png')]
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4]) # Number only

        # Get the mapping for sampled_frame_id and the actual frame id
        # Now start from the sampled frame which corresponds to the first annotated frame and keep the rest of the sampled frames
        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()  # Numbers only

        prediction_save_path = os.path.join(video_save_dir, "predictions.npz")
        if os.path.exists(prediction_save_path):
            predictions = np.load(prediction_save_path, allow_pickle=True)
            print(f"Loaded existing predictions for video {video_id} from {prediction_save_path}")
        else:
            raise NotImplementedError("Prediction generation not implemented in this snippet.")

        # -------------------------
        # NEW: background + per-frame scenes + rerun visualization
        # -------------------------
        try:
            # 1) Static background from 'sparse' points (multi-frame consistency via voxels)
            static_P, static_C = build_static_background(
                predictions,
                conf_min=0.50,  # high-confidence for background
                voxel_size=0.03,  # 3cm voxels
                min_frames=3  # seen in >= 3 frames -> static
            )

            # 2) Per-frame merges (background + frame points)
            S = predictions["points"].shape[0]
            per_frame_P, per_frame_C = [], []
            for fi in range(S):
                Pi, Ci = merge_static_with_frame(
                    predictions,
                    static_P, static_C,
                    frame_idx=fi,
                    conf_min=0.30,  # slightly looser for live frame
                    dedup_voxel=0.02  # 2cm dedup to keep clouds tidy
                )
                per_frame_P.append(Pi)
                per_frame_C.append(Ci)

            # 3) Visualize with rerun (animated over frames)
            visualize_with_rerun(
                predictions,
                static_P, static_C,
                per_frame_P, per_frame_C,
                log_cameras=True,
                spawn=True
            )
        except Exception as viz_e:
            print(f"[WARN] Rerun/static/merge viz pipeline failed: {viz_e}")

        # Cleanup
        del predictions
        gc.collect()
        torch.cuda.empty_cache()


    def infer_all_videos(self, split):
        video_id_list = os.listdir(self.root_dir_path)
        for video_id in tqdm(video_id_list):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            try:
                self.infer_video(video_id)
            except Exception as e:
                print(f"[ERROR] Error processing video {video_id}: {e}")


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from videos based on homography-overlap filtering."
    )
    parser.add_argument(
        "--root_dir_path", type=str, default="/data/rohith/ag/ag4D/static_frames",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    # parser.add_argument(
    #     "--root_dir_path", type=str, default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_frames",
    #     help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    # )
    parser.add_argument(
        "--frames_annotated_dir_path", type=str, default="/data/rohith/ag/frames_annotated",
        help="Path to directory containing annotated frames (with masks)."
    )
    parser.add_argument(
        "--output_dir_path", type=str, default="/data3/rohith/ag/ag4D/static_scenes/pi3_inpaint",
        help="Path to output directory where results will be saved."
    )
    parser.add_argument(
        "--split", type=_parse_split, default="04",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}. "
             "If omitted, processes all videos."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ag_pi3 = AgPi3(
        root_dir_path=args.root_dir_path,
        output_dir_path=args.output_dir_path,
        frame_annotated_dir_path=args.frames_annotated_dir_path,
    )
    ag_pi3.infer_all_videos(args.split)


if __name__ == "__main__":
    main()
