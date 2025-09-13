# frame_filter.py
from __future__ import annotations

import os
import pickle
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

import cv2
import argparse
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm


# ---------------------------
# Utility helpers
# ---------------------------

def _ensure_image(img_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_or_path}")
        return img
    elif isinstance(img_or_path, np.ndarray):
        # If likely RGB (common in PIL / matplotlib), convert to BGR for OpenCV consistency
        if img_or_path.ndim == 3 and img_or_path.shape[2] == 3:
            # Heuristic: assume it's RGB if the red channel has larger mean than blue (often true but harmless)
            # Safe approach: treat as RGB and convert to BGR.
            return cv2.cvtColor(img_or_path, cv2.COLOR_RGB2BGR)
        return img_or_path.copy()
    else:
        raise TypeError("frame must be a file path or a NumPy ndarray")


def _natural_key(path: str):
    base = os.path.basename(path)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', base)]


# ---------------------------
# Core geometry / matching
# ---------------------------

def compute_homography_and_overlap(
        frameA: Union[str, np.ndarray],
        frameB: Union[str, np.ndarray]
) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
    """
    Compute a robust homography between frameB -> frameA using SIFT keypoints + BFMatcher + Lowe's ratio test,
    then estimate overlap by warping frameB's corners into frameA's coordinate frame and computing the polygon
    intersection area with Shapely.
    """
    # Load/normalize images
    imgA = _ensure_image(frameA)
    imgB = _ensure_image(frameB)

    grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    # --- a) Feature Detection & Description (SIFT) ---
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError(
            "SIFT is not available. Please install opencv-contrib-python:\n"
            "  pip install opencv-contrib-python"
        )
    sift = cv2.SIFT_create()
    kpsA, desA = sift.detectAndCompute(grayA, None)
    kpsB, desB = sift.detectAndCompute(grayB, None)

    if desA is None or desB is None or len(kpsA) < 4 or len(kpsB) < 4:
        # Not enough info; assume overlap is unknown => treat as new (overlap=0.0)
        return None, 0.0, {
            "reason": "insufficient keypoints/descriptors",
            "num_kps_A": len(kpsA) if kpsA is not None else 0,
            "num_kps_B": len(kpsB) if kpsB is not None else 0,
        }

    # --- b) Feature Matching (BF + Lowe's ratio test) ---
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = bf.knnMatch(desA, desB, k=2)  # matches from A -> B
    ratio = 0.75
    good = []
    for m, n in raw_matches:
        if m.distance < ratio * n.distance:
            good.append(m)

    if len(good) < 4:
        return None, 0.0, {
            "reason": "insufficient good matches after ratio test",
            "num_matches": len(raw_matches),
            "num_good": len(good),
        }

    # --- c) Homography Estimation (RANSAC) ---
    # Build corresponding points: pointsA ~ H * pointsB
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        ptsB, ptsA, method=cv2.RANSAC, ransacReprojThreshold=4.0, maxIters=2000, confidence=0.995
    )

    if H is None or mask is None or int(mask.sum()) < 4:
        return None, 0.0, {
            "reason": "homography estimation failed",
            "num_good": len(good),
            "inliers": int(mask.sum()) if mask is not None else 0,
        }

    # --- d) Overlap Computation (warp frameB corners into frameA; polygon intersection) ---
    hA, wA = imgA.shape[:2]
    hB, wB = imgB.shape[:2]

    # Corners of frame B in its own coordinate frame
    cornersB = np.float32([
        [0, 0],
        [wB - 1, 0],
        [wB - 1, hB - 1],
        [0, hB - 1],
    ]).reshape(-1, 1, 2)

    warpedB = cv2.perspectiveTransform(cornersB, H).reshape(-1, 2)  # -> A's coordinate frame

    polyA = Polygon([(0, 0), (wA - 1, 0), (wA - 1, hA - 1), (0, hA - 1)])
    polyB_warped = Polygon([(float(x), float(y)) for x, y in warpedB])

    if not polyB_warped.is_valid:
        polyB_warped = polyB_warped.buffer(0)

    inter = polyA.intersection(polyB_warped)
    areaA = polyA.area
    overlap = float(inter.area / areaA) if areaA > 0 else 0.0
    overlap = float(max(0.0, min(1.0, overlap)))

    info: Dict[str, Any] = {
        "num_matches": len(raw_matches),
        "num_good": len(good),
        "inliers": int(mask.sum()),
        "warped_cornersB_in_A": warpedB.tolist(),
        "imageA_shape": (hA, wA),
        "imageB_shape": (hB, wB),
        "H": H.tolist(),
    }

    return H, overlap, info


# ---------------------------
# Batch filtering
# ---------------------------

def filter_frames_by_overlap(
        frames_dir: Optional[str] = None,
        overlap_thresh: float = 0.9
) -> List[str]:
    """
    Scans a directory of frames (defaults to ./frames or the env var FRAMES_DIR),
    compares each frame to the most recent *kept* frame using homography-based overlap,
    and returns the list of kept frame paths (low-overlap = new visual info).
    """
    # Gather and naturally sort image paths
    exts = (".jpg", ".jpeg", ".png", ".bmp")
    all_paths = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(exts)]
    if not all_paths:
        raise FileNotFoundError(f"No images with extensions {exts} found in {frames_dir}")

    all_paths.sort(key=_natural_key)

    kept: List[str] = []
    discarded: List[str] = []

    # Always keep the very first frame
    kept.append(all_paths[0])
    last_kept_img = _ensure_image(all_paths[0])

    for path in tqdm(all_paths[1:], desc="Filtering frames by overlap", unit="frame"):
        curr_img = _ensure_image(path)

        _, overlap, info = compute_homography_and_overlap(last_kept_img, curr_img)

        # Keep if overlap below threshold (i.e., brings new visual info)
        if overlap < overlap_thresh:
            kept.append(path)
            last_kept_img = curr_img  # update reference
        else:
            discarded.append(path)

        # Optional: uncomment to debug per-frame decisions
        # print(f"{os.path.basename(path)} -> overlap={overlap:.3f} ({'KEEP' if overlap < overlap_thresh else 'DROP'})")

    # Optional: final summary
    # print(f"Kept {len(kept)} / {len(all_paths)} frames (threshold={overlap_thresh})")

    return kept


class FeatureDescriptorSampler:

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.frames_dir = self.data_dir / "frames"
        self.max_samples = 150  # hard cap on number of frames to keep

    def init_video_sampler(self, video_id):
        video_frames_dir = self.frames_dir / video_id
        frame_files = sorted(video_frames_dir.glob("*.png"))

        print("---------------------------------------------------------------------")

        if len(frame_files) == 0:
            print(f"[{video_id}] No frames found at {video_frames_dir}")
            return []

        if len(frame_files) == 1:
            print(f"[{video_id}] Only one frame found; returning [0]")
            return [0]

        # Adaptive threshold sampling
        threshold = 0.95
        kept_frame_indices = []

        while threshold > 0.0:
            kept_frame_paths = filter_frames_by_overlap(str(video_frames_dir), overlap_thresh=threshold)
            kept_frame_indices = [int(Path(p).stem) for p in kept_frame_paths if Path(p).stem.isdigit()]

            print(f"[{video_id}] Threshold {threshold:.2f}: {len(kept_frame_indices)} frames")
            if len(kept_frame_indices) <= self.max_samples:
                break
            threshold -= 0.02

        # If still too many, use uniform sampling
        if len(kept_frame_indices) > self.max_samples:
            print(f"[{video_id}] Still exceeds max_samples ({self.max_samples}), using uniform sampling")
            step = len(kept_frame_indices) // self.max_samples
            kept_frame_indices = kept_frame_indices[::step][:self.max_samples]

        print(f"[{video_id}] Preliminary selection: {len(kept_frame_indices)} frames with threshold {threshold:.2f}")

        # --- Add annotated frames before enforcing minimum ---
        annotated_frames_dir = self.data_dir / "frames_annotated" / video_id
        annotated_frame_files = sorted(annotated_frames_dir.glob("*.png"))
        annotated_frame_indices = {int(f.stem) for f in annotated_frame_files if f.stem.isdigit()}
        kept_frame_indices_set = set(kept_frame_indices)
        kept_frame_indices_set.update(annotated_frame_indices)
        kept_frame_indices = sorted(kept_frame_indices_set)

        # 🔹 Ensure at least 17 frames (AFTER adding annotations)
        if len(kept_frame_indices) < 17:
            print(f"[{video_id}] Too few frames ({len(kept_frame_indices)}), enforcing minimum of 17")
            total_frames = len(frame_files)
            step = max(1, total_frames // 17)
            kept_frame_indices = [int(Path(frame_files[i]).stem) for i in range(0, total_frames, step)][:17]
            # Ensure annotated frames are still included
            kept_frame_indices_set = set(kept_frame_indices)
            kept_frame_indices_set.update(annotated_frame_indices)
            kept_frame_indices = sorted(kept_frame_indices_set)

        print(f"[{video_id}] Final selection: {len(kept_frame_indices)} total frames")
        print(kept_frame_indices)

        return kept_frame_indices


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from videos based on homography-overlap filtering."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/rohith/ag",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    feat_desc_sampler = FeatureDescriptorSampler(data_dir)  # uses internal default max_samples=150

    # video_id_list = os.listdir(data_dir / "videos")
    video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]
    output_dir = data_dir / "sampled_frames_idx"
    output_dir.mkdir(parents=True, exist_ok=True)

    sampled_video_dir = data_dir / "sampled_videos"
    sampled_video_dir.mkdir(parents=True, exist_ok=True)

    # Save the sampled frame indices npy file for each video_id
    video_id_frame_idx_dict = {}

    for video_id in tqdm(video_id_list):
        video_stem = video_id[:-4] if video_id.lower().endswith(".mp4") else Path(video_id).stem
        npy_path = output_dir / f"{video_stem}.npy"
        legacy_marker_dir = output_dir / video_id  # optional: legacy directory-as-marker

        already_processed = npy_path.exists() or legacy_marker_dir.is_dir()

        if already_processed:
            try:
                if npy_path.exists():
                    sub_sampled_idx = np.load(npy_path).astype(int).tolist()
                    print(f"[{video_id}] Found existing indices at {npy_path}; loading.")
                else:
                    # Legacy marker present but no npy: attempt to reconstruct from sampled_frames
                    sampled_frames_dir = data_dir / "sampled_frames" / video_id
                    if sampled_frames_dir.exists():
                        sub_sampled_idx = sorted(
                            [int(p.stem) for p in sampled_frames_dir.glob("*.png") if p.stem.isdigit()]
                        )
                        print(
                            f"[{video_id}] Found legacy processed marker dir and derived indices from {sampled_frames_dir}.")
                        np.save(npy_path, np.array(sub_sampled_idx, dtype=np.int32))
                    else:
                        print(
                            f"[{video_id}] Processed marker exists but no indices found; recomputing once to materialize .npy.")
                        sub_sampled_idx = feat_desc_sampler.init_video_sampler(video_id)
                        np.save(npy_path, np.array(sub_sampled_idx, dtype=np.int32))

                video_id_frame_idx_dict[video_id] = sub_sampled_idx
                # Skip copying frames and video writing if already processed
                continue
            except Exception as e:
                print(f"[{video_id}] Failed to load existing indices due to: {e}. Recomputing.")
                sub_sampled_idx = feat_desc_sampler.init_video_sampler(video_id)
                np.save(npy_path, np.array(sub_sampled_idx, dtype=np.int32))
                video_id_frame_idx_dict[video_id] = sub_sampled_idx
                # Fall through to copy/video below

        # Not processed yet: run the pipeline and save artifacts
        if not already_processed:
            sub_sampled_idx = feat_desc_sampler.init_video_sampler(video_id)
            np.save(npy_path, np.array(sub_sampled_idx, dtype=np.int32))
            video_id_frame_idx_dict[video_id] = sub_sampled_idx

        # 2. Store the sampled frames into a separate directory
        sampled_frames_dir = data_dir / "sampled_frames" / video_id
        sampled_frames_dir.mkdir(parents=True, exist_ok=True)

        for idx in sub_sampled_idx:
            src_path = data_dir / "frames" / video_id / f"{idx:06d}.png"
            dst_path = sampled_frames_dir / f"{idx:06d}.png"
            if src_path.exists():
                if not dst_path.exists():
                    shutil.copy(src_path, dst_path)
            else:
                print(f"Warning: Source frame {src_path} does not exist.")

        # 3. Make a video for the sampled frames and place it in sampled_videos
        video_writer = None
        for idx in sub_sampled_idx:
            frame_path = data_dir / "frames" / video_id / f"{idx:06d}.png"
            if frame_path.exists():
                frame = cv2.imread(str(frame_path))
                if video_writer is None:
                    height, width, _ = frame.shape
                    video_writer = cv2.VideoWriter(
                        str(sampled_video_dir / video_id),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        10,  # fps
                        (width, height)
                    )
                video_writer.write(frame)
        if video_writer is not None:
            video_writer.release()

    # 4. Finally, save the dictionary as a pkl file
    with open(data_dir / "video_id_frame_idx_dict.pkl", "wb") as f:
        pickle.dump(video_id_frame_idx_dict, f)


if __name__ == "__main__":
    main()
