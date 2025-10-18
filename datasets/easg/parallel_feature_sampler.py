from __future__ import annotations

import os
import shutil
import re
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import argparse
import cv2
import numpy as np
import subprocess
from shapely.geometry import Polygon
from tqdm import tqdm
import glob
import json
import multiprocessing  # <-- NEW: Module for parallel processing

# --- Configuration for your paths ---
# Configuration is moved to the global scope or passed to main, but fixed here for clarity
DEFAULT_TRIMMED_VIDEOS_DIR = '/data4/rohith/easg/videos/v2/full_scale'
OUTPUT_DIR = '/data4/rohith/easg/data/feature_sampled_data'
TEMP_FRAME_DIR = Path('/data4/rohith/easg/data/temp_full_frames_for_sift')
OVERLAP_THRESHOLD = 0.90  # Frames with overlap above this threshold are discarded
#NUM_WORKERS = os.cpu_count()  # Automatically use all available CPU cores
NUM_WORKERS = 10

# --- Utility helpers from your provided code (omitted for brevity, assume they are present) ---
# ... (Functions: _ensure_image, _natural_key, and FeatureDescriptorSampler class remain UNCHANGED)
def _unpack_and_process(args_tuple):
    """Unpacks (video_path_str, args_namespace) and calls process_single_video."""
    video_path_str, args_namespace = args_tuple
    return process_single_video(video_path_str, args_namespace)
# --------------------------------------------------------------------------------
# NEW: Isolated Function for Parallel Execution
# --------------------------------------------------------------------------------

def process_single_video(video_path_str: str, args: argparse.Namespace) -> Tuple[str, bool]:
    """
    Processes a single video file: extracts frames, runs overlap filter, saves data, and cleans up.
    Returns: (video_id, success_status)
    """
    try:
        # Re-initialize necessary paths from arguments
        videos_dir = Path(args.data_dir)
        output_base_dir = Path(args.output_dir)
        temp_frames_root = Path(TEMP_FRAME_DIR)
        overlap_thresh = args.overlap_thresh

        video_path = Path(video_path_str)
        video_id = video_path.stem
        temp_video_frames_dir = temp_frames_root / video_id

        output_npy_path = output_base_dir / f"{video_id}_sampled_indices.npy"
        output_pose_path = output_base_dir / f"{video_id}_pose_data.json"

        if output_npy_path.exists() and output_pose_path.exists():
            # If BOTH final output files exist, assume the video was successfully processed
            # Print a message (pbar.write) that won't disrupt the tqdm bar
            print(f"Skipping {video_id}: Output files already exist.")
            return video_id, True  # Return True (Success) immediately

            # ------------------------------------------------------------------
            # If the check above fails, proceed with the full pipeline
            # ------------------------------------------------------------------

        temp_frames_root = Path(TEMP_FRAME_DIR)
        temp_video_frames_dir = temp_frames_root / video_id
        overlap_thresh = args.overlap_thresh

        # Since multiprocessing creates independent processes, the sampler must be initialized here
        sampler = FeatureDescriptorSampler()

        # 1. Extract ALL frames to a temporary directory
        if not extract_frames_to_temp_dir(video_path, temp_video_frames_dir):
            return video_id, False

        # 2. Run the overlap filter on the extracted frames
        kept_indices, video_pose_data = sampler.filter_frames_by_overlap(
            video_id,
            temp_video_frames_dir,
            overlap_thresh=overlap_thresh
        )

        # 3. Save the resulting indices (the primary output)
        output_npy_path = output_base_dir / f"{video_id}_sampled_indices.npy"
        np.save(output_npy_path, np.array(kept_indices, dtype=np.int32))

        # 4. Save the Homography (Pose) data to a JSON file
        output_pose_path = output_base_dir / f"{video_id}_pose_data.json"
        with open(output_pose_path, 'w') as f:
            json.dump(video_pose_data, f, indent=4)

        return video_id, True

    except Exception as e:
        # Return False on any unexpected error
        print(f"\n[FATAL ERROR during processing {video_path_str}]: {e}")
        return video_path.stem, False

    finally:
        # CRITICAL: Clean up the temporary directory, regardless of success/failure
        if 'temp_video_frames_dir' in locals() and temp_video_frames_dir.exists():
            shutil.rmtree(temp_video_frames_dir, ignore_errors=True)


# --------------------------------------------------------------------------------
# Main Execution Pipeline (Multiprocessing Pool)
# --------------------------------------------------------------------------------

# The full definitions for the unchanged functions (omitted here for brevity)
# _ensure_image, _natural_key, FeatureDescriptorSampler, extract_frames_to_temp_dir

def _ensure_image(img_or_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img_or_path, str):
        # We read images directly, enforcing BGR for OpenCV
        img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image from path: {img_or_path}")
        return img
    elif isinstance(img_or_path, np.ndarray):
        # We trust the frame is BGR if it came from OpenCV, otherwise convert.
        if img_or_path.ndim == 3 and img_or_path.shape[2] == 3:
            # Simple check to handle potential RGB arrays, assuming they aren't BGR
            if cv2.countNonZero(img_or_path[:, :, 0] - img_or_path[:, :, 2]) > 0:
                return cv2.cvtColor(img_or_path, cv2.COLOR_RGB2BGR)
        return img_or_path.copy()
    else:
        raise TypeError("frame must be a file path or a NumPy ndarray")

def _natural_key(path: str):
    """Sorts file paths naturally (e.g., frame_9.jpg, frame_10.jpg)."""
    base = os.path.basename(path)
    # The frames we extract are named frame_0001.jpg, so we look for the number.
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', base)]

class FeatureDescriptorSampler:

    def __init__(self):
        # Initialize SIFT detector once
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is required. Please install opencv-contrib-python.")
        self.sift = cv2.SIFT_create()
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.ratio = 0.75

    def compute_homography_and_overlap(
            self,
            frameA: Union[str, np.ndarray],
            frameB: Union[str, np.ndarray]
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        """Compute homography (H) between frameB -> frameA and estimate overlap."""
        imgA = _ensure_image(frameA)
        imgB = _ensure_image(frameB)

        grayA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

        # a) Feature Detection & Description
        kpsA, desA = self.sift.detectAndCompute(grayA, None)
        kpsB, desB = self.sift.detectAndCompute(grayB, None)

        if desA is None or desB is None or len(kpsA) < 4 or len(kpsB) < 4:
            return None, 0.0, {"reason": "insufficient keypoints/descriptors"}

        # b) Feature Matching (BF + Lowe's ratio test)
        raw_matches = self.bf.knnMatch(desA, desB, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < self.ratio * n.distance:
                good.append(m)

        if len(good) < 4:
            return None, 0.0, {"reason": "insufficient good matches"}

        # c) Homography Estimation (RANSAC)
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(
            ptsB, ptsA, method=cv2.RANSAC, ransacReprojThreshold=4.0, confidence=0.995
        )

        if H is None or mask is None or int(mask.sum()) < 4:
            return None, 0.0, {"reason": "homography estimation failed"}

        # d) Overlap Computation
        hA, wA = imgA.shape[:2]
        hB, wB = imgB.shape[:2]

        cornersB = np.float32([[0, 0], [wB - 1, 0], [wB - 1, hB - 1], [0, hB - 1]]).reshape(-1, 1, 2)
        warpedB = cv2.perspectiveTransform(cornersB, H).reshape(-1, 2)

        polyA = Polygon([(0, 0), (wA - 1, 0), (wA - 1, hA - 1), (0, hA - 1)])
        # Use buffer(0) to fix non-valid polygons created by warping
        polyB_warped = Polygon([(float(x), float(y)) for x, y in warpedB]).buffer(0)

        inter = polyA.intersection(polyB_warped)
        areaA = polyA.area
        overlap = float(inter.area / areaA) if areaA > 0 else 0.0
        overlap = float(max(0.0, min(1.0, overlap)))

        return H, overlap, {"overlap": overlap}

    def filter_frames_by_overlap(
            self,
            video_id: str,
            frames_dir: Path,
            overlap_thresh: float = 0.9
    ) -> List[int]:
        """Runs the sequential overlap filtering process on a directory of frames."""
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        all_paths = [str(p) for p in frames_dir.glob("*") if p.suffix.lower() in exts]

        if not all_paths:
            return []

        all_paths.sort(key=_natural_key)

        kept_indices: List[int] = []
        pose_data: Dict[int, List[float]] = {}  # NEW DICTIONARY TO STORE H

        # Always keep the very first frame
        kept_indices.append(1)  # Assuming frame indices start at 1
        pose_data[1] = np.identity(3).flatten().tolist()   # NEW DICTIONARY TO STORE H
        last_kept_img = _ensure_image(all_paths[0])

        # We need a progress bar specific to the frames in this video
        frame_paths_to_check = all_paths[1:]

        for i, path in enumerate(tqdm(frame_paths_to_check,
                                      desc=f"Filtering {video_id}",
                                      unit="frame", leave=False)):

            curr_img = _ensure_image(path)

            H, overlap, _ = self.compute_homography_and_overlap(last_kept_img, curr_img)

            # Keep if overlap below threshold (i.e., brings new visual info)
            if overlap < overlap_thresh:
                # Extract the index from the filename (e.g., 'frame_0010.jpg' -> 10)
                frame_index = int(Path(path).stem.split('_')[-1])
                kept_indices.append(frame_index)
                last_kept_img = curr_img  # update reference

                if H is not None:
                    pose_data[frame_index] = H.flatten().tolist()
                else:
                    # Fallback: Treat as an Identity matrix if H failed but frame was kept
                    pose_data[frame_index] = np.identity(3).flatten().tolist()

        return kept_indices, pose_data


# --- Main Execution Pipeline ---

def extract_frames_to_temp_dir(video_path: Path, temp_video_dir: Path):
    """Uses FFmpeg to extract all frames (100% FPS) to a temporary location."""

    # Ensure a clean directory for this video's frames
    if temp_video_dir.exists():
        shutil.rmtree(temp_video_dir)
    os.makedirs(temp_video_dir)

    # FFmpeg command to extract ALL frames
    # -vf fps=1: This extracts 1 frame per second. To extract ALL frames, we use high FPS value or omit fps filter.
    # The standard way is to use a high FPS to capture all video frames.
    command = [
        'ffmpeg',
        '-i', str(video_path),
        '-vf', 'fps=30',  # Extract frames at 30 FPS to ensure we get all of them
        '-q:v', '2',
        '-y',
        str(temp_video_dir / 'frame_%04d.jpg')
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Filter video frames by feature descriptor overlap to remove redundancy."
    )
    # ... (argparse definitions remain the same)
    # FIX 1: Explicitly set dest="data_dir"
    parser.add_argument(
        "--data_dir", type=str, dest="data_dir", default=DEFAULT_TRIMMED_VIDEOS_DIR,
        help="Path to the directory containing the trimmed MP4 videos."
    )

    # FIX 2: Explicitly set dest="output_dir"
    parser.add_argument(
        "--output_dir", type=str, dest="output_dir", default=OUTPUT_DIR,
        help="Path to the directory where the sampled index files will be saved."
    )

    # FIX 3: Explicitly set dest="overlap_thresh"
    parser.add_argument(
        "--overlap_thresh", type=float, dest="overlap_thresh", default=OVERLAP_THRESHOLD,
        help="Overlap threshold (0.0 to 1.0). Frames with overlap > thresh are discarded."
    )

    args = parser.parse_args()

    videos_dir = Path(args.data_dir)
    output_base_dir = Path(args.output_dir)

    # 1. Validation and Setup
    if not videos_dir.exists():
        print(f"Error: Video directory not found at {videos_dir}")
        return

    output_base_dir.mkdir(parents=True, exist_ok=True)
    temp_frames_root = TEMP_FRAME_DIR
    temp_frames_root.mkdir(parents=True, exist_ok=True)  # Ensure root temp folder exists

    video_paths = glob.glob(str(videos_dir / '*.mp4'))

    if not video_paths:
        print(f"No videos found in {videos_dir}. Exiting.")
        return

    print(f"Found {len(video_paths)} videos. Starting parallel sampling process on {NUM_WORKERS} cores...")

    # 2. Create argument tuples for the Pool
    # Each item in the list is the full set of arguments for process_single_video
    pool_args = [(p, args) for p in video_paths]

    results = []

    # 3. Initialize and run the Pool
    try:
        # Initialize Pool with the number of CPU cores
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            # Use imap_unordered for an ordered progress bar on unordered results
            results_iterator = pool.imap_unordered(
                _unpack_and_process,  # Use the new global function
                pool_args
            )

            # Use tqdm to monitor the parallel execution progress
            for result in tqdm(results_iterator, total=len(video_paths), desc="Total Videos Progress", unit="video"):
                results.append(result)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Shutting down pool...")
        pool.terminate()
        pool.join()

    # 4. Final Cleanup and Reporting
    success_count = sum(1 for _, success in results if success)
    fail_count = len(results) - success_count

    # Final cleanup of the root temporary directory
    shutil.rmtree(temp_frames_root, ignore_errors=True)

    print("\n--- Feature Descriptor Sampling Complete ---")
    print(f"Total videos processed: {len(video_paths)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed to process:      {fail_count}")


if __name__ == "__main__":
    # Ensure all utility functions and the class definition are included above main()
    # (The omitted sections must be present in your final file)
    main()