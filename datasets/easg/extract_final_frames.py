import os
import subprocess
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import glob

# --- Configuration ---
DEFAULT_TRIMMED_VIDEOS_DIR = '/data4/rohith/easg/data/trimmed_easg_clips'
DEFAULT_INDEX_DIR = '/data4/rohith/easg/data/feature_sampled_data'
FINAL_FRAME_OUTPUT_DIR = '/data4/rohith/easg/data/feature_sampled_data/final_frames'


def extract_indexed_frames(
        trimmed_videos_dir: str,
        index_dir: str,
        output_dir: str
) -> None:
    """
    Reads sampled index files (*.npy) and extracts the corresponding frames
    from the original trimmed videos using FFmpeg.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Find all index files (the list of frames we want)
    index_files = glob.glob(os.path.join(index_dir, '*_sampled_indices.npy'))

    if not index_files:
        print(f"No index files (*_sampled_indices.npy) found in '{index_dir}'. Exiting.")
        return

    print(f"Found {len(index_files)} index files. Starting final extraction...")

    success_count = 0
    fail_count = 0

    # Use tqdm to track progress over all videos
    with tqdm(total=len(index_files), desc="Extracting Final Frames", unit="video") as pbar:
        for index_path in index_files:
            index_path = Path(index_path)

            # Extract video ID from index filename (e.g., 'clipUID_action0_sampled_indices.npy')
            video_stem = index_path.stem.replace('_sampled_indices', '')
            video_filename = f"{video_stem}.mp4"
            video_path = Path(trimmed_videos_dir) / video_filename

            # Create a dedicated output folder for this video's frames
            output_video_frames_dir = Path(output_dir) / video_stem
            output_video_frames_dir.mkdir(parents=True, exist_ok=True)

            # Check if the source video exists
            if not video_path.exists():
                pbar.write(f"Warning: Source video not found for {video_filename}. Skipping.")
                pbar.update(1)
                fail_count += 1
                continue

            # Load the sampled indices
            try:
                # The indices are 1-based frame numbers (as commonly used)
                sampled_indices = np.load(index_path).astype(int).tolist()
            except Exception as e:
                pbar.write(f"Error loading indices for {video_stem}: {e}. Skipping.")
                pbar.update(1)
                fail_count += 1
                continue

            # Create a filter string for FFmpeg (select specific frame numbers)
            # The select filter works with 1-based frame numbers (n)
            # Example: "select='eq(n,10)+eq(n,25)+eq(n,50)'"
            select_filter = "+".join([f'eq(n,{i})' for i in sampled_indices])

            # 2. FFmpeg command to extract ONLY the selected frames
            command = [
                'ffmpeg',
                '-i', str(video_path),
                '-vf', f"select='{select_filter}'",
                '-vsync', '0',  # Required to output selected frames without duplicates
                '-q:v', '2',  # JPEG quality
                '-y',
                str(output_video_frames_dir / f'frame_%06d.jpg')
            ]

            try:
                # Run FFmpeg
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success_count += 1
            except subprocess.CalledProcessError:
                pbar.write(f"FAILED to extract frames for {video_filename} using FFmpeg.")
                fail_count += 1
            except FileNotFoundError:
                pbar.write("\nFATAL ERROR: FFmpeg not found.")
                break

            pbar.update(1)
            pbar.set_postfix_str(f"Kept: {len(sampled_indices)} frames")

    print("\n--- Final Frame Extraction Complete ---")
    print(f"Total videos processed: {len(index_files)}")
    print(f"Successfully extracted: {success_count}")
    print(f"Failed extractions: {fail_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Extracts only the feature-sampled frames using index files."
    )
    parser.add_argument('--trimmed_videos_dir',
                        default=DEFAULT_TRIMMED_VIDEOS_DIR,
                        help=f"Path to the directory containing the source trimmed MP4 videos (default: '{DEFAULT_TRIMMED_VIDEOS_DIR}')")
    parser.add_argument('--index_dir',
                        default=DEFAULT_INDEX_DIR,
                        help=f"Path to the directory containing the *_sampled_indices.npy files (default: '{DEFAULT_INDEX_DIR}')")
    parser.add_argument('--output_dir',
                        default=FINAL_FRAME_OUTPUT_DIR,
                        help=f"Final directory to save the extracted JPEG frames (default: '{FINAL_FRAME_OUTPUT_DIR}')")

    args = parser.parse_args()
    extract_indexed_frames(args.trimmed_videos_dir, args.index_dir, args.output_dir)