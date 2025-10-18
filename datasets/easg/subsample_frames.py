import os
import subprocess
import argparse
import glob
from tqdm import tqdm  # Import the tqdm library

# --- Configuration ---
DEFAULT_TRIMMED_VIDEOS_DIR = '/data4/rohith/easg/data/trimmed_easg_clips'
DEFAULT_OUTPUT_FRAMES_DIR = '/rohith/easg/data/feature_sampled_data'
TARGET_FPS_FOR_FRAMES = 1  # We want 1 frame per second from each video


def subsample_all_videos_at_1fps(
        trimmed_videos_dir: str,
        output_frames_base_dir: str
) -> None:
    """
    Iterates through all MP4 videos and extracts frames at 1 FPS,
    displaying a progress bar using tqdm.
    """
    # Ensure the base output directory for frames exists
    os.makedirs(output_frames_base_dir, exist_ok=True)

    # Find all MP4 video files in the trimmed videos directory
    video_files = glob.glob(os.path.join(trimmed_videos_dir, '*.mp4'))

    if not video_files:
        print(f"No MP4 video files found in '{trimmed_videos_dir}'. Exiting.")
        return

    total_videos = len(video_files)
    print(f"Found {total_videos} trimmed videos to subsample.")
    print(f"Targeting {TARGET_FPS_FOR_FRAMES} frame(s) per second.")

    success_count = 0
    fail_count = 0

    # Wrap the video_files list with tqdm for the progress bar
    # desc: Description shown next to the bar
    # unit: Unit name for the counter
    with tqdm(total=total_videos, desc="Subsampling Videos", unit="video") as pbar:
        for video_path in video_files:
            video_filename = os.path.basename(video_path)

            # Create a unique folder for each video's frames
            video_name_without_ext = os.path.splitext(video_filename)[0]
            output_video_frames_dir = os.path.join(output_frames_base_dir, video_name_without_ext)
            os.makedirs(output_video_frames_dir, exist_ok=True)

            # FFmpeg command (as before)
            command = [
                'ffmpeg',
                '-i', video_path,
                '-vf', f'fps={TARGET_FPS_FOR_FRAMES}',
                '-q:v', '2',
                '-y',  # Automatically overwrite output files
                os.path.join(output_video_frames_dir, f'frame_%04d.jpg')
            ]

            try:
                # Run FFmpeg, suppressing output
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                success_count += 1
            except subprocess.CalledProcessError:
                # Use pbar.write() to print status without breaking the progress bar
                pbar.write(f"FAILED to process {video_filename}")
                fail_count += 1
            except FileNotFoundError:
                pbar.write("\nFATAL ERROR: FFmpeg not found.")
                break

            # Update the progress bar status/description
            pbar.set_postfix_str(f"Success: {success_count}, Fail: {fail_count}")
            pbar.update(1)  # Increment the progress bar by one task

    print("\n--- Frame Subsampling Complete ---")
    print(f"Total videos processed: {total_videos}")
    print(f"Successfully subsampled: {success_count}")
    print(f"Failed to subsample: {fail_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subsample frames from trimmed videos at a specified FPS.")
    parser.add_argument('--trimmed_videos_dir',
                        default=DEFAULT_TRIMMED_VIDEOS_DIR,
                        help=f"Path to the directory containing the trimmed MP4 videos (default: '{DEFAULT_TRIMMED_VIDEOS_DIR}')")
    parser.add_argument('--output_frames_base_dir',
                        default=DEFAULT_OUTPUT_FRAMES_DIR,
                        help=f"Base directory to save the subsampled frames (default: '{DEFAULT_OUTPUT_FRAMES_DIR}')")
    parser.add_argument('--target_fps',
                        type=float,
                        default=TARGET_FPS_FOR_FRAMES,
                        help=f"Target frames per second to extract from each video (default: {TARGET_FPS_FOR_FRAMES})")

    args = parser.parse_args()
    TARGET_FPS_FOR_FRAMES = args.target_fps
    subsample_all_videos_at_1fps(args.trimmed_videos_dir, args.output_frames_base_dir)