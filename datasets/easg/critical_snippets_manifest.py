import json
import os
from typing import Dict, Any, List

# --- Configuration (Hardcoded Paths and Values) ---

# IMPORTANT: REPLACE THESE PATHS WITH YOUR ACTUAL LOCATIONS ON THE SERVER
EASG_ANNOTATIONS_PATH = '/data4/rohith/easg/EASG_unict_master_final.json'
MAPPING_PATH = '/data4/rohith/easg/clip_video_uids_mapping.json'
EGO4D_VIDEO_DIR = '/data4/rohith/easg/videos/v2/full_scale'

# Output settings
DEFAULT_OUTPUT_DIR = '/data4/rohith/easg/data/critical_snippets_clips'
MANIFEST_FILE = 'critical_snippets_manifest.json'

# Snippet generation settings
FPS = 30.0
WINDOW_SIZE_FRAMES = 50  # 50 frames before and 50 frames after the critical frame (101 frames total)


# --- Core Logic ---

def create_critical_snippet_manifest() -> None:
    """
    Loads EASG data and creates a manifest for trimming 101-frame video snippets
    centered on PRE, PNR, and POST frames.
    """

    output_dir = DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading annotations from: {EASG_ANNOTATIONS_PATH}")
    try:
        with open(EASG_ANNOTATIONS_PATH, 'r') as f:
            easg_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: EASG annotations file not found at {EASG_ANNOTATIONS_PATH}. Please fix the path.")
        return

    print(f"Loading mapping from: {MAPPING_PATH}")
    try:
        with open(MAPPING_PATH, 'r') as f:
            clip_to_video_map = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Mapping file not found at {MAPPING_PATH}. Please fix the path.")
        return

    trimming_tasks: List[Dict[str, Any]] = []

    def generate_task(clip_uid, video_uid, frame_key, frame_number, action_index):

        # 1. Define the snippet's frame boundaries (clamped at 0)
        start_frame = max(0, frame_number - WINDOW_SIZE_FRAMES)
        end_frame = frame_number + WINDOW_SIZE_FRAMES

        # 2. Convert frames to seconds
        start_sec = start_frame / FPS
        end_sec = end_frame / FPS

        # 3. Compile the Task
        task = {
            'clip_uid': clip_uid,
            'video_uid': video_uid,
            'action_index': action_index,
            'frame_key': frame_key.upper(),  # 'PRE', 'PNR', 'POST'
            'center_frame': frame_number,
            'start_sec': round(start_sec, 3),
            'end_sec': round(end_sec, 3),
            'input_path': os.path.join(EGO4D_VIDEO_DIR, f'{video_uid}.mp4'),
            # Output filename includes the frame key (e.g., clip_UID_action0_PRE.mp4)
            'output_path': os.path.join(output_dir, f'{clip_uid}_action{action_index}_{frame_key.lower()}.mp4')
        }
        return task

    # --- Main Loop ---

    for clip_uid, clip_data in easg_data.items():

        try:
            video_uid = clip_to_video_map[clip_uid]
        except KeyError:
            # Skip clips for which we don't have the source video
            continue

        for i, graph_data in enumerate(clip_data.get('graphs', [])):

            # Iterate through the three critical frames defined in EASG
            for frame_key in ['pre', 'pnr', 'post']:
                frame_number = graph_data.get(frame_key)

                if frame_number is not None:
                    new_task = generate_task(
                        clip_uid,
                        video_uid,
                        frame_key,
                        frame_number,
                        i
                    )
                    trimming_tasks.append(new_task)
                # else: print(f"Warning: Missing '{frame_key}' frame for action {i} in clip {clip_uid}. Skipping snippet.")

    print(f"\nSuccessfully compiled {len(trimming_tasks)} critical snippet tasks.")

    # Save the manifest
    manifest_path = os.path.join(output_dir, MANIFEST_FILE)
    with open(manifest_path, 'w') as f:
        json.dump(trimming_tasks, f, indent=4)

    print(f"Manifest saved to: {manifest_path}")


if __name__ == '__main__':
    # Make sure you verify and update the paths at the top of the script!
    create_critical_snippet_manifest()