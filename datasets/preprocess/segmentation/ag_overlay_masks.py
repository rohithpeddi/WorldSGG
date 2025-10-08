import os
from pathlib import Path
import cv2
import numpy as np


def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)


def _list_images(dir_path: str):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [f for f in os.listdir(dir_path) if Path(f).suffix.lower() in exts]

    # natural-ish sort by numeric substrings if present
    def _key(s):
        import re
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

    return sorted(files, key=_key)


def overlay_mask_on_video(
        video_mask_dir,
        video_frame_dir,
        output_dir,
        draw_contour=True,
        contour_bgr=(0, 0, 0),
        contour_thickness=2,
        background_bgr=(0, 0, 0),  # fill color for the masked/removed region (default black)
):
    _ensure_dir(output_dir)

    if not os.path.isdir(video_frame_dir):
        print(f"[overlay_mask_on_video] Frames dir not found: {video_frame_dir}")
        return []

    if not os.path.isdir(video_mask_dir):
        print(f"[overlay_mask_on_video] Masks dir not found:  {video_mask_dir}")
        return []

    frame_files = _list_images(video_frame_dir)
    mask_files = _list_images(video_mask_dir)
    mask_by_stem = {Path(f).stem: f for f in mask_files}

    saved_paths = []
    for frame_name in frame_files:
        stem = Path(frame_name).stem
        mask_name = mask_by_stem.get(stem)
        if mask_name is None:
            continue

        frame_path = os.path.join(video_frame_dir, frame_name)
        mask_path = os.path.join(video_mask_dir, mask_name)

        frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
        if frame is None:
            print(f"[overlay_mask_on_video] Could not read frame: {frame_path}")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[overlay_mask_on_video] Could not read mask:  {mask_path}")
            continue

        # Resize mask to frame if needed
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Ensure binary 0/255
        _, mask255 = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Inverse mask: 255 where we KEEP (background), 0 where we REMOVE (masked/foreground)
        inv_mask255 = cv2.bitwise_not(mask255)

        # Keep background pixels as-is, zero masked region
        kept_bg = cv2.bitwise_and(frame, frame, mask=inv_mask255)

        # Optional: fill masked region with a non-black color (otherwise stays zero)
        out = kept_bg
        if background_bgr != (0, 0, 0):
            bg_img = np.full_like(frame, background_bgr, dtype=np.uint8)
            fill_masked = cv2.bitwise_and(bg_img, bg_img, mask=mask255)
            out = cv2.add(kept_bg, fill_masked)

        # Optional contour for visualization
        if draw_contour:
            # findContours expects 8-bit single-channel binary
            contours, _ = cv2.findContours(mask255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, contour_bgr, contour_thickness)

        out_path = os.path.join(output_dir, frame_name)
        if cv2.imwrite(out_path, out):
            saved_paths.append(os.path.abspath(out_path))
        else:
            print(f"[overlay_mask_on_video] Failed to write: {out_path}")

    return saved_paths


def _write_video_from_frames(frames: list, out_video_path: str, fps: float):
    if not frames:
        print(f"[video] No frames to write for {out_video_path}")
        return

    # Read first frame to get size
    first = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    if first is None:
        print(f"[video] Could not read first frame: {frames[0]}")
        return

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    _ensure_dir(os.path.dirname(out_video_path))
    vw = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
    if not vw.isOpened():
        print(f"[video] Failed to open writer: {out_video_path}")
        return

    for fp in frames:
        im = cv2.imread(fp, cv2.IMREAD_COLOR)
        if im is None:
            print(f"[video] Skipping unreadable frame: {fp}")
            continue
        if im.shape[0] != h or im.shape[1] != w:
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(im)

    vw.release()
    print(f"[video] Wrote: {out_video_path}")


def overlay_masks():
    video_id_list = ["00T1E.mp4", "0DJ6R.mp4"]
    masks_root_dir = "/data/rohith/ag/segmentation/masks/combined_1"
    frames_root_dir = "/data/rohith/ag/sampled_frames/"
    output_frames_root_dir = "/data/rohith/ag/segmentation/masks/overlayed_frames"
    output_videos_root_dir = "/data/rohith/ag/segmentation/masks/overlayed_videos"
    os.makedirs(output_frames_root_dir, exist_ok=True)
    os.makedirs(output_videos_root_dir, exist_ok=True)

    for video_id in video_id_list:
        video_mask_dir = os.path.join(masks_root_dir, video_id)
        video_frame_dir = os.path.join(frames_root_dir, video_id)
        output_frame_dir = os.path.join(output_frames_root_dir, video_id)
        output_video_dir = os.path.join(output_videos_root_dir, video_id)

        if os.path.exists(output_frame_dir) and len(os.listdir(output_frame_dir)) > 0:
            print(f"[main] Skipping existing output for {video_id}: {output_frame_dir}")
            continue

        os.makedirs(output_frame_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)

        if not os.path.isdir(video_frame_dir):
            print(f"[main] No frames directory for {video_id}: {video_frame_dir}")
            continue

        saved = overlay_mask_on_video(
            video_mask_dir=video_mask_dir,
            video_frame_dir=video_frame_dir,
            output_dir=output_frame_dir,
            draw_contour=True,
            contour_bgr=(0, 0, 0),
            contour_thickness=2,
        )

        fps = 10
        out_video_path = os.path.join(output_video_dir, f"{video_id}")
        _write_video_from_frames(saved, out_video_path, fps)


if __name__ == "__main__":
    overlay_masks()
