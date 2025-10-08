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


def overlay_mask_on_video(video_mask_dir, video_frame_dir, output_dir,
                          alpha=0.45, overlay_bgr=(0, 255, 0), draw_contour=True,
                          contour_bgr=(0, 0, 255), contour_thickness=2):
    """
    Overlay per-frame masks onto frames and save results into output_dir.

    Args:
        video_mask_dir (str): directory containing per-frame mask images (0/255).
        video_frame_dir (str): directory containing per-frame RGB frames.
        output_dir (str): directory where overlaid frames will be written.
        alpha (float): blending factor for colored overlay on masked pixels.
        overlay_bgr (tuple): BGR color for mask tint.
        draw_contour (bool): whether to outline mask boundaries.
        contour_bgr (tuple): BGR color for contour lines.
        contour_thickness (int): thickness of contour lines in pixels.

    Returns:
        List[str]: absolute paths to saved overlay frames, in order.
    """
    _ensure_dir(output_dir)

    if not os.path.isdir(video_frame_dir):
        print(f"[overlay_mask_on_video] Frames dir not found: {video_frame_dir}")
        return []

    if not os.path.isdir(video_mask_dir):
        print(f"[overlay_mask_on_video] Masks dir not found:  {video_mask_dir}")
        return []

    frame_files = _list_images(video_frame_dir)
    mask_files = _list_images(video_mask_dir)

    # Map by stem for robust matching
    mask_by_stem = {Path(f).stem: f for f in mask_files}

    saved_paths = []
    for frame_name in frame_files:
        stem = Path(frame_name).stem
        mask_name = mask_by_stem.get(stem, None)
        if mask_name is None:
            # Skip frames without corresponding mask
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
        if (mask.shape[0] != frame.shape[0]) or (mask.shape[1] != frame.shape[1]):
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # Binary mask
        mask_bin = (mask > 127).astype(np.uint8)

        # Create tinted overlay
        overlay = frame.copy()
        tint = np.zeros_like(frame, dtype=np.uint8)
        tint[:, :] = overlay_bgr

        # Alpha blend only on masked pixels
        # overlay = alpha * tint + (1 - alpha) * frame where mask==1
        # Efficient vectorized blend:
        m3 = mask_bin[:, :, None]
        overlay = (overlay.astype(np.float32) * (1 - alpha) +
                   tint.astype(np.float32) * alpha * m3 +
                   overlay.astype(np.float32) * (1 - m3)).astype(np.uint8)

        # Optional: draw contours to sharpen boundaries
        if draw_contour:
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, contour_bgr, contour_thickness)

        out_path = os.path.join(output_dir, frame_name)
        ok = cv2.imwrite(out_path, overlay)
        if ok:
            saved_paths.append(os.path.abspath(out_path))
        else:
            print(f"[overlay_mask_on_video] Failed to write: {out_path}")

    return saved_paths


def _infer_fps_from_dir(frame_dir: str, default_fps: float = 30.0) -> float:
    """
    Try to read FPS from a companion text file (fps.txt). Fallback to default if missing.
    """
    fps_file = os.path.join(frame_dir, "fps.txt")
    if os.path.isfile(fps_file):
        try:
            with open(fps_file, "r") as f:
                v = float(f.read().strip())
                if v > 0:
                    return v
        except Exception:
            pass
    return default_fps


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
    video_id_list = ["00T1E.mp4"]
    masks_root_dir = "data/rohith/ag/segmentation/masks/combined_1"
    frames_root_dir = "data/rohith/ag/sampled_frames/"
    output_frames_root_dir = "data/rohith/ag/segmentation/masks/overlayed_frames"
    output_videos_root_dir = "data/rohith/ag/segmentation/masks/overlayed_videos"
    os.makedirs(output_frames_root_dir, exist_ok=True)
    os.makedirs(output_videos_root_dir, exist_ok=True)

    for video_id in video_id_list:
        video_mask_dir = os.path.join(masks_root_dir, video_id)
        video_frame_dir = os.path.join(frames_root_dir, video_id)
        output_frame_dir = os.path.join(output_frames_root_dir, video_id)
        output_video_dir = os.path.join(output_videos_root_dir, video_id)
        os.makedirs(output_frame_dir, exist_ok=True)
        os.makedirs(output_video_dir, exist_ok=True)

        # Check if there is a folder corresponding to the video id in the frames directory
        if not os.path.isdir(video_frame_dir):
            print(f"[main] No frames directory for {video_id}: {video_frame_dir}")
            continue

        # 1) Create overlayed frames
        saved = overlay_mask_on_video(
            video_mask_dir=video_mask_dir,
            video_frame_dir=video_frame_dir,
            output_dir=output_frame_dir,
            alpha=0.45,
            overlay_bgr=(0, 255, 0),
            draw_contour=True,
            contour_bgr=(0, 0, 255),
            contour_thickness=2,
        )

        # 2) Stitch into a video (overlay.mp4) using FPS if available
        fps = _infer_fps_from_dir(video_frame_dir, default_fps=30.0)
        out_video_path = os.path.join(output_video_dir, "overlay.mp4")
        _write_video_from_frames(saved, out_video_path, fps)

    print("[main] Done.")


if __name__ == "__main__":
    overlay_masks()
