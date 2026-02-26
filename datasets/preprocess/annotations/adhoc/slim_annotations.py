#!/usr/bin/env python3
"""
Create lightweight annotation PKLs for the monocular 3D detector.

Reads the full merged OBB camera annotations (~12 GB) and extracts
ONLY the fields needed for training: intrinsics + bbox_frames.
Saves compressed PKLs to a new directory.

Input:   {ag_root}/world_annotations/bbox_annotations_3d_obb_camera_intrinsics/{stem}.pkl
Output:  {ag_root}/world_annotations/monocular3d_bbox_annotations/{stem}.pkl

Usage:
    python slim_annotations.py \
        --ag_root_directory /data/rohith/ag \
        [--input_dir bbox_annotations_3d_obb_camera_intrinsics] \
        [--overwrite]
"""
import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Numpy version compatibility
# ---------------------------------------------------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Handle numpy 2.x pickles on numpy 1.x."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pkl_compat(path):
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()


# ---------------------------------------------------------------------------
# Slim down a single video PKL
# ---------------------------------------------------------------------------
def _slim_bbox_frame_object(obj: dict) -> dict:
    """Keep only the fields needed for training from a single frame object."""
    slim = {}
    # Label (required for class matching)
    if "label" in obj:
        slim["label"] = obj["label"]

    # 3D corners — try all known formats
    if "obb_corners_final" in obj:
        slim["obb_corners_final"] = obj["obb_corners_final"]
    else:
        raise Exception (f"Missing obb_corners_final in object: {obj.get('label', 'unknown')}")

    return slim


def slim_video_data(video_data: dict) -> dict:
    """Extract only training-relevant fields from a full video PKL."""
    slim: Dict[str, Any] = {}

    # 1. video_id
    if "video_id" in video_data:
        slim["video_id"] = video_data["video_id"]

    # 2. Intrinsics (from attach_intrinsics.py)
    if "intrinsics" in video_data:
        intr = video_data["intrinsics"]
        slim["intrinsics"] = {
            "fx": intr.get("fx"),
            "fy": intr.get("fy"),
            "cx": intr.get("cx"),
            "cy": intr.get("cy"),
            "source": intr.get("source", "unknown"),
        }

    # 3. bbox_frames from frames_final (the only field the dataset reads)
    frames_final = video_data.get("frames_final", {})
    bbox_frames = frames_final.get("bbox_frames", {})

    slim_bbox_frames: Dict[str, Any] = {}
    for frame_name, frame_data in bbox_frames.items():
        objects = frame_data.get("objects", [])
        slim_objects = [_slim_bbox_frame_object(obj) for obj in objects]
        # Only include frames that have at least one object with corners
        if slim_objects:
            slim_bbox_frames[frame_name] = {"objects": slim_objects}

    slim["frames_final"] = {"bbox_frames": slim_bbox_frames}

    return slim


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def create_slim_annotations(
    ag_root_directory: str,
    input_dir_name: str = "bbox_annotations_3d_obb_camera_intrinsics",
    overwrite: bool = False,
) -> None:
    ag_root = Path(ag_root_directory)
    input_dir = ag_root / "world_annotations" / input_dir_name
    output_dir = ag_root / "world_annotations" / "monocular3d_bbox_annotations"
    os.makedirs(output_dir, exist_ok=True)

    if not input_dir.exists():
        print(f"[slim] Input directory not found: {input_dir}")
        return

    pkl_paths = sorted(input_dir.glob("*.pkl"))
    if not pkl_paths:
        print(f"[slim] No PKL files found in {input_dir}")
        return

    print(f"[slim] Input:  {input_dir}  ({len(pkl_paths)} PKLs)")
    print(f"[slim] Output: {output_dir}")
    print()

    stats = {"processed": 0, "skipped": 0, "errors": 0}
    total_input_bytes = 0
    total_output_bytes = 0

    for pkl_path in pkl_paths:
        stem = pkl_path.stem
        out_path = output_dir / f"{stem}.pkl"

        if out_path.exists() and not overwrite:
            stats["skipped"] += 1
            continue

        try:
            video_data = _load_pkl_compat(pkl_path)
        except Exception as e:
            print(f"  [{stem}] ERROR loading: {e}")
            stats["errors"] += 1
            continue

        slim_data = slim_video_data(video_data)

        try:
            with open(out_path, "wb") as f:
                pickle.dump(slim_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  [{stem}] ERROR writing: {e}")
            stats["errors"] += 1
            continue

        in_size = pkl_path.stat().st_size
        out_size = out_path.stat().st_size
        total_input_bytes += in_size
        total_output_bytes += out_size

        stats["processed"] += 1
        if stats["processed"] % 100 == 0:
            ratio = total_output_bytes / max(total_input_bytes, 1) * 100
            print(f"  ... processed {stats['processed']} videos "
                  f"({total_input_bytes / 1e9:.2f} GB → {total_output_bytes / 1e6:.1f} MB, "
                  f"{ratio:.1f}%)")

    # Summary
    ratio = total_output_bytes / max(total_input_bytes, 1) * 100
    print()
    print("=" * 60)
    print("[slim] Summary")
    print(f"  Processed:        {stats['processed']}")
    print(f"  Skipped (exists): {stats['skipped']}")
    print(f"  Errors:           {stats['errors']}")
    print(f"  Input size:       {total_input_bytes / 1e9:.2f} GB")
    print(f"  Output size:      {total_output_bytes / 1e6:.1f} MB")
    print(f"  Compression:      {ratio:.1f}%")
    print(f"  Output dir:       {output_dir}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create lightweight annotation PKLs for monocular 3D detector training."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--input_dir", type=str, default="bbox_annotations_3d_obb_camera_intrinsics",
        help="Name of the input annotation directory under world_annotations/"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_slim_annotations(
        ag_root_directory=args.ag_root_directory,
        input_dir_name=args.input_dir,
        overwrite=args.overwrite,
    )
