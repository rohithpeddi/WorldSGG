#!/usr/bin/env python3
"""
Attach camera intrinsics from the PromptHMR human pipeline to OBB camera-frame annotations.

Reads:
  - Human pipeline results:  {human_dir}/{video_id}/results.pkl   (joblib)
  - OBB camera annotations:  {ag_root}/world_annotations/bbox_annotations_3d_obb_camera/{stem}.pkl  (pickle)

Writes:
  - Merged output:  {ag_root}/world_annotations/bbox_annotations_3d_obb_camera_intrinsics/{stem}.pkl  (pickle)

The merged PKL is the original OBB camera annotation dict with an added "intrinsics" key:
    {
        "fx": float,
        "fy": float,
        "cx": float,
        "cy": float,
        "K":  np.ndarray (3x3),
        "source": "human_pipeline",
        "spec_median_focal": float or None,
    }

Usage:
    python attach_intrinsics.py \
        --ag_root_directory /data/rohith/ag \
        --human_dir /data2/rohith/ag/ag4D/human \
        [--overwrite]
"""
import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np


def extract_intrinsics_from_human_results(
    results: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Extract camera intrinsics from a loaded human pipeline results dict.

    Returns None if the required keys are missing.
    """
    camera = results.get("camera", None)
    if camera is None:
        return None

    img_focal = camera.get("img_focal", None)
    img_center = camera.get("img_center", None)
    if img_focal is None or img_center is None:
        return None

    fx = float(img_focal)
    fy = float(img_focal)  # human pipeline assumes fx == fy
    img_center = np.asarray(img_center, dtype=np.float64)
    cx = float(img_center[0])
    cy = float(img_center[1])

    K = np.eye(3, dtype=np.float64)
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy

    # Optional: SPEC neural camera calibration estimate
    spec_calib = results.get("spec_calib", None)
    spec_median_focal = None
    if spec_calib is not None and isinstance(spec_calib, dict):
        spec_median_focal = spec_calib.get("median_focal_length", None)
        if spec_median_focal is not None:
            spec_median_focal = float(spec_median_focal)

    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "K": K,
        "source": "human_pipeline",
        "spec_median_focal": spec_median_focal,
    }


def attach_intrinsics(
    ag_root_directory: str,
    human_dir: str,
    overwrite: bool = False,
) -> None:
    ag_root = Path(ag_root_directory)
    human_root = Path(human_dir)

    obb_camera_dir = ag_root / "world_annotations" / "bbox_annotations_3d_obb_camera"
    output_dir = ag_root / "world_annotations" / "bbox_annotations_3d_obb_camera_intrinsics"
    os.makedirs(output_dir, exist_ok=True)

    if not obb_camera_dir.exists():
        print(f"[attach_intrinsics] OBB camera annotation directory not found: {obb_camera_dir}")
        return

    # Discover all OBB camera annotation PKLs
    obb_pkl_paths = sorted(obb_camera_dir.glob("*.pkl"))
    if not obb_pkl_paths:
        print(f"[attach_intrinsics] No PKL files found in {obb_camera_dir}")
        return

    print(f"[attach_intrinsics] Found {len(obb_pkl_paths)} OBB camera annotation PKLs")
    print(f"[attach_intrinsics] Human results directory: {human_root}")
    print(f"[attach_intrinsics] Output directory: {output_dir}")
    print()

    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_human": 0, "skipped_no_intrinsics": 0, "errors": 0}

    for obb_pkl_path in obb_pkl_paths:
        stem = obb_pkl_path.stem  # e.g., "00MFE"
        video_id = f"{stem}.mp4"
        out_path = output_dir / f"{stem}.pkl"

        # Skip if already exists and not overwriting
        if out_path.exists() and not overwrite:
            stats["skipped_exists"] += 1
            continue

        # Load human pipeline results (joblib pkl)
        human_results_path = human_root / video_id / "results.pkl"
        if not human_results_path.exists():
            print(f"  [{video_id}] SKIP — no human results at {human_results_path}")
            stats["skipped_no_human"] += 1
            continue

        try:
            human_results = joblib.load(human_results_path)
        except Exception as e:
            print(f"  [{video_id}] ERROR loading human results: {e}")
            stats["errors"] += 1
            continue

        # Extract intrinsics
        intrinsics = extract_intrinsics_from_human_results(human_results)
        if intrinsics is None:
            print(f"  [{video_id}] SKIP — could not extract intrinsics from human results")
            stats["skipped_no_intrinsics"] += 1
            continue

        # Load OBB camera annotation (standard pickle)
        try:
            with open(obb_pkl_path, "rb") as f:
                obb_data = pickle.load(f)
        except Exception as e:
            print(f"  [{video_id}] ERROR loading OBB annotation: {e}")
            stats["errors"] += 1
            continue

        # Merge: add intrinsics to the OBB dict
        obb_data["intrinsics"] = intrinsics

        # Write merged output (standard pickle)
        try:
            with open(out_path, "wb") as f:
                pickle.dump(obb_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"  [{video_id}] ERROR writing output: {e}")
            stats["errors"] += 1
            continue

        stats["processed"] += 1
        if stats["processed"] % 100 == 0:
            print(f"  ... processed {stats['processed']} videos so far")

    # Summary
    print()
    print("=" * 60)
    print("[attach_intrinsics] Summary")
    print(f"  Processed:             {stats['processed']}")
    print(f"  Skipped (exists):      {stats['skipped_exists']}")
    print(f"  Skipped (no human):    {stats['skipped_no_human']}")
    print(f"  Skipped (no intrins):  {stats['skipped_no_intrinsics']}")
    print(f"  Errors:                {stats['errors']}")
    total = sum(stats.values())
    print(f"  Total PKLs scanned:    {total}")
    print("=" * 60)


def inspect_sample(ag_root_directory: str, human_dir: str, video_id: str = "00MFE.mp4") -> None:
    """Quick inspection of what would be merged for a single video."""
    ag_root = Path(ag_root_directory)
    human_root = Path(human_dir)
    stem = video_id[:-4]

    # Human intrinsics
    human_path = human_root / video_id / "results.pkl"
    if human_path.exists():
        results = joblib.load(human_path)
        intrinsics = extract_intrinsics_from_human_results(results)
        print(f"[{video_id}] Human pipeline intrinsics:")
        if intrinsics is not None:
            print(f"  fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}")
            print(f"  cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
            print(f"  spec_median_focal={intrinsics['spec_median_focal']}")
            print(f"  K=\n{intrinsics['K']}")
        else:
            print("  Could not extract intrinsics")
    else:
        print(f"[{video_id}] No human results at {human_path}")

    # OBB annotation
    obb_path = ag_root / "world_annotations" / "bbox_annotations_3d_obb_camera" / f"{stem}.pkl"
    if obb_path.exists():
        with open(obb_path, "rb") as f:
            obb_data = pickle.load(f)
        print(f"\n[{video_id}] OBB camera annotation keys: {list(obb_data.keys())}")
        ff = obb_data.get("frames_final", None)
        if ff is not None:
            print(f"  frames_final keys: {list(ff.keys())}")
            print(f"  num frame_stems: {len(ff.get('frame_stems', []))}")
    else:
        print(f"\n[{video_id}] No OBB camera annotation at {obb_path}")

    # Check output
    out_path = ag_root / "world_annotations" / "bbox_annotations_3d_obb_camera_intrinsics" / f"{stem}.pkl"
    if out_path.exists():
        with open(out_path, "rb") as f:
            merged = pickle.load(f)
        print(f"\n[{video_id}] Merged output keys: {list(merged.keys())}")
        if "intrinsics" in merged:
            intr = merged["intrinsics"]
            print(f"  intrinsics: fx={intr['fx']:.2f}, fy={intr['fy']:.2f}, "
                  f"cx={intr['cx']:.2f}, cy={intr['cy']:.2f}, "
                  f"source={intr['source']}, spec_median={intr['spec_median_focal']}")
    else:
        print(f"\n[{video_id}] No merged output yet at {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach camera intrinsics from human pipeline to OBB camera-frame annotations."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument("--human_dir", type=str, default="/data2/rohith/ag/ag4D/human")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing merged PKLs")
    parser.add_argument(
        "--inspect", type=str, default=None,
        help="Inspect a single video (e.g., '00MFE.mp4') instead of processing all"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.inspect:
        inspect_sample(
            ag_root_directory=args.ag_root_directory,
            human_dir=args.human_dir,
            video_id=args.inspect,
        )
    else:
        attach_intrinsics(
            ag_root_directory=args.ag_root_directory,
            human_dir=args.human_dir,
            overwrite=args.overwrite,
        )
