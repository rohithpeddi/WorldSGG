#!/usr/bin/env python3
"""
Download all manual corrections from Firebase and save as per-video PKL files.

=============================================================================
TRANSFORMATION ORDER — How to apply these corrections to go from raw world-
frame data to the corrected, gravity-aligned canonical frame:
=============================================================================

The full pipeline composes 4 transformations applied IN ORDER:

  x_canonical = T_XY ∘ T_delta ∘ T_auto(x_world)

Step 1: T_auto — Automatic Floor Alignment (from original PKL's global_floor_sim)
    - Extracts floor basis (t1, t2, normal) from R_g columns
    - Builds R_align = F^T where F = [t1 | t2 | n]
    - Applies mirror M = diag(-1, 1, 1)
    - A_world_to_final = M @ R_align
    - origin_world = t_g
    - x_final = A_world_to_final @ (x_world - origin_world)
    → NOT stored here (it's in the original PKL files).

Step 2: T_delta — Manual Floor Correction (floor_corrections)
    - delta_transform = {rx, ry, rz, tx, ty, tz, sx, sy, sz}
    - Rotation: Euler angles (radians) in intrinsic XYZ order → R_delta = Rx @ Ry @ Rz
    - Scale: uniform (sx = sy = sz), applied per-axis to vertices
    - Translation: displacement in the final coordinate frame
    - Applied to the floor mesh: v_corr = R_delta @ (s * v_final) + t_delta
    - The SAME delta should also be applied to bboxes and point clouds.

Step 3: T_XY — Automated XY-Plane Alignment (xy_alignments)
    - alignment_transform = {rx, ry, rz, tx, ty, tz}
    - Rodrigues rotation to align corrected floor normal → +Z axis
    - In-plane rotation to align local X-axis → world X-axis
    - Translation moves floor-origin intersection to world origin
    - R_XY = R_z(alpha) @ R_norm, t_XY = -R_XY @ p_intersection
    - Applied AFTER T_delta: x_xy = R_XY @ x_corr + t_XY

Step 4: Final Alignment — Pre-computed combined transform (final_alignments)
    - combined_transform: 4×4 homogeneous matrix = T_XY ∘ T_delta ∘ T_auto
    - If present, this REPLACES the need to manually compose steps 1–3.
    - Also stores pre-computed per-frame bbox corners and camera poses.
    - x_canonical = combined_transform @ [x_world; 1]

=============================================================================
OUTPUT PKL STRUCTURE (per video):
=============================================================================
{
    "video_id": str,
    "floor_correction": {                    # Step 2 — or None
        "delta_transform": {rx, ry, rz, tx, ty, tz, sx, sy, sz},
        "timestamp": str,
        "version": str,
    },
    "xy_alignment": {                        # Step 3 — or None
        "alignment_transform": {rx, ry, rz, tx, ty, tz},
        "floor_center": [x, y, z],
        "floor_normal": [nx, ny, nz],
        "timestamp": str,
        "version": str,
    },
    "final_alignment": {                     # Step 4 — or None
        "combined_transform": 4×4 list,
        "floor_transform": {...},
        "xy_alignment": {...},
        "frames": {frame_stem: [{id, label, center, corners, color}]},
        "camera_poses": [...],
        "timestamp": str,
        "version": str,
    },
    "download_metadata": {
        "downloaded_at": str,
        "has_floor_correction": bool,
        "has_xy_alignment": bool,
        "has_final_alignment": bool,
    },
}

=============================================================================

Usage:
    python download_manual_corrections.py                              # Download all
    python download_manual_corrections.py --video 001YG.mp4            # Single video
    python download_manual_corrections.py --list                       # List videos with corrections
    python download_manual_corrections.py --output-dir /path/to/out    # Custom output dir
"""

import argparse
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.FirebaseService import FirebaseService


class ManualCorrectionDownloader:
    """Downloads floor corrections, XY alignments, and final alignments from Firebase."""

    # Firebase paths
    FLOOR_CORRECTIONS_PATH = "worldframe_obb/floor_corrections"
    XY_ALIGNMENTS_PATH = "worldframe_obb/xy_alignments"
    FINAL_ALIGNMENTS_PATH = "worldframe_obb/final_alignments"

    def __init__(self, output_dir: Optional[str] = None):
        self.firebase = FirebaseService()

        # Default output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path("/data/rohith/ag/world_annotations/manual_corrections")

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"[ManualCorrectionDownloader] Output directory: {self.output_dir}")

    # ------------------------------------------------------------------
    # Enumerate videos that have ANY correction
    # ------------------------------------------------------------------

    def _get_video_keys_at_path(self, firebase_path: str) -> Set[str]:
        """Get the set of video keys (Firebase-sanitized) at a given path."""
        try:
            keys = self.firebase.get_keys(firebase_path)
            if keys:
                return set(keys)
        except Exception as e:
            print(f"[Warning] Could not list keys at {firebase_path}: {e}")
        return set()

    def get_all_corrected_video_keys(self) -> Dict[str, Set[str]]:
        """
        Return a dict mapping correction_type -> set of video keys
        that have data for that correction type.
        """
        floor_keys = self._get_video_keys_at_path(self.FLOOR_CORRECTIONS_PATH)
        xy_keys = self._get_video_keys_at_path(self.XY_ALIGNMENTS_PATH)
        final_keys = self._get_video_keys_at_path(self.FINAL_ALIGNMENTS_PATH)

        return {
            "floor_corrections": floor_keys,
            "xy_alignments": xy_keys,
            "final_alignments": final_keys,
        }

    def get_all_video_keys_union(self) -> List[str]:
        """Get the union of all video keys that have any correction."""
        by_type = self.get_all_corrected_video_keys()
        union = set()
        for keys in by_type.values():
            union |= keys
        return sorted(union)

    @staticmethod
    def video_key_to_id(video_key: str) -> str:
        """Convert Firebase key (e.g. '001YG') back to video_id (e.g. '001YG.mp4')."""
        return video_key.replace("_", ".") + ".mp4"

    @staticmethod
    def video_id_to_key(video_id: str) -> str:
        """Convert video_id (e.g. '001YG.mp4') to Firebase key (e.g. '001YG')."""
        return video_id.replace(".mp4", "").replace(".", "_")

    # ------------------------------------------------------------------
    # Download individual correction types
    # ------------------------------------------------------------------

    def download_floor_correction(self, video_key: str) -> Optional[Dict[str, Any]]:
        """Download floor correction (delta transform) for a video."""
        path = f"{self.FLOOR_CORRECTIONS_PATH}/{video_key}/latest"
        try:
            data = self.firebase.get_data(path)
            return data
        except Exception as e:
            print(f"  [Warning] Failed to fetch floor correction for {video_key}: {e}")
            return None

    def download_xy_alignment(self, video_key: str) -> Optional[Dict[str, Any]]:
        """Download XY-plane alignment for a video."""
        path = f"{self.XY_ALIGNMENTS_PATH}/{video_key}/latest"
        try:
            data = self.firebase.get_data(path)
            return data
        except Exception as e:
            print(f"  [Warning] Failed to fetch XY alignment for {video_key}: {e}")
            return None

    def download_final_alignment(self, video_key: str) -> Optional[Dict[str, Any]]:
        """Download pre-computed final alignment for a video."""
        path = f"{self.FINAL_ALIGNMENTS_PATH}/{video_key}/latest"
        try:
            data = self.firebase.get_data(path)
            return data
        except Exception as e:
            print(f"  [Warning] Failed to fetch final alignment for {video_key}: {e}")
            return None

    # ------------------------------------------------------------------
    # Process a single video
    # ------------------------------------------------------------------

    def process_video(self, video_key: str) -> bool:
        """
        Download all corrections for a single video and save as PKL.

        Args:
            video_key: Firebase-sanitized video key (e.g. '001YG')

        Returns:
            True if at least one correction was found and saved.
        """
        video_id = self.video_key_to_id(video_key)
        print(f"\n[Processing] {video_id} (key: {video_key})")

        # Download each correction type
        floor_correction = self.download_floor_correction(video_key)
        xy_alignment = self.download_xy_alignment(video_key)
        final_alignment = self.download_final_alignment(video_key)

        has_floor = floor_correction is not None
        has_xy = xy_alignment is not None
        has_final = final_alignment is not None

        if not (has_floor or has_xy or has_final):
            print(f"  [Skip] No corrections found for {video_id}")
            return False

        # Build the output record
        record = {
            "video_id": video_id,
            "video_key": video_key,
            "floor_correction": floor_correction,
            "xy_alignment": xy_alignment,
            "final_alignment": final_alignment,
            "download_metadata": {
                "downloaded_at": datetime.utcnow().isoformat() + "Z",
                "has_floor_correction": has_floor,
                "has_xy_alignment": has_xy,
                "has_final_alignment": has_final,
            },
        }

        # Save PKL
        output_path = self.output_dir / f"{video_key}.pkl"
        try:
            with open(output_path, "wb") as f:
                pickle.dump(record, f)
            print(f"  [Saved] {output_path}")
            corrections = []
            if has_floor:
                corrections.append("floor_correction")
            if has_xy:
                corrections.append("xy_alignment")
            if has_final:
                corrections.append("final_alignment")
            print(f"  [Contains] {', '.join(corrections)}")
            return True
        except Exception as e:
            print(f"  [Error] Failed to save PKL: {e}")
            return False

    # ------------------------------------------------------------------
    # Process all videos
    # ------------------------------------------------------------------

    def process_all_videos(self) -> Dict[str, bool]:
        """Download corrections for all videos that have any correction."""
        video_keys = self.get_all_video_keys_union()

        if not video_keys:
            print("[Info] No videos with corrections found in Firebase")
            return {}

        print(f"[Info] Found {len(video_keys)} videos with corrections")

        results = {}
        for video_key in video_keys:
            results[video_key] = self.process_video(video_key)

        # Summary
        success = sum(1 for v in results.values() if v)
        print(f"\n[Summary] Downloaded corrections for {success}/{len(video_keys)} videos")
        print(f"[Summary] Output directory: {self.output_dir}")

        return results

    # ------------------------------------------------------------------
    # List videos with corrections
    # ------------------------------------------------------------------

    def list_corrected_videos(self):
        """Print a summary of which videos have which corrections."""
        by_type = self.get_all_corrected_video_keys()

        floor_keys = by_type["floor_corrections"]
        xy_keys = by_type["xy_alignments"]
        final_keys = by_type["final_alignments"]
        all_keys = sorted(floor_keys | xy_keys | final_keys)

        print(f"\nVideos with manual corrections:")
        print(f"  Floor corrections: {len(floor_keys)}")
        print(f"  XY alignments:     {len(xy_keys)}")
        print(f"  Final alignments:  {len(final_keys)}")
        print(f"  Total unique:      {len(all_keys)}")
        print()

        # Table
        print(f"{'Video Key':<20} {'Floor':^7} {'XY':^7} {'Final':^7}")
        print("-" * 45)
        for key in all_keys:
            f = "✓" if key in floor_keys else "-"
            x = "✓" if key in xy_keys else "-"
            a = "✓" if key in final_keys else "-"
            print(f"{key:<20} {f:^7} {x:^7} {a:^7}")


def main():
    parser = argparse.ArgumentParser(
        description="Download manual corrections (floor, XY alignment, final alignment) from Firebase"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process a specific video ID (e.g., '001YG.mp4') or key (e.g., '001YG')"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List videos with corrections (don't download)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for PKL files (default: /data/rohith/ag/world_annotations/manual_corrections)"
    )
    args = parser.parse_args()

    downloader = ManualCorrectionDownloader(output_dir=args.output_dir)

    if args.list:
        downloader.list_corrected_videos()
        return

    if args.video:
        # Accept either video_id or video_key format
        video = args.video
        if video.endswith(".mp4"):
            video_key = ManualCorrectionDownloader.video_id_to_key(video)
        else:
            video_key = video
        success = downloader.process_video(video_key)
        sys.exit(0 if success else 1)
    else:
        results = downloader.process_all_videos()
        failed = [v for v, ok in results.items() if not ok]
        if failed:
            print(f"\n[Warning] {len(failed)} videos had no corrections")
            sys.exit(1)


if __name__ == "__main__":
    main()
