#!/usr/bin/env python3
"""
Corrected Frame BBox Generator
================================

Generates frame-level bounding boxes from corrected world bboxes.
Supports two output coordinate frames:
    1. FINAL (canonical) frame — floor-aligned, gravity-aligned
    2. CAMERA frame — per-frame camera coordinate system

Reads from: bbox_annotations_3d_obb_corrected/<video>.pkl
Writes to:
    - bbox_annotations_3d_obb_corrected_final/<video>.pkl
    - bbox_annotations_3d_obb_corrected_camera/<video>.pkl

Usage:
    python corrected_frame_bbox_generator.py                              # Process all
    python corrected_frame_bbox_generator.py --video 001YG.mp4            # Single video
    python corrected_frame_bbox_generator.py --corrections-only           # Only corrected videos
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from annotation_utils import (
    get_video_belongs_to_split,
    _faces_u32,
    _load_pkl_if_exists,
    _npz_open,
    _as_np,
)

from datasets.preprocess.annotations.raw.frame_bbox_3D_base import (
    FrameToWorldAnnotationsBase,
)


# =====================================================================
# CORRECTED FRAME BBOX GENERATOR
# =====================================================================

class CorrectedFrameBBoxGenerator(FrameToWorldAnnotationsBase):
    """
    Generates frame-level bounding boxes from corrected world bboxes.
    Extends FrameToWorldAnnotationsBase for shared utilities
    (point loading, camera transform, path conventions).
    """

    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str):
        super().__init__(ag_root_directory, dynamic_scene_dir_path)

        # Corrected bbox source directory
        self.bbox_3d_obb_corrected_root_dir = (
            self.world_annotations_root_dir / "bbox_annotations_3d_obb_corrected"
        )

        # Output directories for corrected frame-level bboxes
        self.bbox_3d_obb_corrected_final_root_dir = (
            self.world_annotations_root_dir / "bbox_annotations_3d_obb_corrected_final"
        )
        self.bbox_3d_obb_corrected_camera_root_dir = (
            self.world_annotations_root_dir / "bbox_annotations_3d_obb_corrected_camera"
        )

        os.makedirs(self.bbox_3d_obb_corrected_final_root_dir, exist_ok=True)
        os.makedirs(self.bbox_3d_obb_corrected_camera_root_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def get_corrected_video_3d_annotations(
        self, video_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load the corrected world bbox PKL."""
        pkl_path = self.bbox_3d_obb_corrected_root_dir / f"{video_id[:-4]}.pkl"
        return _load_pkl_if_exists(pkl_path)

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_corrected_world_to_final(
        corrected_floor_transform: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Build WORLD→FINAL transform from the corrected floor transform.

        The corrected_floor_transform already contains the combined 4×4 matrix
        that maps world→canonical. We extract R and t from it.

        Returns:
            origin_world: (3,)   — translation origin
            A_world_to_final: (3,3) — rotation matrix
        """
        T_4x4 = np.asarray(
            corrected_floor_transform["combined_transform_4x4"], dtype=np.float32
        )
        R = T_4x4[:3, :3]
        t = T_4x4[:3, 3]

        # For the _apply_world_to_final_points_row interface used in base class:
        #   p_final_row = (p_world_row - origin_row) @ A.T
        # We have: p_final = R @ p_world + t
        # Row-vector form: p_final_row = p_world_row @ R.T + t
        #                              = (p_world_row - 0) @ R.T + t
        # But the base class interface is:
        #   p_final_row = (p_world_row - origin_world) @ A.T
        # So: origin_world = -R_inv @ t, and A = R
        # Or equivalently: origin_world = 0, and we handle it differently.

        # Use a simpler direct transform approach
        return {
            "R": R,
            "t": t,
            "T_4x4": T_4x4,
        }

    @staticmethod
    def _apply_corrected_transform_points(
        pts_world: np.ndarray,
        *,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Transform points from world to final (canonical) frame.
        p_final = pts_world @ R.T + t
        """
        return (pts_world @ R.T) + t[None, :]

    @staticmethod
    def _apply_corrected_transform_camera_pose(
        T_world: np.ndarray,
        *,
        R: np.ndarray,
        t: np.ndarray,
    ) -> np.ndarray:
        """
        Transform camera-to-world pose into camera-to-final pose.

        T_world is (4,4) camera→world.
        Returns (4,4) camera→final.
        """
        T44 = np.eye(4, dtype=np.float32)
        T44[:3, :3] = R
        T44[:3, 3] = t
        return (T44 @ T_world).astype(np.float32)

    # ------------------------------------------------------------------
    # Build FINAL-frame output
    # ------------------------------------------------------------------

    def build_corrected_frames_final_and_store(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
        points_dtype: np.dtype = np.float32,
    ) -> Optional[Path]:
        """
        Build corrected FINAL-frame bounding boxes and points.

        Loads:
          - corrected world bbox PKL (bbox_annotations_3d_obb_corrected/)
          - original Pi3 points + camera poses

        Produces:
          - video_3dgt_updated["frames_final"] with corrected final-frame data
        Writes to:
          - bbox_annotations_3d_obb_corrected_final/<video>.pkl
        """
        out_path = self.bbox_3d_obb_corrected_final_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[corrected-final][{video_id}] exists, skipping.")
            return out_path

        # Load corrected world bboxes
        video_3dgt = self.get_corrected_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[corrected-final][{video_id}] no corrected world bbox PKL. Skipping.")
            return None

        cft = video_3dgt.get("corrected_floor_transform", None)
        if cft is None:
            print(f"[corrected-final][{video_id}] no corrected_floor_transform. Skipping.")
            return None

        # Build transform
        tinfo = self._compute_corrected_world_to_final(cft)
        R = tinfo["R"]
        t = tinfo["t"]

        # Load original points + cameras
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        stems = P["frame_stems"]
        cams_world = P["camera_poses"]

        S, H, W, _ = points_world.shape

        # Transform points to FINAL frame
        pts_flat = points_world.reshape(-1, 3)
        pts_final_flat = self._apply_corrected_transform_points(
            pts_flat, R=R, t=t
        )
        points_final = pts_final_flat.reshape(S, H, W, 3).astype(points_dtype)

        # Transform cameras to FINAL frame
        cams_final = None
        if cams_world is not None:
            cams_final_list = []
            for i in range(min(S, cams_world.shape[0])):
                cams_final_list.append(
                    self._apply_corrected_transform_camera_pose(
                        cams_world[i], R=R, t=t,
                    )
                )
            cams_final = np.stack(cams_final_list, axis=0).astype(np.float32)

        # Transform bbox corners: world → final
        bbox_frames_final: Dict[str, Any] = {}
        frames_map = video_3dgt.get("frames", None)
        if frames_map is not None:
            for frame_name, frame_rec in frames_map.items():
                objs = frame_rec.get("objects", [])
                if not objs:
                    continue
                out_objs = []
                for obj in objs:
                    out_obj = dict(obj)

                    # Transform AABB corners
                    aabb = obj.get("aabb_floor_aligned", None)
                    if aabb is not None and "corners_world" in aabb:
                        corners_world = np.asarray(aabb["corners_world"], dtype=np.float32)
                        corners_final = self._apply_corrected_transform_points(
                            corners_world, R=R, t=t,
                        ).astype(np.float32)
                        out_obj["aabb_final"] = {
                            "corners_final": corners_final,
                            "source": "corrected-aabb",
                        }

                    # Transform OBB floor-parallel corners
                    obb_fp = obj.get("obb_floor_parallel", None)
                    if obb_fp is not None and "corners_world" in obb_fp:
                        corners_world = np.asarray(obb_fp["corners_world"], dtype=np.float32)
                        corners_final = self._apply_corrected_transform_points(
                            corners_world, R=R, t=t,
                        ).astype(np.float32)
                        out_obj["obb_floor_parallel_final"] = {
                            "corners_final": corners_final,
                            "source": "corrected-obb-floor-parallel",
                        }

                    # Transform OBB arbitrary corners
                    obb_arb = obj.get("obb_arbitrary", None)
                    if obb_arb is not None and "corners_world" in obb_arb:
                        corners_world = np.asarray(obb_arb["corners_world"], dtype=np.float32)
                        corners_final = self._apply_corrected_transform_points(
                            corners_world, R=R, t=t,
                        ).astype(np.float32)
                        out_obj["obb_arbitrary_final"] = {
                            "corners_final": corners_final,
                            "source": "corrected-obb-arbitrary",
                        }

                    out_objs.append(out_obj)

                if out_objs:
                    bbox_frames_final[frame_name] = {"objects": out_objs}

        # Floor mesh in FINAL frame (optional)
        floor_final = None
        gv = video_3dgt.get("gv", None)
        gf = video_3dgt.get("gf", None)
        gc = video_3dgt.get("gc", None)
        original_gfs = cft.get("original_global_floor_sim", None)

        if gv is not None and gf is not None and original_gfs is not None:
            gv0 = np.asarray(gv, dtype=np.float32)
            gf0 = _faces_u32(np.asarray(gf))

            s_g = float(original_gfs["s"])
            R_g = np.asarray(original_gfs["R"], dtype=np.float32)
            t_g = np.asarray(original_gfs["t"], dtype=np.float32)

            # Floor mesh: local → world → final
            floor_world = s_g * (gv0 @ R_g.T) + t_g[None, :]
            floor_final_v = self._apply_corrected_transform_points(
                floor_world, R=R, t=t,
            ).astype(np.float32)

            floor_final = {"vertices": floor_final_v, "faces": gf0}
            if gc is not None:
                floor_final["colors"] = np.asarray(gc, dtype=np.uint8)

        # Assemble output PKL
        video_3dgt_updated = dict(video_3dgt)
        video_3dgt_updated["frames_final"] = {
            "frame_stems": stems,
            "camera_poses": cams_final,
            "bbox_frames": bbox_frames_final,
            "floor": floor_final,
        }
        video_3dgt_updated["world_to_final"] = {
            "R": R,
            "t": t,
            "T_4x4": tinfo["T_4x4"],
            "source": cft.get("source", "unknown"),
        }

        # Save
        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(video_3dgt_updated, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[corrected-final][{video_id}] saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Build CAMERA-frame output
    # ------------------------------------------------------------------

    def build_corrected_frames_camera_and_store(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Build corrected CAMERA-frame bounding boxes.

        For each annotated frame, transforms world bboxes + points
        into that frame's camera coordinate system.

        Writes to: bbox_annotations_3d_obb_corrected_camera/<video>.pkl
        """
        out_path = self.bbox_3d_obb_corrected_camera_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[corrected-cam][{video_id}] exists, skipping.")
            return out_path

        # Load corrected world bboxes
        video_3dgt = self.get_corrected_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[corrected-cam][{video_id}] no corrected world bbox PKL. Skipping.")
            return None

        # Load original points + cameras
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        stems = P["frame_stems"]
        cams_world = P["camera_poses"]

        if cams_world is None:
            print(f"[corrected-cam][{video_id}] no camera poses available. Skipping.")
            return None

        S, H, W, _ = points_world.shape

        # Build per-frame camera-space data
        cam_frames: Dict[str, Any] = {}
        frames_map = video_3dgt.get("frames", None)
        stem_to_idx = {stems[i]: i for i in range(len(stems))}

        for sidx, stem in enumerate(stems):
            if sidx >= cams_world.shape[0]:
                break

            # Camera extrinsics: camera→world is cams_world[sidx] (4×4)
            T_cam2world = cams_world[sidx].astype(np.float64)
            # World→camera: T_world2cam = inv(T_cam2world)
            T_world2cam = np.linalg.inv(T_cam2world).astype(np.float32)
            R_w2c = T_world2cam[:3, :3]
            t_w2c = T_world2cam[:3, 3]

            # Points in camera frame
            pts_hw3 = points_world[sidx]  # (H,W,3)
            pts_flat = pts_hw3.reshape(-1, 3)
            pts_cam = (pts_flat @ R_w2c.T) + t_w2c[None, :]
            pts_cam = pts_cam.reshape(H, W, 3).astype(np.float32)

            frame_data = {
                "points_cam": pts_cam,
                "T_world2cam": T_world2cam,
            }

            # Find matching frame in bbox annotations
            frame_name = f"{stem}.png"
            if frames_map is not None and frame_name in frames_map:
                frame_rec = frames_map[frame_name]
                objs = frame_rec.get("objects", [])
                cam_objs = []
                for obj in objs:
                    cam_obj = {
                        "label": obj.get("label"),
                        "gt_bbox_xyxy": obj.get("gt_bbox_xyxy"),
                    }

                    # Transform AABB corners to camera frame
                    aabb = obj.get("aabb_floor_aligned", None)
                    if aabb is not None and "corners_world" in aabb:
                        cw = np.asarray(aabb["corners_world"], dtype=np.float32)
                        cc = (cw @ R_w2c.T) + t_w2c[None, :]
                        cam_obj["aabb_cam"] = {
                            "corners_cam": cc.astype(np.float32),
                        }

                    # Transform OBB floor-parallel corners
                    obb_fp = obj.get("obb_floor_parallel", None)
                    if obb_fp is not None and "corners_world" in obb_fp:
                        cw = np.asarray(obb_fp["corners_world"], dtype=np.float32)
                        cc = (cw @ R_w2c.T) + t_w2c[None, :]
                        cam_obj["obb_floor_parallel_cam"] = {
                            "corners_cam": cc.astype(np.float32),
                        }

                    # Transform OBB arbitrary corners
                    obb_arb = obj.get("obb_arbitrary", None)
                    if obb_arb is not None and "corners_world" in obb_arb:
                        cw = np.asarray(obb_arb["corners_world"], dtype=np.float32)
                        cc = (cw @ R_w2c.T) + t_w2c[None, :]
                        cam_obj["obb_arbitrary_cam"] = {
                            "corners_cam": cc.astype(np.float32),
                        }

                    cam_objs.append(cam_obj)

                frame_data["objects"] = cam_objs

            cam_frames[stem] = frame_data

        # Assemble output
        output = {
            "video_id": video_id,
            "frame_stems": stems,
            "camera_frames": cam_frames,
            "corrected_floor_transform": video_3dgt.get("corrected_floor_transform"),
        }

        os.makedirs(out_path.parent, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump(output, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[corrected-cam][{video_id}] saved to {out_path}")
        return out_path

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def generate_all_final(
        self,
        dataloader,
        split: str,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """Generate all corrected FINAL-frame bboxes for a split."""
        results = {}
        for data in tqdm(dataloader, desc=f"Corrected Final [split={split}]"):
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue
            try:
                out = self.build_corrected_frames_final_and_store(
                    video_id, overwrite=overwrite,
                )
                results[video_id] = out is not None
            except Exception as e:
                print(f"[corrected-final][{video_id}] error: {e}")
                import traceback
                traceback.print_exc()
                results[video_id] = False
        return results

    def generate_all_camera(
        self,
        dataloader,
        split: str,
        *,
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """Generate all corrected CAMERA-frame bboxes for a split."""
        results = {}
        for data in tqdm(dataloader, desc=f"Corrected Camera [split={split}]"):
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue
            try:
                out = self.build_corrected_frames_camera_and_store(
                    video_id, overwrite=overwrite,
                )
                results[video_id] = out is not None
            except Exception as e:
                print(f"[corrected-cam][{video_id}] error: {e}")
                import traceback
                traceback.print_exc()
                results[video_id] = False
        return results

    def generate_from_corrections_only(
        self,
        *,
        mode: str = "both",  # "final", "camera", or "both"
        overwrite: bool = False,
    ) -> Dict[str, bool]:
        """
        Process only videos that have corrected world bboxes.
        Discovers videos from bbox_annotations_3d_obb_corrected/.
        """
        results = {}
        if not self.bbox_3d_obb_corrected_root_dir.exists():
            print(
                f"[corrected-frame] corrected bbox dir not found: "
                f"{self.bbox_3d_obb_corrected_root_dir}"
            )
            return results

        correction_files = sorted(self.bbox_3d_obb_corrected_root_dir.glob("*.pkl"))
        print(f"[corrected-frame] found {len(correction_files)} corrected bbox files")

        for pkl_path in tqdm(correction_files, desc=f"Corrected Frame [{mode}]"):
            video_id = pkl_path.stem + ".mp4"
            try:
                if mode in ("final", "both"):
                    self.build_corrected_frames_final_and_store(
                        video_id, overwrite=overwrite,
                    )
                if mode in ("camera", "both"):
                    self.build_corrected_frames_camera_and_store(
                        video_id, overwrite=overwrite,
                    )
                results[video_id] = True
            except Exception as e:
                print(f"[corrected-frame][{video_id}] error: {e}")
                import traceback
                traceback.print_exc()
                results[video_id] = False
        return results

    # ------------------------------------------------------------------
    # Visualization (from saved PKLs, no recomputation)
    # ------------------------------------------------------------------

    def visualize_from_saved(
        self,
        video_id: str,
        *,
        mode: str = "final",
        app_id: str = "Corrected-FrameBBox",
        img_maxsize: int = 480,
        vis_floor: bool = True,
    ) -> None:
        """Launch rerun visualization from a saved corrected frame bbox PKL.

        Parameters
        ----------
        mode : str
            ``"final"`` to visualize final-frame PKL,
            ``"world"`` to visualize the source world bbox PKL.
        """
        from corrected_bbox_vis import rerun_visualize_corrected_bboxes

        if mode == "final":
            pkl_path = self.bbox_3d_obb_corrected_final_root_dir / f"{video_id[:-4]}.pkl"
            frames_key = "frames_final.bbox_frames"
        else:
            pkl_path = self.bbox_3d_obb_corrected_root_dir / f"{video_id[:-4]}.pkl"
            frames_key = "frames"

        if not pkl_path.exists():
            raise FileNotFoundError(
                f"[vis] Missing corrected frame bbox PKL: {pkl_path}\n"
                f"Run generation first."
            )

        rerun_visualize_corrected_bboxes(
            video_id=video_id,
            pkl_path=str(pkl_path),
            dynamic_scene_dir_path=str(self.dynamic_scene_dir_path),
            idx_to_frame_idx_path_fn=self.idx_to_frame_idx_path,
            app_id=app_id,
            img_maxsize=img_maxsize,
            vis_floor=vis_floor,
            frames_key=frames_key,
        )


# =====================================================================
# CLI
# =====================================================================

def load_dataset(ag_root_directory: str):
    from dataloader.standard.action_genome.ag_dataset import StandardAG
    from torch.utils.data import DataLoader

    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    dataloader_train = DataLoader(
        train_dataset, shuffle=True, collate_fn=lambda b: b[0],
        pin_memory=False, num_workers=0,
    )
    dataloader_test = DataLoader(
        test_dataset, shuffle=False, collate_fn=lambda b: b[0], pin_memory=False,
    )
    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate corrected frame-level 3D bboxes from corrected world bboxes."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path", type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--video", type=str, default=None, help="Process single video")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--mode", type=str, default="both",
        choices=["final", "camera", "both"],
        help="Output mode: 'final' (canonical), 'camera', or 'both'",
    )
    parser.add_argument(
        "--corrections-only", action="store_true",
        help="Only process videos that have corrected world bboxes",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize saved corrected bboxes with rerun (requires --video)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    generator = CorrectedFrameBBoxGenerator(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    if args.visualize:
        if not args.video:
            print("ERROR: --visualize requires --video")
            return
        generator.visualize_from_saved(args.video, mode=args.mode)
        return

    if args.video:
        if args.mode in ("final", "both"):
            generator.build_corrected_frames_final_and_store(
                args.video, overwrite=args.overwrite,
            )
        if args.mode in ("camera", "both"):
            generator.build_corrected_frames_camera_and_store(
                args.video, overwrite=args.overwrite,
            )
        return

    # Default: process only videos that have corrected world bboxes
    # (skip-if-exists is handled inside build_corrected_frames_*_and_store)
    results = generator.generate_from_corrections_only(
        mode=args.mode, overwrite=args.overwrite,
    )
    success = sum(1 for v in results.values() if v)
    print(f"\n[Summary] {success}/{len(results)} videos processed successfully")


def main_sample():
    """Process a single sample video, then launch rerun visualization."""
    args = parse_args()
    video_id = args.video or "001YG.mp4"

    generator = CorrectedFrameBBoxGenerator(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    # Generate both final and camera (skips if already exists unless --overwrite)
    out_final = generator.build_corrected_frames_final_and_store(
        video_id, overwrite=args.overwrite,
    )
    out_cam = generator.build_corrected_frames_camera_and_store(
        video_id, overwrite=args.overwrite,
    )
    print(
        f"[main_sample] Generation result for {video_id}: "
        f"final={'OK' if out_final else 'SKIP'}, "
        f"camera={'OK' if out_cam else 'SKIP'}"
    )

    if out_final is not None:
        generator.visualize_from_saved(video_id, mode="final")


if __name__ == "__main__":
    # main()
    main_sample()

