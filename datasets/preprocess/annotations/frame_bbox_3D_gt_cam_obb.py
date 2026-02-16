#!/usr/bin/env python3
"""
OBB (Oriented Bounding Box) Camera Frame Visualization.

This script is similar to frame_bbox_3D_gt_cam.py but uses Oriented Bounding Boxes (OBB)
instead of Axis-Aligned Bounding Boxes (AABB). All data is transformed to camera coordinates.
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader

import cv2
import numpy as np
import pickle
import rerun as rr

from dataloader.standard.action_genome.ag_dataset import StandardAG
from datasets.preprocess.annotations.frame_bbox_3D_base import FrameToWorldAnnotationsBase


def rerun_frame_vis_camera_obb(
    video_id: str,
    *,
    frames_final: Dict[str, Any],
    frame_annotated_dir_path: Path,
    img_maxsize: int = 320,
    app_id: str = "World4D-CameraFrame-OBB",
    min_conf_default: float = 1e-6,
) -> None:
    """
    Camera-frame visualization with OBB (Oriented Bounding Boxes):
      - Points, bboxes are in CAMERA coordinate frame
      - Camera is at origin, looking along +Z (OpenCV convention: X-right, Y-down, Z-forward)
      - No floor is displayed (it would move every frame in camera coords)
      - Uses OBB corners instead of AABB
    
    Expected frames_final schema:
      frames_final = {
        "frame_stems": List[str],
        "points": (S,H,W,3) float32/float16,
        "colors": (S,H,W,3) uint8 (optional),
        "conf":   (S,H,W) float32 (optional),
        "camera_poses": (S,4,4) float32 - Identity matrices in camera frame,
        "bbox_frames": Dict[str, {"objects": List[{label,color,corners_final,(...)}]}] (optional),
        "floor": None (not used in camera frame)
      }
    """
    rr.init(app_id, spawn=True)
    
    # Use RDF (Right-Down-Forward) for OpenCV camera convention
    rr.log("/", rr.ViewCoordinates.RDF)
    BASE = "camera_frame"
    rr.log(BASE, rr.ViewCoordinates.RDF, static=True)

    # OBB corners from cv2.boxPoints: 0-3 are the bottom face going around perimeter,
    # 4-7 are the top face directly above (i.e., 4 is above 0, 5 is above 1, etc.)
    cuboid_edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]

    def _get_image_for_stem(stem: str) -> Optional[np.ndarray]:
        img_path = frame_annotated_dir_path / video_id / f"{stem}.png"
        if not img_path.exists():
            return None
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            return None
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Static axes at camera origin
    axis_len = 0.5
    rr.log(
        f"{BASE}/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["+X (Right)", "+Y (Down)", "+Z (Forward)"],
        ),
        static=True,
    )

    points_S = np.asarray(frames_final["points"])
    colors_S = frames_final.get("colors", None)
    conf_S = frames_final.get("conf", None)
    stems_S = frames_final["frame_stems"]
    bbox_frames = frames_final.get("bbox_frames", None)

    S, H_grid, W_grid, _ = points_S.shape

    for vis_t in range(S):
        stem = stems_S[vis_t]
        rr.set_time_sequence("frame", vis_t)
        
        # Clear previous frame's entities (especially bboxes which vary per frame)
        rr.log(f"{BASE}/bboxes", rr.Clear(recursive=True))

        # Points (already in camera frame)
        pts = points_S[vis_t].reshape(-1, 3)
        cols = colors_S[vis_t].reshape(-1, 3) if colors_S is not None else None
        conf_flat = conf_S[vis_t].reshape(-1) if conf_S is not None else None

        if conf_flat is not None:
            good = np.isfinite(conf_flat)
            cfs_valid = conf_flat[good]
            thr = min_conf_default if cfs_valid.size == 0 else max(min_conf_default, np.percentile(cfs_valid, 5))
            keep = (conf_flat >= thr) & np.isfinite(pts).all(axis=1)
        else:
            keep = np.isfinite(pts).all(axis=1)

        pts_keep = pts[keep]
        kwargs_pts = {}
        if cols is not None:
            kwargs_pts["colors"] = cols[keep].astype(np.uint8)

        if pts_keep.shape[0] > 0:
            rr.log(f"{BASE}/points", rr.Points3D(pts_keep, **kwargs_pts))

        # Camera frustum at origin (camera is at origin in camera frame)
        rr.log(
            f"{BASE}/camera/frustum",
            rr.Pinhole(
                fov_y=0.7853982,  # ~45 degrees
                aspect_ratio=float(W_grid) / float(H_grid),
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.1,
            ),
        )

        # OBBs (already in camera frame)
        if bbox_frames is not None:
            frame_name = f"{stem}.png"
            rec = bbox_frames.get(frame_name, None)
            if rec is not None:
                for bi, obj in enumerate(rec.get("objects", [])):
                    corners_final = obj.get("obb_corners_final", None)
                    if corners_final is None:
                        continue
                    corners_final = np.asarray(corners_final, dtype=np.float32)  # (8,3)
                    # Use green color for OBB to distinguish from AABB (orange)
                    col = obj.get("color", [255, 180, 0])
                    label = obj.get("label", f"obj_{bi}")
                    
                    strips = [corners_final[[e0, e1], :] for (e0, e1) in cuboid_edges]
                    rr.log(
                        f"{BASE}/bboxes/{label}_{bi}",
                        rr.LineStrips3D(strips=strips, colors=[col] * len(strips)),
                    )

        # Original RGB image
        img = _get_image_for_stem(stem)
        if img is not None:
            rr.log(f"{BASE}/image", rr.Image(img))

    print(f"[camera-frame-obb-vis] running for {video_id}. Scrub the 'frame' timeline in Rerun.")


class FrameToWorldAnnotationsOBB(FrameToWorldAnnotationsBase):
    """
    OBB-based camera frame annotation builder.
    
    Similar to FrameToWorldAnnotations in frame_bbox_3D_gt_cam.py, but uses
    Oriented Bounding Boxes (OBB) instead of AABB.
    """
    
    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str):
        super().__init__(ag_root_directory, dynamic_scene_dir_path)
        # Create a separate directory for OBB camera annotations
        self.bbox_3d_obb_camera_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_obb_camera"
        import os
        os.makedirs(self.bbox_3d_obb_camera_root_dir, exist_ok=True)

    def get_video_rgbd_info_cam(self, video_id: str) -> Optional[Dict[str, Any]]:
        """
        Override to load original points/cameras for a video.
        """
        video_3dgt = self.get_video_3d_obb_annotations(video_id)

        # Load original annotated-frame points/cameras (WORLD frame)
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        stems = P["frame_stems"]
        cams_world = P["camera_poses"]  # (S,4,4) or None

        S, H, W, _ = points_world.shape

        # We will build new arrays for "final" (which here means CAMERA frame)
        points_cam_list = []

        # We also need to transform OBB bboxes per frame
        bbox_frames_cam: Dict[str, Any] = {}
        frames_map = video_3dgt.get("frames", None)

        for i in range(S):
            # 1. Get T_wc for this frame (camera-to-world transform)
            T_wc = cams_world[i].astype(np.float32)
            if T_wc.shape == (3, 4):
                T_wc_4x4 = np.eye(4, dtype=np.float32)
                T_wc_4x4[:3, :4] = T_wc
                T_wc = T_wc_4x4

            # 2. Compute T_cw = inv(T_wc) - transforms WORLD points -> CAMERA points
            T_cw = np.linalg.inv(T_wc)
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]

            # 3. Transform points: p_cam = p_world @ R_cw.T + t_cw
            pts_w_frame = points_world[i].reshape(-1, 3)
            pts_c_frame = pts_w_frame @ R_cw.T + t_cw[None, :]
            points_cam_list.append(pts_c_frame.reshape(H, W, 3))

            # 4. Transform OBB bboxes for this frame
            frame_name = f"{stems[i]}.png"

            if frames_map and frame_name in frames_map:
                frame_rec = frames_map[frame_name]
                objs = frame_rec.get("objects", [])
                out_objs = []
                for obj in objs:
                    obb = obj.get("obb_floor_parallel", None)
                    if obb is None or "corners_world" not in obb:
                        continue

                    corners_w = np.asarray(obb["corners_world"], dtype=np.float32)  # (8,3)

                    # Transform OBB corners to camera frame
                    corners_c = corners_w @ R_cw.T + t_cw[None, :]

                    out_obj = dict(obj)
                    out_obj["obb_corners_final"] = corners_c.astype(np.float32)
                    out_objs.append(out_obj)

                if out_objs:
                    bbox_frames_cam[frame_name] = {"objects": out_objs}

        points_cam = np.stack(points_cam_list, axis=0).astype(np.float32)

        rgbd_info = {
            "points": points_cam,
            "colors": P.get("colors", None),
            "conf": P.get("conf", None),
        }

        return rgbd_info
    
    def build_frames_final_and_store(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
        points_dtype: Any = None,  # kept for signature compatibility
    ) -> Optional[Path]:
        """
        Specialized builder for CAMERA-FRAME OBB output.
        
        Loads:
          - original points/cameras (world frame)
          - bbox_annotations_3d PKL (world frame) - uses OBB corners
        
        Produces:
            - video_3dgt_updated[ "frames_final"] where:
              - points are in CAMERA frame of that frame
              - OBB bboxes are in CAMERA frame of that frame
              - camera_poses are Identity (since we are in camera frame)
              - floor is NOT included
        
        Writes:
          - to bbox_3d_obb_camera_root_dir / <video_id>.pkl
        """
        out_path = self.bbox_3d_obb_camera_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[frames_cam_obb][{video_id}] exists: {out_path} (overwrite=False). Skipping.")
            return out_path

        video_3dgt = self.get_video_3d_obb_annotations(video_id)
        if video_3dgt is None:
            print(f"[frames_cam_obb][{video_id}] missing original bbox_annotations_3d PKL. Skipping.")
            return None

        # Load original annotated-frame points/cameras (WORLD frame)
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        stems = P["frame_stems"]
        cams_world = P["camera_poses"]  # (S,4,4) or None
        
        if cams_world is None:
            print(f"[frames_cam_obb][{video_id}] No camera poses found. Cannot transform to camera frame.")
            return None

        S, H, W, _ = points_world.shape
        
        # We will build new arrays for "final" (which here means CAMERA frame)
        points_cam_list = []
        
        # We also need to transform OBB bboxes per frame
        bbox_frames_cam: Dict[str, Any] = {}
        frames_map = video_3dgt.get("frames", None)
        
        for i in range(S):
            # 1. Get T_wc for this frame (camera-to-world transform)
            T_wc = cams_world[i].astype(np.float32)
            if T_wc.shape == (3, 4):
                T_wc_4x4 = np.eye(4, dtype=np.float32)
                T_wc_4x4[:3, :4] = T_wc
                T_wc = T_wc_4x4
            
            # 2. Compute T_cw = inv(T_wc) - transforms WORLD points -> CAMERA points
            T_cw = np.linalg.inv(T_wc)
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            
            # 3. Transform points: p_cam = p_world @ R_cw.T + t_cw
            pts_w_frame = points_world[i].reshape(-1, 3)
            pts_c_frame = pts_w_frame @ R_cw.T + t_cw[None, :]
            points_cam_list.append(pts_c_frame.reshape(H, W, 3))
            
            # 4. Transform OBB bboxes for this frame
            frame_name = f"{stems[i]}.png"
            
            if frames_map and frame_name in frames_map:
                frame_rec = frames_map[frame_name]
                objs = frame_rec.get("objects", [])
                out_objs = []
                for obj in objs:
                    obb = obj.get("obb_floor_parallel", None)
                    if obb is None or "corners_world" not in obb:
                        continue
                    
                    corners_w = np.asarray(obb["corners_world"], dtype=np.float32)  # (8,3)
                    
                    # Transform OBB corners to camera frame
                    corners_c = corners_w @ R_cw.T + t_cw[None, :]
                    
                    out_obj = dict(obj)
                    out_obj["obb_corners_final"] = corners_c.astype(np.float32)
                    out_objs.append(out_obj)
                
                if out_objs:
                    bbox_frames_cam[frame_name] = {"objects": out_objs}

        points_cam = np.stack(points_cam_list, axis=0).astype(np.float32)
        
        # Construct output structure with identity camera poses
        cams_identity = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
        
        frames_final = {
            "frame_stems": stems,
            # "points": points_cam,
            # "colors": P.get("colors", None),
            # "conf": P.get("conf", None),
            "camera_poses": cams_identity, 
            "bbox_frames": bbox_frames_cam,
            "floor": None  # Floor is not transformed to camera frame
        }
        
        video_3dgt_updated = dict(video_3dgt)
        video_3dgt_updated["frames_final"] = frames_final
        video_3dgt_updated["coordinate_system"] = "camera"
        video_3dgt_updated["bbox_type"] = "obb"  # Mark as OBB
        
        with open(out_path, "wb") as f:
            pickle.dump(video_3dgt_updated, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"[frames_cam_obb][{video_id}] wrote: {out_path}")
        return out_path

    def visualize_camera_frame_obb(self, video_id: str, *, app_id: str = "World4D-CameraFrame-OBB") -> None:
        """
        Visualize camera-frame OBB data (points, OBB bboxes in camera coordinate frame).
        Loads from bbox_3d_obb_camera_root_dir.
        """
        camera_pkl = self.bbox_3d_obb_camera_root_dir / f"{video_id[:-4]}.pkl"
        if not camera_pkl.exists():
            raise FileNotFoundError(f"Camera-frame OBB PKL not found: {camera_pkl}")

        with open(camera_pkl, "rb") as f:
            rec = pickle.load(f)

        frames_final = rec.get("frames_final", None)
        if frames_final is None:
            raise ValueError(f"[camera-frame-obb][{video_id}] frames_final missing in {camera_pkl}")

        rgbd_info = self.get_video_rgbd_info_cam(video_id)
        frames_final["colors"] = rgbd_info.get("colors", None)
        frames_final["conf"] = rgbd_info.get("conf", None)
        frames_final["points"] = rgbd_info["points"]

        rerun_frame_vis_camera_obb(
            video_id,
            frames_final=frames_final,
            frame_annotated_dir_path=self.frame_annotated_dir_path,
            app_id=app_id,
        )


# --------------------------------------------------------------------------------------
# Dataset + CLI
# --------------------------------------------------------------------------------------


def load_dataset(ag_root_directory: str):
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
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "World4D GT OBB helper: "
            "(a) inspect 3D OBB annotations, "
            "(b) visualize OBB bounding boxes in camera coordinate frame."
        )
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument("--split", type=str, default="04")
    return parser.parse_args()


def main():
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotationsOBB(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    frame_to_world_generator.generate_gt_world_3D_bb_annotations(dataloader=dataloader_train, split=args.split)
    frame_to_world_generator.generate_gt_world_3D_bb_annotations(dataloader=dataloader_test, split=args.split)


def main_sample():
    """
    Simple entry point to visualize camera-frame OBB point clouds
    + coordinate frames + camera frustum + 3D OBB bounding boxes for a single video.
    All data is in CAMERA coordinate frame.
    Adjust `video_id` as needed.
    """
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotationsOBB(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    video_id = "00MFE.mp4"
    # frame_to_world_generator.build_frames_final_and_store(video_id=video_id, overwrite=False)
    frame_to_world_generator.visualize_camera_frame_obb(video_id=video_id, app_id="World4D-CameraFrame-OBB-Sample")


if __name__ == "__main__":
    # main()
    main_sample()
