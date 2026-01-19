#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader

from dataloader.standard.action_genome.ag_dataset import StandardAG
from datasets.preprocess.annotations.frame_bbox_3D_base import FrameToWorldAnnotationsBase, rerun_frame_vis_final_only


class FrameToWorldAnnotations(FrameToWorldAnnotationsBase):
    
    def build_frames_final_and_store(
        self,
        video_id: str,
        *,
        overwrite: bool = False,
        points_dtype: Any = None,  # kept for signature compatibility
    ) -> Optional[Path]:
        """
        Specialized builder for CAMERA-FRAME output.
        
        Loads:
          - original points/cameras (world frame)
          - bbox_annotations_3d PKL (world frame)
        
        Produces:
          - video_3dgt_updated["frames_final"] where:
              - points are in CAMERA frame of that frame
              - bboxes are in CAMERA frame of that frame
              - camera_poses are Identity (since we are in camera frame)
              - floor is NOT included (or would need per-frame transformation, usually not needed for mono-3d training)
        
        Writes:
          - to bbox_3d_camera_root_dir / <video_id>.pkl
        """
        import numpy as np
        import pickle
        
        out_path = self.bbox_3d_camera_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[frames_cam][{video_id}] exists: {out_path} (overwrite=False). Skipping.")
            return out_path

        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[frames_cam][{video_id}] missing original bbox_annotations_3d PKL. Skipping.")
            return None

        # Load original annotated-frame points/cameras (WORLD frame)
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        stems = P["frame_stems"]
        cams_world = P["camera_poses"]  # (S,4,4) or None
        
        if cams_world is None:
            print(f"[frames_cam][{video_id}] No camera poses found. Cannot transform to camera frame.")
            return None

        S, H, W, _ = points_world.shape
        
        # We will build new arrays for "final" (which here means CAMERA frame)
        points_cam_list = []
        
        # We also need to transform bboxes per frame
        bbox_frames_cam: Dict[str, Any] = {}
        frames_map = video_3dgt.get("frames", None)
        
        # Pre-calculate inverses of all cameras
        # T_cw = T_wc^-1
        # cams_world[i] is T_wc (camera-to-world) usually? 
        # Let's verify convention. 
        # In _load_original_points_for_video, we just load what's in npz.
        # Usually "camera_poses" in these datasets are T_world_camera (camera-to-world).
        # i.e. p_world = T_wc @ p_cam
        # So p_cam = T_wc^-1 @ p_world.
        
        for i in range(S):
            # 1. Get T_wc for this frame
            T_wc = cams_world[i].astype(np.float32)
            if T_wc.shape == (3, 4):
                T_wc_4x4 = np.eye(4, dtype=np.float32)
                T_wc_4x4[:3, :4] = T_wc
                T_wc = T_wc_4x4
            
            # 2. Compute T_cw = inv(T_wc)
            # This transforms WORLD points -> CAMERA points
            T_cw = np.linalg.inv(T_wc)
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            
            # 3. Transform points: p_cam = (p_world - t_wc_origin) @ R_wc.T ... wait
            # Standard: p_cam = R_cw @ p_world + t_cw
            # For row vectors: p_cam_row = p_world_row @ R_cw.T + t_cw
            
            pts_w_frame = points_world[i].reshape(-1, 3)
            pts_c_frame = pts_w_frame @ R_cw.T + t_cw[None, :]
            points_cam_list.append(pts_c_frame.reshape(H, W, 3))
            
            # 4. Transform bboxes for this frame
            # We need to find the frame name corresponding to index i
            # stems[i] is "000123"
            frame_name = f"{stems[i]}.png"
            
            if frames_map and frame_name in frames_map:
                frame_rec = frames_map[frame_name]
                objs = frame_rec.get("objects", [])
                out_objs = []
                for obj in objs:
                    bb = obj.get("aabb_floor_aligned", None)
                    # Note: "aabb_floor_aligned" actually stores "corners_world" inside it 
                    # (despite the name, it usually has a 'corners_world' field).
                    if bb is None or "corners_world" not in bb:
                        continue
                    
                    corners_w = np.asarray(bb["corners_world"], dtype=np.float32) # (8,3)
                    
                    # Transform corners to camera frame
                    corners_c = corners_w @ R_cw.T + t_cw[None, :]
                    
                    out_obj = dict(obj)
                    out_obj["corners_final"] = corners_c.astype(np.float32)
                    out_objs.append(out_obj)
                
                if out_objs:
                    bbox_frames_cam[frame_name] = {"objects": out_objs}

        points_cam = np.stack(points_cam_list, axis=0).astype(np.float32)
        
        # Construct output structure
        # We set camera_poses to Identity because points are already in camera frame
        # Or we can omit it. Let's set to Identity to be explicit.
        cams_identity = np.tile(np.eye(4, dtype=np.float32), (S, 1, 1))
        
        frames_final = {
            "frame_stems": stems,
            "points": points_cam,
            "colors": P.get("colors", None),
            "conf": P.get("conf", None),
            "camera_poses": cams_identity, 
            "bbox_frames": bbox_frames_cam,
            "floor": None # Floor is not transformed to camera frame (it would move every frame)
        }
        
        video_3dgt_updated = dict(video_3dgt)
        video_3dgt_updated["frames_final"] = frames_final
        
        # We don't really have a single "world_to_final" transform here since it varies per frame.
        # We can store a flag or metadata indicating this is camera-frame data.
        video_3dgt_updated["coordinate_system"] = "camera"
        
        with open(out_path, "wb") as f:
            pickle.dump(video_3dgt_updated, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        return out_path


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
            "World4D GT helper: "
            "(a) inspect 3D bbox annotations, "
            "(b) visualize original Pi3 outputs (points + floor + frames + camera + 3D boxes) "
            "for annotated frames."
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

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    frame_to_world_generator.generate_gt_world_3D_bb_annotations(dataloader=dataloader_train, split=args.split)
    frame_to_world_generator.generate_gt_world_3D_bb_annotations(dataloader=dataloader_test, split=args.split)


def main_sample():
    """
    Simple entry point to visualize original Pi3 point clouds + floor mesh
    + coordinate frames + camera frustum + 3D bounding boxes for a single video.
    Adjust `video_id` as needed.
    """
    args = parse_args()

    frame_to_world_generator = FrameToWorldAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    video_id = "00T1E.mp4"
    # frame_to_world_generator.build_frames_final_and_store(video_id=video_id, overwrite=False)
    frame_to_world_generator.visualize_final_only(video_id=video_id, app_id="World4D-FinalOnly-Sample")


if __name__ == "__main__":
    # main()
    main_sample()