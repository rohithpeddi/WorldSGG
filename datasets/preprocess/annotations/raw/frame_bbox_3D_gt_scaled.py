#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from torch.utils.data import DataLoader

from dataloader.standard.action_genome.ag_dataset import StandardAG
from datasets.preprocess.annotations.raw.frame_bbox_3D_base import FrameToWorldAnnotationsBase, rerun_frame_vis_final_only


class FrameToWorldAnnotations(FrameToWorldAnnotationsBase):
    
    def save_video_3d_annotations_final(self, video_id: str, video_3dgt_updated: Dict[str, Any]) -> Path:
        out_path = self.bbox_3d_final_scaled_root_dir / f"{video_id[:-4]}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(video_3dgt_updated, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path

    def build_frames_final_and_store(
            self,
            video_id: str,
            *,
            overwrite: bool = False,
            points_dtype: np.dtype = np.float32,
    ) -> Optional[Path]:
        """
        Loads:
          - original points/cameras for annotated frames
          - bbox_annotations_3d PKL (video_3dgt)

        Produces:
          - video_3dgt_updated["frames_final"] with final points/cameras/bboxes/floor
        Writes:
          - to bbox_annotations_3d_final/<video_id[:-4]>.pkl
        """
        out_path = self.bbox_3d_final_scaled_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[frames_final][{video_id}] exists: {out_path} (overwrite=False). Skipping.")
            return out_path

        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[frames_final][{video_id}] missing original bbox_annotations_3d PKL. Skipping.")
            return None

        # WARNING: This assumes video_3dgt ALREADY has "frames_final".
        # If running on raw bb3D_generator output, this will KeyError.
        # This script likely expects frame_bbox_3D_gt.py to have run first
        # AND for get_video_3d_annotations to point to that output (which it currently doesn't by default).
        if "frames_final" not in video_3dgt:
             print(f"[frames_final][{video_id}] 'frames_final' key missing. Has frame_bbox_3D_gt.py been run? Skipping.")
             return None

        frame_stems = video_3dgt["frames_final"]["frame_stems"]
        camera_poses = video_3dgt["frames_final"]["camera_poses"]
        floor = video_3dgt["frames_final"]["floor"]
        bbox_frames = video_3dgt["frames_final"]["bbox_frames"]

        # Load original annotated-frame points/cameras
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        S, H, W, _ = points_world.shape

        # TODO: Apply scaling logic here if needed
        # 1. Get the shape for the image grid for the video id
        image_path = self.frame_annotated_dir_path / video_id / f"{frame_stems[0]}.png"
        if image_path.exists():
            img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if img is not None:
                H_img, W_img = img.shape[:2]
                # 2. Compare it with H, W of the loaded points
                scale_x = W / W_img
                scale_y = H / H_img
            else:
                print(f"[frames_final][{video_id}] Could not read image {image_path}")
        else:
             print(f"[frames_final][{video_id}] Image not found {image_path}")

        # 3. Re-adjust the 3D bboxes to the original scale of the image grid
        # 4. Store everything in frames_final_scaled
        bbox_frames_updated = bbox_frames

        # Updated PKL: keep original content intact, add frames_final + world_to_final
        video_3dgt_updated = dict(video_3dgt)
        video_3dgt_updated["frames_final"] = {
            "frame_stems": frame_stems,
            "camera_poses": camera_poses,
            "bbox_frames": bbox_frames_updated,
            "floor": floor,
        }

        saved_path = self.save_video_3d_annotations_final(video_id, video_3dgt_updated)
        print(f"[frames_final][{video_id}] wrote: {saved_path}")
        return saved_path


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