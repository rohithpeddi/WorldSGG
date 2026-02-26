#!/usr/bin/env python3
import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from torch.utils.data import DataLoader

from datasets.preprocess.annotations.raw.frame_bbox_3D_base import FrameToWorldAnnotationsBase, rerun_frame_vis_final_only

sys.path.insert(0, os.path.dirname(__file__) + "/..")

from dataloader.ag_dataset import StandardAG

from datasets.preprocess.annotations.annotation_utils import (
    _faces_u32,
)


class FrameToWorldOBBAnnotations(FrameToWorldAnnotationsBase):

    def visualize_final_only(self, video_id: str, *, app_id: str = "World4D-FinalOnly") -> None:
        final_pkl = self.bbox_3d_obb_final_root_dir / f"{video_id[:-4]}.pkl"
        if not final_pkl.exists():
            raise FileNotFoundError(f"Final PKL not found: {final_pkl}")

        with open(final_pkl, "rb") as f:
            rec = pickle.load(f)

        frames_final = rec.get("frames_final", None)
        if frames_final is None:
            raise ValueError(f"[final-only][{video_id}] frames_final missing in {final_pkl}")

        origin_world = rec["world_to_final"]["origin_world"]
        A = rec["world_to_final"]["A_world_to_final"]

        rgbd_info = self.get_video_rgbd_info(video_id, origin_world=origin_world, A=A)
        frames_final["colors"] = rgbd_info.get("colors", None)
        frames_final["conf"] = rgbd_info.get("conf", None)
        frames_final["points"] = rgbd_info["points"]

        rerun_frame_vis_final_only(
            video_id,
            frames_final=frames_final,
            frame_annotated_dir_path=self.frame_annotated_dir_path,
            app_id=app_id,
            is_obb=True
        )

    def get_video_rgbd_info(
            self,
            video_id,
            origin_world,
            A
    ) -> Optional[Dict[str, Any]]:
        """
        Override to load original points/cameras for a video.
        """
        # Load original annotated-frame points/cameras (WORLD frame)
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        points_dtype = np.float32

        S, H, W, _ = points_world.shape

        pts_flat = points_world.reshape(-1, 3)
        pts_final_flat = self._apply_world_to_final_points_row(pts_flat, origin_world=origin_world, A_world_to_final=A)
        points_final = pts_final_flat.reshape(S, H, W, 3).astype(points_dtype, copy=False)

        rgbd_info = {
            "points": points_final,
            "colors": P.get("colors", None),
            "conf": P.get("conf", None),
        }

        return rgbd_info

    # -------------------------------------------------------------------------
    # Build frames_final and store in updated PKL
    # -------------------------------------------------------------------------

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
        out_path = self.bbox_3d_obb_final_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[frames_final][{video_id}] exists: {out_path} (overwrite=False). Skipping.")
            return out_path

        video_3dgt_obb = self.get_video_3d_obb_annotations(video_id)
        if video_3dgt_obb is None:
            print(f"[frames_final][{video_id}] missing original bbox_annotations_3d PKL. Skipping.")
            return None

        gfs = video_3dgt_obb.get("global_floor_sim", None)
        if gfs is None:
            print(f"[frames_final][{video_id}] global_floor_sim missing in PKL; cannot build FINAL. Skipping.")
            return None

        global_floor_sim = (
            float(gfs["s"]),
            np.asarray(gfs["R"], dtype=np.float32),
            np.asarray(gfs["t"], dtype=np.float32),
        )
        Tinfo = self._compute_world_to_final(global_floor_sim=global_floor_sim)
        origin_world = Tinfo["origin_world"]
        A = Tinfo["A_world_to_final"]

        # Load original annotated-frame points/cameras
        P = self._load_original_points_for_video(video_id)
        points_world = np.asarray(P["points"], dtype=np.float32)  # (S,H,W,3)
        stems = P["frame_stems"]
        cams_world = P["camera_poses"]

        S, H, W, _ = points_world.shape

        # Points FINAL
        pts_flat = points_world.reshape(-1, 3)
        pts_final_flat = self._apply_world_to_final_points_row(pts_flat, origin_world=origin_world, A_world_to_final=A)
        points_final = pts_final_flat.reshape(S, H, W, 3).astype(points_dtype, copy=False)

        # Cameras FINAL
        cams_final = None
        if cams_world is not None:
            cams_final_list = []
            for i in range(min(S, cams_world.shape[0])):
                cams_final_list.append(
                    self._apply_world_to_final_camera_pose(
                        cams_world[i],
                        origin_world=origin_world,
                        A_world_to_final=A,
                    )
                )
            cams_final = np.stack(cams_final_list, axis=0).astype(np.float32)

        # BBoxes FINAL (corners_world -> corners_final)
        bbox_frames_final: Dict[str, Any] = {}
        frames_map = video_3dgt_obb.get("frames", None)
        if frames_map is not None:
            for frame_name, frame_rec in frames_map.items():
                objs = frame_rec.get("objects", [])
                if not objs:
                    continue
                out_objs = []
                for obj in objs:
                    bb = obj.get("obb_floor_parallel", None)
                    if bb is None or "corners_world" not in bb:
                        continue
                    corners_world = np.asarray(bb["corners_world"], dtype=np.float32)  # (8,3)
                    corners_final = self._apply_world_to_final_points_row(
                        corners_world, origin_world=origin_world, A_world_to_final=A
                    ).astype(np.float32)

                    out_obj = dict(obj)
                    out_obj["corners_final"] = corners_final
                    out_objs.append(out_obj)

                if out_objs:
                    bbox_frames_final[frame_name] = {"objects": out_objs}

        # Floor FINAL (optional)
        floor_final = None
        gv = video_3dgt_obb.get("gv", None)
        gf = video_3dgt_obb.get("gf", None)
        gc = video_3dgt_obb.get("gc", None)

        if gv is not None and gf is not None:
            gv0 = np.asarray(gv, dtype=np.float32)
            gf0 = _faces_u32(np.asarray(gf))

            # Move floor mesh into WORLD using global_floor_sim, then into FINAL
            s_g, R_g, t_g = global_floor_sim
            floor_world = s_g * (gv0 @ R_g.T) + t_g[None, :]
            floor_final_v = self._apply_world_to_final_points_row(
                floor_world, origin_world=origin_world, A_world_to_final=A
            ).astype(np.float32)

            floor_final = {"vertices": floor_final_v, "faces": gf0}
            if gc is not None:
                floor_final["colors"] = np.asarray(gc, dtype=np.uint8)

        # Updated PKL: keep original content intact, add frames_final + world_to_final
        video_3dgt_updated = dict(video_3dgt_obb)
        video_3dgt_updated["frames_final"] = {
            "frame_stems": stems,
            # "points": points_final,
            # "colors": P["colors"],
            # "conf": P["conf"],
            "camera_poses": cams_final,
            "bbox_frames": bbox_frames_final,
            "floor": floor_final,
        }
        # Also store the transform used
        video_3dgt_updated["world_to_final"] = {
            "origin_world": origin_world,
            "A_world_to_final": A,
        }

        saved_path = self.save_video_3d_obb_annotations_final(video_id, video_3dgt_updated)
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

    frame_to_world_generator = FrameToWorldOBBAnnotations(
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

    frame_to_world_generator = FrameToWorldOBBAnnotations(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )
    video_id = "00T1E.mp4"
    frame_to_world_generator.build_frames_final_and_store(video_id=video_id, overwrite=False)
    frame_to_world_generator.visualize_final_only(video_id=video_id, app_id="World4D-FinalOnly-Sample")


if __name__ == "__main__":
    main()
    # main_sample()