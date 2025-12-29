#!/usr/bin/env python3
import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
from torch.utils.data import DataLoader
from tqdm import tqdm

from annotation_utils import (
    get_video_belongs_to_split,
    _load_pkl_if_exists,
    _npz_open,
    _faces_u32,
    _is_empty_array,
)
from dataloader.standard.action_genome.ag_dataset import StandardAG


def rerun_frame_vis_final_only(
    video_id: str,
    *,
    frames_final: Dict[str, Any],
    frame_annotated_dir_path: Path,
    img_maxsize: int = 320,
    app_id: str = "World4D-FinalOnly",
    min_conf_default: float = 1e-6,
) -> None:
    """
    FINAL-ONLY viewer:
      - Assumes points/cameras/bboxes/floor are ALREADY in FINAL coords.
      - Performs NO transformations internally.

    Expected frames_final schema (stored in updated video_3dgt):
      frames_final = {
        "frame_stems": List[str],
        "points": (S,H,W,3) float32/float16,
        "colors": (S,H,W,3) uint8 (optional),
        "conf":   (S,H,W) float32 (optional),
        "camera_poses": (S,4,4) float32 (optional),
        "bbox_frames": Dict[str, {"objects": List[{label,color,corners_final,(...)}]}] (optional),
        "floor": {"vertices": (V,3), "faces": (F,3), "colors": (V,3) optional} (optional)
      }
    """
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)
    BASE = "world_final"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
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
        return img

    # Static axes
    axis_len = 0.5
    rr.log(
        f"{BASE}/axes",
        rr.Arrows3D(
            origins=[[0, 0, 0]] * 3,
            vectors=[[axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]],
            colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            labels=["+X", "+Y", "+Z"],
        ),
        timeless=True,
    )

    # Floor (already final)
    floor = frames_final["floor"]
    if floor is not None:
        v = np.asarray(floor["vertices"], dtype=np.float32)
        f = _faces_u32(np.asarray(floor["faces"]))
        kwargs = {}
        if floor.get("colors", None) is not None:
            kwargs["vertex_colors"] = np.asarray(floor["colors"], dtype=np.uint8)
        else:
            kwargs["albedo_factor"] = [160, 160, 160]
        rr.log(f"{BASE}/floor", rr.Mesh3D(vertex_positions=v, triangle_indices=f, **kwargs), timeless=True)

    points_S = np.asarray(frames_final["points"])
    colors_S = frames_final["colors"]
    conf_S = frames_final["conf"]
    camera_poses_S = frames_final["camera_poses"]
    stems_S = frames_final["frame_stems"]
    bbox_frames = frames_final["bbox_frames"]

    S, H_grid, W_grid, _ = points_S.shape

    for vis_t in range(S):
        stem = stems_S[vis_t]
        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # Points (already final)
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

        # Camera (already final)
        if camera_poses_S is not None and vis_t < camera_poses_S.shape[0]:
            T = np.asarray(camera_poses_S[vis_t], dtype=np.float32)
            if T.shape == (3, 4):
                T4 = np.eye(4, dtype=np.float32)
                T4[:3, :4] = T
                T = T4
            cam_origin = T[:3, 3]
            R_cam = T[:3, :3]

            rr.log(
                f"{BASE}/camera/frustum",
                rr.Pinhole(
                    fov_y=0.7853982,
                    aspect_ratio=float(W_grid) / float(H_grid),
                    camera_xyz=rr.ViewCoordinates.RUB,
                    image_plane_distance=0.1,
                ),
                rr.Transform3D(translation=cam_origin.tolist(), mat3x3=R_cam),
            )

        # BBoxes (already final)
        if bbox_frames is not None:
            frame_name = f"{stem}.png"
            rec = bbox_frames.get(frame_name, None)
            if rec is not None:
                for bi, obj in enumerate(rec.get("objects", [])):
                    corners_final = np.asarray(obj["corners_final"], dtype=np.float32)  # (8,3)
                    col = obj.get("color", [255, 180, 0])
                    strips = [corners_final[[e0, e1], :] for (e0, e1) in cuboid_edges]
                    rr.log(
                        f"{BASE}/bboxes/bbox_{bi}",
                        rr.LineStrips3D(strips=strips, colors=[col] * len(strips)),
                    )

        # Original RGB image (not transformed)
        img = _get_image_for_stem(stem)
        if img is not None:
            rr.log(f"{BASE}/image", rr.Image(img))

    print(f"[final-only-vis] running for {video_id}. Scrub the 'frame' timeline in Rerun.")

# ----------------------------------------------------------------
# FrameToWorldAnnotations
#   - loads 3D bbox annotations (.pkl produced by BBox3DGenerator)
#   - can visualize ORIGINAL Pi3 points + floor mesh + 3D boxes for annotated frames
# --------------------------------------------------------------------------------------


class FrameToWorldAnnotations:

    def __init__(self, ag_root_directory: str, dynamic_scene_dir_path: str):
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)

        self.dataset_classnames = [
            "__background__",
            "person",
            "bag",
            "bed",
            "blanket",
            "book",
            "box",
            "broom",
            "chair",
            "closet/cabinet",
            "clothes",
            "cup/glass/bottle",
            "dish",
            "door",
            "doorknob",
            "doorway",
            "floor",
            "food",
            "groceries",
            "laptop",
            "light",
            "medicine",
            "mirror",
            "paper/notebook",
            "phone/camera",
            "picture",
            "pillow",
            "refrigerator",
            "sandwich",
            "shelf",
            "shoe",
            "sofa/couch",
            "table",
            "television",
            "towel",
            "vacuum",
            "window",
        ]
        self.name_to_catid = {
            name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0
        }
        self.catid_to_name_map = {v: k for k, v in self.name_to_catid.items()}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        # Detection directories (same layout as BBox3DGenerator)
        self.dynamic_detections_root_path = (
            self.ag_root_directory / "detection" / "gdino_bboxes"
        )
        self.static_detections_root_path = (
            self.ag_root_directory / "detection" / "gdino_bboxes_static"
        )

        # Annotated frame and sampling info
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = (
            self.ag_root_directory / "sampled_frames_idx"
        )

        # World annotation dirs (3D bboxes already generated by BBox3DGenerator)
        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        self.bbox_4d_root_dir = self.world_annotations_root_dir / "bbox_annotations_4d"
        self.bbox_3d_final_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_final"
        self.bbox_3d_final_scaled_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d_final_scaled"
        os.makedirs(self.bbox_4d_root_dir, exist_ok=True)
        os.makedirs(self.bbox_3d_final_root_dir, exist_ok=True)
        os.makedirs(self.bbox_3d_final_scaled_root_dir, exist_ok=True)

        # GT annotations
        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        # Segmentation dirs (if you later want to build label-wise masks)
        self.dynamic_masked_frames_im_dir_path = (
            self.ag_root_directory
            / "segmentation"
            / "masked_frames"
            / "image_based"
        )
        self.dynamic_masked_frames_vid_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_frames" / "video_based"
        )
        self.dynamic_masked_frames_combined_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_frames" / "combined"
        )
        self.dynamic_masked_videos_dir_path = (
            self.ag_root_directory / "segmentation" / "masked_videos"
        )

        self.dynamic_masks_im_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "image_based"
        )
        self.dynamic_masks_vid_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "video_based"
        )
        self.dynamic_masks_combined_dir_path = (
            self.ag_root_directory / "segmentation" / "masks" / "combined"
        )

        self.static_masks_im_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        )
        self.static_masks_vid_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        )
        self.static_masks_combined_dir_path = (
            self.ag_root_directory / "segmentation_static" / "masks" / "combined"
        )

        # Mirror transform about ZY plane (x -> -x) in the aligned frame
        M_mirror = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)

    # ----------------------------------------------------------------------------------
    # GT + GDINO + 3D annotations loaders
    # ----------------------------------------------------------------------------------

    def get_video_3d_annotations(self, video_id: str):
        """
        Load the floor-aligned 3D bbox annotations created by BBox3DGenerator.
        """
        out_path = self.bbox_3d_final_root_dir / f"{video_id[:-4]}.pkl"
        if not out_path.exists():
            print(f"[world4d][{video_id}] 3D bbox annotations not found at {out_path}")
            return None

        with open(out_path, "rb") as f:
            video_3d_annotations = pickle.load(f)
        return video_3d_annotations

    # ----------------------------------------------------------------------------------
    # Indexing helper — mirror BBox3DGenerator.idx_to_frame_idx_path
    # ----------------------------------------------------------------------------------

    def _bbox_idx_to_frame_idx_path(
        self, video_id: str
    ) -> Tuple[Dict[int, str], List[int], List[int], List[str], List[int]]:
        """
        Replica of BBox3DGenerator.idx_to_frame_idx_path, so that
        the points we load here match exactly the subset used
        when constructing the bbox_3D .pkl files.
        """
        video_frames_annotated_dir_path = self.frame_annotated_dir_path / video_id
        annotated_frame_id_list = [
            f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith(".png")
        ]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))

        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        video_sampled_frames_npy_path = (
            self.sampled_frames_idx_root_dir / f"{video_id[:-4]}.npy"
        )
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(
            annotated_first_frame_id
        )
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(
            annotated_last_frame_id
        )
        sample_idx = list(
            range(
                an_first_id_in_vid_sam_frame_id_list,
                an_last_id_in_vid_sam_frame_id_list + 1,
            )
        )

        annotated_frame_idx_in_sample_idx: List[int] = []
        for frame_name in annotated_frame_id_list:
            frame_id = int(frame_name[:-4])
            if frame_id in video_sampled_frame_id_list:
                idx_in_sampled = video_sampled_frame_id_list.index(frame_id)
                annotated_frame_idx_in_sample_idx.append(sample_idx.index(idx_in_sampled))

        chosen_frames = [video_sampled_frame_id_list[i] for i in sample_idx]
        frame_idx_frame_path_map = {
            i: f"{frame_id:06d}.png" for i, frame_id in enumerate(chosen_frames)
        }

        return (
            frame_idx_frame_path_map,
            sample_idx,
            video_sampled_frame_id_list,
            annotated_frame_id_list,
            annotated_frame_idx_in_sample_idx,
        )

    # -------------------------------------------------------------------------
    # Load original points (annotated frames only)
    # -------------------------------------------------------------------------

    def _load_original_points_for_video(self, video_id: str) -> Dict[str, Any]:
        """
        Load original Pi3 outputs for annotated frames only, using same slicing logic
        as bbox generation.

        Returns:
            {
                "points": (S,H,W,3) float32,
                "conf":   (S,H,W) float32 or None,
                "frame_stems": List[str],       # "000123", ...
                "colors": (S,H,W,3) uint8,
                "camera_poses": (S,4,4) or None
            }
        """
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        if not video_dynamic_3d_scene_path.exists():
            raise FileNotFoundError(f"[original-points] predictions.npz not found: {video_dynamic_3d_scene_path}")

        with _npz_open(video_dynamic_3d_scene_path) as pred:
            points = pred["points"].astype(np.float32)  # (S,H,W,3)

            imgs_f32 = pred["images"]
            colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

            conf = None
            if "conf" in pred:
                conf = pred["conf"]
                if conf.ndim == 4 and conf.shape[-1] == 1:
                    conf = conf[..., 0]
                conf = conf.astype(np.float32)

            camera_poses = None
            if "camera_poses" in pred:
                camera_poses = pred["camera_poses"].astype(np.float32)

            S, H, W, _ = points.shape

            (
                _frame_idx_frame_path_map,
                sample_idx,
                video_sampled_frame_id_list,
                annotated_frame_id_list,
                _annotated_frame_idx_in_sample_idx,
            ) = self._bbox_idx_to_frame_idx_path(video_id)

            assert S == len(sample_idx), (
                f"[original-points] points axis ({S}) != sample_idx range ({len(sample_idx)}) for {video_id}"
            )

            sampled_idx_frame_name_map: Dict[int, str] = {}
            frame_name_sampled_idx_map: Dict[str, int] = {}
            for idx_in_s, frame_idx in enumerate(sample_idx):
                frame_name = f"{video_sampled_frame_id_list[frame_idx]:06d}.png"
                sampled_idx_frame_name_map[idx_in_s] = frame_name
                frame_name_sampled_idx_map[frame_name] = idx_in_s

            annotated_idx_in_sampled_idx: List[int] = []
            for frame_name in annotated_frame_id_list:
                if frame_name in frame_name_sampled_idx_map:
                    annotated_idx_in_sampled_idx.append(frame_name_sampled_idx_map[frame_name])

            points_sub = points[annotated_idx_in_sampled_idx]
            colors_sub = colors[annotated_idx_in_sampled_idx]
            conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
            camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx] if camera_poses is not None else None
            stems_sub = [sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx]

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub,
        }

    # -------------------------------------------------------------------------
    # Write updated PKL (separate directory)
    # -------------------------------------------------------------------------

    def save_video_3d_annotations_final(self, video_id: str, video_3dgt_updated: Dict[str, Any]) -> Path:
        out_path = self.bbox_3d_final_scaled_root_dir / f"{video_id[:-4]}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(video_3dgt_updated, f, protocol=pickle.HIGHEST_PROTOCOL)
        return out_path

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
        out_path = self.bbox_3d_final_scaled_root_dir / f"{video_id[:-4]}.pkl"
        if out_path.exists() and not overwrite:
            print(f"[frames_final][{video_id}] exists: {out_path} (overwrite=False). Skipping.")
            return out_path

        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is None:
            print(f"[frames_final][{video_id}] missing original bbox_annotations_3d PKL. Skipping.")
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
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        H_img, W_img = img.shape[:2]

        # 2. Compare it with H, W of the loaded points
        scale_x = W / W_img
        scale_y = H / H_img

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

    # -------------------------------------------------------------------------
    # FINAL-only visualization entry
    # -------------------------------------------------------------------------

    def visualize_final_only(self, video_id: str, *, app_id: str = "World4D-FinalOnly") -> None:
        final_pkl = self.bbox_3d_final_root_dir / f"{video_id[:-4]}.pkl"
        if not final_pkl.exists():
            raise FileNotFoundError(f"Final PKL not found: {final_pkl}")

        with open(final_pkl, "rb") as f:
            rec = pickle.load(f)

        frames_final = rec.get("frames_final", None)
        if frames_final is None:
            raise ValueError(f"[final-only][{video_id}] frames_final missing in {final_pkl}")

        # Load original points + cameras for annotated frames
        P = self._load_original_points_for_video(video_id)
        points_original = P["points"]
        cols = P["colors"]

        # Apply the transform to original points to get final points
        world_to_final = rec["world_to_final"]
        origin_world = world_to_final["origin_world"]
        A = world_to_final["A_world_to_final"]
        S, H, W, _ = points_original.shape
        pts_flat = points_original.reshape(-1, 3)
        pts_final_flat = self._apply_world_to_final_points_row(pts_flat, origin_world=origin_world, A_world_to_final=A)
        points_final = pts_final_flat.reshape(S, H, W, 3).astype(np.float32)
        frames_final["points"] = points_final
        frames_final["colors"] = cols
        frames_final["conf"] = P["conf"]

        rerun_frame_vis_final_only(
            video_id,
            frames_final=frames_final,
            frame_annotated_dir_path=self.frame_annotated_dir_path,
            app_id=app_id,
        )

    # -------------------------------------------------------------------------
    # Batch processing over a dataloader
    # -------------------------------------------------------------------------

    def generate_gt_world_3D_bb_annotations(
            self,
            dataloader: DataLoader,
            split: str,
            *,
            overwrite: bool = False,
            points_dtype: np.dtype = np.float32,
    ) -> None:
        """
        Iterate over an AG dataloader and build frames_final PKLs for videos in the given split.
        """
        for data in tqdm(dataloader):
            video_id = data["video_id"]
            if get_video_belongs_to_split(video_id) != split:
                continue

            print("-----------------------------------------------------------------")
            print(f"[frames_final] processing video {video_id}...")

            try:
                self.build_frames_final_and_store(
                    video_id,
                    overwrite=overwrite,
                    points_dtype=points_dtype,
                )
            except Exception as e:
                print(f"[frames_final] failed video {video_id}: {e}")
            print(f"[frames_final] done video {video_id}.")


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
    frame_to_world_generator.build_frames_final_and_store(video_id=video_id, overwrite=False)
    frame_to_world_generator.visualize_final_only(video_id=video_id, app_id="World4D-FinalOnly-Sample")


if __name__ == "__main__":
    # main()
    main_sample()