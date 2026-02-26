# This block of code takes in each annotations frame and the corresponding 3D points from the world.
# For each annotated 2D bounding box, estimates its corresponding 3D bounding box in the world coordinate system.
# It uses two types (a) Axis Aligned Bounding Boxes (AABB) and (b) Oriented Bounding Boxes (OBB).
# It also helps in visualization of the 3D bounding boxes.
import argparse
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.ag_dataset import StandardAG
from datasets.preprocess.human.prompt_hmr.vis.traj import align_meshes_to_ground
from datasets.preprocess.annotations.raw.bb3D_base import BBox3DBase
from datasets.preprocess.annotations.annotation_utils import (
    _load_pkl_if_exists,
    _is_empty_array,
    get_video_belongs_to_split,
)





def build_scene_floor(scene_pts_xyz: np.ndarray,
                      floor_scale=2.0,
                      floor_color=None,
                      device="cpu"):
    """
    scene_pts_xyz: (N, 3) numpy, raw scene points in *current* frame/world
    returns: R (3,3), offset (3,), floor_v, floor_f, floor_c
    """
    if scene_pts_xyz.ndim == 2:
        scene_pts_xyz = scene_pts_xyz[None, ...]  # (1, N, 3)

    verts = torch.from_numpy(scene_pts_xyz).float().to(device)  # (1, N, 3)

    # re-use your helper
    verts_world, floor_data, R, offset = align_meshes_to_ground(
        verts, floor_scale=floor_scale, floor_color=floor_color
    )
    # floor_data is [gv, gf, gc]
    floor_v, floor_f, floor_c = floor_data  # numpy-ish from your helper

    # R, offset are torch -> make them numpy for logging
    R = R.detach().cpu().numpy()  # (3,3)
    offset = offset.detach().cpu().numpy()  # (3,)

    return R, offset, floor_v, floor_f, floor_c


class BBox3DGenerator(BBox3DBase):

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
    ) -> None:
        super().__init__(dynamic_scene_dir_path, ag_root_directory)



    def create_gt_annotations_map(self, dataloader, split):
        video_id_gt_annotations_map = {}
        video_id_gt_bboxes_map = {}
        for data in tqdm(dataloader):
            video_id = data['video_id']

            if get_video_belongs_to_split(video_id) == split:
                gt_annotations = data['gt_annotations']
                video_id_gt_annotations_map[video_id] = gt_annotations

        # video_id, gt_bboxes for the gt detections
        for video_id, gt_annotations in video_id_gt_annotations_map.items():
            video_gt_bboxes = {}
            for frame_idx, frame_items in enumerate(gt_annotations):
                frame_name = frame_items[0]["frame"].split("/")[-1]
                boxes = []
                labels = []
                for item in frame_items:
                    if 'person_bbox' in item:
                        boxes.append(item['person_bbox'][0])
                        labels.append('person')
                        continue
                    category_id = item['class']
                    category_name = self.catid_to_name_map[category_id]
                    if category_name:
                        if category_name == "closet/cabinet":
                            category_name = "closet"
                        elif category_name == "cup/glass/bottle":
                            category_name = "cup"
                        elif category_name == "paper/notebook":
                            category_name = "paper"
                        elif category_name == "sofa/couch":
                            category_name = "sofa"
                        elif category_name == "phone/camera":
                            category_name = "phone"
                        boxes.append(item['bbox'])
                        labels.append(category_name)
                if boxes:
                    video_gt_bboxes[frame_name] = {
                        'boxes': boxes,
                        'labels': labels
                    }
            video_id_gt_bboxes_map[video_id] = video_gt_bboxes

        return video_id_gt_bboxes_map, video_id_gt_annotations_map

    def create_gdino_annotations_map(self, dataloader, split):
        video_id_gdino_annotations_map = {}
        for data in tqdm(dataloader):
            video_id = data["video_id"]

            if get_video_belongs_to_split(video_id) != split:
                continue

            try:
                combined_gdino_predictions = self.get_video_gdino_annotations(video_id)
                video_id_gdino_annotations_map[video_id] = combined_gdino_predictions
            except ValueError as e:
                # Handle the case where no predictions are found, similar to original logic
                # Original logic raised ValueError if both empty.
                # Here we can just skip or re-raise.
                # The original code raised ValueError, so we should probably let it bubble up or catch it.
                # But wait, original code iterated over all videos.
                raise e

        return video_id_gdino_annotations_map



    # ------------------------------ (5) Load 3D points for frames ------------------------------ #
    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        video_dynamic_predictions = np.load(video_dynamic_3d_scene_path, allow_pickle=True)

        points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
        imgs_f32 = video_dynamic_predictions["images"]  # float32 in [0, 1]
        camera_poses = video_dynamic_predictions["camera_poses"]  # (S,4,4)
        colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)  # (S, H, W, 3)

        conf = None
        if "conf" in video_dynamic_predictions:
            conf = video_dynamic_predictions["conf"]
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]
        S, H, W, _ = points.shape

        # Dynamic Scene Predictions will be of length S where S -->
        # Begin from first annotated frame to last annotated frame in the sampled video frames.
        # But we need dynamic points for specific annotated frames.
        # So, we need to sample the points accordingly.
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        annotated_frame_id_list = os.listdir(video_frames_annotated_dir_path)
        annotated_frame_id_list = [f for f in annotated_frame_id_list if f.endswith('.png')]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        # Get the mapping for sampled_frame_id and the actual frame id
        # Now start from the sampled frame which corresponds to the first annotated frame and keep the rest of the sampled frames
        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()  # Numbers only

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        assert S == len(sample_idx)

        # Indices corresponding to the annotated frames in the sampled frames
        sampled_idx_frame_name_map = {}
        frame_name_sampled_idx_map = {}
        for idx_in_s, frame_idx in enumerate(sample_idx):
            frame_name = f"{video_sampled_frame_id_list[frame_idx]:06d}.png"
            sampled_idx_frame_name_map[idx_in_s] = frame_name
            frame_name_sampled_idx_map[frame_name] = idx_in_s

        annotated_idx_in_sampled_idx = []
        for frame_name in annotated_frame_id_list:
            if frame_name in frame_name_sampled_idx_map:
                annotated_idx_in_sampled_idx.append(frame_name_sampled_idx_map[frame_name])

        # Return 3D points corresponding to the annotated frames only
        points_sub = points[annotated_idx_in_sampled_idx]  # (S,H,W,3)
        conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None  # (S,H,W) or None
        stems_sub = [sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx]  # len S
        colors_sub = colors[annotated_idx_in_sampled_idx]  # (S,H,W,3)
        camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx]  # (S,4,4)

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub
        }

    # ------------------------------ (6–9) Per-video BB generation ------------------------------ #
    def generate_video_bb_annotations(
            self,
            video_id: str,
            video_gt_annotations: List[Any],
            video_gdino_predictions: Dict[str, Any],
            *,
            min_points: int = 50,
            iou_thr: float = 0.3,
            visualize: bool = False
    ) -> None:
        P = self._load_points_for_video(video_id)
        points_S = P["points"]  # (S,H,W,3)
        conf_S = P["conf"]  # (S,H,W) or None
        stems_S = P["frame_stems"]  # len S
        colors = P["colors"]  # (S,H,W,3)
        camera_poses = P["camera_poses"]  # (S,4,4)
        S, H, W, _ = points_S.shape

        stem_to_idx = {stems_S[i]: i for i in range(S)}
        R_floor, offset_floor, floor_v, floor_f, floor_c = build_scene_floor(points_S[0])

        # ------------------------------------------------------------------
        # (A) init rerun
        # ------------------------------------------------------------------
        if visualize:
            base = f"world_bb/{video_id}"
            rr.init("world_bb", spawn=True)

        out_frames: Dict[str, Dict[str, Any]] = {}
        video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels = self.create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_gt_annotations
        )

        for frame_idx, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]  # '000123.png'
            stem = Path(frame_name).stem
            if stem not in stem_to_idx:
                continue
            sidx = stem_to_idx[stem]
            pts_hw3 = points_S[sidx]  # (H,W,3)
            colors_hw3 = colors[sidx]
            conf_hw = conf_S[sidx] if conf_S is not None else None

            frame_non_zero_pts = self._finite_and_nonzero(pts_hw3)

            # Build per-frame GT object list (normalize to xyxy)
            gt_objects: List[Tuple[str, List[float]]] = []
            for item in frame_items:
                if "person_bbox" in item:
                    xywh = item["person_bbox"][0]  # list
                    gt_objects.append(("person", self._xywh_to_xyxy(xywh)))
                    continue
                cid = item["class"]
                label = self.catid_to_name_map.get(cid, None)
                if not label:
                    continue
                if label == "closet/cabinet":
                    label = "closet"
                elif label == "cup/glass/bottle":
                    label = "cup"
                elif label == "paper/notebook":
                    label = "paper"
                elif label == "sofa/couch":
                    label = "sofa"
                elif label == "phone/camera":
                    label = "phone"
                # GT is xyxy for objects
                gt_objects.append((label, [float(v) for v in item["bbox"]]))

            # Pull GDINO predictions for this frame (already combined dyn+stat)
            gd = video_gdino_predictions.get(frame_name, None)
            if gd is None:
                gd_boxes, gd_labels, gd_scores = [], [], []
            else:
                gd_boxes = [list(map(float, b)) for b in gd["boxes"]]
                gd_labels = gd["labels"]
                gd_scores = [float(s) for s in gd["scores"]]

            frame_rec = {"objects": []}

            # ------------------------------------------------------------------
            # (B) per-frame time + clear so previous frame disappears
            # ------------------------------------------------------------------
            if visualize:
                rr.set_time_sequence("frame", int(frame_idx))
                # clear everything previously logged so this frame stands alone
                rr.log("/", rr.Clear(recursive=True))
                # self.log_floor_rr(floor_v, floor_f, floor_c)

                # # Also log points with colors
                # transformed_pts = self.transform_pts_R_offset(
                #     points_S[int(frame_idx)].reshape(-1, 3), R_floor, offset_floor
                # )  # (S*H*W, 3)
                # rr.log(
                #     f"{base}/points",
                #     rr.Points3D(
                #         transformed_pts,
                #         colors=colors[int(frame_idx)].reshape(-1, 3)
                #     )
                # )

                # Log points without transformation
                rr.log(
                    f"{base}/points",
                    rr.Points3D(
                        points_S[int(frame_idx)].reshape(-1, 3),
                        colors=colors[int(frame_idx)].reshape(-1, 3)
                    )
                )

            # Extract 3D for each GT object
            for (label, gt_xyxy) in gt_objects:
                chosen_gd_xyxy = self._match_gdino_to_gt(
                    label, gt_xyxy,
                    gd_boxes, gd_labels, gd_scores,
                    iou_thr=iou_thr
                )

                # Build mask: prefer segmentation union; fallback to bbox mask
                frame_label_mask = video_to_frame_to_label_mask[video_id][stem][label]
                if frame_label_mask is None:
                    # mask fallback -> use chosen GDINO box, else GT
                    box = chosen_gd_xyxy if chosen_gd_xyxy is not None else gt_xyxy
                    frame_label_mask = self._mask_from_bbox(H, W, box)
                else:
                    frame_label_mask = self._resize_mask_to(frame_label_mask, (H, W))

                sel = frame_label_mask & frame_non_zero_pts
                if conf_hw is not None:
                    sel &= (conf_hw > 1e-6)

                if sel.sum() < min_points:
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": [float(v) for v in gt_xyxy],
                        "gdino_bbox_xyxy": [float(v) for v in chosen_gd_xyxy] if chosen_gd_xyxy is not None else None,
                        "num_points": int(sel.sum()),
                        "aabb": None,
                        "obb": None
                    })
                    continue

                label_non_zero_pts = pts_hw3[sel].reshape(-1, 3).astype(np.float32)
                label_colors = colors_hw3[sel].reshape(-1, 3).astype(np.uint8)

                # AABB & OBB in original world coords (for saving)
                aabb = self._aabb(label_non_zero_pts)
                obb = self._pca_obb(label_non_zero_pts)

                if visualize:
                    obj_base = f"{base}/{stem}/{label}"

                    # transform OBB corners
                    # corners = np.asarray(obb["corners"], dtype=np.float32)  # (8,3)
                    # corners_aligned = self.transform_pts_R_offset(corners, R_floor, offset_floor)
                    # self._log_box_lines_rr(
                    #     f"{obj_base}/obb",
                    #     corners_aligned,
                    #     rgba=(0, 255, 0, 255),
                    # )

                    # transform AABB corners
                    mn = np.asarray(aabb["min"], dtype=np.float32)
                    mx = np.asarray(aabb["max"], dtype=np.float32)
                    aabb_corners = np.array([
                        [mn[0], mn[1], mn[2]],
                        [mn[0], mn[1], mx[2]],
                        [mn[0], mx[1], mn[2]],
                        [mn[0], mx[1], mx[2]],
                        [mx[0], mn[1], mn[2]],
                        [mx[0], mn[1], mx[2]],
                        [mx[0], mx[1], mn[2]],
                        [mx[0], mx[1], mx[2]],
                    ], dtype=np.float32)

                    # aabb_corners_aligned = self.transform_pts_R_offset(aabb_corners, R_floor, offset_floor)
                    self._log_box_lines_rr(
                        f"{obj_base}/aabb",
                        aabb_corners,
                        rgba=(255, 255, 0, 255),
                    )

                frame_rec["objects"].append({
                    "label": label,
                    "gt_bbox_xyxy": [float(v) for v in gt_xyxy],
                    "gdino_bbox_xyxy": [float(v) for v in chosen_gd_xyxy] if chosen_gd_xyxy is not None else None,
                    "num_points": int(label_non_zero_pts.shape[0]),
                    "aabb": aabb,
                    "obb": obb
                })

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        # ------------------------------ Persist to disk ------------------------------ #
        out_path = self.bbox_3d_root_dir / f"{video_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "video_id": video_id,
                "frames": out_frames
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

    def generate_gt_world_bb_annotations(self, dataloader, split) -> None:
        # For every frame in the video, Person bbox is in xywh format and the object bbox is in xyxy format.
        # For the category of the object in the frame we have to get the 3D points corresponding to that object.

        # 1. Ground truth annotations for specific frames.
        # This primarily includes bounding boxes for persons and objects in the frame.
        print("Creating GT annotations map...")
        video_id_gt_bboxes_map, video_id_gt_annotations_map = self.create_gt_annotations_map(dataloader, split)

        # 2. Grounding Dino bounding boxes for specific frames.
        # Combined detections of dynamic objects and static objects.
        print("Creating GDINO annotations map...")
        video_id_gdino_annotations_map = self.create_gdino_annotations_map(dataloader, split)

        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                self.generate_video_bb_annotations(
                    video_id,
                    video_id_gt_annotations_map[video_id],
                    video_id_gdino_annotations_map.get(video_id, {}),
                    visualize=True
                )


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize static + per-frame 3D points with Rerun (AG-Pi3 unified)."
    )
    # Paths
    parser.add_argument(
        "--ag_root_directory",
        type=str,
        default="/data/rohith/ag",
        help="Optional: directory containing annotated frames (unused here).",
    )
    parser.add_argument(
        "--static_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_static",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    # Selection
    parser.add_argument(
        "--split",
        type=_parse_split,
        default="QT",
        help="Shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}.",
    )
    return parser.parse_args()


def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
        pin_memory=False,
        num_workers=0
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
        pin_memory=False
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def main() -> None:
    args = parse_args()
    bbox_3d_generator = BBox3DGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    bbox_3d_generator.generate_gt_world_bb_annotations(dataloader=dataloader_train, split=args.split)


if __name__ == "__main__":
    main()
