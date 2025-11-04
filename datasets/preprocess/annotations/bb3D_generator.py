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
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.standard.action_genome.ag_dataset import StandardAG


def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """
    Get the split that the video belongs to based on its ID.
    Accepts either a bare ID (e.g., '0DJ6R') or a filename (e.g., '0DJ6R.mp4').
    """
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"
    return None


def _is_empty_array(x):
    # Handles None, list, tuple, torch.Tensor, np.ndarray
    if x is None:
        return True
    # list/tuple
    if isinstance(x, (list, tuple)):
        return len(x) == 0
    # try tensor-like / ndarray-like
    try:
        return getattr(x, "numel", None) and x.numel() == 0
    except Exception:
        pass
    try:
        return hasattr(x, "size") and hasattr(x, "shape") and x.size == 0
    except Exception:
        pass
    return False


def _to_len(x):
    if x is None:
        return 0
    if isinstance(x, (list, tuple)):
        return len(x)
    # torch / np
    try:
        return int(x.shape[0])
    except Exception:
        return 0

def _load_pkl_if_exists(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


class BBox3DGenerator:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
    ) -> None:
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)

        self.dataset_classnames = [
            '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
            'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
            'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
            'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe',
            'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window'
        ]
        self.name_to_catid = {name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0}
        self.catid_to_name_map = {v: k for k, v in self.name_to_catid.items()}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        # ------------------------------ Directory Paths ------------------------------ #
        # Detections paths
        self.dynamic_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.static_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = self.ag_root_directory / "sampled_frames_idx"
        self.bbox_3d_root_dir = self.ag_root_directory / "world_bb_annotations"
        os.makedirs(self.bbox_3d_root_dir, exist_ok=True)

        # Segmentation masks paths
        self.dynamic_masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.dynamic_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.dynamic_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.dynamic_masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        # Internal (per-object) mask stores
        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.static_masked_frames_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'image_based'
        self.static_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'video_based'
        self.static_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'combined'
        self.static_masked_videos_dir_path = self.ag_root_directory / "segmentation_static" / "masked_videos"

        # Internal (per-object) mask stores
        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

    # ------------------------------ Utilities ------------------------------ #
    @staticmethod
    def _xywh_to_xyxy(b):  # [x,y,w,h] -> [x1,y1,x2,y2]
        x, y, w, h = [float(v) for v in b]
        return [x, y, x + w, y + h]

    @staticmethod
    def _area_xyxy(b):
        x1, y1, x2, y2 = b
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @staticmethod
    def _iou_xyxy(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = BBox3DGenerator._area_xyxy(a) + BBox3DGenerator._area_xyxy(b) - inter
        return inter / max(ua, 1e-8)

    @staticmethod
    def _union_boxes_xyxy(boxes: List[List[float]]) -> Optional[List[float]]:
        if not boxes:
            return None
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)
        return [x1, y1, x2, y2]

    @staticmethod
    def _mask_from_bbox(h: int, w: int, xyxy: List[float]) -> np.ndarray:
        m = np.zeros((h, w), dtype=bool)
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, w), min(y2, h)
        if x2 > x1 and y2 > y1:
            m[y1:y2, x1:x2] = True
        return m

    @staticmethod
    def _resize_mask_to(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        th, tw = target_hw
        if mask.shape == (th, tw):
            return mask.astype(bool)
        # nearest-neighbor for binary masks
        return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)

    @staticmethod
    def _finite_and_nonzero(pts: np.ndarray) -> np.ndarray:
        good = np.isfinite(pts).all(axis=-1)
        if pts.ndim == 2:  # (N,3)
            nz = np.linalg.norm(pts, axis=-1) > 1e-12
        else:  # (H,W,3)
            nz = np.linalg.norm(pts, axis=-1) > 1e-12
        return good & nz

    @staticmethod
    def _aabb(pts_n3: np.ndarray) -> Dict[str, Any]:
        mins = pts_n3.min(axis=0).tolist()
        maxs = pts_n3.max(axis=0).tolist()
        return {"min": mins, "max": maxs}

    @staticmethod
    def _pca_obb(pts_n3: np.ndarray) -> Dict[str, Any]:
        # center
        c = pts_n3.mean(axis=0)
        X = pts_n3 - c
        # PCA via SVD
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        R = Vt  # (3,3) rows are principal axes in world coords
        # project points to PCA basis
        Y = X @ R.T  # (N,3)
        mins = Y.min(axis=0)
        maxs = Y.max(axis=0)
        extents = (maxs - mins)  # box size along each axis
        # corners in PCA frame (8 corners)
        corners_local = np.array([[mins[0], mins[1], mins[2]],
                                  [mins[0], mins[1], maxs[2]],
                                  [mins[0], maxs[1], mins[2]],
                                  [mins[0], maxs[1], maxs[2]],
                                  [maxs[0], mins[1], mins[2]],
                                  [maxs[0], mins[1], maxs[2]],
                                  [maxs[0], maxs[1], mins[2]],
                                  [maxs[0], maxs[1], maxs[2]]], dtype=np.float32)
        # back to world frame
        corners_world = corners_local @ R + c
        return {
            "center": c.tolist(),
            "axes": R.tolist(),  # rows are axis directions in world frame
            "extents": extents.tolist(),  # full lengths along each axis
            "corners": corners_world.tolist()
        }

    @staticmethod
    def _box_edges_from_corners(corners: np.ndarray) -> List[np.ndarray]:
        # 8 corners -> 12 edges as 2-point line segments
        idx_pairs = [
            (0, 1), (0, 2), (0, 4),
            (7, 6), (7, 5), (7, 3),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (4, 5), (4, 6)
        ]
        return [np.vstack([corners[i], corners[j]]) for (i, j) in idx_pairs]

    def _log_box_lines_rr(self, path: str, corners: np.ndarray,
                          rgba=(255, 255, 255, 255), radius=0.002):
        edges = self._box_edges_from_corners(corners)
        for k, e in enumerate(edges):
            e = np.asarray(e, dtype=np.float32)
            rr.log(
                f"{path}/edge_{k}",
                rr.LineStrips3D(
                    [e],  # list of strips
                    radii=radius,
                    colors=[rgba],
                ),
            )

    def labels_for_frame(self, video_id: str, stem: str, is_static: bool) -> List[str]:
        lbls = set()
        if is_static:
            image_root_dir_list = [self.static_masks_im_dir_path, self.static_masks_vid_dir_path]
        else:
            image_root_dir_list = [self.dynamic_masks_im_dir_path, self.dynamic_masks_vid_dir_path]
        for root in image_root_dir_list:
            vdir = root / video_id
            if not vdir.exists():
                continue
            for fn in os.listdir(vdir):
                if not fn.endswith(".png"):
                    continue
                if "__" in fn:
                    st, lbl = fn.split("__", 1)
                    lbl = lbl.rsplit(".png", 1)[0]
                    if st == stem:
                        lbls.add(lbl)
        return sorted(lbls)

    def get_union_mask(self, video_id: str, stem: str, label: str, is_static) -> Optional[np.ndarray]:
        if is_static:
            im_p = self.static_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.static_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
        else:
            im_p = self.dynamic_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.dynamic_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
        m_im = cv2.imread(str(im_p), cv2.IMREAD_GRAYSCALE) if im_p.exists() else None
        m_vd = cv2.imread(str(vd_p), cv2.IMREAD_GRAYSCALE) if vd_p.exists() else None
        if m_im is None and m_vd is None:
            return None
        if m_im is None:
            m = (m_vd > 127)
        elif m_vd is None:
            m = (m_im > 127)
        else:
            m = (m_im > 127) | (m_vd > 127)
        return m.astype(bool)

    def update_frame_map(
            self,
            frame_stems,
            video_id,
            frame_map: Dict[str, Dict[str, np.ndarray]],
            is_static
    ):
        all_labels = set()
        for stem in frame_stems:
            lbls = self.labels_for_frame(video_id, stem, is_static)
            if not lbls:
                continue
            all_labels.update(lbls)
            if stem not in frame_map:
                frame_map[stem] = {}
            for lbl in lbls:
                m = self.get_union_mask(video_id, stem, lbl, is_static)
                if m is not None:
                    frame_map[stem][lbl] = m
        return frame_map, all_labels

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

            # 1. Load dynamic gdino annotations
            video_dynamic_gdino_prediction_file_path = self.dynamic_detections_root_path / f"{video_id}.pkl"
            video_dynamic_predictions = _load_pkl_if_exists(video_dynamic_gdino_prediction_file_path)

            # 2. Load static gdino annotations
            video_static_gdino_prediction_file_path = self.static_detections_root_path / f"{video_id}.pkl"
            video_static_predictions = _load_pkl_if_exists(video_static_gdino_prediction_file_path)

            # Normalize None to empty dict to simplify logic
            if video_dynamic_predictions is None:
                video_dynamic_predictions = {}
            if video_static_predictions is None:
                video_static_predictions = {}

            # If both are empty, that's an error for this video
            if not video_dynamic_predictions and not video_static_predictions:
                raise ValueError(
                    f"No GDINO predictions found for video {video_id} "
                    f"in both dynamic ({video_dynamic_gdino_prediction_file_path}) "
                    f"and static ({video_static_gdino_prediction_file_path}) paths."
                )

            # Collect all frame names seen in either dict
            all_frame_names = set(video_dynamic_predictions.keys()) | set(video_static_predictions.keys())

            combined_gdino_predictions = {}
            for frame_name in all_frame_names:
                dyn_pred = video_dynamic_predictions.get(frame_name, None)
                stat_pred = video_static_predictions.get(frame_name, None)
                if dyn_pred is None:
                    dyn_pred = {"boxes": [], "labels": [], "scores": []}
                if stat_pred is None:
                    stat_pred = {"boxes": [], "labels": [], "scores": []}

                if _is_empty_array(dyn_pred["boxes"]) and _is_empty_array(stat_pred["boxes"]):
                    combined_gdino_predictions[frame_name] = {
                        "boxes": [],
                        "labels": [],
                        "scores": [],
                    }
                    continue

                combined_boxes = []
                combined_labels = []
                combined_scores = []

                if not _is_empty_array(dyn_pred["boxes"]):
                    combined_boxes += list(dyn_pred["boxes"])
                    combined_labels += list(dyn_pred["labels"])
                    combined_scores += list(dyn_pred["scores"])

                if not _is_empty_array(stat_pred["boxes"]):
                    combined_boxes += list(stat_pred["boxes"])
                    combined_labels += list(stat_pred["labels"])
                    combined_scores += list(stat_pred["scores"])

                final_pred = {
                    "boxes": combined_boxes,
                    "labels": combined_labels,
                    "scores": combined_scores,
                }

                combined_gdino_predictions[frame_name] = final_pred

            # At this point, combined_gdino_predictions has per-frame dicts
            # if video_dynamic_predictions and video_static_predictions:
            #     print(f"[{video_id}] Combined GDINO dynamic and static predictions.")
            # elif video_dynamic_predictions:
            #     print(f"[{video_id}] Using only GDINO dynamic predictions (validated).")
            # else:
            #     print(f"[{video_id}] Using only GDINO static predictions (validated).")
            video_id_gdino_annotations_map[video_id] = combined_gdino_predictions

        return video_id_gdino_annotations_map

    def create_label_wise_masks_map(
            self,
            video_id,
            gt_annotations
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], set, set]:
        video_to_frame_to_label_mask: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        frame_stems = []
        for frame_items in gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]  # e.g., '000123.png'
            stem = Path(frame_name).stem
            frame_stems.append(stem)

        frame_map: Dict[str, Dict[str, np.ndarray]] = {}
        frame_map, all_static_labels = self.update_frame_map(
            frame_stems=frame_stems,
            video_id=video_id,
            frame_map=frame_map,
            is_static=True
        )
        frame_map, all_dynamic_labels = self.update_frame_map(
            frame_stems=frame_stems,
            video_id=video_id,
            frame_map=frame_map,
            is_static=False
        )
        if frame_map:
            video_to_frame_to_label_mask[video_id] = frame_map

        return video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels

    # ------------------------------ (4) Match GDINO to GT ------------------------------ #
    def _match_gdino_to_gt(
        self,
        gt_label: str,
        gt_xyxy: List[float],
        gd_boxes: List[List[float]],
        gd_labels: List[str],
        gd_scores: List[float],
        iou_thr: float = 0.3,
    ) -> List[float]:
        candidates = [
            (b, s) for b, l, s in zip(gd_boxes, gd_labels, gd_scores)
            if (l == gt_label)
        ]
        if not candidates:
            return gt_xyxy

        # keep boxes with IoU >= iou_thr (or top-1 if none pass)
        passing = [b for (b, s) in candidates if self._iou_xyxy(b, gt_xyxy) >= iou_thr]
        if passing:
            box = self._union_boxes_xyxy(passing)
            return box if box is not None else gt_xyxy

        # no IoU pass -> pick highest-score of same label
        best = max(candidates, key=lambda t: t[1])[0]
        return best

    # ------------------------------ (5) Load 3D points for frames ------------------------------ #
    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        """
        Returns a dict with:
           - 'points': (S,H,W,3) float32
           - 'conf'  : (S,H,W) float32 or None
           - 'frame_stems': List[str] length S (best-effort)
        """
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
        points_S = P["points"]          # (S,H,W,3)
        conf_S   = P["conf"]            # (S,H,W) or None
        stems_S  = P["frame_stems"]     # len S
        colors = P["colors"]          # (S,H,W,3)
        camera_poses = P["camera_poses"]  # (S,4,4)
        S, H, W, _ = points_S.shape

        stem_to_idx = {stems_S[i]: i for i in range(S)}
        if visualize:
            base = f"world_bb/{video_id}"
            rr.init(f"world_bb", spawn=True)

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
                if label == "closet/cabinet": label = "closet"
                elif label == "cup/glass/bottle": label = "cup"
                elif label == "paper/notebook": label = "paper"
                elif label == "sofa/couch": label = "sofa"
                elif label == "phone/camera": label = "phone"
                # GT is xyxy for objects
                gt_objects.append((label, [float(v) for v in item["bbox"]]))

            # Pull GDINO predictions for this frame (already combined dyn+stat)
            gd = video_gdino_predictions.get(frame_name, None)
            if gd is None:
                gd_boxes, gd_labels, gd_scores = [], [], []
            else:
                gd_boxes  = [list(map(float, b)) for b in gd["boxes"]]
                gd_labels = gd["labels"]
                gd_scores = [float(s) for s in gd["scores"]]

            frame_rec = {"objects": []}

            if visualize:
                rr.set_time_sequence("frame", int(frame_idx))

            # Extract 3D for each GT object
            for (label, gt_xyxy) in gt_objects:
                chosen_gd_xyxy = self._match_gdino_to_gt(label, gt_xyxy, gd_boxes, gd_labels, gd_scores, iou_thr=iou_thr)

                # Build mask: prefer segmentation union; fallback to bbox mask (chosen GDINO > GT)
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
                    # too few points -> skip box (but still record minimal info)
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": [float(v) for v in gt_xyxy],
                        "gdino_bbox_xyxy": [float(v) for v in chosen_gd_xyxy],
                        "num_points": int(sel.sum()),
                        "aabb": None,
                        "obb": None
                    })
                    continue

                label_non_zero_pts = pts_hw3[sel].reshape(-1, 3).astype(np.float32)
                label_colors = colors_hw3[sel].reshape(-1, 3).astype(np.uint8)

                # AABB & OBB
                aabb = self._aabb(label_non_zero_pts)
                obb  = self._pca_obb(label_non_zero_pts)
                if visualize:
                    # make path unique per frame + label
                    obj_base = f"{base}/{stem}/{label}"

                    # 1) points
                    rr.log(
                        f"{obj_base}/points",
                        rr.Points3D(
                            positions=label_non_zero_pts,
                            colors=label_colors,
                            radii=0.01,
                        ),
                    )

                    # 2) OBB
                    corners = np.asarray(obb["corners"], dtype=np.float32)
                    self._log_box_lines_rr(
                        f"{obj_base}/obb",
                        corners,
                        rgba=(0, 255, 0, 255),
                    )

                    # 3) AABB
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
                    self._log_box_lines_rr(
                        f"{obj_base}/aabb",
                        aabb_corners,
                        rgba=(255, 255, 0, 255),
                    )

                frame_rec["objects"].append({
                    "label": label,
                    "gt_bbox_xyxy": [float(v) for v in gt_xyxy],
                    "gdino_bbox_xyxy": [float(v) for v in chosen_gd_xyxy],
                    "num_points": int(label_non_zero_pts.shape[0]),
                    "aabb": aabb,
                    "obb": obb
                })

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        # ------------------------------ (9) Persist to disk ------------------------------ #
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