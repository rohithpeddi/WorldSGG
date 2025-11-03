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
from tqdm import tqdm


class WorldBBGenerator:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
    ) -> None:
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = dynamic_scene_dir_path

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

        # Segmentation masks paths
        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation_static" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

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
            ua = WorldBBGenerator._area_xyxy(a) + WorldBBGenerator._area_xyxy(b) - inter
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

        def _log_box_lines_rr(self, path: str, corners: np.ndarray, rgba=(255, 255, 255, 255), radius=0.002):
            if rr is None:
                return
            edges = self._box_edges_from_corners(corners)
            for k, e in enumerate(edges):
                rr.log(
                    f"{path}/edge_{k}",
                    rr.LineStrips3D(positions=[e.astype(np.float32)],
                                    radii=radius,
                                    colors=[rgba])
                )
    def create_gt_annotations_map(self, dataloader):
        video_id_gt_annotations_map = {}
        video_id_gt_bboxes_map = {}
        for data in tqdm(dataloader):
            video_id = data['video_id']
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

    def create_gdino_annotations_map(self, dataloader):
        video_id_gdino_annotations_map = {}
        for data in tqdm(dataloader):
            video_id = data['video_id']

            # 1. Load dynamic gdino annotations
            video_dynamic_gdino_prediction_file_path = self.dynamic_detections_root_path / f"{video_id}.pkl"
            video_dynamic_predictions = None
            with open(video_dynamic_gdino_prediction_file_path, 'rb') as f:
                video_dynamic_predictions = pickle.load(f)

            # 2. Load static gdino annotations
            video_static_gdino_prediction_file_path = self.static_detections_root_path / f"{video_id}.pkl"
            video_static_predictions = None
            with open(video_static_gdino_prediction_file_path, 'rb') as f:
                video_static_predictions = pickle.load(f)

            # 3. Frame wise combined gdino annotations, use frame_id as the key for the map
            combined_gdino_predictions = {}
            for frame_name, dynamic_pred in video_dynamic_predictions.items():
                static_pred = video_static_predictions.get(frame_name, None)
                if static_pred:
                    combined_boxes = dynamic_pred['boxes'] + static_pred['boxes']
                    combined_labels = dynamic_pred['labels'] + static_pred['labels']
                    combined_scores = dynamic_pred['scores'] + static_pred['scores']
                else:
                    combined_boxes = dynamic_pred['boxes']
                    combined_labels = dynamic_pred['labels']
                    combined_scores = dynamic_pred['scores']
                combined_gdino_predictions[frame_name] = {
                    'boxes': combined_boxes,
                    'labels': combined_labels,
                    'scores': combined_scores
                }
            video_id_gdino_annotations_map[video_id] = combined_gdino_predictions

        return video_id_gdino_annotations_map

    def create_label_wise_masks_map(
            self,
            video_id,
            gt_annotations
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Returns:
            video_id -> frame_stem -> label -> bool mask
        Notes:
            - Merges image/video masks if both exist (logical OR), thresholded at >127.
            - Only builds entries for frames present in the dataloader's GT annotations.
        """
        video_to_frame_to_label_mask: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

        def labels_for_frame(video_id: str, stem: str) -> List[str]:
            lbls = set()
            for root in [self.masks_im_dir_path, self.masks_vid_dir_path]:
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

        frame_stems = []
        for frame_items in gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]  # e.g., '000123.png'
            stem = Path(frame_name).stem
            frame_stems.append(stem)

        frame_map: Dict[str, Dict[str, np.ndarray]] = {}
        for stem in frame_stems:
            lbls = labels_for_frame(video_id, stem)
            if not lbls:
                continue
            frame_map[stem] = {}
            for lbl in lbls:
                im_p = self.masks_im_dir_path / video_id / f"{stem}__{lbl}.png"
                vd_p = self.masks_vid_dir_path / video_id / f"{stem}__{lbl}.png"
                m_im = cv2.imread(str(im_p), cv2.IMREAD_GRAYSCALE) if im_p.exists() else None
                m_vd = cv2.imread(str(vd_p), cv2.IMREAD_GRAYSCALE) if vd_p.exists() else None
                if m_im is None and m_vd is None:
                    continue
                if m_im is None:
                    m = (m_vd > 127)
                elif m_vd is None:
                    m = (m_im > 127)
                else:
                    m = (m_im > 127) | (m_vd > 127)
                frame_map[stem][lbl] = m.astype(bool)

        if frame_map:
            video_to_frame_to_label_mask[video_id] = frame_map

        return video_to_frame_to_label_mask

    def generate_video_bb_annotations(self, video_id: str, video_gt_annotations, video_gdino_predictions) -> None:
        # 3. Label wise masks for each object in specific frames.

        # 4. For every ground truth bounding box detection, we need to make sure that we have corresponding gdino bounding box may be some union of boxes.

        # 5. Load 3D points for specific frames.
        # We need to match specific frames with the subsampled frames for the complete video

        # 6. We need to extract the 3D points corresponding to each object in the frame using the masks.

        # 7. Using the 3D points, we need to estimate the Axis Aligned Bounding Box (AABB) and Oriented Bounding Box (OBB) for each object in the frame.

        # 8. We need to run a rerun visualization for all the things frame by frame.
        # Gdino detections, Ground truth detection, Final label wise masks, 3D points, AABB and OBB boxes.

        # 9. Finally, we need to save the world bounding box annotations in a pkl file.
        pass

    def generate_gt_world_bb_annotations(self, dataloader) -> None:
        # For every frame in the video, Person bbox is in xywh format and the object bbox is in xyxy format.
        # For the category of the object in the frame we have to get the 3D points corresponding to that object.

        # 1. Ground truth annotations for specific frames.
        # This primarily includes bounding boxes for persons and objects in the frame.
        video_id_gt_bboxes_map, video_id_gt_annotations_map = self.create_gt_annotations_map(dataloader)

        # 2. Grounding Dino bounding boxes for specific frames.
        # Combined detections of dynamic objects and static objects.
        video_id_gdino_annotations_map = self.create_gdino_annotations_map(dataloader)

        for data in tqdm(dataloader):
            video_id = data['video_id']
            self.generate_video_bb_annotations(
                video_id,
                video_id_gt_annotations_map[video_id],
                video_id_gdino_annotations_map.get(video_id, {})
            )

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
        """Return a chosen GDINO bbox (xyxy) for this GT object; union of candidates if multiple, else fallback to GT."""
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
        cand_dirs = [
            Path(self.dynamic_scene_dir_path) / f"{video_id}_10",
            Path(self.dynamic_scene_dir_path) / video_id / "10",
            Path(self.dynamic_scene_dir_path) / video_id,
        ]
        pred = None
        for d in cand_dirs:
            p = d / "predictions.npz"
            if p.exists():
                arr = np.load(str(p), allow_pickle=True)
                pred = {k: arr[k] for k in arr.files}
                break
        if pred is None:
            raise FileNotFoundError(f"predictions.npz not found for video {video_id} in {cand_dirs}")

        points = pred["points"].astype(np.float32)  # (S,H,W,3)
        conf = None
        if "conf" in pred:
            c = pred["conf"]
            if c.ndim == 4 and c.shape[-1] == 1:
                c = c[..., 0]
            conf = c.astype(np.float32)

        # Map stems from sampled_frames dir if available
        frames_dir = self.ag_root_directory / "sampled_frames" / video_id
        stems = sorted(
            {Path(fn).stem for fn in os.listdir(frames_dir) if fn.lower().endswith((".png", ".jpg", ".jpeg"))}
        ) if frames_dir.exists() else []

        # best-effort: if lengths match, assume aligned by sort order
        if stems and len(stems) == points.shape[0]:
            frame_stems = stems
        else:
            # fallback: sequential index strings
            frame_stems = [f"{i:06d}" for i in range(points.shape[0])]

        return {"points": points, "conf": conf, "frame_stems": frame_stems}

    # ------------------------------ (6–9) Per-video BB generation ------------------------------ #
    def generate_video_bb_annotations(
        self,
        video_id: str,
        video_gt_annotations: List[Any],
        video_gdino_predictions: Dict[str, Any],
        *,
        min_points: int = 50,
        iou_thr: float = 0.3,
        visualize: bool = False,
        rr_app_id: Optional[str] = None,
    ) -> None:
        """
        Produces world 3D BB annotations per object per frame and writes:
            <ag_root>/world_bb_annotations/<video_id>.pkl
        Structure:
            {
              'video_id': str,
              'frames': {
                  '<frame_name>': {
                      'objects': [
                           { 'label', 'gt_bbox_xyxy', 'gdino_bbox_xyxy',
                             'num_points',
                             'aabb': {'min','max'},
                             'obb':  {'center','axes','extents','corners'}
                           }, ...
                      ]
                  }, ...
              }
            }
        """
        # Load points
        P = self._load_points_for_video(video_id)
        points_S = P["points"]          # (S,H,W,3)
        conf_S   = P["conf"]            # (S,H,W) or None
        stems_S  = P["frame_stems"]     # len S
        S, H, W, _ = points_S.shape

        # Build mapping: frame_name -> index in S
        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # optional Rerun init
        if visualize and rr is not None:
            rr.init(rr_app_id or f"world_bb_{video_id}", spawn=False)

        # Preload label-wise mask helper that unions image & video routes on demand
        def get_union_mask(video_id: str, stem: str, label: str) -> Optional[np.ndarray]:
            im_p = self.masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.masks_vid_dir_path / video_id / f"{stem}__{label}.png"
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

        # Output structure
        out_frames: Dict[str, Dict[str, Any]] = {}

        # Iterate frames using GT list (authoritative for which frames matter)
        for frame_items in video_gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]  # '000123.png'
            stem = Path(frame_name).stem
            if stem not in stem_to_idx:
                # frame missing in the sampled S sequence
                continue
            sidx = stem_to_idx[stem]
            pts_hw3 = points_S[sidx]  # (H,W,3)
            conf_hw = conf_S[sidx] if conf_S is not None else None

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

            # Extract 3D for each GT object
            for (label, gt_xyxy) in gt_objects:
                chosen_gd_xyxy = self._match_gdino_to_gt(
                    label, gt_xyxy, gd_boxes, gd_labels, gd_scores, iou_thr=iou_thr
                )

                # Build mask: prefer segmentation union; fallback to bbox mask (chosen GDINO > GT)
                m = get_union_mask(video_id, stem, label)
                if m is None:
                    # mask fallback -> use chosen GDINO box, else GT
                    box = chosen_gd_xyxy if chosen_gd_xyxy is not None else gt_xyxy
                    m = self._mask_from_bbox(H, W, box)
                else:
                    m = self._resize_mask_to(m, (H, W))

                # Select 3D points by mask + finiteness (+ optional confidence)
                sel = m & self._finite_and_nonzero(pts_hw3)
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

                pts_n3 = pts_hw3[sel].reshape(-1, 3).astype(np.float32)

                # AABB & OBB
                aabb = self._aabb(pts_n3)
                obb  = self._pca_obb(pts_n3)

                # Visualization (optional)
                if visualize and rr is not None:
                    base = f"world_bb/{video_id}/{stem}/{label}"
                    rr.log(f"{base}/points", rr.Points3D(positions=pts_n3))
                    # draw OBB as line strips
                    corners = np.asarray(obb["corners"], dtype=np.float32)
                    self._log_box_lines_rr(f"{base}/obb", corners, rgba=(0, 255, 0, 255))
                    # draw AABB as line strips
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
                    self._log_box_lines_rr(f"{base}/aabb", aabb_corners, rgba=(255, 255, 0, 255))

                frame_rec["objects"].append({
                    "label": label,
                    "gt_bbox_xyxy": [float(v) for v in gt_xyxy],
                    "gdino_bbox_xyxy": [float(v) for v in chosen_gd_xyxy],
                    "num_points": int(pts_n3.shape[0]),
                    "aabb": aabb,
                    "obb": obb
                })

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        # ------------------------------ (9) Persist to disk ------------------------------ #
        out_dir = self.ag_root_directory / "world_bb_annotations"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{video_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "video_id": video_id,
                "frames": out_frames
            }, f, protocol=pickle.HIGHEST_PROTOCOL)




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
        "--frames_annotated_dir_path",
        type=str,
        default="/data/rohith/ag/frames_annotated",
        help="Optional: directory containing annotated frames (unused here).",
    )
    parser.add_argument(
        "--mask_dir_path",
        type=str,
        default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_masks",
        help="Path to directory containing trained model checkpoints.",
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
    parser.add_argument(
        "--grounded_dynamic_scene_dir_path",
        type=str,
        default="/data2/rohith/ag/ag4D/dynamic_scenes/pi3_grounded_dynamic"
    )
    # Selection
    parser.add_argument(
        "--split",
        type=_parse_split,
        default="04",
        help="Shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}.",
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()


if __name__ == "__main__":
    main()