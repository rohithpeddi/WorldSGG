#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rerun as rr
import torch
from scipy.spatial.transform import Rotation as SciRot
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__) + '/..')

# from AG / human pipeline codebase
from dataloader.standard.action_genome.ag_dataset import StandardAG

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline
from datasets.preprocess.human.data_config import SMPLX_PATH
from datasets.preprocess.human.prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from datasets.preprocess.human.prompt_hmr.vis.traj import (
    get_floor_mesh,
)
from datasets.preprocess.human.pipeline.kp_utils import (
    get_openpose_joint_names,
    get_smpl_joint_names,
)


# =====================================================================
# COMMON HELPERS
# =====================================================================
def get_video_belongs_to_split(video_id: str) -> Optional[str]:
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


def _faces_u32(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces)
    if faces.dtype != np.uint32:
        faces = faces.astype(np.uint32)
    return faces


def _pinhole_from_fov(w: int, h: int, fov_y: float):
    fy = 0.5 * h / np.tan(0.5 * fov_y)
    fx = fy
    cx = w / 2.0
    cy = h / 2.0
    return fx, fy, cx, cy

def _is_empty_array(x):
    if x is None:
        return True
    if isinstance(x, (list, tuple)):
        return len(x) == 0
    try:
        return getattr(x, "numel", None) and x.numel() == 0
    except Exception:
        pass
    try:
        return hasattr(x, "size") and hasattr(x, "shape") and x.size == 0
    except Exception:
        pass
    return False

def _load_pkl_if_exists(path: Path):
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

def _xywh_to_xyxy(b):  # [x,y,w,h] -> [x1,y1,x2,y2]
    x, y, w, h = [float(v) for v in b]
    return [x, y, x + w, y + h]

def _area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = _area_xyxy(a) + _area_xyxy(b) - inter
    return inter / max(ua, 1e-8)

def _mask_from_bbox(h: int, w: int, xyxy: List[float]) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m

def _resize_mask_to(mask: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if mask.shape == (th, tw):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), (tw, th), interpolation=cv2.INTER_NEAREST).astype(bool)

def _finite_and_nonzero(pts: np.ndarray) -> np.ndarray:
    good = np.isfinite(pts).all(axis=-1)
    nz = np.linalg.norm(pts, axis=-1) > 1e-12
    return good & nz

def transform_pts_R_offset(pts: np.ndarray, R: np.ndarray, offset: np.ndarray):
    return (R @ pts.T).T + offset[None, :]

def inv_transform_pts_R_offset(pts: np.ndarray, R: np.ndarray, offset: np.ndarray):
    # inverse of x' = R x + o  -> x = R^T (x' - o)
    return (R.T @ (pts - offset[None, :]).T).T

def _box_edges_from_corners(corners: np.ndarray) -> List[np.ndarray]:
    idx_pairs = [
        (0, 1), (0, 2), (0, 4),
        (7, 6), (7, 5), (7, 3),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (4, 5), (4, 6)
    ]
    return [np.vstack([corners[i], corners[j]]) for (i, j) in idx_pairs]


def _lift_2d_to_3d(frame_points_hw3: np.ndarray, u: float, v: float):
    H, W, _ = frame_points_hw3.shape
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or ui >= W or vi < 0 or vi >= H:
        return None
    p3d = frame_points_hw3[vi, ui]
    if not np.isfinite(p3d).all() or np.abs(p3d).sum() < 1e-6:
        return None
    return p3d

def _build_frame_to_kps_map(results: dict, primary_person_id_1: Optional[str]):
        frame_to_kps = {}
        people_results = results.get("people", {})
        if not people_results:
            return frame_to_kps
        if primary_person_id_1 is None:
            primary_person_id_1 = list(people_results.keys())[0]
        pdata = people_results.get(primary_person_id_1, None)
        if pdata is None:
            return frame_to_kps
        frames = pdata.get("frames", None)
        kp_maps = pdata.get("keypoints_2d_map", None)
        if frames is None or kp_maps is None:
            return frame_to_kps
        for i, fidx in enumerate(frames):
            frame_to_kps[fidx] = kp_maps[i]
        return frame_to_kps

# ---------------------------------------------------------------------
# RERUN VIS HELPERS
# ---------------------------------------------------------------------

def _log_box_lines_rr(path: str, corners: np.ndarray,
                      rgba=(255, 255, 255, 255), radius=0.002):
    edges = _box_edges_from_corners(corners)
    for k, e in enumerate(edges):
        e = np.asarray(e, dtype=np.float32)
        rr.log(
            f"{path}/edge_{k}",
            rr.LineStrips3D(
                [e],
                radii=radius,
                colors=[rgba],
            ),
        )

# ------------------------------------------------------------------
# gdino <-> gt match
# ------------------------------------------------------------------
def _match_gdino_to_gt(
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

    passing = [b for (b, s) in candidates if _iou_xyxy(b, gt_xyxy) >= iou_thr]
    if passing:
        x1 = min(p[0] for p in passing)
        y1 = min(p[1] for p in passing)
        x2 = max(p[2] for p in passing)
        y2 = max(p[3] for p in passing)
        return [x1, y1, x2, y2]

    best = max(candidates, key=lambda t: t[1])[0]
    return best

# ------------------------------------------------------------------
# HUMAN MESH ALIGNMENT RELATED FUNCTIONS
# ------------------------------------------------------------------
def _choose_primary_actor(results: dict, world4d: Dict[int, dict]) -> Tuple[Optional[str], Optional[int]]:
    people = results.get("people", {})
    if people:
        counts = {pid: len(pdata.get("frames", [])) for pid, pdata in people.items()}
        primary_person_id_1 = max(counts.items(), key=lambda x: x[1])[0]
        primary_track_id_0 = int(primary_person_id_1) - 1
        return primary_person_id_1, primary_track_id_0

    for _, frame_data in world4d.items():
        tids = frame_data.get("track_id", [])
        if tids:
            track_id_0 = int(tids[0]) if not isinstance(tids[0], torch.Tensor) else int(tids[0].item())
            person_id_1 = str(track_id_0 + 1)
            return person_id_1, track_id_0
    return None, None

def _find_actor_index_in_frame(frame_data: dict, primary_track_id_0: Optional[int]) -> Optional[int]:
    track_ids = frame_data.get("track_id", [])
    if len(track_ids) == 0:
        return None
    if primary_track_id_0 is None:
        return 0
    for idx, tid in enumerate(track_ids):
        tid_val = int(tid.item()) if isinstance(tid, torch.Tensor) else int(tid)
        if tid_val == int(primary_track_id_0):
            return idx
    return None

# ------------------------------------------------------------------
# similarity estimation
# ------------------------------------------------------------------
def _similarity_umeyama(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt
    var_src = (src_c ** 2).sum() / n
    s = (S * np.array([1, 1, np.linalg.det(U @ Vt)])).sum() / var_src
    t = mu_dst - s * (R @ mu_src)
    return s, R, t

def _robust_similarity_ransac(
        src: np.ndarray,
        dst: np.ndarray,
        *,
        max_iters: int = 800,
        inlier_thresh: float = 0.03,
        min_inliers: int = 4,
        scale_bounds: Tuple[float, float] = (0.4, 3.0),
):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape
    N = src.shape[0]
    if N < 3:
        return _similarity_umeyama(src, dst)

    best_num = -1
    best_model = None
    iters = min(max_iters, 100 + 30 * N)
    for _ in range(iters):
        idx = np.random.choice(N, 3, replace=False)
        s_cand, R_cand, t_cand = _similarity_umeyama(src[idx], dst[idx])
        s_cand = float(np.clip(s_cand, scale_bounds[0], scale_bounds[1]))
        src_tf = s_cand * (src @ R_cand.T) + t_cand
        err = np.linalg.norm(dst - src_tf, axis=1)
        inliers = err < inlier_thresh
        num_inl = int(inliers.sum())
        if num_inl > best_num and num_inl >= min_inliers:
            s_ref, R_ref, t_ref = _similarity_umeyama(src[inliers], dst[inliers])
            s_ref = float(np.clip(s_ref, scale_bounds[0], scale_bounds[1]))
            best_num = num_inl
            best_model = (s_ref, R_ref, t_ref)
    if best_model is None:
        return _similarity_umeyama(src, dst)
    return best_model

def _mad_based_mask(values: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """
    values: (N,) array
    returns: boolean mask where True = keep (not an outlier)
    """
    if values.size == 0:
        return np.ones_like(values, dtype=bool)
    median = np.median(values)
    abs_dev = np.abs(values - median)
    mad = np.median(abs_dev) + 1e-8  # to avoid div by zero
    modified_z = 0.6745 * abs_dev / mad
    return modified_z < thresh


def _average_sims_robust(per_frame_sims: Dict[int, Dict[str, Any]],
                         rot_thresh_deg: float = 15.0,
                         trans_thresh_scale: float = 3.5
                         ) -> Tuple[float, np.ndarray, np.ndarray]:
    if len(per_frame_sims) == 0:
        return 1.0, np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

    ws, scales, trans, quats, rotvecs = [], [], [], [], []

    for _, d in per_frame_sims.items():
        w = float(d.get("w", 1.0))
        s = float(d["s"])
        R = np.asarray(d["R"], dtype=np.float64)
        t = np.asarray(d["t"], dtype=np.float64)

        rot = SciRot.from_matrix(R)
        quat = rot.as_quat()
        rotvec = rot.as_rotvec()  # axis * angle

        ws.append(w)
        scales.append(s)
        trans.append(t)
        quats.append(quat)
        rotvecs.append(rotvec)

    ws = np.asarray(ws, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)     # (N, 3)
    quats = np.asarray(quats, dtype=np.float64)     # (N, 4)
    rotvecs = np.asarray(rotvecs, dtype=np.float64) # (N, 3)

    # --- 1) scale outliers (MAD)
    scale_mask = _mad_based_mask(scales, thresh=3.5)

    # --- 2) translation outliers:
    # use L2 norm deviation from median
    trans_norms = np.linalg.norm(trans, axis=1)
    trans_mask = _mad_based_mask(trans_norms, thresh=trans_thresh_scale)

    # --- 3) rotation outliers:
    # compare each rotvec magnitude to median magnitude
    rot_mags = np.linalg.norm(rotvecs, axis=1)  # in radians
    rot_mags_deg = np.degrees(rot_mags)
    # simpler than MAD: hard threshold on degrees
    rot_mask = rot_mags_deg < rot_thresh_deg

    # combine masks
    keep_mask = scale_mask & trans_mask & rot_mask

    if not np.any(keep_mask):
        # fallback to naive average
        ws = ws / (ws.sum() + 1e-8)
        s_avg = float((ws * scales).sum())
        t_avg = (ws[:, None] * trans).sum(axis=0)
        q_avg = (ws[:, None] * quats).sum(axis=0)
        q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-8)
        R_avg = SciRot.from_quat(q_avg).as_matrix().astype(np.float32)
        return s_avg, R_avg, t_avg.astype(np.float32)

    # filter
    ws_f  = ws[keep_mask]
    scales_f = scales[keep_mask]
    trans_f  = trans[keep_mask]
    quats_f  = quats[keep_mask]

    # reweight to sum to 1
    ws_f = ws_f / (ws_f.sum() + 1e-8)

    # scale
    s_avg = float((ws_f * scales_f).sum())

    # translation
    t_avg = (ws_f[:, None] * trans_f).sum(axis=0)

    # rotation: weighted quat average on inliers
    q_avg = (ws_f[:, None] * quats_f).sum(axis=0)
    q_avg = q_avg / (np.linalg.norm(q_avg) + 1e-8)
    R_avg = SciRot.from_quat(q_avg).as_matrix().astype(np.float32)

    return s_avg, R_avg, t_avg.astype(np.float32)

# =====================================================================
# BOUNDING BOX GENERATOR (AABB is floor-aligned)
# =====================================================================
class BBox3DGenerator:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
            output_human_dir_path: Optional[str] = None,
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

        self.dynamic_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.static_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = self.ag_root_directory / "sampled_frames_idx"

        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        os.makedirs(self.bbox_3d_root_dir, exist_ok=True)

        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"
        self.pipeline = AgPipeline(static_cam=False, dynamic_scene_dir_path=self.dynamic_scene_dir_path)
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()

        # segmentation dirs
        self.dynamic_masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.dynamic_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.dynamic_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.dynamic_masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masks' / 'image_based'
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masks' / 'video_based'
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        self.output_human_dir_path = Path(output_human_dir_path)

    def get_video_gt_annotations(self, video_id):
        video_gt_annotations_json_path = self.gt_annotations_root_dir / video_id / "gt_annotations.json"
        if not video_gt_annotations_json_path.exists():
            raise FileNotFoundError(f"GT annotations file not found: {video_gt_annotations_json_path}")

        with open(video_gt_annotations_json_path, "r") as f:
            video_gt_annotations = json.load(f)

        video_gt_bboxes = {}
        for frame_idx, frame_items in enumerate(video_gt_annotations):
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

        return video_gt_bboxes, video_gt_annotations

    def get_video_gdino_annotations(self, video_id):
        video_dynamic_gdino_prediction_file_path = self.dynamic_detections_root_path / f"{video_id}.pkl"
        video_dynamic_predictions = _load_pkl_if_exists(video_dynamic_gdino_prediction_file_path)

        video_static_gdino_prediction_file_path = self.static_detections_root_path / f"{video_id}.pkl"
        video_static_predictions = _load_pkl_if_exists(video_static_gdino_prediction_file_path)

        if video_dynamic_predictions is None:
            video_dynamic_predictions = {}
        if video_static_predictions is None:
            video_static_predictions = {}

        if not video_dynamic_predictions and not video_static_predictions:
            raise ValueError(
                f"No GDINO predictions found for video {video_id}"
            )

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
        return combined_gdino_predictions

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

    def create_label_wise_masks_map(
            self,
            video_id,
            gt_annotations
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], set, set]:
        video_to_frame_to_label_mask: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        frame_stems = []
        for frame_items in gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
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

    def idx_to_frame_idx_path(self, video_id: str):
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        annotated_frame_id_list = [f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith('.png')]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))

        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        annotated_frame_idx_in_sample_idx = []
        for frame_name in annotated_frame_id_list:
            frame_id = int(frame_name[:-4])
            if frame_id in video_sampled_frame_id_list:
                idx_in_sampled = video_sampled_frame_id_list.index(frame_id)
                annotated_frame_idx_in_sample_idx.append(sample_idx.index(idx_in_sampled))


        chosen_frames = [video_sampled_frame_id_list[i] for i in sample_idx]
        frame_idx_frame_path_map = {i: f"{frame_id:06d}.png" for i, frame_id in enumerate(chosen_frames)}
        return frame_idx_frame_path_map, sample_idx, video_sampled_frame_id_list, annotated_frame_id_list, annotated_frame_idx_in_sample_idx

    def annotated_idx_to_frame_idx_path(self, video_id: str):
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        annotated_frame_id_list = [f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith('.png')]
        annotated_frame_id_list.sort(key=lambda x: int(x[:-4]))

        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()

        sample_idx = []
        for frame_name in annotated_frame_id_list:
            frame_id = int(frame_name[:-4])
            if frame_id in video_sampled_frame_id_list:
                idx_in_sampled = video_sampled_frame_id_list.index(frame_id)
                sample_idx.append(idx_in_sampled)

        chosen_frames = [video_sampled_frame_id_list[i] for i in sample_idx]
        frame_idx_frame_path_map = {i: f"{frame_id:06d}.png" for i, frame_id in enumerate(chosen_frames)}
        return frame_idx_frame_path_map, sample_idx, video_sampled_frame_id_list, annotated_frame_id_list, sample_idx

    def _load_points_for_video(self, video_id: str) -> Dict[str, Any]:
        video_dynamic_3d_scene_path = self.dynamic_scene_dir_path / f"{video_id[:-4]}_10" / "predictions.npz"
        video_dynamic_predictions = np.load(video_dynamic_3d_scene_path, allow_pickle=True)

        points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
        imgs_f32 = video_dynamic_predictions["images"]
        camera_poses = video_dynamic_predictions["camera_poses"]
        colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

        conf = None
        if "conf" in video_dynamic_predictions:
            conf = video_dynamic_predictions["conf"]
            if conf.ndim == 4 and conf.shape[-1] == 1:
                conf = conf[..., 0]

        S, H, W, _ = points.shape

        (frame_idx_frame_path_map, sample_idx, video_sampled_frame_id_list,
         annotated_frame_id_list, annotated_frame_idx_in_sample_idx) = self.idx_to_frame_idx_path(video_id)
        assert S == len(sample_idx), "dynamic predictions length must match annotated sampled range"

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

        points_sub = points[annotated_idx_in_sampled_idx]
        conf_sub = conf[annotated_idx_in_sampled_idx] if conf is not None else None
        stems_sub = [sampled_idx_frame_name_map[idx][:-4] for idx in annotated_idx_in_sampled_idx]
        colors_sub = colors[annotated_idx_in_sampled_idx]
        camera_poses_sub = camera_poses[annotated_idx_in_sampled_idx]

        return {
            "points": points_sub,
            "conf": conf_sub,
            "frame_stems": stems_sub,
            "colors": colors_sub,
            "camera_poses": camera_poses_sub
        }


    # ------------------------------------------------------------------
    # lifting and partial pointcloud
    # ------------------------------------------------------------------
    def _get_partial_pointcloud(self, video_id: str, frame_idx: int, frame_idx_frame_path_map,
                                label: str = "person") -> np.ndarray:
        pts_hw3 = self.pipeline.points[frame_idx]
        H, W, _ = pts_hw3.shape
        stem = frame_idx_frame_path_map[frame_idx][:-4]
        mask = self.get_union_mask(video_id, stem, label, is_static=False)
        if mask is not None:
            mask = mask.astype(bool)
            if mask.shape[0] != H or mask.shape[1] != W:
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            pts = pts_hw3[mask]
        else:
            pts = pts_hw3.reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        nonzero = ~(np.abs(pts).sum(axis=1) < 1e-6)
        pts = pts[nonzero]
        return pts

    def _collect_kp_corr_for_frame(self, frame_idx: int, frame_data: dict, frame_to_kps: dict,
                                   primary_track_id_0: Optional[int]):
        if frame_idx not in frame_to_kps:
            return None, None
        actor_idx = _find_actor_index_in_frame(frame_data, primary_track_id_0)
        if actor_idx is None:
            return None, None
        kps_2d = frame_to_kps[frame_idx]
        rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]

        smpl_out = self.smplx(
            global_orient=rotmat_actor[:, :1].cuda(),
            body_pose=rotmat_actor[:, 1:22].cuda(),
            betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
            transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
        )
        joints = smpl_out.joints.cpu().numpy()
        joints = joints[:, :24, :]
        smpl_joints_actor = joints[0]

        frame_points_hw3 = self.pipeline.points[frame_idx]
        smpl_joint_names = get_smpl_joint_names()
        openpose_joint_names = get_openpose_joint_names()
        OPENPOSE_TO_SMPL = {
            "OP Nose": "head",
            "OP Neck": "neck",
            "OP MidHip": "hips",
            "OP LHip": "leftUpLeg",
            "OP RHip": "rightUpLeg",
            "OP LKnee": "leftLeg",
            "OP RKnee": "rightLeg",
            "OP LAnkle": "leftFoot",
            "OP RAnkle": "rightFoot",
            "OP LShoulder": "leftShoulder",
            "OP RShoulder": "rightShoulder",
            "OP LElbow": "leftForeArm",
            "OP RElbow": "rightForeArm",
            "OP LWrist": "leftHand",
            "OP RWrist": "rightHand",
            "OP LBigToe": "leftToeBase",
            "OP RBigToe": "rightToeBase",
        }
        if not smpl_joint_names or len(smpl_joint_names) < smpl_joints_actor.shape[0]:
            return None, None
        smpl_name_to_pt = {smpl_joint_names[j]: smpl_joints_actor[j] for j in range(smpl_joints_actor.shape[0])}

        smpl_list = []
        scene_list = []
        for op_name, kp in kps_2d.items():
            smpl_name = OPENPOSE_TO_SMPL.get(op_name, None)
            if smpl_name is None:
                continue
            if smpl_name not in smpl_name_to_pt:
                continue
            u = float(kp[0]);
            v = float(kp[1])
            scene_p = _lift_2d_to_3d(frame_points_hw3, u, v)
            if scene_p is None:
                continue
            smpl_list.append(smpl_name_to_pt[smpl_name])
            scene_list.append(scene_p)

        if len(smpl_list) == 0:
            return None, None
        return np.stack(smpl_list, axis=0), np.stack(scene_list, axis=0)

    def _collect_dense_corr_for_frame(
            self,
            video_id: str,
            frame_idx: int,
            frame_data: dict,
            frame_idx_frame_path_map: dict,
            primary_track_id_0: Optional[int],
            num_smpl_samples_per_person: int = 400,
            num_scene_subsample: int = 800,
    ):
        actor_idx = _find_actor_index_in_frame(frame_data, primary_track_id_0)
        if actor_idx is None:
            return None, None
        rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
        rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]
        smpl_out = self.smplx(
            global_orient=rotmat_actor[:, :1].cuda(),
            body_pose=rotmat_actor[:, 1:22].cuda(),
            betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
            transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
        )
        verts = smpl_out.vertices.cpu().numpy()
        smpl_verts = verts[0]

        scene_pts = self._get_partial_pointcloud(video_id, frame_idx, frame_idx_frame_path_map, label="person")
        if scene_pts.shape[0] == 0:
            return None, None

        if scene_pts.shape[0] > num_scene_subsample:
            choice = np.random.choice(scene_pts.shape[0], num_scene_subsample, replace=False)
            scene_pts = scene_pts[choice]

        if smpl_verts.shape[0] > num_smpl_samples_per_person:
            idx = np.random.choice(smpl_verts.shape[0], num_smpl_samples_per_person, replace=False)
            smpl_sampled = smpl_verts[idx]
        else:
            smpl_sampled = smpl_verts

        matched_scene = []
        for sp in smpl_sampled:
            dists = np.sum((scene_pts - sp[None, :]) ** 2, axis=1)
            nn_idx = np.argmin(dists)
            matched_scene.append(scene_pts[nn_idx])
        matched_scene = np.asarray(matched_scene, dtype=np.float64)
        return smpl_sampled, matched_scene

    def process_video(self, video_id: str, include_dense: bool = False, use_consistent_transformation: bool = False):
        # 0) run human/scene pipeline
        self.pipeline.__call__(video_id, save_only_essential=False)
        self.pipeline.estimate_2d_keypoints()
        results = self.pipeline.results
        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        # make frame indices 0..N-1
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        # sampled / annotated indices
        (frame_idx_frame_path_map,
         sample_idx,
         _,
         _,
         annotated_frame_idx_in_sample_idx) = self.idx_to_frame_idx_path(video_id)
        sampled_frame_indices = sorted(frame_idx_frame_path_map.keys())

        # choose primary actor
        primary_person_id_1, primary_track_id_0 = _choose_primary_actor(results, world4d)
        print(f"[align] Using primary actor -> results={primary_person_id_1}, world4d_track={primary_track_id_0}")

        # map frame -> keypoints for that actor
        frame_to_kps = _build_frame_to_kps_map(results, primary_person_id_1)

        # containers
        frame_kp_corr: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        per_frame_sims: Dict[int, Dict[str, Any]] = {}

        # ------------------------------------------------------------------
        # 1) estimate per-frame similarity *for sampled frames*
        # ------------------------------------------------------------------
        for frame_idx in sampled_frame_indices:
            frame_data = world4d.get(frame_idx, None)
            if frame_data is None:
                continue

            smpl_s, scene_s = self._collect_kp_corr_for_frame(
                frame_idx, frame_data, frame_to_kps, primary_track_id_0
            )

            smpl_all = []
            scene_all = []

            if smpl_s is not None and scene_s is not None and smpl_s.shape[0] > 0:
                frame_kp_corr[frame_idx] = (smpl_s, scene_s)
                smpl_all.append(smpl_s)
                scene_all.append(scene_s)

            if include_dense:
                smpl_d, scene_d = self._collect_dense_corr_for_frame(
                    video_id, frame_idx, frame_data, frame_idx_frame_path_map, primary_track_id_0
                )
                if smpl_d is not None and scene_d is not None:
                    smpl_all.append(smpl_d)
                    scene_all.append(scene_d)

            if len(smpl_all) == 0:
                print(f"[align][{video_id}] no corr for sampled frame {frame_idx}")
                continue

            smpl_all = np.concatenate(smpl_all, axis=0)
            scene_all = np.concatenate(scene_all, axis=0)

            if smpl_all.shape[0] < 3:
                print(f"[align][{video_id}] insufficient corr for frame {frame_idx}")
                continue

            # solve per-frame sim
            s_f, R_f, t_f = _robust_similarity_ransac(
                smpl_all, scene_all,
                max_iters=800,
                inlier_thresh=0.03,
                min_inliers=4,
                scale_bounds=(0.4, 3.0),
            )

            per_frame_sims[frame_idx] = {
                "s": float(s_f),
                "R": R_f,
                "t": t_f,
                "w": float(smpl_all.shape[0]),
            }

        # ------------------------------------------------------------------
        # 2) compute robust average over sampled frames
        # ------------------------------------------------------------------
        sampled_per_frame_sims = {k: v for k, v in per_frame_sims.items() if k in sampled_frame_indices}
        s_avg, R_avg, t_avg = _average_sims_robust(sampled_per_frame_sims)

        if use_consistent_transformation:
            per_frame_sims = {
                fidx: {
                    "s": float(s_avg),
                    "R": R_avg,
                    "t": t_avg,
                    "w": 1.0,
                }
                for fidx in sampled_frame_indices
            }

        # ------------------------------------------------------------------
        # 3) build verts per sampled frame and apply either per-frame or avg sim
        # ------------------------------------------------------------------
        all_verts_for_floor = []
        for frame_idx in sampled_frame_indices:
            frame_data = world4d.get(frame_idx, None)
            if frame_data is None:
                continue
            actor_idx = _find_actor_index_in_frame(frame_data, primary_track_id_0)
            if actor_idx is None:
                frame_data['track_id'] = []
                continue

            rotmat_all = axis_angle_to_matrix(frame_data['pose'].reshape(-1, 55, 3))
            rotmat_actor = rotmat_all[actor_idx:actor_idx + 1]
            smpl_out = self.smplx(
                global_orient=rotmat_actor[:, :1].cuda(),
                body_pose=rotmat_actor[:, 1:22].cuda(),
                betas=frame_data['shape'][actor_idx:actor_idx + 1].cuda(),
                transl=frame_data['trans'][actor_idx:actor_idx + 1].cuda()
            )
            verts = smpl_out.vertices.cpu().numpy()

            frame_data['track_id'] = [primary_track_id_0]
            frame_data['vertices_orig'] = [verts[0].copy()]

            all_verts_for_floor.append(torch.tensor(verts, dtype=torch.bfloat16))

            if use_consistent_transformation and sampled_per_frame_sims:
                # use the robust average for EVERY frame
                s_use, R_use, t_use = s_avg, R_avg, t_avg
                verts_flat = verts.reshape(-1, 3)
                verts_tf = s_use * (verts_flat @ R_use.T) + t_use
                verts_tf = verts_tf.reshape(verts.shape)
            else:
                # fall back to per-frame sim if we have it
                if frame_idx in per_frame_sims:
                    s_f = per_frame_sims[frame_idx]["s"]
                    R_f = per_frame_sims[frame_idx]["R"]
                    t_f = per_frame_sims[frame_idx]["t"]
                    verts_flat = verts.reshape(-1, 3)
                    verts_tf = s_f * (verts_flat @ R_f.T) + t_f
                    verts_tf = verts_tf.reshape(verts.shape)
                else:
                    verts_tf = verts
            frame_data['vertices'] = [verts_tf[0]]

        # ------------------------------------------------------------------
        # 4) floor mesh from all transformed verts' originals
        # ------------------------------------------------------------------
        if len(all_verts_for_floor) > 0:
            all_verts_for_floor = torch.cat(all_verts_for_floor)
            gv, gf, gc = get_floor_mesh(all_verts_for_floor, scale=2)
        else:
            gv, gf, gc = None, None, None

        return (
            images,
            world4d,
            sampled_frame_indices,
            per_frame_sims,
            s_avg,
            R_avg,
            t_avg,
            gv,
            gf,
            gc,
            annotated_frame_idx_in_sample_idx,
            primary_track_id_0
        )

    def generate_video_bb_annotations(
            self,
            video_id: str,
            video_gt_annotations: List[Any],
            video_gdino_predictions: Dict[str, Any],
            *,
            min_points: int = 50,
            iou_thr: float = 0.3,
            visualize: bool = False,
            use_consistent_transformation: bool = False,
    ) -> None:
        # 1) load dynamic points (already sub-sampled to annotated frames)
        P = self._load_points_for_video(video_id)
        points_S = P["points"]  # (S, H, W, 3)
        conf_S = P["conf"]  # (S, H, W) or None
        stems_S = P["frame_stems"]  # list[str], len S
        colors = P["colors"]
        camera_poses = P["camera_poses"]
        S, H, W, _ = points_S.shape

        stem_to_idx = {stems_S[i]: i for i in range(S)}

        # 2) build label-wise masks (static + dynamic) for this video
        video_to_frame_to_label_mask, _, _ = self.create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_gt_annotations
        )

        # 3) run the human/scene pipeline & get the global floor sim
        (
            images,
            world4d,
            sampled_frame_indices,
            per_frame_sims,
            s_avg,
            R_avg,
            t_avg,
            gv,
            gf,
            gc,
            annotated_frame_idx_in_sampled_idx,
            primary_track_id_0
        ) = self.process_video(video_id, use_consistent_transformation)

        # we will collect bboxes for visualization here
        frame_bbox_meshes: Dict[int, List[Dict[str, Any]]] = {}

        # ----- helper to make a box mesh from 8 world corners -----
        def _make_box_mesh(corners_world: np.ndarray):
            faces = np.array([
                [0, 1, 2], [1, 3, 2],  # min-x side
                [4, 6, 5], [5, 6, 7],  # max-x side
                [0, 4, 1], [1, 4, 5],  # min-y side
                [2, 3, 6], [3, 7, 6],  # max-y side
                [0, 2, 4], [2, 6, 4],  # min-z side
                [1, 5, 3], [3, 5, 7],  # max-z side
            ], dtype=np.uint32)
            return corners_world.astype(np.float32), faces

        # ----- floor transform -----
        has_floor = gv is not None and gf is not None
        s_floor = float(s_avg) if s_avg is not None else 1.0
        R_floor = np.asarray(R_avg, dtype=np.float32) if R_avg is not None else np.eye(3, dtype=np.float32)
        t_floor = np.asarray(t_avg, dtype=np.float32) if t_avg is not None else np.zeros(3, dtype=np.float32)

        # ----- helpers specific to humans -----
        def _get_human_verts_world(world4d: Dict[int, dict], frame_idx: int, track_id: int) -> Optional[np.ndarray]:
            """
            Try a few common keys to get the human mesh vertices for this frame.
            Adjust this if your world4d uses a different key.
            """
            f = world4d.get(frame_idx, None)
            if f is None:
                return None
            # common patterns from your earlier scripts
            if "vertices" in f:
                return np.asarray(f["vertices"][track_id], dtype=np.float32)
            else:
                print(f"[bbox][{video_id}][frame {frame_idx}] no 'vertices' key in world4d frame data")
            return None

        def _floor_align_points(points_world: np.ndarray) -> np.ndarray:
            # world -> floor-local
            return ((points_world - t_floor[None, :]) / s_floor) @ R_floor

        def _floor_to_world(points_floor: np.ndarray) -> np.ndarray:
            # floor-local -> world
            return (points_floor @ R_floor.T) * s_floor + t_floor

        def _corners_from_mins_maxs(mins: np.ndarray, maxs: np.ndarray) -> np.ndarray:
            return np.array([
                [mins[0], mins[1], mins[2]],
                [mins[0], mins[1], maxs[2]],
                [mins[0], maxs[1], mins[2]],
                [mins[0], maxs[1], maxs[2]],
                [maxs[0], mins[1], mins[2]],
                [maxs[0], mins[1], maxs[2]],
                [maxs[0], maxs[1], mins[2]],
                [maxs[0], maxs[1], maxs[2]],
            ], dtype=np.float32)

        def _corners_from_center_dims(center_floor: np.ndarray, dims: np.ndarray) -> np.ndarray:
            """Make floor-aligned cuboid using center + (dx,dy,dz)."""
            half = 0.5 * dims
            mins = center_floor - half
            maxs = center_floor + half
            return _corners_from_mins_maxs(mins, maxs)

        out_frames: Dict[str, Dict[str, Any]] = {}

        for frame_idx_anno, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]
            stem = Path(frame_name).stem

            if stem not in stem_to_idx:
                continue

            sidx = stem_to_idx[stem]
            pts_hw3 = points_S[sidx]
            colors_hw3 = colors[sidx]
            conf_hw = conf_S[sidx] if conf_S is not None else None

            ann_frame_id_in_sampled = annotated_frame_idx_in_sampled_idx[sidx]
            frame_non_zero_pts = _finite_and_nonzero(pts_hw3)

            # gdino per-frame predictions
            gd = video_gdino_predictions.get(frame_name, None)
            if gd is None:
                gd_boxes, gd_labels, gd_scores = [], [], []
            else:
                gd_boxes = [list(map(float, b)) for b in gd["boxes"]]
                gd_labels = gd["labels"]
                gd_scores = [float(s) for s in gd["scores"]]

            frame_rec = {"objects": []}

            # try to read per-frame human mesh (we'll reuse for all "person" in this frame)
            human_mesh_floor_aabb = None
            human_mesh_dims = None
            human_mesh_volume = None
            human_mesh_available = False
            if has_floor:
                human_verts_world = _get_human_verts_world(world4d, ann_frame_id_in_sampled, primary_track_id_0)
                if human_verts_world is not None and human_verts_world.size > 0:
                    human_verts_floor = _floor_align_points(human_verts_world)
                    hmins = human_verts_floor.min(axis=0)
                    hmaxs = human_verts_floor.max(axis=0)
                    human_mesh_dims = (hmaxs - hmins).astype(np.float32)
                    # very small guard to avoid 0 volume
                    human_mesh_volume = float(np.prod(np.maximum(human_mesh_dims, 1e-4)))
                    human_mesh_floor_aabb = (hmins, hmaxs)
                    human_mesh_available = True
                    print(f"[bbox][{video_id}][{frame_name}] human mesh floor-aabb dims {human_mesh_dims}, volume {human_mesh_volume:.4f}")
            else:
                print(f"[bbox][{video_id}][{frame_name}] no floor mesh available; skipping human mesh bbox")

            # iterate over GT objects in this frame
            for item in frame_items:
                if "person_bbox" in item:
                    label = "person"
                    gt_xyxy = _xywh_to_xyxy(item["person_bbox"][0])
                else:
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
                    gt_xyxy = [float(v) for v in item["bbox"]]

                # match GDINO
                chosen_gd_xyxy = _match_gdino_to_gt(
                    label,
                    gt_xyxy,
                    gd_boxes,
                    gd_labels,
                    gd_scores,
                    iou_thr=iou_thr
                )

                # get mask for this label
                frame_label_mask = video_to_frame_to_label_mask[video_id][stem].get(label, None)
                if frame_label_mask is None:
                    box = chosen_gd_xyxy if chosen_gd_xyxy is not None else gt_xyxy
                    frame_label_mask = _mask_from_bbox(H, W, box)
                else:
                    frame_label_mask = _resize_mask_to(frame_label_mask, (H, W))

                sel = frame_label_mask & frame_non_zero_pts
                if conf_hw is not None:
                    sel &= (conf_hw > 1e-6)

                if sel.sum() < min_points:
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "gdino_bbox_xyxy": chosen_gd_xyxy,
                        "num_points": int(sel.sum()),
                        "aabb_floor_aligned": None
                    })
                    continue

                # actual 3D points for this object (world space)
                label_non_zero_pts = pts_hw3[sel].reshape(-1, 3).astype(np.float32)

                if has_floor:
                    # world → floor-local
                    pts_floor = _floor_align_points(label_non_zero_pts)
                    mins = pts_floor.min(axis=0)
                    maxs = pts_floor.max(axis=0)

                    # default corners from point cloud
                    corners_floor = _corners_from_mins_maxs(mins, maxs)
                    corners_world = _floor_to_world(corners_floor)

                    # ---------------- PERSON SPECIAL CASE ----------------
                    if label == "person" and human_mesh_available:
                        # volume from sparse PC box
                        pc_dims = (maxs - mins)
                        pc_volume = float(np.prod(np.maximum(pc_dims, 1e-4)))

                        # allow some slack over mesh volume
                        volume_scale = 1.5
                        use_mesh_like_box = (pc_volume > volume_scale * human_mesh_volume)

                        # center of observed points (in floor coords)
                        pc_center_floor = pts_floor.mean(axis=0)

                        if use_mesh_like_box:
                            print(f"[bbox][{video_id}][{frame_name}] using mesh-shaped box for person "
                                  f"(pc volume {pc_volume:.4f} > {volume_scale} x mesh volume {human_mesh_volume:.4f})")
                            corners_floor = _corners_from_center_dims(pc_center_floor, human_mesh_dims)
                            corners_world = _floor_to_world(corners_floor)
                            frame_rec["objects"].append({
                                "label": label,
                                "gt_bbox_xyxy": gt_xyxy,
                                "gdino_bbox_xyxy": chosen_gd_xyxy,
                                "num_points": int(label_non_zero_pts.shape[0]),
                                "aabb_floor_aligned": {
                                    "mins_floor": (pc_center_floor - 0.5 * human_mesh_dims).tolist(),
                                    "maxs_floor": (pc_center_floor + 0.5 * human_mesh_dims).tolist(),
                                    "corners_world": corners_world.tolist(),
                                    "source": "mesh-shaped-from-volume",
                                    "mesh_volume": human_mesh_volume,
                                    "pc_volume": pc_volume,
                                },
                            })

                            verts_box, faces_box = _make_box_mesh(corners_world)
                            frame_bbox_meshes.setdefault(sidx, []).append({
                                "verts": verts_box,
                                "faces": faces_box,
                                "color": [0, 255, 0],
                                "label": label,
                            })
                            continue

                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "gdino_bbox_xyxy": chosen_gd_xyxy,
                        "num_points": int(label_non_zero_pts.shape[0]),
                        "aabb_floor_aligned": {
                            "mins_floor": mins.tolist(),
                            "maxs_floor": maxs.tolist(),
                            "corners_world": corners_world.tolist(),
                            "source": "pc-aabb",
                        },
                    })

                    verts_box, faces_box = _make_box_mesh(corners_world)
                    frame_bbox_meshes.setdefault(sidx, []).append({
                        "verts": verts_box,
                        "faces": faces_box,
                        "color": [255, 180, 0] if label != "person" else [0, 255, 0],
                        "label": label,
                    })

                else:
                    # fallback: just world AABB
                    mins = label_non_zero_pts.min(axis=0)
                    maxs = label_non_zero_pts.max(axis=0)
                    corners_world = _corners_from_mins_maxs(mins, maxs)
                    frame_rec["objects"].append({
                        "label": label,
                        "gt_bbox_xyxy": gt_xyxy,
                        "gdino_bbox_xyxy": chosen_gd_xyxy,
                        "num_points": int(label_non_zero_pts.shape[0]),
                        "aabb_floor_aligned": {
                            "mins_world": mins.tolist(),
                            "maxs_world": maxs.tolist(),
                            "corners_world": corners_world.tolist(),
                            "source": "world-aabb",
                        },
                    })
                    verts_box, faces_box = _make_box_mesh(corners_world)
                    frame_bbox_meshes.setdefault(sidx, []).append({
                        "verts": verts_box,
                        "faces": faces_box,
                        "color": [255, 0, 0],
                        "label": label,
                    })

            if frame_rec["objects"]:
                out_frames[frame_name] = frame_rec

        # --------------- visualization ---------------
        if visualize:
            rerun_vis_world4d(
                video_id=video_id,
                images=images,
                world4d=world4d,
                faces=self.smplx.faces,
                sampled_indices=sampled_frame_indices,
                annotated_frame_idx_in_sample_idx=annotated_frame_idx_in_sampled_idx,
                dynamic_prediction_path=str(self.dynamic_scene_dir_path),
                per_frame_sims=per_frame_sims,
                global_floor_sim=(s_avg, R_avg, t_avg),
                floor=(gv, gf, gc) if gv is not None else None,
                img_maxsize=480,
                app_id="World4D-Combined",
                frame_bbox_meshes=frame_bbox_meshes,
            )

            print("Visualization running. Press Ctrl+C to stop.")
            while True:
                time.sleep(1)

        # --------------- dump to disk ---------------
        out_path = self.bbox_3d_root_dir / f"{video_id}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump({
                "video_id": video_id,
                "frames": out_frames
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[bbox] saved floor-aligned 3D bboxes to {out_path}")

    def generate_gt_world_bb_annotations(self, dataloader, split) -> None:
        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
                video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
                self.generate_video_bb_annotations(
                    video_id,
                    video_id_gt_annotations,
                    video_id_gdino_annotations,
                    visualize=True
                )

    def generate_sample_gt_world_bb_annotations(self, video_id: str) -> None:
        video_id_gt_bboxes, video_id_gt_annotations = self.get_video_gt_annotations(video_id)
        video_id_gdino_annotations = self.get_video_gdino_annotations(video_id)
        self.generate_video_bb_annotations(
            video_id,
            video_id_gt_annotations,
            video_id_gdino_annotations,
            visualize=True
        )

def rerun_vis_world4d(
        video_id: str,
        images: List[Optional[np.ndarray]],
        world4d: Dict[int, dict],
        faces: np.ndarray,
        *,
        annotated_frame_idx_in_sample_idx: List[int],
        sampled_indices: List[int],
        dynamic_prediction_path: str,
        per_frame_sims: Optional[Dict[int, Dict[str, Any]]] = None,
        global_floor_sim: Optional[Tuple[float, np.ndarray, np.ndarray]] = None,
        floor: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None,
        img_maxsize: int = 320,
        app_id: str = "World4D",
        frame_bbox_meshes: Optional[Dict[int, List[Dict[str, Any]]]] = None,
):
    faces_u32 = _faces_u32(faces)
    rr.init(app_id, spawn=True)
    rr.log("/", rr.ViewCoordinates.RUB)

    video_dynamic_prediction_path = os.path.join(dynamic_prediction_path, f"{video_id[:-4]}_10", "predictions.npz")
    video_dynamic_predictions = np.load(video_dynamic_prediction_path, allow_pickle=True)
    video_dynamic_predictions = {k: video_dynamic_predictions[k] for k in video_dynamic_predictions.files}
    points = video_dynamic_predictions["points"].astype(np.float32)  # (S,H,W,3)
    imgs_f32 = video_dynamic_predictions["images"]
    colors = (imgs_f32 * 255.0).clip(0, 255).astype(np.uint8)

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # floor
    floor_vertices_tf = None
    floor_faces = None
    floor_kwargs = None
    if floor is not None:
        floor_verts0, floor_faces0, floor_colors0 = floor
        floor_verts0 = np.asarray(floor_verts0, dtype=np.float32)
        floor_faces0 = _faces_u32(np.asarray(floor_faces0))
        if global_floor_sim is not None:
            s_g, R_g, t_g = global_floor_sim
            floor_vertices_tf = s_g * (floor_verts0 @ R_g.T) + t_g
        else:
            floor_vertices_tf = floor_verts0
        floor_kwargs = {}
        if floor_colors0 is not None:
            floor_colors0 = np.asarray(floor_colors0, dtype=np.uint8)
            floor_kwargs["vertex_colors"] = floor_colors0
        else:
            floor_kwargs["albedo_factor"] = [160, 160, 160]
        floor_faces = floor_faces0

    def _get_image_for_time(i: int) -> Optional[np.ndarray]:
        if images is None:
            return None
        if i < 0 or i >= len(images):
            return None
        img = images[i]
        if img is None:
            return None
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    # edges for a cuboid (8 vertices) — indices match the order we stored earlier
    cuboid_edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # bottom rectangle
        (4, 5), (5, 7), (7, 6), (6, 4),  # top rectangle
        (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
    ]

    # ------------------------------------------------------------------
    # use only the annotated frames (indices into sampled_indices)
    # ------------------------------------------------------------------
    # if list is empty, we can fall back to showing all sampled_indices
    if annotated_frame_idx_in_sample_idx:
        iter_indices = annotated_frame_idx_in_sample_idx
    else:
        iter_indices = list(range(len(sampled_indices)))

    for vis_t, sample_idx in enumerate(iter_indices):
        # sample_idx is an index into sampled_indices
        if sample_idx < 0 or sample_idx >= len(sampled_indices):
            continue

        frame_idx = sampled_indices[sample_idx]

        # set timeline to a dense 0..N-1 sequence of annotated frames
        rr.set_time_sequence("frame", vis_t)
        rr.log("/", rr.Clear(recursive=True))

        # floor (constant per frame)
        if floor_vertices_tf is not None and floor_faces is not None:
            rr.log(
                f"{BASE}/floor",
                rr.Mesh3D(
                    vertex_positions=floor_vertices_tf.astype(np.float32),
                    triangle_indices=floor_faces,
                    **(floor_kwargs or {}),
                ),
            )

        # per-frame sim
        s_i = None
        R_i = None
        t_i = None
        if per_frame_sims is not None and frame_idx in per_frame_sims:
            s_i = float(per_frame_sims[frame_idx]["s"])
            R_i = np.asarray(per_frame_sims[frame_idx]["R"], dtype=np.float32)
            t_i = np.asarray(per_frame_sims[frame_idx]["t"], dtype=np.float32)

        frame_data = world4d.get(frame_idx, None)
        if frame_data is None:
            continue

        # human meshes (orig is already stored in world4d; we only show transformed)
        track_ids = frame_data.get("track_id", [])
        verts_orig_list = frame_data.get("vertices_orig", [])
        if track_ids and verts_orig_list:
            tid = int(track_ids[0])
            verts_orig = np.asarray(verts_orig_list[0], dtype=np.float32)

            if s_i is not None:
                verts_flat = verts_orig.reshape(-1, 3)
                verts_tf = s_i * (verts_flat @ R_i.T) + t_i
                verts_tf = verts_tf.reshape(verts_orig.shape)
                rr.log(
                    f"{BASE}/humans_xform/h{tid}",
                    rr.Mesh3D(
                        vertex_positions=verts_tf.astype(np.float32),
                        triangle_indices=faces_u32,
                        albedo_factor=[0, 255, 0],
                    ),
                )

        # --- dynamic points: NOTE we index by sample_idx, not vis_t ---
        if sample_idx < points.shape[0]:
            rr.log(
                f"{BASE}/points",
                rr.Points3D(
                    points[sample_idx].reshape(-1, 3),
                    colors=colors[sample_idx].reshape(-1, 3),
                ),
            )

        # --- per-frame cuboid bboxes ---
        if frame_bbox_meshes is not None and vis_t in frame_bbox_meshes:
            for bi, bbox_m in enumerate(frame_bbox_meshes[vis_t]):
                verts_world = bbox_m["verts"].astype(np.float32)  # (8,3)
                col = bbox_m.get("color", [255, 180, 0])

                strips = []
                for e0, e1 in cuboid_edges:
                    strips.append(verts_world[[e0, e1], :])

                rr.log(
                    f"{BASE}/bboxes/frame_{vis_t}/bbox_{bi}",
                    rr.LineStrips3D(
                        strips=strips,
                        colors=[col] * len(strips),
                    ),
                )

        # camera
        cam_3x4 = np.asarray(frame_data["camera"], dtype=np.float32)
        R_wc = cam_3x4[:3, :3]
        t_wc = cam_3x4[:3, 3]
        image = _get_image_for_time(frame_idx)
        if image is not None:
            H_img, W_img = image.shape[:2]
        else:
            H_img, W_img = 480, 640
        fov_y = 0.96
        fx, fy, cx, cy = _pinhole_from_fov(W_img, H_img, fov_y)
        quat_xyzw = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)
        frus_path = f"{BASE}/frustum"
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=t_wc.astype(np.float32),
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            )
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W_img, H_img)),
        )
        if image is not None:
            rr.log(f"{frus_path}/image", rr.Image(image))

    print("Rerun visualization running. Scrub the 'frame' timeline.")


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
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined: (a) floor-aligned 3D bbox generator + (b) SMPL↔PI3 human mesh aligner (sampled frames only)."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument("--dynamic_scene_dir_path", type=str,
                        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic")
    parser.add_argument("--output_human_dir_path", type=str, default="/data/rohith/ag/ag4D/human/")
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument("--include_dense", action="store_true",
                        help="use dense correspondences for human aligner")
    return parser.parse_args()

def main():
    args = parse_args()

    bbox_3d_generator = BBox3DGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_human_dir_path=args.output_human_dir_path,
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    bbox_3d_generator.generate_gt_world_bb_annotations(dataloader=dataloader_train, split=args.split)
    bbox_3d_generator.generate_gt_world_bb_annotations(dataloader=dataloader_test, split=args.split)

def main_sample():
    args = parse_args()

    bbox_3d_generator = BBox3DGenerator(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_human_dir_path=args.output_human_dir_path,
    )
    video_id = "GYAR9.mp4"
    bbox_3d_generator.generate_sample_gt_world_bb_annotations(video_id=video_id)


if __name__ == "__main__":
    main_sample()
