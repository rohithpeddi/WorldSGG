import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, Any, List, DefaultDict, Optional

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from datasets.preprocess.segmentation.ag_dynamic_detection import BaseAgActor


# ---------------------------
# Split logic (yours)
# ---------------------------

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


class AgSegmentation(BaseAgActor):

    def __init__(self, ag_root_directory):
        super().__init__(ag_root_directory)

        self._use_amp = None
        self.sam2_video_predictor = None
        self.sam2_image_predictor = None

        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation_static" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        for p in [
            self.masked_frames_im_dir_path,
            self.masked_frames_vid_dir_path,
            self.masked_frames_combined_dir_path,
            self.masked_videos_dir_path,
            self.masks_im_dir_path,
            self.masks_vid_dir_path,
            self.masks_combined_dir_path,
            self.sampled_frames_jpg,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self.load_sam2_model()

    # -------------------------------------- LOADING INFORMATION -------------------------------------- #
    def load_sam2_model(self):
        # Use a balanced checkpoint (you can switch to 'facebook/sam2-hiera-large' if you have headroom)
        self.sam2_image_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
        self.sam2_video_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-base-plus")

        # Prefer bfloat16 autocast on CUDA; fall back to regular inference on CPU.
        self._use_amp = (self.device.type == "cuda")

    # -------------------------------------- SEGMENTATION (SAM2) -------------------------------------- #

    def segment_with_sam2(self, video_id):
        frames_dir = Path(self.ag_root_directory) / "sampled_frames" / video_id
        pkl_path = Path(self.bbox_static_dir_path) / f"{video_id}.pkl"
        if not pkl_path.exists():
            print(f"[segment_with_sam2][{video_id}] Missing detections ({pkl_path}). Skipping.")
            return

        with open(pkl_path, "rb") as f:
            dets: Dict[str, Dict[str, Any]] = pickle.load(f)

        out_mask_dir = self.masks_im_dir_path / video_id
        out_frames_dir = self.masked_frames_im_dir_path / video_id
        self._ensure_dir(out_mask_dir)
        self._ensure_dir(out_frames_dir)

        frame_names = sorted([fn for fn in dets.keys() if (frames_dir / fn).exists()])
        for fn in tqdm(frame_names, desc=f"SAM2 (image) {video_id}"):
            img_p = frames_dir / fn
            img = Image.open(img_p).convert("RGB")
            img_np = np.array(img)

            boxes: torch.Tensor = dets[fn]["boxes"]
            labels: List[str] = dets[fn]["labels"]
            scores: torch.Tensor = dets[fn]["scores"]

            # Group detections by label
            by_label: DefaultDict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
            for b, l, s in zip(boxes.cpu().numpy(), labels, scores.cpu().numpy()):
                by_label[l].append((b.astype(np.float32), float(s)))

            # Run predictor per frame cast in AMP on CUDA (bf16), else normal mode
            if self._use_amp:
                amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                class _Noop:
                    def __enter__(self): return None

                    def __exit__(self, *args): return False

                amp_ctx = _Noop()

            union_mask = np.zeros(img_np.shape[:2], dtype=bool)
            with torch.inference_mode(), amp_ctx:
                self.sam2_image_predictor.set_image(img_np)
                for lbl, items in by_label.items():
                    lbl_mask = np.zeros(img_np.shape[:2], dtype=bool)
                    for box_px, _ in items:
                        # SAM2 returns (N, H, W) masks; we pick the best mask (index 0 if multimask_output=False)
                        masks, _, _ = self.sam2_image_predictor.predict(
                            box=box_px[None, :], multimask_output=False
                        )
                        m = np.array(masks[0]).astype(bool)
                        lbl_mask |= m
                    # save per-object binary mask
                    save_p = out_mask_dir / f"{Path(fn).stem}__{lbl}.png"
                    cv2.imwrite(str(save_p), self._binary_to_png(lbl_mask))
                    union_mask |= lbl_mask

            # save union-masked frame (all objects together)
            masked_np = self._apply_mask(img_np, union_mask)
            cv2.imwrite(str(out_frames_dir / fn), cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR))

    def segment_with_sam2_video_mode(self, video_id: str, mask_threshold: float = 0.5, min_area: int = 0) -> None:
        # -------------------------------
        # Robust binarization helper
        # -------------------------------
        def _as_bool_mask(x, thr: float = mask_threshold) -> np.ndarray:
            """
            Convert torch/numpy mask (logits or probabilities; shapes [H,W], [1,H,W], [H,W,1], [N,H,W], etc.)
            into a 2D boolean mask with robust handling.
            """
            # to numpy float32 on CPU
            if isinstance(x, torch.Tensor):
                x = x.detach().to(dtype=torch.float32, device="cpu").numpy()
            x = np.asarray(x)

            # squeeze singleton dimensions
            x = np.squeeze(x)

            # if still 3D (e.g., [C,H,W] or [H,W,C]), reduce to single channel
            if x.ndim == 3:
                # prefer channel-first common case [C,H,W]
                if x.shape[0] in (1, 3):
                    x = x[0] if x.shape[0] == 1 else x.mean(axis=0)
                # else handle channel-last [H,W,C]
                elif x.shape[-1] in (1, 3):
                    x = x[..., 0] if x.shape[-1] == 1 else x.mean(axis=-1)
                else:
                    # fallback: average all channels
                    x = x.mean(axis=0)

            # sanitize NaNs/Inf
            x = np.nan_to_num(x, copy=False)

            # if outside [0,1], treat as logits (apply sigmoid) or rescale if narrow range
            x_min, x_max = float(x.min()), float(x.max())
            if not (0.0 <= x_min and x_max <= 1.0):
                # large dynamic range -> likely logits
                if (x_max - x_min) > 6.0 or x_max > 2.0 or x_min < -2.0:
                    x = 1.0 / (1.0 + np.exp(-x))
                else:
                    denom = (x_max - x_min) if x_max != x_min else 1.0
                    x = (x - x_min) / denom

            mask = (x >= thr)

            # optional: remove tiny components
            if min_area > 0:
                m8 = mask.astype(np.uint8)
                num, labels, stats, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
                keep = np.zeros_like(mask, dtype=bool)
                for lab in range(1, num):
                    if stats[lab, cv2.CC_STAT_AREA] >= min_area:
                        keep |= (labels == lab)
                mask = keep
            return mask

        # -------------------------------
        # Paths & detections
        # -------------------------------
        frames_dir = Path(self.ag_root_directory) / "sampled_frames" / video_id
        video_annotated_frames_dir = Path(self.ag_root_directory) / "frames_annotated" / video_id

        pkl_path = Path(self.bbox_static_dir_path) / f"{video_id}.pkl"
        if not pkl_path.exists():
            print(f"[segment_with_sam2_video_mode][{video_id}] Missing detections ({pkl_path}). Skipping.")
            return

        with open(pkl_path, "rb") as f:
            dets: Dict[str, Dict[str, Any]] = pickle.load(f)

        # -------------------------------
        # First-occurrence fallback map
        # -------------------------------
        first_occurrence: Dict[str, Tuple[str, np.ndarray]] = {}
        frame_names_sorted = sorted([fn for fn in dets.keys() if (frames_dir / fn).exists()])

        seen_labels = set()
        for fn in frame_names_sorted:
            rec = dets[fn]
            labels_raw = rec["labels"]
            # Normalize labels to Python list[str]
            if isinstance(labels_raw, torch.Tensor):
                labels_list = labels_raw.cpu().tolist()
            elif isinstance(labels_raw, (list, tuple, np.ndarray)):
                labels_list = list(labels_raw)
            else:
                labels_list = [labels_raw]
            labels = [str(l) for l in labels_list]

            boxes = rec["boxes"]
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy()
            boxes = np.asarray(boxes, dtype=np.float32)

            scores = rec["scores"]
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            scores = np.asarray(scores, dtype=np.float32)

            # best per label on this frame
            per_label_best: Dict[str, Tuple[np.ndarray, float]] = {}
            for b, l, s in zip(boxes, labels, scores):
                if l not in per_label_best or s > per_label_best[l][1]:
                    per_label_best[l] = (b, float(s))

            for l, (b, s) in per_label_best.items():
                if l not in seen_labels:
                    first_occurrence[l] = (fn, b.astype(np.float32))
                    seen_labels.add(l)

        if not first_occurrence:
            print(f"[segment_with_sam2_video_mode][{video_id}] No objects found in detections.")
            return

        # Mirror PNG->JPG for stable indexing (jpg_order drives frame_idx)
        jpg_dir, jpg_order = self._mirror_pngs_to_jpg(frames_dir, video_id)
        stem_to_idx = {Path(n).stem: idx for idx, n in enumerate(jpg_order)}

        # -------------------------------
        # Output dirs
        # -------------------------------
        out_mask_dir = self.masks_vid_dir_path / video_id
        out_frames_dir = self.masked_frames_vid_dir_path / video_id
        self._ensure_dir(out_mask_dir)
        self._ensure_dir(out_frames_dir)

        # -------------------------------
        # Collect anchor frames (if any)
        # -------------------------------
        annotated_stems: set = set()
        if video_annotated_frames_dir.exists():
            for fn in os.listdir(video_annotated_frames_dir):
                p = Path(fn)
                if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    annotated_stems.add(p.stem)

        # seeds_by_frame: stem -> [(label, box[x1,y1,x2,y2], score)]
        from collections import defaultdict
        seeds_by_frame: Dict[str, List[Tuple[str, np.ndarray, float]]] = defaultdict(list)

        # Fallback: use detections on annotated frames
        if annotated_stems:
            for fn, rec in dets.items():
                stem = Path(fn).stem
                if stem not in annotated_stems:
                    continue

                labels_raw = rec["labels"]
                if isinstance(labels_raw, torch.Tensor):
                    labels_list = labels_raw.cpu().tolist()
                elif isinstance(labels_raw, (list, tuple, np.ndarray)):
                    labels_list = list(labels_raw)
                else:
                    labels_list = [labels_raw]
                labels = [str(l) for l in labels_list]

                boxes = rec["boxes"]
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                boxes = np.asarray(boxes, dtype=np.float32)

                scores = rec["scores"]
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                scores = np.asarray(scores, dtype=np.float32)

                best = {}
                for b, l, s in zip(boxes, labels, scores):
                    if (l not in best) or s > best[l][1]:
                        best[l] = (b, float(s))
                for l, (b, s) in best.items():
                    seeds_by_frame[stem].append((l, b.astype(np.float32), s))

        use_anchor_seeds = len(seeds_by_frame) > 0

        # Label universe & obj-id mapping
        if use_anchor_seeds:
            all_labels = sorted({l for items in seeds_by_frame.values() for (l, _, __) in items})
        else:
            all_labels = sorted(first_occurrence.keys())

        if not all_labels:
            print(f"[segment_with_sam2_video_mode][{video_id}] No objects to seed.")
            return

        lbl_to_objid = {lbl: i for i, lbl in enumerate(all_labels)}
        objid_to_lbl = {i: lbl for lbl, i in lbl_to_objid.items()}

        # -------------------------------
        # Select spaced/high-quality seeds
        # -------------------------------
        max_seeds_per_label = getattr(self, "max_seeds_per_label", 10)  # tune as needed
        anchor_min_gap = getattr(self, "anchor_min_gap", 10)  # frames between seeds per label

        # Candidates as (label, frame_idx, box, score)
        candidates: List[Tuple[str, int, np.ndarray, float]] = []
        if use_anchor_seeds:
            for stem, items in seeds_by_frame.items():
                if stem not in stem_to_idx:
                    continue
                idx = stem_to_idx[stem]
                for (l, b, s) in items:
                    if l in lbl_to_objid:
                        candidates.append((l, idx, b, float(s)))

        # If no anchor candidates, synthesize from first_occurrence (one per label)
        if not candidates and not use_anchor_seeds:
            for lbl, (first_fn, box_px) in first_occurrence.items():
                stem = Path(first_fn).stem
                if stem in stem_to_idx:
                    idx = stem_to_idx[stem]
                    candidates.append((lbl, idx, box_px.astype(np.float32), 1.0))

        # Per-label selection: highest score, spaced by >= anchor_min_gap
        from collections import defaultdict as dd
        selected_by_frame: Dict[int, List[Tuple[str, np.ndarray]]] = dd(list)
        if candidates:
            by_label: Dict[str, List[Tuple[int, np.ndarray, float]]] = dd(list)
            for l, idx, b, s in candidates:
                by_label[l].append((idx, b, s))
            for l, lst in by_label.items():
                lst.sort(key=lambda x: (-x[2], x[0]))  # score desc, frame idx asc
                chosen: List[Tuple[int, np.ndarray]] = []
                used: List[int] = []
                for idx, b, _s in lst:
                    if all(abs(idx - u) >= anchor_min_gap for u in used):
                        chosen.append((idx, b))
                        used.append(idx)
                    if len(chosen) >= max_seeds_per_label:
                        break
                if not chosen and lst:
                    chosen.append((lst[0][0], lst[0][1]))
                for idx, b in chosen:
                    selected_by_frame[idx].append((l, b.astype(np.float32)))
        else:
            # Edge case: nothing selected -> first occurrence fallback
            for lbl, (first_fn, box_px) in first_occurrence.items():
                stem = Path(first_fn).stem
                if stem in stem_to_idx:
                    idx = stem_to_idx[stem]
                    selected_by_frame[idx].append((lbl, box_px.astype(np.float32)))

        # -------------------------------
        # Init predictor & add ALL seeds
        # -------------------------------
        # Use JPG directory so frame_idx aligns with jpg_order/stem_to_idx
        video_path = str(self.ag_root_directory / "sampled_videos" / video_id)
        state = self.sam2_video_predictor.init_state(video_path=video_path)
        # state = self.sam2_video_predictor.init_state(video_path=str(jpg_dir))

        # Add seeds for multiple frames/labels up-front; one object_id per label
        for frame_idx in sorted(selected_by_frame.keys()):
            for (lbl, box_px) in selected_by_frame[frame_idx]:
                self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=int(frame_idx),
                    obj_id=int(lbl_to_objid[lbl]),
                    box=box_px.astype(np.float32),
                )

        # AMP guard
        if getattr(self, "_use_amp", False):
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Noop:
                def __enter__(self): return None

                def __exit__(self, *args): return False

            amp_ctx = _Noop()

        # -------------------------------
        # Single pass propagation & saving
        # -------------------------------
        with torch.inference_mode(), amp_ctx:
            for frame_idx, object_ids, masks in self.sam2_video_predictor.propagate_in_video(state):
                # Resolve original frame name
                jpg_name = jpg_order[frame_idx]
                stem = Path(jpg_name).stem
                candidate_png = frames_dir / f"{stem}.png"
                candidate_jpg = frames_dir / f"{stem}.jpg"
                src_img_path = candidate_png if candidate_png.exists() else candidate_jpg
                if not src_img_path.exists():
                    src_img_path = Path(jpg_dir) / jpg_name  # fallback

                img = Image.open(src_img_path).convert("RGB")
                img_np = np.array(img)  # HxWx3 (RGB)
                H, W = img_np.shape[:2]
                union_mask = np.zeros((H, W), dtype=bool)

                # Iterate over masks aligned with object_ids
                K = len(object_ids)
                for k in range(K):
                    # object id -> label
                    oid = object_ids[k]
                    oid_i = int(oid.item() if hasattr(oid, "item") else oid)
                    lbl = objid_to_lbl.get(oid_i, f"obj{oid_i}")

                    # Convert the k-th mask to boolean with robust binarization
                    m_bool = _as_bool_mask(masks[k])

                    # Ensure mask matches frame size
                    if m_bool.shape != (H, W):
                        m_bool = cv2.resize(m_bool.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(
                            bool)

                    union_mask |= m_bool

                    # Save per-object binary mask as 0/255 PNG
                    save_p = out_mask_dir / f"{stem}__{lbl}.png"
                    cv2.imwrite(str(save_p), (m_bool.astype(np.uint8) * 255))

                # Save union-masked frame
                masked_np = self._apply_mask(img_np, union_mask)  # expects a boolean mask
                cv2.imwrite(str(out_frames_dir / f"{stem}.png"), cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR))

    def combine_masks(self, video_id):
        """
        For each frame, read per-label masks from both sources (image- and video-mode),
        threshold to boolean, take their union, and WRITE A BINARY MASK (0/255) so all
        foreground pixels are white (255). Also export the union-masked RGB frame.
        """
        frames_dir = Path(self.ag_root_directory) / "sampled_frames" / video_id
        out_masks_combined_dir = self.masks_combined_dir_path / video_id
        out_frames_combined_dir = self.masked_frames_combined_dir_path / video_id
        self._ensure_dir(out_masks_combined_dir)
        self._ensure_dir(out_frames_combined_dir)

        # collect all frame stems present
        stems = sorted(
            {Path(fn).stem for fn in os.listdir(frames_dir) if fn.lower().endswith((".png", ".jpg", ".jpeg"))}
        )

        # collect all label names that appear in either route for this video
        def _labels_in(mask_dir: Path) -> set:
            lbls = set()
            vid_dir = mask_dir / video_id
            if not vid_dir.exists():
                return lbls
            for fn in os.listdir(vid_dir):
                if "__" in fn and fn.endswith(".png"):
                    lbls.add(fn.split("__", 1)[1].rsplit(".png", 1)[0])
            return lbls

        labels_all = _labels_in(self.masks_im_dir_path) | _labels_in(self.masks_vid_dir_path)

        for stem in tqdm(stems, desc=f"Combine masks {video_id}"):
            # build union across labels for frame image export
            union_frame = None  # keep as boolean

            for lbl in labels_all:
                im_mask_p = self.masks_im_dir_path / video_id / f"{stem}__{lbl}.png"
                vd_mask_p = self.masks_vid_dir_path / video_id / f"{stem}__{lbl}.png"

                m_im = cv2.imread(str(im_mask_p), cv2.IMREAD_GRAYSCALE) if im_mask_p.exists() else None
                m_vd = cv2.imread(str(vd_mask_p), cv2.IMREAD_GRAYSCALE) if vd_mask_p.exists() else None

                if m_im is None and m_vd is None:
                    continue

                # --- threshold to boolean ---
                if m_im is None:
                    m_union = (m_vd > 127)
                elif m_vd is None:
                    m_union = (m_im > 127)
                else:
                    m_union = (m_im > 127) | (m_vd > 127)

                # --- save combined per-object mask as STRICT BINARY (0 or 255) ---
                # m_union_u8 = (m_union.astype(np.uint8) * 255)  # foreground=255 (white), background=0 (black)
                # save_p = out_masks_combined_dir / f"{stem}__{lbl}.png"
                # cv2.imwrite(str(save_p), m_union_u8)

                # accumulate into per-frame union (boolean)
                if union_frame is None:
                    union_frame = m_union.copy()
                else:
                    union_frame |= m_union

            # save union-masked frame (combined)
            if union_frame is not None:
                # find original frame
                png_p = frames_dir / f"{stem}.png"
                jpg_p = frames_dir / f"{stem}.jpg"
                src = png_p if png_p.exists() else jpg_p
                img = Image.open(src).convert("RGB")
                img_np = np.array(img)
                masked_np = self._apply_mask(img_np, union_frame.astype(bool))
                cv2.imwrite(str(out_frames_combined_dir / f"{stem}.png"), cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR))

                # (optional) also save the per-frame union mask itself as binary:
                union_mask_path = out_masks_combined_dir / f"{stem}.png"
                cv2.imwrite(str(union_mask_path), (union_frame.astype(np.uint8) * 255))
            else:
                print(f"[combine_masks][{video_id}][{stem}] No masks found in either route. Writing placeholders.")
                # Edge case: no masks from either route for this frame -> write empty mask + blacked frame
                # Steps:
                # No masks from either route for this frame: write placeholders
                # 1) Find the original frame (try .png, .jpg, .jpeg)
                src = None
                for ext in (".png", ".jpg", ".jpeg"):
                    cand = frames_dir / f"{stem}{ext}"
                    if cand.exists():
                        src = cand
                        break
                if src is None:
                    print(f"[combine_masks][{video_id}][{stem}] Frame not found (no .png/.jpg/.jpeg). Skipping.")
                    continue

                # 2) Load image to get H, W and keep RGB for consistency
                img = Image.open(src).convert("RGB")
                h, w = img.height, img.width

                # 3) Save an EMPTY (all-zero) union mask in the combined masks dir
                empty_mask_u8 = np.zeros((h, w), dtype=np.uint8)  # 0 = background (black)
                cv2.imwrite(str(out_masks_combined_dir / f"{stem}.png"), empty_mask_u8)

                # 4) Save a union-masked RGB frame placeholder
                #    (semantics match the masked frame logic: applying an empty mask -> all black)
                masked_np = self._apply_mask(np.array(img), empty_mask_u8.astype(bool))
                cv2.imwrite(
                    str(out_frames_combined_dir / f"{stem}.png"),
                    cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR),
                )

    def save_masked_frames_and_videos(self, video_id):
        routes = [
            ("image_based", self.masked_frames_im_dir_path / video_id),
            ("video_based", self.masked_frames_vid_dir_path / video_id),
            ("combined_masks", self.masks_combined_dir_path / video_id),
            ("combined_frames", self.masked_frames_combined_dir_path / video_id),
        ]
        for route_name, frames_dir in routes:
            if not frames_dir.exists():
                continue

            mp4_video_directory = self.masked_videos_dir_path / f"{route_name}"
            mp4_video_directory.mkdir(parents=True, exist_ok=True)

            # write video from frames
            out_mp4 = mp4_video_directory / f"{Path(video_id).stem}.mp4"
            if out_mp4.exists() and out_mp4.stat().st_size > 0:
                print(f"[save_masked_frames_and_videos][{video_id}][{route_name}] Skipping (already done: {out_mp4})")
                continue

            self._write_video_from_frames(frames_dir, out_mp4, fps=10)

    def process(self, split):
        # video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4", "0A8CF.mp4"]
        # for video_id in tqdm(video_id_list):
        #     # Skip if already done
        #     out_mp4 = self.masked_videos_dir_path / "combined_frames" / f"{video_id[:-4]}.mp4"
        #     if out_mp4.exists() and out_mp4.stat().st_size > 0:
        #         print(f"[process][{video_id}] Skipping (already done: {out_mp4})")
        #         continue
        #
        #     self.segment_with_sam2(video_id)
        #     self.segment_with_sam2_video_mode(video_id)
        #     self.combine_masks(video_id)
        #     self.save_masked_frames_and_videos(video_id)

        for data in tqdm(self._dataloader_train):
            video_id = data['video_id']
            out_mp4 = self.masked_videos_dir_path / "combined_frames" / f"{video_id[:-4]}.mp4"
            if out_mp4.exists() and out_mp4.stat().st_size > 0:
                print(f"[process][{video_id}] Skipping (already done: {out_mp4})")
                continue
            if get_video_belongs_to_split(video_id) == split:
                self.segment_with_sam2(video_id)
                self.segment_with_sam2_video_mode(video_id)
                self.combine_masks(video_id)
                self.save_masked_frames_and_videos(video_id)
        for data in tqdm(self._dataloader_test):
            video_id = data['video_id']
            out_mp4 = self.masked_videos_dir_path / "combined_frames" / f"{video_id[:-4]}.mp4"
            if out_mp4.exists() and out_mp4.stat().st_size > 0:
                print(f"[process][{video_id}] Skipping (already done: {out_mp4})")
                continue
            if get_video_belongs_to_split(video_id) == split:
                self.segment_with_sam2(video_id)
                self.segment_with_sam2_video_mode(video_id)
                self.combine_masks(video_id)
                self.save_masked_frames_and_videos(video_id)


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from videos based on homography-overlap filtering."
    )
    parser.add_argument(
        "--data_dir", type=str, default="/data/rohith/ag",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    parser.add_argument(
        "--split", type=_parse_split, default="04",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}. "
             "If omitted, processes all videos."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    split = args.split
    ag_actor_segmentation = AgSegmentation(data_dir)
    ag_actor_segmentation.process(split)

if __name__ == "__main__":
    main()
