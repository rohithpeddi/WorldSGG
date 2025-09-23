import argparse
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, DefaultDict

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from datasets.preprocess.samplers.feature_descripter_sampler import get_video_belongs_to_split
from datasets.preprocess.segmentation.ag_detection import BaseAgActor


# TODO: Apply dilation to expand masks slightly to cover object boundaries better?

class AgSegmentation(BaseAgActor):

    def __init__(self, ag_root_directory):
        super().__init__(ag_root_directory)

        self._use_amp = None
        self.sam2_video_predictor = None
        self.sam2_image_predictor = None

        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

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
        pkl_path = Path(self.bbox_dir_path) / f"{video_id}.pkl"
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
        """
        Runs SAM2 in video mode and writes per-object binary masks + union-masked frames.
        Args:
            video_id: ID/name of the video (directory under sampled_frames)
            mask_threshold: probability/logit threshold used to binarize masks
            min_area: remove connected components smaller than this (in pixels); 0 disables
        """

        import pickle
        from typing import Dict, Tuple, Any
        from pathlib import Path

        import cv2
        import numpy as np
        import torch
        from PIL import Image

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

        frames_dir = Path(self.ag_root_directory) / "sampled_frames" / video_id
        pkl_path = Path(self.bbox_dir_path) / f"{video_id}.pkl"
        if not pkl_path.exists():
            print(f"[segment_with_sam2_video_mode][{video_id}] Missing detections ({pkl_path}). Skipping.")
            return

        with open(pkl_path, "rb") as f:
            dets: Dict[str, Dict[str, Any]] = pickle.load(f)

        # Build a map: label -> (first_frame_name, best_box_on_that_frame)
        first_occurrence: Dict[str, Tuple[str, np.ndarray]] = {}
        frame_names_sorted = sorted([fn for fn in dets.keys() if (frames_dir / fn).exists()])

        seen_labels = set()
        for fn in frame_names_sorted:
            labels = dets[fn]["labels"]
            boxes = dets[fn]["boxes"].cpu().numpy().astype(np.float32)
            scores = dets[fn]["scores"].cpu().numpy().astype(np.float32)

            # for each label on this frame, if unseen, record the highest-score box
            per_label_best: Dict[str, Tuple[np.ndarray, float]] = {}
            for b, l, s in zip(boxes, labels, scores):
                if l not in per_label_best or s > per_label_best[l][1]:
                    per_label_best[l] = (b, s)

            for l, (b, s) in per_label_best.items():
                if l not in seen_labels:
                    first_occurrence[l] = (fn, b)
                    seen_labels.add(l)

        if not first_occurrence:
            print(f"[segment_with_sam2_video_mode][{video_id}] No objects found in detections.")
            return

        # If your predictor truly needs JPEGs on disk, mirror PNG->JPG. Otherwise you can pass a video file.
        jpg_dir, jpg_order = self._mirror_pngs_to_jpg(frames_dir, video_id)

        # Map frame name -> index in jpg_order (strip extension)
        stem_to_idx = {Path(n).stem: idx for idx, n in enumerate(jpg_order)}

        out_mask_dir = self.masks_vid_dir_path / video_id
        out_frames_dir = self.masked_frames_vid_dir_path / video_id
        self._ensure_dir(out_mask_dir)
        self._ensure_dir(out_frames_dir)

        # Prepare object id mapping (stable small ints)
        labels_sorted = sorted(first_occurrence.keys())
        lbl_to_objid = {lbl: i for i, lbl in enumerate(labels_sorted)}
        objid_to_lbl = {i: lbl for lbl, i in lbl_to_objid.items()}

        # AMP guard
        if getattr(self, "_use_amp", False):
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Noop:
                def __enter__(self): return None

                def __exit__(self, *args): return False

            amp_ctx = _Noop()

        with torch.inference_mode(), amp_ctx:
            # Either pass a directory of JPG frames or a video file; choose what your predictor expects.
            # state = self.sam2_video_predictor.init_state(video_path=str(jpg_dir))
            video_path = str(self.ag_root_directory / "sampled_videos" / video_id)
            state = self.sam2_video_predictor.init_state(video_path=video_path)

            # Seed with one box per object at its first occurrence frame
            for lbl, (first_fn, box_px) in first_occurrence.items():
                obj_id = lbl_to_objid[lbl]
                ann_frame_idx = stem_to_idx[Path(first_fn).stem]
                self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    box=box_px.astype(np.float32),
                )

            # Propagate through video
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
                # masks is typically a torch.Tensor with shape [K, H, W] or [K, 1, H, W]
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
            self._write_video_from_frames(frames_dir, out_mp4, fps=10)

    def process(self, split):
        video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]
        for video_id in tqdm(video_id_list):
            self.segment_with_sam2(video_id)
            self.segment_with_sam2_video_mode(video_id)
            self.combine_masks(video_id)
            self.save_masked_frames_and_videos(video_id)

        # for data in tqdm(self._dataloader_train):
        #     video_id = data['video_id']
        #     if get_video_belongs_to_split(video_id) == split:
        #         self.segment_with_sam2(video_id)
        #         self.segment_with_sam2_video_mode(video_id)
        #         self.combine_masks(video_id)
        #         self.save_masked_frames_and_videos(video_id)
        # for data in tqdm(self._dataloader_test):
        #     video_id = data['video_id']
        #     if get_video_belongs_to_split(video_id) == split:
        #         self.segment_with_sam2(video_id)
        #         self.segment_with_sam2_video_mode(video_id)
        #         self.combine_masks(video_id)
        #         self.save_masked_frames_and_videos(video_id)


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
        "--split", type=_parse_split, default=None,
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
