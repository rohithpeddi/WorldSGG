import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, DefaultDict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
# Image predictor (promptable with boxes/points)
from sam2.sam2_image_predictor import SAM2ImagePredictor
# Video predictor (stateful propagation across frames)
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from dataloader.coco.action_genome.ag_dataset import StandardAGCoCoDataset
from utils import get_color_map


# 1. Load the dataset, get the objects present in the dataset annotations.
# 2. Use gdino to extract bounding boxes.
# 3. Segmentation Route 1: Use SAM2 to get the masks for the objects in individual frames.
# 4. Segmentation Route 2:
#       (a) Identify the first frame occurrence of each object from annotations.
#       (b) Use SAM2 video mode to propagate and get the masks for each frame.
# 5. Take union of masks from both routes to get the final masks for each object in each frame.
# 6. Save masked frames and masked videos.


class AgActorSegmentation:

    def __init__(self, ag_root_directory):
        self._use_amp = None
        self.sam2_video_predictor = None
        self.sam2_image_predictor = None
        self.gdino_object_labels = None
        self.gdino_model = None
        self.gdino_processor = None
        self.gdino_device = None
        self.gdino_model_id = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._train_dataset = None

        self.ag_root_directory = Path(ag_root_directory)
        self.bbox_dir_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.gdino_vis_path = self.ag_root_directory / "detection" / 'gdino_vis'
        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        # temp JPG mirror for SAM2 video predictor (expects JPEG frames)
        self.sam2_jpg_tmp = self.ag_root_directory / "segmentation" / "tmp_jpg"

        for p in [
            self.masked_frames_im_dir_path,
            self.masked_frames_vid_dir_path,
            self.masked_frames_combined_dir_path,
            self.masked_videos_dir_path,
            self.masks_im_dir_path,
            self.masks_vid_dir_path,
            self.masks_combined_dir_path,
            self.sam2_jpg_tmp,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.load_dataset()
        self.load_gdino_model()
        self.load_sam2_model()  # >>> filled

        self.video_id_active_objects_map = {}
        self.process_video_id_active_objects_map()

    # -------------------------------------- LOADING INFORMATION -------------------------------------- #

    def load_dataset(self):
        self._train_dataset = StandardAGCoCoDataset(
            phase="train",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=True
        )

        self._test_dataset = StandardAGCoCoDataset(
            phase="test",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=True
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
            pin_memory=False,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=lambda b: b[0],  # you use batch_size=1; just pass the item through,
            pin_memory=False
        )

    def process_video_id_active_objects_map(self):
        for data in self._dataloader_train:
            video_id = data['video_id']
            gt_annotations = data['gt_annotations']
            active_objects = set()
            for frame_items in gt_annotations:
                for item in frame_items:
                    category_id = item['class']
                    category_name = self._train_dataset.catid_to_name_map[category_id]
                    if category_name:
                        active_objects.add(category_name)

            active_objects.add("person")  # Ensure 'person' is always included
            self.video_id_active_objects_map[video_id] = sorted(list(active_objects))

    def load_gdino_model(self):
        # Load GDINO model for bounding box extraction
        self.gdino_model_id = "IDEA-Research/grounding-dino-base"
        self.gdino_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model_id).to(self.device)
        self.gdino_object_labels = [
            "a person", "a bag", "a blanket", "a book", "a box", "a broom", "a chair", "a clothes",
            "a cup", "a dish", "a food", "a laptop", "a paper", "a phone", "a picture", "a pillow",
            "a sandwich", "a shoe", "a towel", "a vacuum", "a glass", "a bottle", "a notebook", "a camera",
            "a bed", "a closet", "a cabinet", "a door", "a doorknob", "a groceries", "a mirror", "a refrigerator",
            "a sofa", "a couch", "a table", "a television", "a window"
        ]

    def load_sam2_model(self):
        # Use a balanced checkpoint (you can switch to 'facebook/sam2-hiera-large' if you have headroom)
        self.sam2_image_predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus")
        self.sam2_video_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-base-plus")

        # Prefer bfloat16 autocast on CUDA; fall back to regular inference on CPU.
        self._use_amp = (self.device.type == "cuda")

    # -------------------------------------- DETECTION MODULES -------------------------------------- #

    @staticmethod
    def _normalize_label(s: str) -> str:
        s = s.lower().strip()
        for art in ("a ", "an ", "the "):
            if s.startswith(art):
                s = s[len(art):]
                break
        return s

    @staticmethod
    def _ensure_dir(p: Path):
        p.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _img_to_np(img: Image.Image) -> np.ndarray:
        return np.array(img)

    @staticmethod
    def _binary_to_png(mask_bool: np.ndarray) -> np.ndarray:
        return (mask_bool.astype(np.uint8) * 255)

    @staticmethod
    def _apply_mask(img_np: np.ndarray, mask_bool: np.ndarray) -> np.ndarray:
        # keep RGB where mask==1, black elsewhere
        out = img_np.copy()
        if out.ndim == 2:
            out = np.stack([out, out, out], axis=-1)
        out[~mask_bool] = 0
        return out

    def draw_and_save_bboxes(
            self,
            image_path: str,
            boxes: torch.Tensor,
            labels: List[str],
            output_dir: str,
            frame_name: str
    ):
        if not os.path.exists(image_path): return
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        unique_labels = sorted(list(set(labels)))

        if len(unique_labels) == 0: return

        color_map = get_color_map(len(unique_labels))
        label_to_color = {label: tuple(c) for label, c in zip(unique_labels, color_map)}

        for box, label in zip(boxes.tolist(), labels):
            color = label_to_color.get(label, "red")
            draw.rectangle(box, outline=color, width=3)
            draw.text((box[0], box[1] - 10), label, fill=color)

        os.makedirs(output_dir, exist_ok=True)
        image.save(os.path.join(output_dir, frame_name))

    def extract_bounding_boxes(self, video_id, visualize=False):
        # Use GDINO to extract bounding boxes for objects in frames
        video_frames_dir_path = os.path.join(self.ag_root_directory, "sampled_frames", video_id)
        video_output_file_path = os.path.join(self.bbox_dir_path, f"{video_id}.pkl")

        # Loads object labels corresponding to active objects in the dataset
        video_object_labels = self.video_id_active_objects_map[video_id]

        if os.path.exists(video_output_file_path):
            print(f"Bounding boxes for video {video_id} already exist. Skipping detection...")
            return

        self._ensure_dir(Path(self.gdino_vis_path) / video_id)
        self._ensure_dir(Path(self.bbox_dir_path))

        video_predictions = {}
        video_frames = sorted([f for f in os.listdir(video_frames_dir_path) if f.endswith('.png') or f.endswith('.jpg')])
        for video_frame_name in tqdm(video_frames, desc=f"Detecting objects in {video_id}"):
            frame_path = os.path.join(video_frames_dir_path, video_frame_name)
            if not os.path.exists(frame_path): continue
            image = Image.open(frame_path).convert("RGB")
            inputs = self.gdino_processor(
                images=image,
                text=". ".join(video_object_labels),
                return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]])[0]

            # normalize labels (strip 'a/the')
            labels = [self._normalize_label(l) for l in results['labels']]

            video_predictions[video_frame_name] = {
                'boxes': results['boxes'],
                'scores': results['scores'],
                'labels': labels
            }

            if visualize:
                vis_dir = os.path.join(self.gdino_vis_path, video_id)
                self.draw_and_save_bboxes(frame_path, results['boxes'], labels, vis_dir, video_frame_name)

        with open(video_output_file_path, 'wb') as file:
            pickle.dump(video_predictions, file)

    # -------------------------------------- SEGMENTATION (SAM2) -------------------------------------- #

    def segment_with_sam2(self, video_id):
        """Image-mode SAM2: per-frame segmentation with box prompts from GDINO.  # >>> filled
        Saves per-object binary masks and union-masked frames.
        """
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
            img_np = self._img_to_np(img)

            boxes: torch.Tensor = dets[fn]["boxes"]
            labels: List[str] = dets[fn]["labels"]
            scores: torch.Tensor = dets[fn]["scores"]

            # group detections by label
            by_label: DefaultDict[str, List[Tuple[np.ndarray, float]]] = defaultdict(list)
            for b, l, s in zip(boxes.cpu().numpy(), labels, scores.cpu().numpy()):
                by_label[l].append((b.astype(np.float32), float(s)))

            # run predictor per frame
            # cast in AMP on CUDA (bf16), else normal mode
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

    def _mirror_pngs_to_jpg(self, frames_dir: Path, video_id: str) -> Tuple[Path, List[str]]:
        """Ensure a JPG folder exists with same ordering as sampled_frames.  # >>> helper
        Returns (jpg_dir, ordered_jpg_filenames)."""
        jpg_dir = self.sam2_jpg_tmp / video_id
        self._ensure_dir(jpg_dir)

        fn_png = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        jpg_names = []
        for fn in fn_png:
            src = frames_dir / fn
            stem = Path(fn).stem
            dst = jpg_dir / f"{stem}.jpg"
            if not dst.exists():
                # convert (and also unify color)
                img = Image.open(src).convert("RGB")
                img.save(dst, format="JPEG", quality=95)
            jpg_names.append(dst.name)
        return jpg_dir, sorted(jpg_names)

    def segment_with_sam2_video_mode(self, video_id):
        """Video-mode SAM2: add a single box prompt on the first observed frame per object,
        then propagate throughout the video to get per-frame masks.  # >>> filled
        """
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

        # SAM2 video predictor expects a directory of JPEG frames
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

        # init state and add prompts
        if self._use_amp:
            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class _Noop:
                def __enter__(self): return None
                def __exit__(self, *args): return False
            amp_ctx = _Noop()

        with torch.inference_mode(), amp_ctx:
            # The official API uses init_state(video_path=...)
            state = self.sam2_video_predictor.init_state(video_path=str(jpg_dir))

            # Add one box per object at its first occurrence frame
            for lbl, (first_fn, box_px) in first_occurrence.items():
                obj_id = lbl_to_objid[lbl]
                ann_frame_idx = stem_to_idx[Path(first_fn).stem]
                # API: add_new_points_or_box(inference_state=..., frame_idx=..., obj_id=..., box=...)
                self.sam2_video_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=ann_frame_idx,
                    obj_id=obj_id,
                    box=box_px.astype(np.float32),
                )

            # Now propagate across the video, saving per-frame masks
            # We'll also build per-frame union masks to export masked frames.
            for frame_idx, object_ids, masks in self.sam2_video_predictor.propagate_in_video(state):
                # masks: (num_objects_on_frame, H, W) — map each obj_id to its mask
                # Resolve original frame name (PNG path)
                jpg_name = jpg_order[frame_idx]
                stem = Path(jpg_name).stem
                # locate original PNG/JPG frame from sampled_frames
                candidate_png = frames_dir / f"{stem}.png"
                candidate_jpg = frames_dir / f"{stem}.jpg"
                src_img_path = candidate_png if candidate_png.exists() else candidate_jpg
                if not src_img_path.exists():
                    # fallback to temp jpg
                    src_img_path = jpg_dir / jpg_name

                img = Image.open(src_img_path).convert("RGB")
                img_np = np.array(img)
                union_mask = np.zeros(img_np.shape[:2], dtype=bool)

                for k, obj_id in enumerate(object_ids):
                    lbl = objid_to_lbl[int(obj_id)]
                    m = np.array(masks[k]).astype(bool)
                    union_mask |= m
                    # save per-object binary mask
                    save_p = out_mask_dir / f"{stem}__{lbl}.png"
                    cv2.imwrite(str(save_p), self._binary_to_png(m))

                # save union-masked frame
                masked_np = self._apply_mask(img_np, union_mask)
                cv2.imwrite(str(out_frames_dir / f"{stem}.png"), cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR))

    def combine_masks(self, video_id):
        """Union per-object masks from image route and video route, write combined masks and frames.  # >>> filled
        """
        frames_dir = Path(self.ag_root_directory) / "sampled_frames" / video_id
        out_mask_dir = self.masks_combined_dir_path / video_id
        out_frames_dir = self.masked_frames_combined_dir_path / video_id
        self._ensure_dir(out_mask_dir)
        self._ensure_dir(out_frames_dir)

        # collect all frame stems present
        stems = sorted({Path(fn).stem for fn in os.listdir(frames_dir) if fn.lower().endswith((".png", ".jpg", ".jpeg"))})

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
            union_frame = None

            for lbl in labels_all:
                im_mask_p = self.masks_im_dir_path / video_id / f"{stem}__{lbl}.png"
                vd_mask_p = self.masks_vid_dir_path / video_id / f"{stem}__{lbl}.png"

                m_im = cv2.imread(str(im_mask_p), cv2.IMREAD_GRAYSCALE) if im_mask_p.exists() else None
                m_vd = cv2.imread(str(vd_mask_p), cv2.IMREAD_GRAYSCALE) if vd_mask_p.exists() else None

                if m_im is None and m_vd is None:
                    continue

                if m_im is None:
                    m_union = (m_vd > 127)
                elif m_vd is None:
                    m_union = (m_im > 127)
                else:
                    m_union = (m_im > 127) | (m_vd > 127)

                # save combined per-object mask
                save_p = out_mask_dir / f"{stem}__{lbl}.png"
                cv2.imwrite(str(save_p), self._binary_to_png(m_union))

                # accumulate into per-frame union
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
                cv2.imwrite(str(out_frames_dir / f"{stem}.png"), cv2.cvtColor(masked_np, cv2.COLOR_RGB2BGR))

    def _write_video_from_frames(self, frames_dir: Path, out_path: Path, fps: int = 15):
        img_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not img_files:
            return
        first = cv2.imread(str(frames_dir / img_files[0]), cv2.IMREAD_COLOR)
        h, w = first.shape[:2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
        for fn in img_files:
            frame = cv2.imread(str(frames_dir / fn), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[0] != h or frame.shape[1] != w:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
        writer.release()

    def save_masked_frames_and_videos(self, video_id):
        routes = [
            ("image_based", self.masked_frames_im_dir_path / video_id),
            ("video_based", self.masked_frames_vid_dir_path / video_id),
            ("combined", self.masked_frames_combined_dir_path / video_id),
        ]
        for route_name, frames_dir in routes:
            if not frames_dir.exists():
                continue
            out_mp4 = self.masked_videos_dir_path / f"{Path(video_id).stem}__{route_name}.mp4"
            self._write_video_from_frames(frames_dir, out_mp4, fps=15)

    def process(self):
        self.load_dataset()

        # video_id_list = os.listdir(self.data_dir_path / "videos")
        video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]

        for video_id in tqdm(video_id_list):
            self.extract_bounding_boxes(video_id)
            self.segment_with_sam2(video_id)
            self.segment_with_sam2_video_mode(video_id)
            self.combine_masks(video_id)
            self.save_masked_frames_and_videos(video_id)


def main():
    data_dir_path = "/data/rohith/ag/"
    ag_actor_segmentation = AgActorSegmentation(data_dir_path)
    # ag_actor_segmentation.process()


if __name__ == "__main__":
    main()
