import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Any, DefaultDict

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from fontTools.ttx import process
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM, AutoTokenizer

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

class BaseAgActor:

    def __init__(self, ag_root_directory):
        self.ag_root_directory = Path(ag_root_directory)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # temp JPG mirror for SAM2 video predictor (expects JPEG frames)
        self.sampled_frames_jpg = self.ag_root_directory / "sampled_frames_jpg"
        self.bbox_dir_path = self.ag_root_directory / "detection" / 'gdino_bboxes'

        self._ensure_dir(self.bbox_dir_path)
        self._ensure_dir(self.sampled_frames_jpg)

        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._train_dataset = None

        self.load_dataset()

    # -------------------------------------- LOADING INFORMATION -------------------------------------- #
    def load_dataset(self):
        self._train_dataset = StandardAGCoCoDataset(
            phase="train",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=False
        )

        self._test_dataset = StandardAGCoCoDataset(
            phase="test",
            mode="sgdet",
            datasize="large",
            data_path=self.ag_root_directory,
            filter_nonperson_box_frame=True,
            filter_small_box=False
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
    def _binary_to_png(mask_bool: np.ndarray) -> np.ndarray:
        return mask_bool.astype(np.uint8) * 255

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

    def _mirror_pngs_to_jpg(self, frames_dir: Path, video_id: str) -> Tuple[Path, List[str]]:
        jpg_dir = self.sampled_frames_jpg / video_id
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


class AgActorDetection(BaseAgActor):

    def __init__(self, ag_root_directory):
        super().__init__(ag_root_directory)
        self.caption_data = None
        self.bbox_dir_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.gdino_vis_path = self.ag_root_directory / "detection" / 'gdino_vis'
        self.active_objects_b_annotations_path = self.ag_root_directory / 'active_objects' / 'annotations'
        self.active_objects_b_reasoned_path = self.ag_root_directory / 'active_objects' / 'sampled_videos'

        for p in [self.bbox_dir_path, self.gdino_vis_path, self.active_objects_b_reasoned_path,
                  self.active_objects_b_annotations_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = None
        self.llama_model = None
        self.model_id = None
        self.gdino_object_labels = None
        self.gdino_model = None
        self.gdino_processor = None
        self.gdino_device = None
        self.gdino_model_id = None

        self.load_gdino_model()
        self.load_llama_model()
        self.load_caption_data()

        self.video_id_active_objects_annotations_map = {}
        self.video_id_active_objects_b_reasoned_map = {}
        self.process_video_id_active_objects_map()

    def load_caption_data(self):
        caption_json = self.ag_root_directory / "captions" / "charades.json"
        if not caption_json.exists():
            print(f"Warning: Caption file not found: {caption_json}")
            return

        with open(caption_json, 'r') as f:
            self.caption_data = json.load(f)

    def load_llama_model(self):
        self.model_id = "meta-llama/Llama-3.1-8B-Instruct"

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        # Ensure we have a pad token to avoid warnings during generation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    # ---------------------------
    # Low-level generation helpers
    # ---------------------------
    def _generate(self, messages: List[Dict[str, str]], max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        # Keep tensors on CPU and let accelerate handle device placement for sharded models
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            do_sample=False,
            eos_token_id=terminators,
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            out_ids = self.llama_model.generate(inputs, **gen_kwargs)
        gen_only = out_ids[:, inputs.shape[1]:]
        text = self.tokenizer.batch_decode(
            gen_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return text.strip()

    @staticmethod
    def _safe_json_loads(s: str) -> Dict[str, Any]:
        """Try to parse JSON; if there is extra text, extract the first JSON object candidate."""
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            # Attempt to extract the first {...} block
            m = re.search(r"\{[\s\S]*\}", s)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    def process_video_id_active_objects_map(self, process_raw=False):

        def fetch_active_objects_in_videos(dataloader, process_raw=False):
            for data in dataloader:
                video_id = data['video_id']
                gt_annotations = data['gt_annotations']
                video_id_object_reasoning_path = self.active_objects_b_reasoned_path / f"{video_id[:-4]}.txt"
                video_id_object_annotations_path = self.active_objects_b_annotations_path / f"{video_id[:-4]}.txt"
                if not process_raw:
                    if video_id_object_annotations_path.exists():
                        with open(video_id_object_annotations_path, "r") as f:
                            annotated_objects = [line.strip() for line in f if line.strip()]

                        if video_id_object_reasoning_path.exists():
                            with open(video_id_object_reasoning_path, "r") as f:
                                video_reasoned_objects = [line.strip() for line in f if line.strip()]

                            # Ensure presence of "person", as it's always active
                            # If there is a television in annotated objects, add it to reasoned objects
                            # If there is a mirror in annotated objects, add it to reasoned objects
                            video_reasoned_objects = set(video_reasoned_objects)
                            video_reasoned_objects.add("person")

                            if "television" in annotated_objects:
                                video_reasoned_objects.add("television")
                            if "mirror" in annotated_objects:
                                video_reasoned_objects.add("mirror")
                            self.video_id_active_objects_b_reasoned_map[video_id] = sorted(list(video_reasoned_objects))
                        else:
                            self.video_id_active_objects_annotations_map[video_id] = sorted(annotated_objects)
                    else:
                        print("Warning: Missing annotation file for video:", video_id)
                else:
                    active_objects = set()
                    for frame_items in gt_annotations:
                        for item in frame_items:
                            if 'person_bbox' in item:
                                continue
                            category_id = item['class']
                            category_name = self._train_dataset.catid_to_name_map[category_id]
                            if category_name:
                                if category_name == "closet/cabinet":
                                    active_objects.add("closet")
                                    active_objects.add("cabinet")
                                elif category_name == "cup/glass/bottle":
                                    active_objects.add("cup")
                                    active_objects.add("glass")
                                    active_objects.add("bottle")
                                elif category_name == "paper/notebook":
                                    active_objects.add("paper")
                                    active_objects.add("notebook")
                                elif category_name == "sofa/couch":
                                    active_objects.add("sofa")
                                    active_objects.add("couch")
                                elif category_name == "phone/camera":
                                    active_objects.add("phone")
                                    active_objects.add("camera")
                                else:
                                    active_objects.add(category_name)

                    active_objects.add("person")
                    self.video_id_active_objects_annotations_map[video_id] = sorted(list(active_objects))

        fetch_active_objects_in_videos(self._dataloader_train, process_raw=process_raw)
        fetch_active_objects_in_videos(self._dataloader_test, process_raw=process_raw)

        if process_raw:
            new_objects_list = []
            error_videos_list = []

            # list of objects corresponding to each video id in a text file
            for video_id, objects in tqdm(self.video_id_active_objects_annotations_map.items()):
                with open(self.active_objects_b_annotations_path / f"{video_id[:-4]}.txt", "w") as f:
                    for obj in objects:
                        f.write(f"{obj}\n")

                video_caption = self.caption_data[video_id[:-4]]

                # Given active objects and video caption, use LLaMA to reason about objects that result in
                # movement due to the interaction from the active objects
                prompt = (
                    f"Given the video caption: \"{video_caption}\", and the list of objects present in the video: "
                    f"\"{', '.join(objects)}\", identify ONLY the objects that are likely to be involved in "
                    f"movements or actions. Exclude static/background items. "
                    f"Return STRICT JSON with DOUBLE QUOTES only, exactly in the form:\n"
                    f"{{\"reasoned_objects\": [\"obj1\", \"obj2\", ...]}}\n"
                    f"Use ONLY names from the provided list; do not invent new ones. No extra text."
                )

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert video-understanding assistant. "
                            "Output must be a SINGLE JSON object with key \"reasoned_objects\" "
                            "whose value is a list of object names from the provided candidates. "
                            "Use ONLY double quotes and ONLY provided object names. No prose."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ]

                try:
                    raw = self._generate(messages, max_new_tokens=128)
                    parsed = self._safe_json_loads(raw)
                    reasoned_objects = set(parsed["reasoned_objects"])
                    reasoned_objects.add("person")
                    with open(self.active_objects_b_reasoned_path / f"{video_id[:-4]}.txt", "w") as f:
                        for obj in reasoned_objects:
                            if obj in objects:
                                f.write(f"{obj}\n")
                            else:
                                new_objects_list.append((video_id, obj))
                    self.video_id_active_objects_b_reasoned_map[video_id] = sorted(list(reasoned_objects))
                except Exception as e:
                    print(f"Error processing video {video_id}: {e}")
                    error_videos_list.append(video_id)
                    continue
            if new_objects_list:
                with open(self.ag_root_directory / "new_objects_b_reasoned.txt", "w") as f:
                    for video_id, obj in new_objects_list:
                        f.write(f"{video_id}: {obj}\n")
            if error_videos_list:
                with open(self.ag_root_directory / "error_videos_b_reasoned.txt", "w") as f:
                    for video_id in error_videos_list:
                        f.write(f"{video_id}\n")

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

    # -------------------------------------- DETECTION MODULES -------------------------------------- #
    def extract_bounding_boxes(self, video_id, visualize=True):
        """
        Run Grounding DINO on sampled frames, apply class-wise NMS, and (optionally) save visualizations.
        Saves a pickle with per-frame {boxes, scores, labels} to self.bbox_dir_path.
        """

        # ---------- helpers (self-contained) ----------
        def _box_iou_single_vs_many(box_i: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
            # boxes are xyxy; box_i is shape (4,), boxes is (N,4)
            tl = torch.maximum(box_i[:2], boxes[:, :2])  # (N,2)
            br = torch.minimum(box_i[2:], boxes[:, 2:])  # (N,2)
            wh = (br - tl).clamp(min=0)  # (N,2)
            inter = wh[:, 0] * wh[:, 1]  # (N,)

            area_i = (box_i[2] - box_i[0]).clamp(min=0) * (box_i[3] - box_i[1]).clamp(min=0)
            area_n = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
            union = area_i + area_n - inter
            return inter / union.clamp(min=1e-9)

        def _nms_single_class(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float = 0.5) -> torch.Tensor:
            # Returns indices (w.r.t. input tensors) to keep
            if boxes.numel() == 0:
                return torch.empty(0, dtype=torch.long)
            order = scores.argsort(descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                rest = order[1:]
                ious = _box_iou_single_vs_many(boxes[i], boxes[rest])
                order = rest[ious <= iou_thr]
            return torch.tensor(keep, dtype=torch.long)

        def _nms_classwise(boxes: torch.Tensor, scores: torch.Tensor, labels: list,
                           iou_thr: float = 0.5, min_score: float = 0.0):
            # Run NMS per label string; return filtered (boxes, scores, labels)
            if boxes.numel() == 0:
                return boxes, scores, labels

            boxes = boxes.detach().cpu().float()
            scores = scores.detach().cpu().float()
            labels = list(labels)

            # pre-filter on score if requested
            if min_score > 0:
                keep0 = torch.nonzero(scores >= min_score, as_tuple=False).squeeze(1)
                boxes, scores = boxes[keep0], scores[keep0]
                labels = [labels[i] for i in keep0.tolist()]
                if boxes.numel() == 0:
                    return boxes, scores, labels

            kept_boxes, kept_scores, kept_labels = [], [], []
            for lbl in sorted(set(labels)):
                idx = [i for i, l in enumerate(labels) if l == lbl]
                if not idx:
                    continue
                b = boxes[idx]
                s = scores[idx]
                keep_idx_local = _nms_single_class(b, s, iou_thr=iou_thr).tolist()
                for k in keep_idx_local:
                    kept_boxes.append(b[k].unsqueeze(0))
                    kept_scores.append(s[k].unsqueeze(0))
                    kept_labels.append(lbl)

            if kept_boxes:
                boxes_out = torch.cat(kept_boxes, dim=0)
                scores_out = torch.cat(kept_scores, dim=0)
                labels_out = kept_labels
            else:
                boxes_out = torch.empty((0, 4), dtype=torch.float32)
                scores_out = torch.empty((0,), dtype=torch.float32)
                labels_out = []
            return boxes_out, scores_out, labels_out

        # ---------------------------------------------

        # Use GDINO to extract bounding boxes for objects in frames
        video_frames_dir_path = os.path.join(self.ag_root_directory, "sampled_frames", video_id)
        video_output_file_path = os.path.join(self.bbox_dir_path, f"{video_id}.pkl")

        # Loads object labels corresponding to active objects in the dataset
        video_object_labels = self.video_id_active_objects_b_reasoned_map[video_id]

        if os.path.exists(video_output_file_path):
            print(f"Bounding boxes for video {video_id} already exist. Skipping detection...")
            return

        self._ensure_dir(Path(self.gdino_vis_path) / video_id)
        self._ensure_dir(Path(self.bbox_dir_path))

        video_predictions = {}
        video_frames = sorted([f for f in os.listdir(video_frames_dir_path)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for video_frame_name in tqdm(video_frames, desc=f"Detecting objects in {video_id}"):
            frame_path = os.path.join(video_frames_dir_path, video_frame_name)
            if not os.path.exists(frame_path):
                continue

            image = Image.open(frame_path).convert("RGB")
            inputs = self.gdino_processor(
                images=image,
                text=". ".join(video_object_labels),
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )[0]

            # normalize labels (strip 'a/the')
            labels = [self._normalize_label(l) for l in results['labels']]

            # ---- class-wise NMS (tune thresholds as needed) ----
            boxes_nms, scores_nms, labels_nms = _nms_classwise(
                results['boxes'], results['scores'], labels,
                iou_thr=0.5,  # IoU threshold for suppression
                min_score=0.0  # optional pre-filter; set >0 to drop low scores
            )

            video_predictions[video_frame_name] = {
                'boxes': boxes_nms,
                'scores': scores_nms,
                'labels': labels_nms
            }

            if visualize and boxes_nms.numel() > 0 and len(labels_nms) > 0:
                vis_dir = os.path.join(self.gdino_vis_path, video_id)
                self.draw_and_save_bboxes(frame_path, boxes_nms, labels_nms, vis_dir, video_frame_name)

        with open(video_output_file_path, 'wb') as file:
            pickle.dump(video_predictions, file)

    def process(self):
        # video_id_list = os.listdir(self.data_dir_path / "videos")
        video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]

        for video_id in tqdm(video_id_list):
            self.extract_bounding_boxes(video_id, visualize=True)


class AgActorSegmentation(BaseAgActor):

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
        frames_dir = Path(self.ag_root_directory) / "sampled_frames" / video_id
        out_mask_dir = self.masks_combined_dir_path / video_id
        out_frames_dir = self.masked_frames_combined_dir_path / video_id
        self._ensure_dir(out_mask_dir)
        self._ensure_dir(out_frames_dir)

        # collect all frame stems present
        stems = sorted(
            {Path(fn).stem for fn in os.listdir(frames_dir) if fn.lower().endswith((".png", ".jpg", ".jpeg"))})

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
        # video_id_list = os.listdir(self.data_dir_path / "videos")
        video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]

        for video_id in tqdm(video_id_list):
            # self.segment_with_sam2(video_id)
            self.segment_with_sam2_video_mode(video_id)
            # self.combine_masks(video_id)
            # self.save_masked_frames_and_videos(video_id)


def main():
    data_dir_path = "/data/rohith/ag/"

    # ag_actor_detection = AgActorDetection(data_dir_path)
    # ag_actor_detection.process()

    ag_actor_segmentation = AgActorSegmentation(data_dir_path)
    ag_actor_segmentation.process()


if __name__ == "__main__":
    main()
