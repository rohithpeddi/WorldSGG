import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM, AutoTokenizer

from datasets.preprocess.segmentation.base_ag_actor import BaseAgActor, get_video_belongs_to_split


class AgDetection(BaseAgActor):

    def __init__(self, ag_root_directory, process_raw=False):
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
        self.process_video_id_active_objects_map(process_raw=process_raw)

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

    def fetch_raw_active_objects_in_videos(self, dataloader):
        for data in dataloader:
            video_id = data['video_id']
            gt_annotations = data['gt_annotations']
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

    def process_raw_active_objects_in_videos(self):
        self.fetch_raw_active_objects_in_videos(self._dataloader_train)
        self.fetch_raw_active_objects_in_videos(self._dataloader_test)

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

    def fetch_stored_active_objects_in_videos(self, dataloader):
        for data in dataloader:
            video_id = data['video_id']
            video_id_object_reasoning_path = self.active_objects_b_reasoned_path / f"{video_id[:-4]}.txt"
            video_id_object_annotations_path = self.active_objects_b_annotations_path / f"{video_id[:-4]}.txt"
            if video_id_object_annotations_path.exists():
                with open(video_id_object_annotations_path, "r") as f:
                    annotated_objects = [line.strip() for line in f if line.strip()]
                self.video_id_active_objects_annotations_map[video_id] = sorted(annotated_objects)
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
                    print(f"Video {video_id} has no reasoned objects. Loading annotated objects instead.")
                    self.video_id_active_objects_b_reasoned_map[video_id] = sorted(annotated_objects)
            else:
                print("Warning: Missing annotation file for video:", video_id)

    def process_video_id_active_objects_map(self, process_raw=False):
        if process_raw:
            print("Processing raw active objects in videos...")
            self.process_raw_active_objects_in_videos()
        else:
            print("Processing stored active objects in videos...")
            self.fetch_stored_active_objects_in_videos(self._dataloader_train)
            self.fetch_stored_active_objects_in_videos(self._dataloader_test)

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

    def process(self, split):
        video_id_list = ["0DJ6R.mp4", "00HFP.mp4", "00NN7.mp4", "00T1E.mp4", "00X3U.mp4", "00ZCA.mp4", "0ACZ8.mp4"]
        for video_id in tqdm(video_id_list):
            self.extract_bounding_boxes(video_id, visualize=True)

        # for data in tqdm(self._dataloader_train):
        #     video_id = data['video_id']
        #     if get_video_belongs_to_split(video_id) == split:
        #         self.extract_bounding_boxes(video_id, visualize=True)
        # for data in tqdm(self._dataloader_test):
        #     video_id = data['video_id']
        #     if get_video_belongs_to_split(video_id) == split:
        #         self.extract_bounding_boxes(video_id, visualize=True)


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

    ag_actor_detection = AgDetection(data_dir, process_raw=False)
    ag_actor_detection.process(split)


if __name__ == "__main__":
    main()
