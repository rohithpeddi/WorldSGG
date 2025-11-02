import argparse
import json
import os
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModelForCausalLM, AutoTokenizer

from datasets.preprocess.segmentation.base_ag_actor import BaseAgActor


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


class AgDetection(BaseAgActor):

    def __init__(self, ag_root_directory, process_raw=False):
        super().__init__(ag_root_directory)
        self.caption_data = None
        self.bbox_dir_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'
        self.gdino_vis_path = self.ag_root_directory / "detection" / 'gdino_vis_static'
        self.gt_vis_path = self.ag_root_directory / "detection" / 'gt_vis_static'
        self.active_objects_b_annotations_path = self.ag_root_directory / 'active_objects' / 'annotations'
        self.active_objects_b_reasoned_path = self.ag_root_directory / 'active_objects' / 'sampled_videos'

        for p in [self.bbox_dir_path, self.gdino_vis_path, self.gt_vis_path, self.active_objects_b_reasoned_path,
                  self.active_objects_b_annotations_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_id = None
        self.gdino_model = None
        self.gdino_processor = None
        self.gdino_device = None
        self.gdino_model_id = None

        self.load_gdino_model()

        self.video_id_active_objects_annotations_map = {}
        self.video_id_active_objects_b_reasoned_map = {}

        self.video_id_gt_annotations_map = {}
        self.video_id_gt_bboxes_map = {}
        self.process_video_id_active_objects_map()

        self.create_gt_annotations_map()
        self.req_gt_bbox_format = "xyxy"  # set to "xywh" if your GT is COCO-style

    def create_gt_annotations_map(self):
        # Create a mapping from video_id to its ground truth annotations
        for data in self._dataloader_train:
            video_id = data['video_id']
            gt_annotations = data['gt_annotations']
            self.video_id_gt_annotations_map[video_id] = gt_annotations
        for data in self._dataloader_test:
            video_id = data['video_id']
            gt_annotations = data['gt_annotations']
            self.video_id_gt_annotations_map[video_id] = gt_annotations

        # video_id, gt_bboxes for the gt detections
        for video_id, gt_annotations in self.video_id_gt_annotations_map.items():
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
                    category_name = self._train_dataset.catid_to_name_map[category_id]
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
            self.video_id_gt_bboxes_map[video_id] = video_gt_bboxes

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

    def process_video_id_active_objects_map(self):
        print("Processing stored active objects in videos...")
        self.fetch_stored_active_objects_in_videos(self._dataloader_train)
        self.fetch_stored_active_objects_in_videos(self._dataloader_test)

    def load_gdino_model(self):
        self.gdino_model_id = "IDEA-Research/grounding-dino-base"
        self.gdino_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gdino_processor = AutoProcessor.from_pretrained(self.gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model_id).to(self.device)

    # -------------------------------------- DETECTION MODULES -------------------------------------- #

    def _prepare_gt_for_frame(
            self,
            video_id: str,
            frame_name: str,
            video_object_labels: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        frame_map = self.video_id_gt_bboxes_map.get(video_id, {})
        rec = frame_map.get(frame_name)
        if rec is None or len(rec.get("boxes", [])) == 0:
            return (torch.empty((0, 4), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.float32), [])

        gt_boxes = [torch.tensor(b, dtype=torch.float32) for b in rec["boxes"]]
        if self.req_gt_bbox_format == "xywh":
            gt_boxes = self._xywh_to_xyxy(gt_boxes)

        gt_labels = [self._normalize_label(l) for l in rec["labels"]]
        gt_scores = [torch.tensor(1.001, dtype=torch.float32) for _ in gt_labels]  # slightly >1 to ensure GT is always kept in NMS

        # Filter GT to only include objects in video_object_labels
        filtered_boxes, filtered_scores, filtered_labels = [], [], []
        for box, score, label in zip(gt_boxes, gt_scores, gt_labels):
            if label in video_object_labels:
                filtered_boxes.append(box.unsqueeze(0))
                filtered_scores.append(score.unsqueeze(0))
                filtered_labels.append(label)

        if filtered_boxes:
            return (torch.cat(filtered_boxes, dim=0),
                    torch.cat(filtered_scores, dim=0),
                    filtered_labels)
        else:
            return (torch.empty((0, 4), dtype=torch.float32),
                    torch.empty((0,), dtype=torch.float32), [])

    def extract_bounding_boxes(self, video_id, visualize=True):

        # ---------- helpers (self-contained) ----------
        def _classwise_nms(
                boxes: torch.Tensor,
                scores: torch.Tensor,
                labels: list[str],
                iou_thr: float = 0.5,
                min_score: float = 0.0
        ):
            if boxes.numel() == 0:
                return boxes, scores, labels

            boxes = boxes.detach().cpu().float()
            scores = scores.detach().cpu().float()
            labels = list(labels)

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
                # per-class greedy NMS
                order = s.argsort(descending=True)
                keep = []
                while order.numel() > 0:
                    i = order[0].item()
                    keep.append(i)
                    if order.numel() == 1:
                        break
                    rest = order[1:]
                    tl = torch.maximum(b[i, :2], b[rest, :2])
                    br = torch.minimum(b[i, 2:], b[rest, 2:])
                    inter = (br - tl).clamp(min=0)
                    inter = inter[:, 0] * inter[:, 1]
                    area_i = (b[i, 2] - b[i, 0]).clamp(min=0) * (b[i, 3] - b[i, 1]).clamp(min=0)
                    area_r = (b[rest, 2] - b[rest, 0]).clamp(min=0) * (b[rest, 3] - b[rest, 1]).clamp(min=0)
                    iou = inter / (area_i + area_r - inter).clamp(min=1e-9)
                    order = rest[iou <= iou_thr]
                for k in keep:
                    kept_boxes.append(b[k].unsqueeze(0))
                    kept_scores.append(s[k].unsqueeze(0))
                    kept_labels.append(lbl)

            if kept_boxes:
                return torch.cat(kept_boxes, dim=0), torch.cat(kept_scores, dim=0), kept_labels
            return torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32), []

        # ---------------------------------------------

        # Use GDINO to extract bounding boxes for objects in frames
        video_frames_dir_path = os.path.join(self.ag_root_directory, "sampled_frames", video_id)
        video_output_file_path = os.path.join(self.bbox_dir_path, f"{video_id}.pkl")

        # Loads object labels corresponding to active objects in the dataset
        video_active_object_labels = self.video_id_active_objects_annotations_map[video_id]
        video_reasoned_active_object_labels = self.video_id_active_objects_b_reasoned_map[video_id]
        non_moving_objects = ["floor", "sofa", "couch", "bed", "doorway", "table", "chair"]
        video_dynamic_object_labels = [obj for obj in video_reasoned_active_object_labels if obj not in non_moving_objects]

        # We want to look for all the objects ignored above but are part of video active object labels.
        video_static_object_labels = [obj for obj in video_active_object_labels if obj not in video_dynamic_object_labels]

        if len(video_static_object_labels) == 0:
            print(f"No static objects to detect in video {video_id}. Skipping detection...")
            return

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
                text=". ".join(video_static_object_labels),
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            results = self.gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.25,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]]
            )[0]

            # Normalize labels (strip 'a/the')
            labels = [self._normalize_label(l) for l in results['labels']]

            # 1) First-stage NMS on GDINO outputs (you already have this)
            boxes_nms, scores_nms, labels_nms = _classwise_nms(
                results['boxes'], results['scores'], labels,
                iou_thr=0.5,
                min_score=0.0
            )

            # 2) Pull GT for this frame
            gt_boxes, gt_scores, gt_labels = self._prepare_gt_for_frame(video_id, video_frame_name, video_static_object_labels)

            # If the gt_boxes is not empty, store the frame in gt_vis directory for visualization
            if visualize and gt_boxes.numel() > 0 and len(gt_labels) > 0:
                vid_gt_vis_dir = Path(self.gt_vis_path) / video_id
                self._ensure_dir(vid_gt_vis_dir)
                self.draw_and_save_bboxes(frame_path, gt_boxes, gt_labels, vid_gt_vis_dir, video_frame_name)

            # 3) Concatenate predicted + GT and run final per-class NMS
            if gt_boxes.numel() > 0:
                all_boxes = torch.cat([boxes_nms, gt_boxes], dim=0) if boxes_nms.numel() else gt_boxes
                all_scores = torch.cat([scores_nms, gt_scores], dim=0) if scores_nms.numel() else gt_scores
                all_labels = list(labels_nms) + gt_labels
            else:
                all_boxes, all_scores, all_labels = boxes_nms, scores_nms, labels_nms

            final_boxes, final_scores, final_labels = _classwise_nms(
                all_boxes, all_scores, all_labels,
                iou_thr=0.5,  # you can use a slightly higher IoU here if you want fewer near-duplicates
                min_score=0.0
            )

            video_predictions[video_frame_name] = {
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            }

            if visualize and boxes_nms.numel() > 0 and len(labels_nms) > 0:
                vis_dir = Path(self.gdino_vis_path) / video_id
                self._ensure_dir(vis_dir)
                self.draw_and_save_bboxes(frame_path, final_boxes, final_labels, vis_dir, video_frame_name)

        with open(video_output_file_path, 'wb') as file:
            pickle.dump(video_predictions, file)

    def process(self, split):
        for data in tqdm(self._dataloader_train):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                self.extract_bounding_boxes(video_id, visualize=True)
        for data in tqdm(self._dataloader_test):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                self.extract_bounding_boxes(video_id, visualize=True)


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

    ag_actor_detection = AgDetection(data_dir, process_raw=False)
    ag_actor_detection.process(split)


if __name__ == "__main__":
    main()
