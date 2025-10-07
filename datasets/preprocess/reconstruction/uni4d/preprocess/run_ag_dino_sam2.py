import os
import argparse
import pickle

import cv2
import json
import torch
import imageio
import numpy as np
import supervision as sv
from pathlib import Path

from PIL import Image
from supervision.draw.color import ColorPalette
import sys

sys.path.insert(0, './Grounded-SAM-2')

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision
from tqdm import tqdm

torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def id_to_colors(id):  # id to color
    rgb = np.zeros((3,), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb


class AgGDinoSam2:

    def __init__(
            self,
            args,
            ag_root_dir
    ):
        self.ag_root_dir = ag_root_dir

        self.ag_frames_dir = os.path.join(ag_root_dir, "frames")
        self.ag_4D_dir = os.path.join(ag_root_dir, "ag4D")
        self.uni4D_dir = os.path.join(self.ag_4D_dir, "uni4D")
        self.unidepth_dir = os.path.join(self.uni4D_dir, "unidepth")
        os.makedirs(self.unidepth_dir, exist_ok=True)

        self.frames_path = os.path.join(self.ag_root_dir, "frames")
        self.annotations_path = os.path.join(self.ag_root_dir, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

        self.idx_to_id = [i for i in range(256 * 256 * 256)]
        np.random.shuffle(self.idx_to_id)  # mapping to randomize idx to id to get random color

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        video_id_frame_id_list_pkl_file_path = os.path.join(self.ag_root_dir, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)
        else:
            assert False, f"Please generate {video_id_frame_id_list_pkl_file_path} first"

        self.object_labels = [
            "person", "bag", "blanket", "book", "box", "broom", "chair", "clothes", "cup", "dish", "food", "laptop",
            "paper", "phone", "picture", "pillow", "sandwich", "shoe", "towel", "vacuum", "glass", "bottle", "notebook",
            "camera"
        ]
        self.gdino_output_dir = os.path.join(self.uni4D_dir, "gdino")
        self.sam2_output_dir = os.path.join(self.uni4D_dir, "sam2")

        # ------------- SAM2 settings -------------
        self.sam2_checkpoint = args.sam2_checkpoint
        self.model_cfg = args.sam2_model_config
        self.sam2_model = build_sam2(self.model_cfg, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)

    def get_frames(self, video_name):
        frames = []
        video_frames_path = os.path.join(self.frames_path, video_name)
        frame_id_list = self.video_id_frame_id_list[video_name]
        frame_id_list = sorted(np.unique(frame_id_list))
        for frame_id in frame_id_list:
            frames.append(imageio.imread(os.path.join(video_frames_path, f"{frame_id:06d}.png")))
        frames = np.array(frames)  # F x H x W x C
        return frames

    def get_frame_paths(self, video_name):
        frame_paths = []
        video_frames_path = os.path.join(self.frames_path, video_name)
        frame_id_list = self.video_id_frame_id_list[video_name]
        frame_id_list = sorted(np.unique(frame_id_list))
        for frame_id in frame_id_list:
            frame_paths.append(os.path.join(video_frames_path, f"{frame_id:06d}.png"))
        return frame_id_list, frame_paths

    def fetch_gdino_detections(self, video_name):
        gdino_output_path = os.path.join(self.gdino_output_dir, f"{video_name[:-4]}.pkl")
        if not os.path.exists(gdino_output_path):
            raise FileNotFoundError(f"Grounding DINO detections not found for {video_name}. Please run GDino first.")
        with open(gdino_output_path, 'rb') as f:
            gdino_detections = torch.load(f)

        # Map object detections to frame IDs
        object_to_frame_id_map = {}
        for frame_id, detections in gdino_detections.items():
            objects = detections['labels']
            for obj in objects:
                if obj not in object_to_frame_id_map:
                    object_to_frame_id_map[obj] = []
                object_to_frame_id_map[obj].append(frame_id)

        # Frame id to information maps
        frame_id_to_bbox_map = {}
        frame_id_to_scores_map = {}
        frame_id_to_labels_map = {}
        for frame_id, detections in gdino_detections.items():
            bboxes = detections['boxes']
            frame_id_to_bbox_map[frame_id] = bboxes
            scores = detections['scores']
            frame_id_to_scores_map[frame_id] = scores
            labels = detections['labels']
            frame_id_to_labels_map[frame_id] = labels

        return object_to_frame_id_map, frame_id_to_bbox_map, frame_id_to_scores_map, frame_id_to_labels_map

    def get_sam_masks_for_video(self, object_to_frame_id_map, video_name):
        """
        Get SAM masks for each object in the video.
        This function should be implemented to use the SAM2 model to get masks based on the bounding boxes.
        """
        # Placeholder for actual implementation
        pass

    def get_sam_masks_framewise(
            self,
            video_name,
            frame_id_to_bbox_map,
            frame_id_to_scores_map,
            frame_id_to_labels_map,
    ):
        """
        Get SAM masks for each frame in the video based on the bounding boxes.
        This function should be implemented to use the SAM2 model to get masks based on the bounding boxes.
        """
        video_output_path_vis = os.path.join(self.sam2_output_dir, video_name, "vis")
        video_output_path_mask = os.path.join(self.sam2_output_dir, video_name, "mask")
        os.makedirs(video_output_path_vis, exist_ok=True)
        os.makedirs(video_output_path_mask, exist_ok=True)

        frame_id_list, frame_paths = self.get_frame_paths(video_name)
        for frame_id, frame_path in tqdm(zip(frame_id_list, frame_paths), total=len(frame_id_list)):
            # Process each frame in the video
            image_pil = Image.open(frame_path).convert("RGB")
            image = np.array(image_pil)

            # Get GDINO detections for the current frame
            bboxes = frame_id_to_bbox_map.get(frame_id, None)
            if bboxes is None or len(bboxes) == 0:
                print(f"No bounding boxes found for frame {frame_id} in video {video_name}. Skipping.")
                continue

            self.sam2_predictor.set_image(image)
            input_boxes = bboxes.cpu().numpy()
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = frame_id_to_scores_map.get(frame_id, None).cpu().numpy().tolist()
            class_names = frame_id_to_labels_map.get(frame_id, None)
            # confidences = results[0]["scores"].cpu().numpy().tolist()
            # class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            img = cv2.imread(frame_path)
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids,
                confidence=np.array(confidences)
            )

            assert (len(detections.class_id) > 0)

            nms_idx = torchvision.ops.nms(
                torch.from_numpy(detections.xyxy).float(),
                torch.from_numpy(detections.confidence).float(),
                0.5
            ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.mask = detections.mask[nms_idx]

            labels = [
                f"{class_names[id]} {confidence:.2f}"
                for id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

            masks = detections.mask
            labels = detections.class_id

            assert (np.sum(labels == -1) == 0)  # check if any label == -1? concept graph has a bug with this

            color_mask = np.zeros(image.shape, dtype=np.uint8)
            obj_info_json = []

            # sort masks according to size
            mask_size = [np.sum(mask) for mask in masks]
            sorted_mask_idx = np.argsort(mask_size)[::-1]
            for idx in sorted_mask_idx:
                mask = masks[idx]
                color_mask[mask] = id_to_colors(self.idx_to_id[idx])
                obj_info_json.append({
                    "id": self.idx_to_id[idx],
                    "label": class_names[labels[idx]],
                    "score": float(detections.confidence[idx]),
                })

            frame_name = f"{frame_id:06d}.png"
            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(video_output_path_vis, frame_name), annotated_frame)  # VISUALIZATION
            cv2.imwrite(os.path.join(video_output_path_mask, frame_name), color_mask)
            with open(os.path.join(video_output_path_mask, frame_name.replace(".png", ".json")), "w") as f:
                json.dump(obj_info_json, f)

    def process_ag_video(self, video_name):
        video_output_dir = os.path.join(self.sam2_output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        (object_to_frame_id_map, frame_id_to_bbox_map,
         frame_id_to_scores_map, frame_id_to_labels_map) = self.fetch_gdino_detections(video_name)

        self.get_sam_masks_framewise(
            video_name=video_name,
            frame_id_to_bbox_map=frame_id_to_bbox_map,
            frame_id_to_scores_map=frame_id_to_scores_map,
            frame_id_to_labels_map=frame_id_to_labels_map
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--workdir", default="/data/rohith/ag")
    parser.add_argument("--sam2-checkpoint", default="./preprocess/pretrained/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="gsam2")
    parser.add_argument("--input-dir", default="ram")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    ag_gdino_sam2 = AgGDinoSam2(
        ag_root_dir=args.workdir,
        args=args
    )

    # for video_name in tqdm(ag_unidepth.video_list):
    #     ag_unidepth.process_ag_video(video_name)

    video_name = "00T1E.mp4"  # Replace with an actual video name from your dataset
    ag_gdino_sam2.process_ag_video(video_name)


if __name__ == "__main__":
    main()
