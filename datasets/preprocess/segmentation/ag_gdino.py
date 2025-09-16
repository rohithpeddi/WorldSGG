import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    Sam2Processor,
    Sam2Model,
)
import cv2
import matplotlib.pyplot as plt





def draw_and_save_masks(
        frame_path: str,
        masks: List[np.ndarray],
        labels: List[str],
        output_dir: str,
        frame_number: int,
        overlay: bool = False
):
    """Draws masks on an image and saves the output."""
    if not os.path.exists(frame_path): return

    image = cv2.imread(frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a blank canvas for pure masks or a copy for overlay
    output_image = np.zeros_like(image) if not overlay else image.copy()

    unique_labels = sorted(list(set(labels)))
    color_map = get_color_map(len(unique_labels))
    label_to_color = {label: color for label, color in zip(unique_labels, color_map)}

    for i, mask in enumerate(masks):
        if mask.ndim == 3: mask = np.squeeze(mask)  # Ensure 2D mask

        color = label_to_color.get(labels[i], (255, 255, 255))
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = color

        if overlay:
            # Blend the mask with the original image
            output_image[mask > 0] = cv2.addWeighted(image[mask > 0], 0.4, colored_mask[mask > 0], 0.6, 0)
        else:
            # Add the colored mask to the blank canvas
            output_image[mask > 0] = colored_mask[mask > 0]

    # Convert back to BGR for OpenCV saving
    output_image_bgr = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"{frame_number:06d}.png"), output_image_bgr)


# --- Bounding Box Estimator (with Visualization) ---
class BoundingBoxEstimator:

    def __init__(self, ag_root_directory: str):
        # ... (initialization code is the same)
        self.ag_root_directory = ag_root_directory
        self.model_id = "IDEA-Research/grounding-dino-base"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.model_id).to(self.device)
        self.object_labels = ["a person", "a bag", "a blanket", "a book", "a box", "a broom", "a chair", "a clothes",
                              "a cup", "a dish", "a food", "a laptop", "a paper", "a phone", "a picture", "a pillow",
                              "a sandwich", "a shoe", "a towel", "a vacuum", "a glass", "a bottle", "a notebook", "a camera"]

        self.gdino_output_path = os.path.join(self.ag_root_directory, "detection", "gdino_bboxes")
        self.vis_output_path = os.path.join(self.ag_root_directory, "detection", "gdino_bboxes_vis")  # Vis path
        os.makedirs(self.gdino_output_path, exist_ok=True)

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

    def process_video_bboxes(self, video_id: str, visualize: bool = False):
        video_frames_dir_path = os.path.join(self.ag_root_directory, "sampled_frames", video_id)
        video_output_file_path = os.path.join(self.gdino_output_path, f"{video_id}.pkl")

        if os.path.exists(video_output_file_path):
            print(f"Bounding boxes for video {video_id} already exist. Skipping detection...")
            return

        video_predictions = {}
        video_frames = sorted([f for f in os.listdir(video_frames_dir_path) if f.endswith('.png')])
        for video_frame_name in tqdm(video_frames, desc=f"Detecting objects in {video_id}"):
            frame_path = os.path.join(video_frames_dir_path, video_frame_name)
            if not os.path.exists(frame_path): continue
            image = Image.open(frame_path).convert("RGB")
            inputs = self.processor(
                images=image,
                text=". ".join(self.object_labels),
                return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.gdino_model(**inputs)

            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=0.4,
                text_threshold=0.3,
                target_sizes=[image.size[::-1]])[0]

            video_predictions[video_frame_name] = {
                'boxes': results['boxes'],
                'scores': results['scores'],
                'labels': results['labels']
            }

            if visualize:
                vis_dir = os.path.join(self.vis_output_path, video_id)
                self.draw_and_save_bboxes(frame_path, results['boxes'], results['labels'], vis_dir, video_frame_name)

        with open(video_output_file_path, 'wb') as file:
            pickle.dump(video_predictions, file)

    def process_all_videos(self, visualize: bool = False):
        video_ids = sorted([d for d in os.listdir(os.path.join(self.ag_root_directory, "sampled_frames")) if
                            os.path.isdir(os.path.join(self.ag_root_directory, "sampled_frames", d))])
        for video_id in video_ids:
            self.process_video_bboxes(video_id, visualize)


# --- Segmentation Generator (with Visualization) ---
class SegmentationGenerator:

    def __init__(self, ag_root_directory: str):
        self.ag_root_directory = ag_root_directory
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SegmentationGenerator is using device: {self.device}")

        # --- MODIFIED LINES START ---
        # 1. Using the SAM-2 model ID
        self.sam_model_id = "facebook/sam2-hiera-base-400m"

        # 2. Changed class to the base Sam2Model
        self.sam_model = Sam2Model.from_pretrained(self.sam_model_id).to(self.device)

        # 3. The Sam2Processor remains the same
        self.sam_processor = Sam2Processor.from_pretrained(self.sam_model_id)
        # --- MODIFIED LINES END ---

        # Configure I/O paths (this part is unchanged)
        self.bbox_input_path = os.path.join(ag_root_directory, "detection", "gdino_bboxes")
        self.image_seg_output_path = os.path.join(ag_root_directory, "detection", "image_sam_segmentations")
        self.video_seg_output_path = os.path.join(ag_root_directory, "detection", "video_sam_segmentations")
        self.image_vis_path = os.path.join(ag_root_directory, "detection", "image_sam_masks_vis")
        self.video_vis_path = os.path.join(ag_root_directory, "detection", "video_sam_masks_vis")
        os.makedirs(self.image_seg_output_path, exist_ok=True);
        os.makedirs(self.video_seg_output_path, exist_ok=True)

        json_file_path = Path("./4d_video_frame_id_list.json")
        if json_file_path.exists():
            with open(json_file_path, 'r') as file:
                self.video_id_frames_dict = json.load(file)
        else:
            raise FileNotFoundError("Could not find '4d_video_frame_id_list.json'. Please ensure it exists.")

    # No changes are needed in the methods below, as the processor handles the output.

    def _load_video_frame_dict(self) -> Dict[str, Any]:
        json_file_path = Path("./4d_video_frame_id_list.json")
        if not json_file_path.exists(): raise FileNotFoundError("Could not find '4d_video_frame_id_list.json'.")
        with open(json_file_path, 'r') as file: return json.load(file)

    def run_image_based_segmentation(self, video_id: str, visualize: bool = False):
        bbox_file_path = os.path.join(self.bbox_input_path, f"{video_id}.pkl")
        output_file_path = os.path.join(self.image_seg_output_path, f"{video_id}.pkl")
        video_frames_path = os.path.join(self.ag_root_directory, "frames", video_id)

        if os.path.exists(output_file_path):
            print(f"Image-based segmentations for {video_id} already exist. Skipping segmentation...")
            if visualize:
                with open(output_file_path, 'rb') as f:
                    video_segmentations = pickle.load(f)
                vis_dir = os.path.join(self.image_vis_path, video_id)
                for frame_num, data in tqdm(video_segmentations.items(),
                                            desc=f"Visualizing Image Masks for {video_id}"):
                    frame_path = os.path.join(video_frames_path, f"{frame_num:06d}.png")
                    draw_and_save_masks(frame_path, data['masks'], data['labels'], vis_dir, frame_num)
            return

        if not os.path.exists(bbox_file_path): return
        with open(bbox_file_path, 'rb') as file:
            bbox_data = pickle.load(file)
        video_segmentations = {}

        for frame_number, detections in tqdm(bbox_data.items(), desc=f"Segmenting frames for {video_id}"):
            frame_path = os.path.join(video_frames_path, f"{frame_number:06d}.png")
            if not os.path.exists(frame_path) or detections['boxes'].shape[0] == 0: continue
            image = Image.open(frame_path).convert("RGB")
            inputs = self.sam_processor(image, input_boxes=[[detections['boxes'].tolist()]], return_tensors="pt").to(
                self.device)
            with torch.no_grad():
                # The output object from the base model also contains `pred_masks`
                outputs = self.sam_model(**inputs)

            # The processor's post-processing function works directly with the base model's output
            masks = self.sam_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )[0].cpu().numpy()

            video_segmentations[frame_number] = {'masks': masks, 'labels': detections['labels'],
                                                 'scores': detections['scores'].cpu().numpy()}

            if visualize:
                vis_dir = os.path.join(self.image_vis_path, video_id)
                draw_and_save_masks(frame_path, masks, detections['labels'], vis_dir, frame_number)

        with open(output_file_path, 'wb') as file:
            pickle.dump(video_segmentations, file)

    def run_video_based_segmentation(self, video_id: str, visualize: bool = False):
        bbox_file_path = os.path.join(self.bbox_input_path, f"{video_id}.pkl")
        output_file_path = os.path.join(self.video_seg_output_path, f"{video_id}.pkl")

        if os.path.exists(output_file_path):
            print(f"Video-based segmentations for {video_id} already exist. Skipping segmentation...")
            if visualize:
                with open(output_file_path, 'rb') as f:
                    video_tracked_segmentations = pickle.load(f)
                vis_dir = os.path.join(self.video_vis_path, video_id)
                video_frames_path = os.path.join(self.ag_root_directory, "frames", video_id)
                frame_list = self.video_id_frames_dict.get(video_id, [])
                for frame_num in tqdm(frame_list, desc=f"Visualizing Video Masks for {video_id}"):
                    frame_path = os.path.join(video_frames_path, f"{frame_num:06d}.png")
                    masks_in_frame, labels_in_frame = [], []
                    for label, tracked_masks in video_tracked_segmentations.items():
                        if frame_num in tracked_masks:
                            masks_in_frame.append(tracked_masks[frame_num]);
                            labels_in_frame.append(label)
                    if masks_in_frame:
                        draw_and_save_masks(frame_path, masks_in_frame, labels_in_frame, vis_dir, frame_num)
            return

        if not os.path.exists(bbox_file_path): return
        with open(bbox_file_path, 'rb') as file:
            bbox_data = pickle.load(file)
        video_frames_path = os.path.join(self.ag_root_directory, "frames", video_id)
        objects_by_label = {}
        for fn, dets in bbox_data.items():
            for i, lbl in enumerate(dets['labels']):
                if lbl not in objects_by_label: objects_by_label[lbl] = []
                objects_by_label[lbl].append({'frame': fn, 'box': dets['boxes'][i], 'score': dets['scores'][i]})
        video_tracked_segmentations = {}

        for label, detections in objects_by_label.items():
            if not detections: continue
            ref_detection = max(detections, key=lambda x: x['score'])
            ref_frame_num, ref_box = ref_detection['frame'], ref_detection['box'].unsqueeze(0)
            ref_image = Image.open(os.path.join(video_frames_path, f"{ref_frame_num:06d}.png")).convert("RGB")
            inputs = self.sam_processor(ref_image, input_boxes=[[ref_box.tolist()]], return_tensors="pt").to(
                self.device)
            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            ref_mask = self.sam_processor.post_process_masks(
                outputs.pred_masks,
                inputs["original_sizes"],
                inputs["reshaped_input_sizes"]
            )[0]

            propagated_masks = {}
            frame_list = self.video_id_frames_dict.get(video_id, [])
            for frame_number in tqdm(frame_list, desc=f"Tracking '{label}' in {video_id}"):
                frame_path = os.path.join(video_frames_path, f"{frame_number:06d}.png")
                if not os.path.exists(frame_path): continue
                current_image = Image.open(frame_path).convert("RGB")
                inputs = self.sam_processor(current_image, input_masks=[[[ref_mask.cpu().numpy()]]],
                                            return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.sam_model(**inputs)

                p_mask = self.sam_processor.post_process_masks(
                    outputs.pred_masks,
                    inputs["original_sizes"],
                    inputs["reshaped_input_sizes"]
                )[0].cpu().numpy()

                propagated_masks[frame_number] = p_mask
            video_tracked_segmentations[label] = propagated_masks

        with open(output_file_path, 'wb') as file:
            pickle.dump(video_tracked_segmentations, file)

        if visualize:
            vis_dir = os.path.join(self.video_vis_path, video_id)
            for frame_num in tqdm(frame_list, desc=f"Visualizing Video Masks for {video_id}"):
                frame_path = os.path.join(video_frames_path, f"{frame_num:06d}.png")
                masks_in_frame, labels_in_frame = [], []
                for label, tracked_masks in video_tracked_segmentations.items():
                    if frame_num in tracked_masks:
                        masks_in_frame.append(tracked_masks[frame_num]);
                        labels_in_frame.append(label)
                if masks_in_frame:
                    draw_and_save_masks(frame_path, masks_in_frame, labels_in_frame, vis_dir, frame_num)

    def process_all_videos(self, run_image_based=False, run_video_based=False, visualize=False):
        video_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.bbox_input_path) if f.endswith('.pkl')])
        for video_id in video_ids:
            if run_image_based: self.run_image_based_segmentation(video_id, visualize)
            if run_video_based: self.run_video_based_segmentation(video_id, visualize)


# --- Mask Combiner (with Visualization and Video Creation) ---
class MaskCombiner:
    def __init__(self, ag_root_directory: str):
        self.ag_root_directory = ag_root_directory
        self.image_seg_path = os.path.join(ag_root_directory, "detection", "image_sam_segmentations")
        self.video_seg_path = os.path.join(ag_root_directory, "detection", "video_sam_segmentations")
        self.combined_output_path = os.path.join(ag_root_directory, "detection", "combined_segmentations")
        self.masked_frames_path = os.path.join(ag_root_directory, "masked_frames")
        self.overlayed_frames_path = os.path.join(ag_root_directory, "overlayed_frames")
        self.masked_videos_path = os.path.join(ag_root_directory, "masked_videos")
        os.makedirs(self.combined_output_path, exist_ok=True)
        os.makedirs(self.masked_videos_path, exist_ok=True)

    def create_video_from_frames(self, frames_dir: str, video_path: str, fps: int = 30):
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        if not frame_files: return

        frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

        for frame_file in tqdm(frame_files, desc=f"Creating video {os.path.basename(video_path)}"):
            video_writer.write(cv2.imread(os.path.join(frames_dir, frame_file)))
        video_writer.release()

    def combine_masks_for_video(self, video_id: str, visualize: bool = False):
        # ... (combination logic is the same, with added visualize call)
        image_seg_file = os.path.join(self.image_seg_path, f"{video_id}.pkl")
        video_seg_file = os.path.join(self.video_seg_path, f"{video_id}.pkl")
        output_file = os.path.join(self.combined_output_path, f"{video_id}.pkl")
        video_frames_path = os.path.join(self.ag_root_directory, "frames", video_id)

        if os.path.exists(output_file):
            print(f"Combined masks for {video_id} already exist. Skipping combination...")
            if visualize:
                with open(output_file, 'rb') as f:
                    final_segmentations = pickle.load(f)
            else:
                return  # if not visualizing and exists, we are done
        else:
            if not os.path.exists(image_seg_file) or not os.path.exists(video_seg_file): return
            with open(image_seg_file, 'rb') as f:
                image_data = pickle.load(f)
            with open(video_seg_file, 'rb') as f:
                video_data = pickle.load(f)
            final_segmentations = {}
            for frame_number, frame_detections in tqdm(image_data.items(), desc=f"Combining masks for {video_id}"):
                final_masks, final_labels = [], []
                image_masks_by_label = {}
                for i, label in enumerate(frame_detections['labels']):
                    if label not in image_masks_by_label: image_masks_by_label[label] = []
                    image_masks_by_label[label].append(np.squeeze(frame_detections['masks'][i]))
                for label, img_masks in image_masks_by_label.items():
                    video_mask = video_data.get(label, {}).get(frame_number)
                    combined_mask = np.logical_or.reduce(img_masks) if video_mask is None else np.squeeze(video_mask)
                    if video_mask is not None:
                        for img_mask in img_masks: combined_mask = np.logical_or(combined_mask, img_mask)
                    final_masks.append(combined_mask.astype(np.uint8));
                    final_labels.append(label)
                final_segmentations[frame_number] = {'masks': final_masks, 'labels': final_labels}
            with open(output_file, 'wb') as f:
                pickle.dump(final_segmentations, f)

        if visualize:
            masked_dir = os.path.join(self.masked_frames_path, video_id)
            overlay_dir = os.path.join(self.overlayed_frames_path, video_id)
            for frame_num, data in tqdm(final_segmentations.items(), desc=f"Visualizing final masks for {video_id}"):
                frame_path = os.path.join(video_frames_path, f"{frame_num:06d}.png")
                # Save pure masks
                draw_and_save_masks(frame_path, data['masks'], data['labels'], masked_dir, frame_num, overlay=False)
                # Save overlay masks
                draw_and_save_masks(frame_path, data['masks'], data['labels'], overlay_dir, frame_num, overlay=True)

            # Create videos from the generated frames
            self.create_video_from_frames(masked_dir, os.path.join(self.masked_videos_path, f"{video_id}_masked.mp4"))
            self.create_video_from_frames(overlay_dir,
                                          os.path.join(self.masked_videos_path, f"{video_id}_overlayed.mp4"))

    def process_all_videos(self, visualize: bool = False):
        video_ids = sorted([os.path.splitext(f)[0] for f in os.listdir(self.image_seg_path) if f.endswith('.pkl')])
        for video_id in video_ids:
            self.combine_masks_for_video(video_id, visualize)


def main():
    parser = argparse.ArgumentParser(description="Run object detection, segmentation, and mask combination.")
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag/",
                        help="Root directory with a 'frames/' subfolder.")
    parser.add_argument("--run_detection", action='store_true', help="Run the bounding box detection step.")
    parser.add_argument("--run_image_segmentation", action='store_true',
                        help="Run image-based (frame-by-frame) segmentation.")
    parser.add_argument("--run_video_segmentation", action='store_true',
                        help="Run video-based (object tracking) segmentation.")
    parser.add_argument("--run_combination", action='store_true',
                        help="Combine image and video masks into a final result.")
    parser.add_argument("--visualize", action='store_true',
                        help="Generate and save visual outputs for all selected steps.")
    args = parser.parse_args()

    if args.run_detection:
        print("\n--- 🚀 Starting Bounding Box Estimation ---")
        bbox_estimator = BoundingBoxEstimator(args.ag_root_directory)
        bbox_estimator.process_all_videos(visualize=args.visualize)

    if args.run_image_segmentation or args.run_video_segmentation:
        print("\n--- 🚀 Starting Segmentation Generation ---")
        segmentation_generator = SegmentationGenerator(args.ag_root_directory)
        segmentation_generator.process_all_videos(
            run_image_based=args.run_image_segmentation,
            run_video_based=args.run_video_segmentation,
            visualize=args.visualize
        )

    if args.run_combination:
        print("\n--- 🚀 Starting Mask Combination ---")
        mask_combiner = MaskCombiner(args.ag_root_directory)
        mask_combiner.process_all_videos(visualize=args.visualize)


if __name__ == "__main__":
    main()
