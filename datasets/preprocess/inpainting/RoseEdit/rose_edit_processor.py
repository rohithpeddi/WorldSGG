import numpy as np
from PIL import Image

import cv2
import os
import torch
import pickle
import matplotlib.pyplot as plt


class RoseEditProcessor:

    def __init__(
            self,
            data_directory,
            sam2_directory,
            monst3r_directory,
            mask_prefix="mask_"
    ):
        self.pipeline = None
        self.data_directory = data_directory
        self.sam2_directory = sam2_directory
        self.monst3r_directory = monst3r_directory
        self.mask_video_directory = os.path.join(self.data_directory, "mask_videos")
        self.sampled_video_directory = os.path.join(self.data_directory, "sampled_videos")
        self.static_video_directory = os.path.join(self.data_directory, "static_videos")
        self.static_frames_directory = os.path.join(self.data_directory, "static_frames")
        self.mask_prefix = mask_prefix

        os.makedirs(self.mask_video_directory, exist_ok=True)
        os.makedirs(self.sampled_video_directory, exist_ok=True)
        os.makedirs(self.static_video_directory, exist_ok=True)
        os.makedirs(self.static_frames_directory, exist_ok=True)

        self.debug = False

        # Load the video_id_frame_id_list.pkl file that contains the list of (video_id, frame_id) tuples.
        video_id_frame_id_list_pkl_file_path = os.path.join(self.data_directory, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)

    def get_sam2_masks(self, video_id):
        frame_ids = self.video_id_frame_id_list[video_id]
        frame_id_list = sorted(list(np.unique(frame_ids)))

        sam_video_mask_dir = os.path.join(self.sam2_directory, video_id, "mask")
        frame_files = sorted([f for f in os.listdir(sam_video_mask_dir) if f.endswith('.png')])
        frame_ids = [os.path.splitext(f)[0] for f in frame_files]

        sam_masks_dict = {}
        for frame_file, frame_id in zip(frame_files, frame_ids):
            mask_path = os.path.join(sam_video_mask_dir, frame_file)
            mask_image = Image.open(mask_path).convert("RGB")
            mask_image_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
            # Change it into binary mask if any pixel value is greater than 0, set it to 1.0 else 0.0
            mask_image_tensor = (mask_image_tensor > 0).float()
            sam_masks_dict[int(frame_id)] = mask_image_tensor

        sam_mask_keys = list(sam_masks_dict.keys())
        sam_mask_keys.sort()
        print(f"[{video_id}] Loaded {len(sam_mask_keys)} SAM2 masks: {sam_mask_keys[:5]} ... {sam_mask_keys[-5:]}")

        sam2_masks = []
        for frame_id in frame_id_list:
            sam_mask = sam_masks_dict[frame_id]

            # Convert the mask tensor to a PIL image
            sam_mask = Image.fromarray((sam_mask.numpy() * 255).astype(np.uint8))
            sam2_masks.append(sam_mask)
        return sam2_masks

    def get_overlayed_sam2_masks(self, video_id):
        frame_ids = self.video_id_frame_id_list[video_id]
        frame_id_list = sorted(list(np.unique(frame_ids)))

        sam_video_mask_dir = os.path.join(self.sam2_directory, video_id, "mask")
        sam_video_frame_dir = os.path.join(self.data_directory, "frames", video_id)
        frame_files = sorted([f for f in os.listdir(sam_video_mask_dir) if f.endswith('.png')])
        frame_ids = [os.path.splitext(f)[0] for f in frame_files]

        sam_masks_dict = {}
        for frame_file, frame_id in zip(frame_files, frame_ids):
            mask_path = os.path.join(sam_video_mask_dir, frame_file)
            mask_image = Image.open(mask_path).convert("RGB")
            mask_image_tensor = torch.from_numpy(np.array(mask_image)).float() / 255.0
            # Change it into binary mask if any pixel value is greater than 0, set it to 1.0 else 0.0
            mask_image_tensor = (mask_image_tensor > 0).float()
            sam_masks_dict[int(frame_id)] = mask_image_tensor

        sam_mask_keys = list(sam_masks_dict.keys())
        sam_mask_keys.sort()
        print(f"[{video_id}] Loaded {len(sam_mask_keys)} SAM2 masks: {sam_mask_keys[:5]} ... {sam_mask_keys[-5:]}")

        overlayed_sam2_masks = []
        for frame_id in frame_id_list:
            sam_mask = sam_masks_dict[frame_id]

            # Load the original frame
            original_frame_path = os.path.join(sam_video_frame_dir, f"{frame_id:06d}.png")
            original_frame = Image.open(original_frame_path).convert("RGB")
            original_frame_tensor = torch.from_numpy(np.array(original_frame)).float() / 255.0

            # Ensure the mask is binary
            binary_mask = (sam_mask > 0.5).float()

            # Create the masked image by setting the masked region to white (1.0)
            masked_frame_tensor = original_frame_tensor * (1 - binary_mask) + binary_mask * 1.0
            masked_frame = Image.fromarray((masked_frame_tensor.numpy() * 255).astype(np.uint8))
            overlayed_sam2_masks.append(masked_frame)
        return overlayed_sam2_masks

    def store_overlayed_sam2_masks(self, video_id, overlayed_sam2_masks):
        overlayed_mask_dir = os.path.join(self.data_directory, "overlayed_sam2_masks", video_id)
        os.makedirs(overlayed_mask_dir, exist_ok=True)
        for idx, overlayed_mask in enumerate(overlayed_sam2_masks):
            overlayed_mask_path = os.path.join(overlayed_mask_dir, f"{idx:06d}.png")
            overlayed_mask.save(overlayed_mask_path)
        print(f"[{video_id}] Saved {len(overlayed_sam2_masks)} overlayed SAM2 masks to {overlayed_mask_dir}")

    def store_masked_frames(self, video_id, mask_frames):
        masked_frame_dir = os.path.join(self.data_directory, "masked_frames", video_id)
        os.makedirs(masked_frame_dir, exist_ok=True)
        for idx, mask_frame in enumerate(mask_frames):
            mask_frame_path = os.path.join(masked_frame_dir, f"{idx:06d}.png")
            mask_frame.save(mask_frame_path)
        print(f"[{video_id}] Saved {len(mask_frames)} masked frames to {masked_frame_dir}")

    def store_sampled_frames(self, video_id, sampled_frames):
        sampled_frame_dir = os.path.join(self.data_directory, "sampled_frames", video_id)
        os.makedirs(sampled_frame_dir, exist_ok=True)
        for idx, sampled_frame in enumerate(sampled_frames):
            sampled_frame_path = os.path.join(sampled_frame_dir, f"{idx:06d}.png")
            sampled_frame.save(sampled_frame_path)
        print(f"[{video_id}] Saved {len(sampled_frames)} sampled frames to {sampled_frame_dir}")

    def get_sampled_frames(self, video_id):
        frame_ids = self.video_id_frame_id_list[video_id]
        frame_id_list = sorted(list(np.unique(frame_ids)))

        sampled_frames = []
        for frame_id in frame_id_list:
            # Load the original frame
            original_frame_path = os.path.join(self.data_directory, "frames", video_id, f"{frame_id:06d}.png")
            original_frame = Image.open(original_frame_path).convert("RGB")
            sampled_frames.append(original_frame)
        return sampled_frames

    def store_mask_videos(self, video_id, mask_frames):
        mask_video_path = os.path.join(self.mask_video_directory, video_id)
        # Convert the mask frames which is a list of PIL images to a video and save it
        height, width = mask_frames[0].size[1], mask_frames[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(mask_video_path, fourcc, 30, (width, height))
        for frame in mask_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"[{video_id}] Saved mask video to {mask_video_path}")

    def store_sampled_videos(self, video_id, sampled_frames):
        sampled_video_path = os.path.join(self.sampled_video_directory, video_id)
        # Convert the sampled frames which is a list of PIL images to a video and save it
        height, width = sampled_frames[0].size[1], sampled_frames[0].size[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(sampled_video_path, fourcc, 30, (width, height))
        for frame in sampled_frames:
            frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)
        video_writer.release()
        print(f"[{video_id}] Saved sampled video to {sampled_video_path}")

    def resize_frames(self, frames, target_size=(480, 720)):
        resized_frames = []
        for frame in frames:
            resized_frame = frame.resize(target_size, Image.LANCZOS)
            resized_frames.append(resized_frame)
        return resized_frames

    def process(self, video_id):
        print(f"Processing video_id: {video_id}")

        # Get SAM2 masks
        # sam2_mask_frames = self.get_sam2_masks(video_id)

        # Get overlayed SAM2 masks
        overlayed_sam2_masks = self.get_overlayed_sam2_masks(video_id)
        self.store_overlayed_sam2_masks(video_id, overlayed_sam2_masks)

        # Resize SAM2 masks to 480x720
        # resized_sam2_mask_frames = self.resize_frames(sam2_mask_frames, target_size=(480, 720))
        # sam2_mask_frames = resized_sam2_mask_frames

        # self.store_masked_frames(video_id, sam2_mask_frames)
        # self.store_mask_videos(video_id, sam2_mask_frames)

        # Get sampled original frames
        # sampled_frames = self.get_sampled_frames(video_id)

        # Resize sampled frames to 480x720
        # resized_sampled_frames = self.resize_frames(sampled_frames, target_size=(480, 720))
        # sampled_frames = resized_sampled_frames

        # self.store_sampled_frames(video_id, sampled_frames)
        # self.store_sampled_videos(video_id, sampled_frames)

    def static_video_to_frames(self, video_id):
        static_video_path = os.path.join(self.static_video_directory, video_id)
        if not os.path.exists(static_video_path):
            print(f"[{video_id}] Static video not found at {static_video_path}")
            return

        cap = cv2.VideoCapture(static_video_path)
        frame_idx = 0
        static_frame_dir = os.path.join(self.static_frames_directory, video_id)
        os.makedirs(static_frame_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frame_path = os.path.join(static_frame_dir, f"{frame_idx:06d}.png")
            frame_pil.save(frame_path)
            frame_idx += 1

        cap.release()
        print(f"[{video_id}] Extracted {frame_idx} frames to {static_frame_dir}")


def main():
    sam2_directory = "/data/rohith/ag/ag4D/uni4D/sam2"
    monst3r_directory = "/data/rohith/ag/ag4D/monst3r"
    monst3r_mask_prefix = "dynamic_mask_"
    data_directory = "/data/rohith/ag/"

    video_id_list = ["00T1E.mp4"]

    processor = RoseEditProcessor(
        data_directory=data_directory,
        sam2_directory=sam2_directory,
        monst3r_directory=monst3r_directory,
        mask_prefix=monst3r_mask_prefix
    )

    for video_id in video_id_list:
        processor.process(video_id)


def main_static_video_to_frames():
    data_directory = "/data/rohith/ag/"
    processor = RoseEditProcessor(
        data_directory=data_directory,
        sam2_directory="",
        monst3r_directory="",
        mask_prefix=""
    )

    video_id_list = ["00T1E.mp4"]

    for video_id in video_id_list:
        processor.static_video_to_frames(video_id)


if __name__ == "__main__":
    main_static_video_to_frames()
