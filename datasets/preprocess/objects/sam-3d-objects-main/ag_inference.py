import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any

import imageio
import pickle

import numpy as np
from IPython.display import Image as ImageDisplay

from notebook.inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer


class AgSam3DInference:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
    ):
        # -------------------- Data Folders --------------------
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)
        self.ag_root_directory = Path(ag_root_directory)

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

        # -------------------- Inference Setup --------------------
        PATH = os.getcwd()
        TAG = "hf"
        self.config_path = f"{PATH}/checkpoints/{TAG}/pipeline.yaml"
        self.inference = Inference(self.config_path, compile=False)

    def process_frame(
            self,
            frame_name: str,
            image_path: str,
            masks: List[np.ndarray],
            video_output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run SAM3D on a single frame, save all outputs for that frame, and return basic metadata.

        Saved artifacts (inside video_output_dir / frame_name):
        - {frame_name}.gif                : rendered SAM3D turntable video
        - {frame_name}_sam3d.pkl          : dict with image_path, masks, raw outputs, scene_gaussians, video_path
        """

        # ------------------------------------------------------------------
        # 0) Prepare per-frame directory
        # ------------------------------------------------------------------
        frame_output_dir = video_output_dir / frame_name
        gaussians_dir = frame_output_dir / "gaussians"

        os.makedirs(frame_output_dir, exist_ok=True)
        os.makedirs(gaussians_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Load image & run inference per mask
        # ------------------------------------------------------------------
        image = load_image(image_path)
        display_image(image, masks)

        # Run SAM3D for each mask and collect outputs
        outputs = []
        for mask in masks:
            out = self.inference(image, mask, seed=42)
            outputs.append(out)

        # ------------------------------------------------------------------
        # 2) Build scene gaussians and render video
        # ------------------------------------------------------------------
        scene_gs = make_scene(*outputs)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        video = render_video(scene_gs, r=1, fov=60, resolution=512)["color"]

        # ------------------------------------------------------------------
        # 3) Save rendered video (GIF)
        # ------------------------------------------------------------------
        video_frame_file_path = frame_output_dir / f"{frame_name}.gif"
        imageio.mimsave(
            video_frame_file_path,
            video,
            format="GIF",
            duration=1000 / 30,  # assume 30fps
            loop=0,  # 0 means loop indefinitely
        )

        # ------------------------------------------------------------------
        # 4) Save all outputs + gaussians in a single pickle
        # ------------------------------------------------------------------
        pkl_output_path = frame_output_dir / f"{frame_name}_sam3d.pkl"

        frame_dump: Dict[str, Any] = {
            "frame_name": frame_name,
            "image_path": str(image_path),
            "masks": masks,
            "outputs": outputs,  # list of per-mask SAM3D outputs
            "scene_gaussians": scene_gs,  # combined gaussian scene
        }

        with open(pkl_output_path, "wb") as f:
            pickle.dump(frame_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

        # (Optional) if you want to also save gaussians separately per frame, you
        # could add something like:
        # gaussians_pkl_path = gaussians_dir / f"{frame_name}_scene_gaussians.pkl"
        # with open(gaussians_pkl_path, "wb") as f:
        #     pickle.dump(scene_gs, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ------------------------------------------------------------------
        # 5) Return basic metadata (optional)
        # ------------------------------------------------------------------
        return {
            "frame_name": frame_name,
            "frame_output_dir": frame_output_dir,
            "pickle_path": pkl_output_path,
            "gif_path": video_frame_file_path,
        }

