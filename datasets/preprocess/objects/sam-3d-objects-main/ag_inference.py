import os
import uuid
import imageio
import numpy as np
from IPython.display import Image as ImageDisplay

from notebook.inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer


class AgInference:

    def __init__(self):
        PATH = os.getcwd()
        TAG = "hf"
        self.config_path = f"{PATH}/checkpoints/{TAG}/pipeline.yaml"
        self.inference = Inference(self.config_path, compile=False)

    def process_frame(
            self,
            image_path,
            masks
    ):
        image = load_image(image_path)
        display_image(image, masks)

        outputs = [self.inference(image, mask, seed=42) for mask in masks]

        scene_gs = make_scene(*outputs)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

