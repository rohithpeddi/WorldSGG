import argparse
import gc
import logging
import math
import os
import pickle
import random
import shutil
import sys
import copy
import warnings
warnings.filterwarnings("ignore")

import accelerate
import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
import transformers
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDIMScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (EMAModel,
                                      compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from diffusers.utils import check_min_version, deprecate, is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from omegaconf import OmegaConf
from packaging import version
from PIL import Image
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from transformers import AutoTokenizer
from transformers.utils import ContextManagers

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

from rose.data.bucket_sampler import (ASPECT_RATIO_512,
                                           ASPECT_RATIO_RANDOM_CROP_512,
                                           ASPECT_RATIO_RANDOM_CROP_PROB,
                                           AspectRatioBatchImageVideoSampler,
                                           RandomSampler, get_closest_ratio)
from rose.data.dataset_image_video import (ImageVideoControlDataset,
                                                ImageVideoSampler,
                                                get_random_mask)
from rose.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel,
                               WanTransformer3DModel)
from rose.pipeline import WanFunInpaintPipeline
from rose.utils.discrete_sampler import DiscreteSampling
from rose.utils.utils import (get_video_to_video_latent,
                                    get_video_and_mask,
                                    save_videos_grid)

def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs

def parse_args():
    parser = argparse.ArgumentParser(description="Run video inpainting pipeline")

    parser.add_argument("--validation_videos", type=str, nargs='+', default=['data/eval/example1/video.mp4'],
                        help="Path(s) to validation videos.")
    parser.add_argument("--validation_masks", type=str, nargs='+', default=['data/eval/example1/mask.mp4'],
                        help="Path(s) to validation masks.")
    parser.add_argument("--validation_prompts", type=str, nargs='+', default=[""],
                        help="Validation prompts.")
    parser.add_argument("--output_dir", type=str, default="results/",
                        help="Output directory.")
    parser.add_argument("--video_length", type=int, default=49,
                        help="Number of frames in video.") # The length of videos needs to be 16n+1.
    parser.add_argument("--sample_size", type=int, nargs=2, default=[480, 720],
                        help="Video frame size: height width.")

    return parser.parse_args()

def main():
    args = parse_args()
    pretrained_model_name_or_path = "models/Wan2.1-Fun-1.3B-InP"
    pretrained_transformer_path = "weights/transformer"
    config_path = "configs/wan2.1/wan_civitai.yaml"
    config = OmegaConf.load(config_path)
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(pretrained_model_name_or_path, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
    )

    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(pretrained_model_name_or_path, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    )
    
    transformer3d = WanTransformer3DModel.from_pretrained(
        os.path.join(pretrained_transformer_path, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
    )

    pipeline = WanFunInpaintPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer3d,
        scheduler=noise_scheduler,
        clip_image_encoder=clip_image_encoder
    ).to("cuda", torch.float16)

    with torch.no_grad():
        for i, (validation_prompt, validation_video, validation_mask) in enumerate(zip(args.validation_prompts, args.validation_videos, args.validation_masks), start=1):
            input_video, input_mask, ref_image, clip_image = get_video_and_mask(
                input_video_path=validation_video,
                video_length=args.video_length,
                sample_size=args.sample_size,
                input_mask_path=validation_mask
            )      
            
            result = pipeline(
                prompt=validation_prompt,
                video=input_video,
                mask_video=input_mask,
                num_frames=args.video_length,
                num_inference_steps=50
            ).videos

            os.makedirs(args.output_dir, exist_ok=True)
            out_path = os.path.join(args.output_dir, f"example-{i}.mp4")
            save_videos_grid(result, out_path)


if __name__ == "__main__":
    main()





    
