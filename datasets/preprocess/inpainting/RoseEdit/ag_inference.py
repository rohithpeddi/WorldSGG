import os
import sys
import math
import glob
import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

warnings.filterwarnings("ignore")

import torch
import cv2
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import OmegaConf
from transformers import AutoTokenizer

# Insert project roots so rose.* imports resolve just like your original script
current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path),
                 os.path.dirname(os.path.dirname(current_file_path)),
                 os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# rose imports (same as your script)
from rose.utils.utils import get_video_and_mask, save_videos_grid
from rose.pipeline import WanFunInpaintPipeline
from rose.models import (AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel)
from diffusers import FlowMatchEulerDiscreteScheduler


@dataclass
class ExtractorConfig:
    data_dir: str = "/data/rohith/ag"
    videos_dir: str = "/data/rohith/ag/videos"
    sampled_dir: str = "/data/rohith/ag/sampled_videos"
    masked_dir: str = "/data/rohith/ag/mask_videos"
    static_dir: str = "/data/rohith/ag/static_videos"

    # Model/config paths (same as your original script)
    pretrained_model_name_or_path: str = "models/Wan2.1-Fun-1.3B-InP"
    pretrained_transformer_path: str = "weights/transformer"
    config_path: str = "configs/wan2.1/wan_civitai.yaml"

    # Inference
    work_size_hw: Tuple[int, int] = (480, 720)  # (H, W) for working resolution
    max_chunk_len: int = 129  # ≤ 128 per your requirement
    num_inference_steps: int = 50
    device: str = "cuda"
    dtype = torch.float16

    # I/O behavior
    save_resized_work_clips: bool = True  # also save the (480,720) sampled & masked intermediates
    overwrite_outputs: bool = True  # overwrite existing outputs

    # If your transformer needs 16n+1, set this True
    ENFORCE_16N_PLUS_1: bool = True


class StaticAgSceneExtractor:

    def __init__(self, cfg: ExtractorConfig = ExtractorConfig()):
        self.cfg = cfg
        os.makedirs(self.cfg.sampled_dir, exist_ok=True)
        os.makedirs(self.cfg.masked_dir, exist_ok=True)
        os.makedirs(self.cfg.static_dir, exist_ok=True)
        self._init_models()

    # --------------------------
    # Model / pipeline init
    # --------------------------
    def _init_models(self):
        cfg = self.cfg
        config = OmegaConf.load(cfg.config_path)
        self.config = config

        # Tokenizer
        print("[StaticAgSceneExtractor] Loading tokenizer ....")
        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(cfg.pretrained_model_name_or_path,
                         config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
        )

        # Text encoder
        print("[StaticAgSceneExtractor] Loading text_encoder ....")
        text_encoder = WanT5EncoderModel.from_pretrained(
            os.path.join(cfg.pretrained_model_name_or_path,
                         config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
            additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
            low_cpu_mem_usage=True,
        )

        # CLIP image encoder
        print("[StaticAgSceneExtractor] Loading clip_image_encoder ....")
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(cfg.pretrained_model_name_or_path,
                         config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
        )

        # Scheduler
        def filter_kwargs(cls, kwargs):
            import inspect
            sig = inspect.signature(cls.__init__)
            valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
            return {k: v for k, v in kwargs.items() if k in valid_params}

        noise_scheduler = FlowMatchEulerDiscreteScheduler(
            **filter_kwargs(FlowMatchEulerDiscreteScheduler, OmegaConf.to_container(config['scheduler_kwargs']))
        )

        # VAE
        print("[StaticAgSceneExtractor] Loading vae ....")
        vae = AutoencoderKLWan.from_pretrained(
            os.path.join(cfg.pretrained_model_name_or_path, config['vae_kwargs'].get('vae_subpath', 'vae')),
            additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
        )

        # 3D transformer
        print("[StaticAgSceneExtractor] Loading transformer3d ....")
        transformer3d = WanTransformer3DModel.from_pretrained(
            os.path.join(cfg.pretrained_transformer_path,
                         config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
            transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        )

        # Pipeline
        pipe = WanFunInpaintPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer3d,
            scheduler=noise_scheduler,
            clip_image_encoder=clip_image_encoder
        ).to(self.cfg.device, self.cfg.dtype)

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.clip_image_encoder = clip_image_encoder
        self.noise_scheduler = noise_scheduler
        self.vae = vae
        self.transformer3d = transformer3d
        self.pipeline = pipe

    # --------------------------
    # Utilities
    # --------------------------
    @staticmethod
    def _stem(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def _get_video_hw(path: str) -> Tuple[int, int]:
        """Return (H, W) from the source video using OpenCV without loading full frames."""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return h, w

    def _compute_chunks(self, total_frames: int) -> List[Tuple[int, int]]:
        """
        Partition [0, total_frames) into chunks where:
          - Each chunk length is one of {17, 33, 49, 65, 81, 97, 113, 129}.
          - No chunk > 129.
          - If the final leftover < 17, merge it into the previous chunk
            by expanding the last chunk's end to cover the total_frames.
        """
        if total_frames <= 0:
            return []

        allowed = [129, 113, 97, 81, 65, 49, 33, 17]
        chunks: List[Tuple[int, int]] = []
        start = 0
        remaining = total_frames

        while remaining >= 17:
            # pick the largest allowed size that fits
            for size in allowed:
                if size <= remaining:
                    chunks.append((start, start + size))
                    start += size
                    remaining -= size
                    break

        if remaining > 0 and chunks:
            # leftover < 17 → merge into the last chunk
            last_start, last_end = chunks[-1]
            chunks[-1] = (last_start, total_frames)
        elif remaining > 0 and not chunks:
            # edge case: total_frames < 17, just one chunk [0, total_frames)
            raise ValueError("Total frames less than 17; cannot form a valid chunk.")

        return chunks

    @staticmethod
    def _resize_video_tensor(video: torch.Tensor, out_hw: Tuple[int, int]) -> torch.Tensor:
        """
        Resize a video tensor to (H_out, W_out).
        Accepts shapes [B, C, F, H, W] or [B, F, C, H, W].
        Returns the same order as input.
        """
        if video.ndim != 5:
            raise ValueError(f"Expected 5D video tensor, got {video.shape}")

        order = "BFCHW"
        B, F_, C, H, W = video.shape
        vid = rearrange(video, "b f c h w -> (b f) c h w")
        vid = F.interpolate(vid, size=out_hw, mode="bilinear", align_corners=False)
        vid = rearrange(vid, "(b f) c h w -> b f c h w", b=B, f=F_)
        return vid

    @staticmethod
    def _ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)

    # --------------------------
    # Core processing
    # --------------------------
    def _load_inputs_for_chunk(
            self,
            video_path: str,
            mask_path: str,
            work_frames: int,
            work_hw: Tuple[int, int],
            start_idx: int,
            end_idx: int,
    ):
        """
        Use rose.utils.utils.get_video_and_mask to fetch a window [start_idx, end_idx)
        resized to work_hw and clamped to work_frames.
        """
        # The rose helper loads a continuous clip; we pass the requested length explicitly.
        # It returns: input_video, input_mask, ref_image, clip_image
        # NOTE: If ENFORCE_16N_PLUS_1 is True and window length doesn't match that,
        # you may need to pad/trim externally. For now we rely on window selection.
        window_len = end_idx - start_idx
        num_frames = min(work_frames, window_len)

        input_video, input_mask, ref_image, clip_image = get_video_and_mask(
            input_video_path=video_path,
            video_length=num_frames,
            sample_size=list(work_hw),  # expects [H, W]
            input_mask_path=mask_path,
            start_idx=start_idx,
        )
        return input_video, input_mask, num_frames

    def process_single_video(self, video_path: str, prompt: str = ""):
        """
        - Finds the matching mask video in masked_videos/ (same stem).
        - Splits into chunks per _compute_chunks().
        - Runs the ROSE inpaint pipeline per chunk at (480,720),
          restores each chunk to original HxW, and assembles frames
          by global frame index.
        - Writes a single stitched output video to static_videos/.
        """
        stem = self._stem(video_path)
        mask_video_path = os.path.join(self.cfg.masked_dir, f"{stem}.mp4")
        sample_video_path = os.path.join(self.cfg.sampled_dir, f"{stem}.mp4")

        if not os.path.exists(mask_video_path):
            raise FileNotFoundError(f"Mask video missing for {stem}: {mask_video_path}")

        # Original resolution (H, W)
        orig_h, orig_w = self._get_video_hw(video_path)

        # Get total frame count from the MASK video (assumed aligned with source video)
        import cv2
        cap = cv2.VideoCapture(mask_video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Compute chunk windows [(start, end)], end exclusive
        chunks = self._compute_chunks(total_frames)

        # Buffer to hold restored frames at global positions
        # Each entry will be a numpy array [H, W, C] (uint8, RGB)
        frame_buffer = [None] * total_frames

        # Process each chunk
        for ci, (s, e) in enumerate(chunks):
            window_len = e - s
            if window_len <= 0:
                continue

            # Load working-resolution tensors (expects start_idx support)
            input_video, input_mask, num_frames = self._load_inputs_for_chunk(
                video_path=sample_video_path,
                mask_path=mask_video_path,
                work_frames=min(self.cfg.max_chunk_len, window_len),
                work_hw=self.cfg.work_size_hw,
                start_idx=s,
                end_idx=e,
            )

            # Run the edit pipeline
            with torch.no_grad():
                result = self.pipeline(
                    prompt=prompt,
                    video=input_video,
                    mask_video=input_mask,
                    num_frames=num_frames,
                    num_inference_steps=self.cfg.num_inference_steps
                ).videos  # [B, C, F, H, W]

            # out_result_path = os.path.join(self.cfg.static_dir, f"{stem}_result.mp4")
            # save_videos_grid(result, out_result_path)
            #
            # # Ensure [B, C, F, H, W]
            # # Resize to the original resolution
            restored = self._resize_video_tensor(result, (orig_h, orig_w))  # [B,C,F,H,W]
            # out_restored_path = os.path.join(self.cfg.static_dir, f"{stem}_restored.mp4")
            # save_videos_grid(restored, out_restored_path)

            # Convert to numpy frames [F,H,W,C] uint8 for placement
            restored_np = (
                restored[0]
                .permute(1, 2, 3, 0)  # [C,F,H,W] -> [F,H,W,C]
                .cpu()
                .numpy()
            )

            # Place frames into the global buffer
            # Use min in case num_frames < (e - s) for any reason (e.g., short read)
            place_count = min(restored_np.shape[0], e - s)
            for k in range(place_count):
                frame_buffer[s + k] = restored_np[k]

        # Drop any None (e.g., if the last chunk was short); keep order
        final_frames = [f for f in frame_buffer if f is not None]
        if not final_frames:
            raise RuntimeError(f"No frames produced for {video_path}")

        # Stack to [F, H, W, C] → torch [1, C, F, H, W]
        final_np = np.stack(final_frames, axis=0)  # [F,H,W,C]
        final_tensor = torch.from_numpy(final_np) # [F,H,W,C]
        final_tensor = final_tensor.permute(3, 0, 1, 2).unsqueeze(0)  # [1,C,F,H,W], uint8

        # Save once
        out_path = os.path.join(self.cfg.static_dir, f"{stem}.mp4")
        save_videos_grid(final_tensor, out_path)

    def process_all(self, prompt: str = ""):
        # video_paths = sorted(glob.glob(os.path.join(self.cfg.videos_dir, "*.mp4")))
        # if not video_paths:
        #     raise FileNotFoundError(f"No videos found in {self.cfg.videos_dir}")

        video_paths = ["/data/rohith/ag/videos/00T1E.mp4"]

        for vp in video_paths:
            print(f"[StaticAgSceneExtractor] Processing: {vp}")
            self.process_single_video(vp, prompt=prompt)
        print("[StaticAgSceneExtractor] Done.")


def main():
    extractor = StaticAgSceneExtractor()
    extractor.process_all(prompt="")


# --------------------------
# CLI entry
# --------------------------
if __name__ == "__main__":
    main()
