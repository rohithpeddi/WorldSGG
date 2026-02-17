"""InternVL2.5 model wrapper – OpenGVLab/InternVL2_5-8B.

Supports two inference backends:
  • vLLM  – fast batched inference via the vLLM engine.
  • Direct – native HuggingFace model.chat() API with a custom
            generate() that bypasses InternLM2's broken GenerationMixin
            on transformers >= 4.50.
"""

from backend.pseudo.models.base_model import BaseVideoModel
import torch
import numpy as np
import logging
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#                        Image / video preprocessing                          #
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num <= i * j <= max_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _frames_to_pixel_values(frames, input_size=448, max_num=1):
    """Convert list of PIL Images to (pixel_values, num_patches_list)."""
    transform = _build_transform(input_size)
    pixel_values_list = []
    num_patches_list = []
    for frame in frames:
        tiles = _dynamic_preprocess(frame, image_size=input_size,
                                    use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(tile) for tile in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# --------------------------------------------------------------------------- #
#                            InternVLModel class                              #
# --------------------------------------------------------------------------- #

class InternVLModel(BaseVideoModel):
    """Wrapper for OpenGVLab/InternVL2_5-8B.

    Supports two inference backends:
      • vLLM  – fast batched inference via the vLLM engine.
      • Direct – native HuggingFace ``model.chat()`` API.
    """

    MAX_NUM_FRAMES = 16  # keep visual tokens manageable (16 * 256 = 4096)

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)

        if self.use_vllm:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": self.MAX_NUM_FRAMES},
                tensor_parallel_size=self.args.tensor_parallel_size,
            )
        else:
            self._load_direct()

    # ------------------------------------------------------------------ #
    #  Direct (non-vLLM) loading with all compat patches                  #
    # ------------------------------------------------------------------ #
    def _load_direct(self):
        import sys
        import types
        from transformers import AutoModel, AutoTokenizer, AutoConfig

        logger.info(f"Loading InternVL2.5 model directly from {self.model_name}")

        # --- Patch 1: InternVLChatConfig.to_dict() crash on transformers >= 4.50 ---
        try:
            AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        except AttributeError as exc:
            if "llm_config" not in str(exc):
                raise
            logger.info("Patching InternVLChatConfig for transformers >= 4.50 compat")
            self._patch_config(sys.modules)
            AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, use_fast=False,
        )
        self.model = (
            AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            .half().cuda().to(torch.bfloat16)
        )
        self.model.eval()

        # --- Patch 2: Replace model.generate() to bypass broken
        #     language_model.generate() (GenerationMixin / prepare_inputs
        #     issues on transformers >= 4.50).
        #     Our replacement builds input embeddings the same way the
        #     original does, then runs a simple greedy loop with
        #     language_model.forward() – no GenerationMixin needed. ---
        self.model.generate = types.MethodType(
            self._greedy_generate, self.model
        )
        logger.info("InternVL2.5 model loaded directly (no vLLM)")

    # ------------------------------------------------------------------ #
    #  Config monkey-patch (identical issue to InternVideo)                #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _patch_config(modules):
        from transformers import PretrainedConfig
        for mod_name, mod in modules.items():
            if "configuration_internvl_chat" not in mod_name:
                continue
            Cls = getattr(mod, "InternVLChatConfig", None)
            if Cls is None:
                continue

            _orig_init = Cls.__init__

            def _patched_init(self_cfg, *args, _orig=_orig_init, **kwargs):
                self_cfg.llm_config = None
                self_cfg.vision_config = None
                _orig(self_cfg, *args, **kwargs)
            Cls.__init__ = _patched_init

            def _safe_to_dict(self_cfg):
                output = PretrainedConfig.to_dict(self_cfg)
                for key in ("llm_config", "vision_config"):
                    val = getattr(self_cfg, key, None)
                    output[key] = val.to_dict() if val is not None else {}
                output["model_type"] = self_cfg.model_type
                return output
            Cls.to_dict = _safe_to_dict
            break

    # ------------------------------------------------------------------ #
    #  Custom greedy generate – bypasses GenerationMixin entirely         #
    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.no_grad()
    def _greedy_generate(
        self_model,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        visual_features=None,
        generation_config=None,
        output_hidden_states=None,
        **generate_kwargs,
    ):
        """Drop-in replacement for InternVLChatModel.generate().

        Builds input embeddings exactly like the original, then runs a
        simple greedy token-by-token loop using language_model.forward()
        instead of language_model.generate().  This completely avoids
        GenerationMixin / prepare_inputs_for_generation compat issues.
        """
        assert self_model.img_context_token_id is not None

        # --- Build input embeddings (mirrors original generate) --- #
        if pixel_values is not None:
            vit_embeds = (visual_features if visual_features is not None
                          else self_model.extract_feature(pixel_values))
            input_embeds = self_model.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            flat_embeds = input_embeds.reshape(B * N, C)
            flat_ids = input_ids.reshape(B * N)
            selected = (flat_ids == self_model.img_context_token_id)
            assert selected.sum() != 0
            flat_embeds[selected] = vit_embeds.reshape(-1, C).to(flat_embeds.device)
            input_embeds = flat_embeds.reshape(B, N, C)
        else:
            input_embeds = self_model.language_model.get_input_embeddings()(input_ids)

        # --- Extract generation parameters --- #
        max_new_tokens = generate_kwargs.pop("max_new_tokens", 512)
        eos_token_id = generate_kwargs.pop("eos_token_id", None)
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        elif eos_token_id is None:
            eos_token_id = []

        B = input_embeds.shape[0]
        device = input_embeds.device
        generated = torch.zeros(B, 0, dtype=torch.long, device=device)

        # --- Greedy loop using forward() only --- #
        past_kv = None
        current_embeds = input_embeds
        current_mask = attention_mask

        for _ in range(max_new_tokens):
            outputs = self_model.language_model(
                inputs_embeds=current_embeds,
                attention_mask=current_mask,
                past_key_values=past_kv,
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :]
            next_ids = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)
            generated = torch.cat([generated, next_ids], dim=-1)

            # Check EOS for batch-size-1 (our typical case)
            if B == 1 and next_ids.item() in eos_token_id:
                break

            # Next step: one-token embedding, extend attention mask
            past_kv = outputs.past_key_values
            current_embeds = self_model.language_model.get_input_embeddings()(next_ids)
            if current_mask is not None:
                current_mask = torch.cat([
                    current_mask,
                    torch.ones(B, 1, device=device, dtype=current_mask.dtype),
                ], dim=-1)

        return generated

    # ------------------------------------------------------------------ #
    #  Main dispatch                                                      #
    # ------------------------------------------------------------------ #
    def mllm_response(self, text, video_inputs, max_new_tokens=512,
                      size_list=None, fps=None):
        if self.use_vllm:
            return self._vllm_response(text, video_inputs, max_new_tokens)
        else:
            return self._direct_response(text, video_inputs, max_new_tokens)

    # ------------------------------------------------------------------ vLLM
    def _vllm_response(self, text, video_inputs, max_new_tokens):
        from vllm import SamplingParams, TextPrompt

        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)

        current_video = video_inputs
        if isinstance(video_inputs, list):
            current_video = video_inputs[0]

        # Convert tensor → PIL frames
        frames = self._tensor_to_pil_frames(current_video)
        if not frames:
            logger.warning("No frames extracted for InternVL2.5 vLLM")
            return ""

        if len(frames) > self.MAX_NUM_FRAMES:
            indices = np.linspace(0, len(frames) - 1, self.MAX_NUM_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]

        tokenizer = self.model.get_tokenizer()

        # Build prompt with one <image> per frame
        content = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompt_input = TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"image": frames},
        )
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # -------------------------------------------------------------- Direct
    def _direct_response(self, text, video_inputs, max_new_tokens):
        """Direct inference using the official InternVL2.5 model.chat() API."""
        current_video = video_inputs
        if isinstance(video_inputs, list):
            current_video = video_inputs[0]

        if current_video is None:
            generation_config = dict(
                do_sample=False, temperature=0.0,
                max_new_tokens=max_new_tokens, top_p=0.1, num_beams=1,
            )
            output, _ = self.model.chat(
                self.tokenizer, None, text, generation_config,
                history=None, return_history=True,
            )
            return output

        # --- Convert tensor → PIL frames --- #
        frames = self._tensor_to_pil_frames(current_video)
        if not frames:
            logger.warning("Could not convert video input to PIL frames")
            return ""

        # --- Subsample --- #
        if len(frames) > self.MAX_NUM_FRAMES:
            indices = np.linspace(0, len(frames) - 1, self.MAX_NUM_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]

        # --- Preprocess (official pipeline) --- #
        pixel_values, num_patches_list = _frames_to_pixel_values(
            frames, input_size=448, max_num=1
        )
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)

        # --- Build frame-prefix prompt --- #
        video_prefix = "".join(
            [f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))]
        )
        question = video_prefix + text

        generation_config = dict(
            do_sample=False, temperature=0.0,
            max_new_tokens=max_new_tokens, top_p=0.1, num_beams=1,
        )

        output, _ = self.model.chat(
            self.tokenizer, pixel_values, question, generation_config,
            num_patches_list=num_patches_list,
            history=None, return_history=True,
        )
        return output

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _tensor_to_pil_frames(video):
        """Convert a video tensor (T, C, H, W) or numpy array to list[PIL.Image]."""
        if isinstance(video, torch.Tensor):
            arr = video.cpu()
            if arr.dtype in (torch.float32, torch.float16, torch.bfloat16):
                if arr.max() <= 1.0:
                    arr = (arr * 255).clamp(0, 255)
                arr = arr.to(torch.uint8)
            arr = arr.permute(0, 2, 3, 1).numpy()
            return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        elif isinstance(video, np.ndarray):
            if video.ndim == 4 and video.shape[1] in (1, 3):
                video = np.transpose(video, (0, 2, 3, 1))
            return [Image.fromarray(video[i].astype(np.uint8))
                    for i in range(video.shape[0])]
        else:
            return []

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        if not self.use_vllm:
            return super().mllm_batch_response(prompts)

        from vllm import SamplingParams, TextPrompt

        tokenizer = self.model.get_tokenizer()
        prompt_inputs = []
        max_tokens_list = []

        for p in prompts:
            text = p["text"]
            video_inputs = p.get("video_inputs", None)
            max_new_tokens = p.get("max_new_tokens", 512)
            max_tokens_list.append(max_new_tokens)

            if video_inputs is not None:
                current_video = video_inputs
                if isinstance(video_inputs, list):
                    current_video = video_inputs[0]
                frames = self._tensor_to_pil_frames(current_video)
                if not frames:
                    # Fallback: empty prompt that will produce empty output
                    prompt_inputs.append(TextPrompt(prompt=""))
                    continue
                if len(frames) > self.MAX_NUM_FRAMES:
                    indices = np.linspace(0, len(frames) - 1, self.MAX_NUM_FRAMES, dtype=int)
                    frames = [frames[i] for i in indices]
                content = [{"type": "image"} for _ in frames]
                content.append({"type": "text", "text": text})
                messages = [{"role": "user", "content": content}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_inputs.append(
                    TextPrompt(prompt=prompt_text, multi_modal_data={"image": frames})
                )
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_inputs.append(TextPrompt(prompt=prompt_text))

        sampling_params = SamplingParams(
            temperature=0.2, max_tokens=max(max_tokens_list)
        )
        outputs = self.model.generate(prompt_inputs, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]
