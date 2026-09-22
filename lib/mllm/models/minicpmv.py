"""MiniCPM-V/O model wrapper.

Serves:
  • openbmb/MiniCPM-V-4_5
  • openbmb/MiniCPM-o-2_6

vLLM-only implementation.

NOTE: MiniCPM-V's vLLM integration expects video data as a **list of
PIL Images** (one per frame), NOT a raw (T, C, H, W) tensor.  Passing a
tensor causes the internal multimodal processor to misinterpret the
input type, leading to ``'list object' has no attribute 'startswith'``.
All entry-points therefore convert tensors via ``_prepare_video``.
"""
from .base_model import BaseVideoModel
import logging

logger = logging.getLogger(__name__)


class MiniCPMVModel(BaseVideoModel):

    FRAME_SIZE = 448  # limit resolution to prevent context length explosion

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)
        if not self.use_vllm:
            raise NotImplementedError(
                "MiniCPM-V direct (non-vLLM) inference is not implemented. "
                "Use --use_vllm (default)."
            )
        from vllm import LLM
        self.model = LLM(
            model=self.model_name,
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1},
            tensor_parallel_size=self.args.tensor_parallel_size,
            **self._vllm_engine_kwargs(),
        )

    # ------------------------------------------------------------- helpers
    def _prepare_video(self, video_inputs):
        """Convert video tensor → list[PIL.Image] for MiniCPM-V's vLLM
        multimodal processor.

        Accepts:
          • list wrapping a tensor: [tensor(T,C,H,W)]
          • a bare tensor: tensor(T,C,H,W)
          • already a list[PIL.Image]: returned as-is
        """
        import torch
        from PIL import Image as PILImage

        if video_inputs is None:
            return None

        current = video_inputs[0] if isinstance(video_inputs, list) else video_inputs

        # Already PIL images — pass through
        if isinstance(current, list) and len(current) > 0 and isinstance(current[0], PILImage.Image):
            return current

        sz = self.FRAME_SIZE

        # Tensor → list of PIL Images
        if isinstance(current, torch.Tensor):
            frames = self._tensor_to_pil_frames(current)
            return [f.resize((sz, sz)) for f in frames]

        # numpy array (T, C, H, W) or (T, H, W, C)
        import numpy as np
        if isinstance(current, np.ndarray):
            frames = self._tensor_to_pil_frames(current)
            return [f.resize((sz, sz)) for f in frames]

        # Unknown — return as-is and let vLLM deal with it
        logger.warning(
            f"_prepare_video: unexpected type {type(current)}, passing through"
        )
        return current

    # --------------------------------------------------------- helpers
    _VIDEO_PLACEHOLDER = "(<video>./</video>)"

    def _build_prompt(self, text, has_video=True, tokenizer=None):
        """Build prompt string with MiniCPM-V's native placeholder.

        MiniCPM-V's chat template expects ``content`` to be a plain
        **string** — passing a list of dicts causes
        ``'list object' has no attribute 'startswith'``.

        Accepts an optional *tokenizer* to avoid repeated
        ``self.model.get_tokenizer()`` lookups in batch methods.
        """
        if tokenizer is None:
            tokenizer = self.model.get_tokenizer()
        if has_video:
            content = f"{self._VIDEO_PLACEHOLDER}\n{text}"
        else:
            content = text
        messages = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # --------------------------------------------------------- Single vLLM
    def mllm_response(self, text, video_inputs, max_new_tokens=512,
                      size_list=None, fps=None):
        if video_inputs is None:
            return self._vllm_text_only(text, max_new_tokens)
        from vllm import SamplingParams, TextPrompt
        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        frames = self._prepare_video(video_inputs)
        prompt_text = self._build_prompt(text, has_video=True)
        prompt_input = TextPrompt(
            prompt=prompt_text, multi_modal_data={"video": [frames]}
        )
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()

        # Cache video conversion per unique tensor identity
        video_cache = {}
        prompt_inputs, max_tokens_list = [], []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            max_tokens_list.append(p.get("max_new_tokens", 512))
            if video_inputs is not None:
                vid_key = id(video_inputs[0] if isinstance(video_inputs, list) else video_inputs)
                if vid_key not in video_cache:
                    video_cache[vid_key] = self._prepare_video(video_inputs)
                frames = video_cache[vid_key]
                pt = self._build_prompt(text, has_video=True, tokenizer=tokenizer)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"video": [frames]}))
            else:
                pt = self._build_prompt(text, has_video=False, tokenizer=tokenizer)
                prompt_inputs.append(TextPrompt(prompt=pt))
        sp = SamplingParams(temperature=0.2, max_tokens=max(max_tokens_list))
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [o.outputs[0].text for o in outputs]

    def mllm_yes_no_batch(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()

        video_cache = {}
        prompt_inputs = []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            if video_inputs is not None:
                vid_key = id(video_inputs[0] if isinstance(video_inputs, list) else video_inputs)
                if vid_key not in video_cache:
                    video_cache[vid_key] = self._prepare_video(video_inputs)
                frames = video_cache[vid_key]
                pt = self._build_prompt(text, has_video=True, tokenizer=tokenizer)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"video": [frames]}))
            else:
                pt = self._build_prompt(text, has_video=False, tokenizer=tokenizer)
                prompt_inputs.append(TextPrompt(prompt=pt))
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [self._extract_yes_prob(o) for o in outputs]
