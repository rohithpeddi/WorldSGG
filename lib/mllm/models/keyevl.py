"""Keye-VL 1.5 model wrapper – Kwai-Keye/Keye-VL-1_5-8B.

vLLM-only implementation.  Keye-VL 1.5 uses a Qwen3-8B backbone with
SigLIP vision encoder and Qwen-style chat template.

KeyeVL requires ``keye_vl_utils.process_vision_info`` to pre-process
video inputs.  Raw tensors/numpy arrays are converted to PIL frames
before being passed through the standard processing pipeline.
"""
from .base_model import BaseVideoModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KeyeVLModel(BaseVideoModel):

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)
        if not self.use_vllm:
            raise NotImplementedError(
                "Keye-VL direct (non-vLLM) inference is not implemented. "
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _prepare_video(self, video_input):
        """Convert a raw tensor/ndarray into the format KeyeVL expects.

        Returns ``(video_data, mm_processor_kwargs)`` where *video_data*
        is a list of PIL frames and *mm_processor_kwargs* carries FPS
        metadata required by the KeyeVL1_5Processor.
        """
        # Use base class tensor→PIL conversion (handles all dtype cases)
        pil_frames = self._tensor_to_pil_frames(video_input)
        # KeyeVL's processor expects mm_processor_kwargs with fps info.
        # When frames are pre-sampled (our case), use fps=1.0 as a
        # safe default — the frame count is already controlled upstream.
        mm_processor_kwargs = {"fps": [1.0]}
        return pil_frames, mm_processor_kwargs

    @staticmethod
    def _build_prompt(tokenizer, text, has_video=True):
        """Build the tokenized prompt string."""
        if has_video:
            messages = [{"role": "user", "content": [
                {"type": "video"}, {"type": "text", "text": text}
            ]}]
        else:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": text}
            ]}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    # ------------------------------------------------------------------
    # Single inference
    # ------------------------------------------------------------------

    def mllm_response(self, text, video_inputs, max_new_tokens=512,
                      size_list=None, fps=None):
        if video_inputs is None:
            return self._vllm_text_only(text, max_new_tokens)

        from vllm import SamplingParams

        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        current_video = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
        tokenizer = self.model.get_tokenizer()
        prompt_text = self._build_prompt(tokenizer, text, has_video=True)

        pil_frames, mm_kwargs = self._prepare_video(current_video)
        llm_input = {
            "prompt": prompt_text,
            "multi_modal_data": {"video": pil_frames},
            "mm_processor_kwargs": mm_kwargs,
        }
        outputs = self.model.generate(llm_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # ------------------------------------------------------------------
    # Batch vLLM
    # ------------------------------------------------------------------

    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams

        tokenizer = self.model.get_tokenizer()
        video_cache = {}
        prompt_inputs, max_tokens_list = [], []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            max_tokens_list.append(p.get("max_new_tokens", 512))
            if video_inputs is not None:
                cv = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                vid_key = id(cv)
                if vid_key not in video_cache:
                    video_cache[vid_key] = self._prepare_video(cv)
                pil_frames, mm_kwargs = video_cache[vid_key]
                prompt_text = self._build_prompt(tokenizer, text, has_video=True)
                prompt_inputs.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"video": pil_frames},
                    "mm_processor_kwargs": mm_kwargs,
                })
            else:
                prompt_text = self._build_prompt(tokenizer, text, has_video=False)
                prompt_inputs.append({"prompt": prompt_text})

        sp = SamplingParams(temperature=0.2, max_tokens=max(max_tokens_list))
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [o.outputs[0].text for o in outputs]

    def mllm_yes_no_batch(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams

        tokenizer = self.model.get_tokenizer()
        video_cache = {}
        prompt_inputs = []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            if video_inputs is not None:
                cv = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                vid_key = id(cv)
                if vid_key not in video_cache:
                    video_cache[vid_key] = self._prepare_video(cv)
                pil_frames, mm_kwargs = video_cache[vid_key]
                prompt_text = self._build_prompt(tokenizer, text, has_video=True)
                prompt_inputs.append({
                    "prompt": prompt_text,
                    "multi_modal_data": {"video": pil_frames},
                    "mm_processor_kwargs": mm_kwargs,
                })
            else:
                prompt_text = self._build_prompt(tokenizer, text, has_video=False)
                prompt_inputs.append({"prompt": prompt_text})

        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [self._extract_yes_prob(o) for o in outputs]
