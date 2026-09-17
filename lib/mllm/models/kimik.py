"""KimiVL model wrapper – moonshotai/Kimi-VL-A3B-Instruct.

Supports vLLM and direct HuggingFace backends.
"""
from .base_model import BaseVideoModel
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


class KimikModel(BaseVideoModel):

    MAX_NUM_FRAMES = 128  # stay well within limit_mm_per_prompt=256
    FRAME_SIZE = 448

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)
        if self.use_vllm:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name, trust_remote_code=True,
                limit_mm_per_prompt={"image": 256},
                tensor_parallel_size=self.args.tensor_parallel_size,
                **self._vllm_engine_kwargs(),
            )
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor
            logger.info(f"Loading KimiVL directly from {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, trust_remote_code=True,
                torch_dtype=torch.bfloat16, device_map="auto",
            )
            self.model.eval()
            logger.info("KimiVL model loaded directly (no vLLM)")

    def _frames_from_input(self, video_inputs):
        """Convert video_inputs to a list of PIL images, subsampled
        to at most MAX_NUM_FRAMES."""
        current_video = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
        # Use base class helper — handles all tensor/ndarray dtypes
        if isinstance(current_video, torch.Tensor) or (hasattr(current_video, 'shape') and hasattr(current_video, 'dtype')):
            frames = self._tensor_to_pil_frames(current_video)
        elif isinstance(current_video, list):
            frames = current_video
        else:
            frames = [current_video]
        # Subsample if needed to stay within vLLM image limit
        if len(frames) > self.MAX_NUM_FRAMES:
            indices = np.linspace(0, len(frames) - 1, self.MAX_NUM_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        sz = self.FRAME_SIZE
        frames = [f.resize((sz, sz)) for f in frames]
        return frames

    def mllm_response(self, text, video_inputs, max_new_tokens=512, size_list=None, fps=None):
        if video_inputs is None:
            if self.use_vllm:
                return self._vllm_text_only(text, max_new_tokens)
            else:
                return self._direct_text_only(text, max_new_tokens)
        frames = self._frames_from_input(video_inputs)
        if self.use_vllm:
            return self._vllm_response(text, frames, max_new_tokens)
        else:
            return self._direct_response(text, frames, max_new_tokens)

    # ------------------------------------------------------------------ vLLM
    def _vllm_response(self, text, frames, max_new_tokens):
        from vllm import SamplingParams, TextPrompt
        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        tokenizer = self.model.get_tokenizer()
        content = [{"type": "image"} for _ in frames]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_input = TextPrompt(prompt=prompt_text, multi_modal_data={"image": frames})
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def _vllm_text_only(self, text, max_new_tokens):
        from vllm import SamplingParams, TextPrompt
        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        tokenizer = self.model.get_tokenizer()
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.model.generate(TextPrompt(prompt=prompt_text), sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # -------------------------------------------------------------- Direct
    def _direct_response(self, text, frames, max_new_tokens):
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(prompt_text, images=frames, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _direct_text_only(self, text, max_new_tokens):
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(prompt_text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        if not self.use_vllm:
            return super().mllm_batch_response(prompts)
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()

        frame_cache = {}
        prompt_inputs, max_tokens_list = [], []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            max_tokens_list.append(p.get("max_new_tokens", 512))
            if video_inputs is not None:
                vid_key = id(video_inputs[0] if isinstance(video_inputs, list) else video_inputs)
                if vid_key not in frame_cache:
                    frame_cache[vid_key] = self._frames_from_input(video_inputs)
                frames = frame_cache[vid_key]
                content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": text}]
                messages = [{"role": "user", "content": content}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"image": frames}))
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt))
        sp = SamplingParams(temperature=0.2, max_tokens=max(max_tokens_list))
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [o.outputs[0].text for o in outputs]

    def mllm_yes_no_batch(self, prompts):
        if not prompts:
            return []
        if not self.use_vllm:
            return super().mllm_yes_no_batch(prompts)
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()

        frame_cache = {}
        prompt_inputs = []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            if video_inputs is not None:
                vid_key = id(video_inputs[0] if isinstance(video_inputs, list) else video_inputs)
                if vid_key not in frame_cache:
                    frame_cache[vid_key] = self._frames_from_input(video_inputs)
                frames = frame_cache[vid_key]
                content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": text}]
                messages = [{"role": "user", "content": content}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"image": frames}))
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt))
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [self._extract_yes_prob(o) for o in outputs]