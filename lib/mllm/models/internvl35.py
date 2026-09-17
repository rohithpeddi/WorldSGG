"""InternVL3.5 model wrapper – OpenGVLab/InternVL3_5-{8B,14B}.

vLLM-only implementation.  InternVL3.5 uses the same <image> token
prompt format as InternVL2.5 with multi-image input.
"""
from .base_model import BaseVideoModel
import numpy as np
import logging

logger = logging.getLogger(__name__)


class InternVL35Model(BaseVideoModel):

    MAX_NUM_FRAMES = 16  # keep visual tokens manageable

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)
        if not self.use_vllm:
            raise NotImplementedError(
                "InternVL3.5 direct (non-vLLM) inference is not implemented. "
                "Use --use_vllm (default)."
            )
        from vllm import LLM
        self.model = LLM(
            model=self.model_name,
            tokenizer_mode="slow",
            trust_remote_code=True,
            limit_mm_per_prompt={"image": self.MAX_NUM_FRAMES},
            tensor_parallel_size=self.args.tensor_parallel_size,
            **self._vllm_engine_kwargs(),
        )

    def _prepare_frames(self, video_inputs):
        """Extract and subsample PIL frames from video tensor."""
        current_video = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
        frames = self._tensor_to_pil_frames(current_video)
        if len(frames) > self.MAX_NUM_FRAMES:
            indices = np.linspace(0, len(frames) - 1, self.MAX_NUM_FRAMES, dtype=int)
            frames = [frames[i] for i in indices]
        # Limit resolution to prevent massive dynamic patching context explosion in vLLM
        frames = [f.resize((448, 448)) for f in frames]
        return frames

    def mllm_response(self, text, video_inputs, max_new_tokens=512,
                      size_list=None, fps=None):
        if video_inputs is None:
            return self._vllm_text_only(text, max_new_tokens)
        from vllm import SamplingParams, TextPrompt
        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        frames = self._prepare_frames(video_inputs)
        if not frames:
            return ""
        tokenizer = self.model.get_tokenizer()
        image_prefix = "".join(
            [f"Frame{i+1}: <image>\n" for i in range(len(frames))]
        )
        messages = [{"role": "user", "content": image_prefix + text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_input = TextPrompt(
            prompt=prompt_text, multi_modal_data={"image": frames}
        )
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()

        # Cache frames + image_prefix per unique tensor
        frame_cache = {}
        prompt_inputs, max_tokens_list = [], []
        skipped = set()
        for i, p in enumerate(prompts):
            text, video_inputs = p["text"], p.get("video_inputs")
            max_tokens_list.append(p.get("max_new_tokens", 512))
            if video_inputs is not None:
                vid_key = id(video_inputs[0] if isinstance(video_inputs, list) else video_inputs)
                if vid_key not in frame_cache:
                    frames = self._prepare_frames(video_inputs)
                    if frames:
                        prefix = "".join(f"Frame{j+1}: <image>\n" for j in range(len(frames)))
                        frame_cache[vid_key] = (frames, prefix)
                    else:
                        frame_cache[vid_key] = None
                cached = frame_cache[vid_key]
                if cached is None:
                    skipped.add(i)
                    continue
                frames, image_prefix = cached
                messages = [{"role": "user", "content": image_prefix + text}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"image": frames}))
            else:
                messages = [{"role": "user", "content": text}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt))
        if not prompt_inputs:
            return [""] * len(prompts)
        sp = SamplingParams(temperature=0.2, max_tokens=max(max_tokens_list))
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        gen_results = [o.outputs[0].text for o in outputs]
        if skipped:
            results, gen_idx = [], 0
            for i in range(len(prompts)):
                if i in skipped:
                    results.append("")
                else:
                    results.append(gen_results[gen_idx])
                    gen_idx += 1
            return results
        return gen_results

    def mllm_yes_no_batch(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()

        frame_cache = {}
        prompt_inputs = []
        skipped = set()
        for i, p in enumerate(prompts):
            text, video_inputs = p["text"], p.get("video_inputs")
            if video_inputs is not None:
                vid_key = id(video_inputs[0] if isinstance(video_inputs, list) else video_inputs)
                if vid_key not in frame_cache:
                    frames = self._prepare_frames(video_inputs)
                    if frames:
                        prefix = "".join(f"Frame{j+1}: <image>\n" for j in range(len(frames)))
                        frame_cache[vid_key] = (frames, prefix)
                    else:
                        frame_cache[vid_key] = None
                cached = frame_cache[vid_key]
                if cached is None:
                    skipped.add(i)
                    continue
                frames, image_prefix = cached
                messages = [{"role": "user", "content": image_prefix + text}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"image": frames}))
            else:
                messages = [{"role": "user", "content": text}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt))
        if not prompt_inputs:
            return [0.5] * len(prompts)
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        gen_results = [self._extract_yes_prob(o) for o in outputs]
        if skipped:
            results, gen_idx = [], 0
            for i in range(len(prompts)):
                if i in skipped:
                    results.append(0.5)
                else:
                    results.append(gen_results[gen_idx])
                    gen_idx += 1
            return results
        return gen_results
