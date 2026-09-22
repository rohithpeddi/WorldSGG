"""Tarsier2 model wrapper – omni-research/Tarsier2-7b-0115.

vLLM-only implementation.  Tarsier2 is built on Qwen2-VL and uses the
same video-as-tensor prompt format.
"""
from .base_model import BaseVideoModel
import logging

logger = logging.getLogger(__name__)


class Tarsier2Model(BaseVideoModel):

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)
        if not self.use_vllm:
            raise NotImplementedError(
                "Tarsier2 direct (non-vLLM) inference is not implemented. "
                "Use --use_vllm (default)."
            )
        from vllm import LLM
        self.model = LLM(
            model=self.model_name,
            trust_remote_code=True,
            limit_mm_per_prompt={"video": 1},
            tensor_parallel_size=self.args.tensor_parallel_size,
            hf_overrides={
                "architectures": ["Tarsier2ForConditionalGeneration"],
            },
            **self._vllm_engine_kwargs(),
        )

    def mllm_response(self, text, video_inputs, max_new_tokens=512,
                      size_list=None, fps=None):
        if video_inputs is None:
            return self._vllm_text_only(text, max_new_tokens)
        from vllm import SamplingParams, TextPrompt
        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        current_video = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
        tokenizer = self.model.get_tokenizer()
        messages = [{"role": "user", "content": [
            {"type": "video"},
            {"type": "text", "text": text},
        ]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_input = TextPrompt(
            prompt=prompt_text, multi_modal_data={"video": current_video}
        )
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()
        prompt_inputs, max_tokens_list = [], []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            max_tokens_list.append(p.get("max_new_tokens", 512))
            if video_inputs is not None:
                cv = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                messages = [{"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": text}
                ]}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"video": cv}))
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
        from vllm import SamplingParams, TextPrompt
        tokenizer = self.model.get_tokenizer()
        prompt_inputs = []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            if video_inputs is not None:
                cv = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                messages = [{"role": "user", "content": [
                    {"type": "video"}, {"type": "text", "text": text}
                ]}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt, multi_modal_data={"video": cv}))
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
                pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompt_inputs.append(TextPrompt(prompt=pt))
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [self._extract_yes_prob(o) for o in outputs]
