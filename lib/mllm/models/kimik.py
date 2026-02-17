from backend.pseudo.models.base_model import BaseVideoModel
import torch
from torchvision import transforms
import logging

logger = logging.getLogger(__name__)

class KimikModel(BaseVideoModel):

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)

        if self.use_vllm:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 256},
                tensor_parallel_size=self.args.tensor_parallel_size,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoProcessor
            logger.info(f"Loading KimiVL model directly from {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()
            logger.info("KimiVL model loaded directly (no vLLM)")

    def _frames_from_input(self, video_inputs):
        """Convert video_inputs to a list of PIL images."""
        current_video = video_inputs
        if isinstance(video_inputs, list):
            current_video = video_inputs[0]

        frames = []
        if isinstance(current_video, torch.Tensor):
            to_pil = transforms.ToPILImage()
            for i in range(current_video.shape[0]):
                frames.append(to_pil(current_video[i]))
        elif isinstance(current_video, list):
            frames = current_video
        else:
            frames = [current_video]
        return frames

    def mllm_response(self, text, video_inputs, max_new_tokens=512, size_list=None, fps=None):
        if video_inputs is None:
            # Text-only query
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

        content = [{"type": "image"} for _ in range(len(frames))]
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_input = TextPrompt(
            prompt=prompt_text, multi_modal_data={"image": frames}
        )
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    def _vllm_text_only(self, text, max_new_tokens):
        from vllm import SamplingParams, TextPrompt

        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        tokenizer = self.model.get_tokenizer()

        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_input = TextPrompt(prompt=prompt_text)
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # -------------------------------------------------------------- Direct
    def _direct_response(self, text, frames, max_new_tokens):
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": text})

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(
            prompt_text, images=frames, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # Slice off the prompt tokens
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    def _direct_text_only(self, text, max_new_tokens):
        messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )
        inputs = self.processor(prompt_text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        """
        Batched inference via vLLM.  Falls back to sequential calls when
        not using vLLM or when the batch is empty.
        """
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
                frames = self._frames_from_input(video_inputs)
                content = [{"type": "image"} for _ in range(len(frames))]
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

        # Use the max of all requested token limits for the batch
        sampling_params = SamplingParams(
            temperature=0.2, max_tokens=max(max_tokens_list)
        )
        outputs = self.model.generate(prompt_inputs, sampling_params=sampling_params)
        return [o.outputs[0].text for o in outputs]