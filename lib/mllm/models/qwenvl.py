from backend.pseudo.models.base_model import BaseVideoModel
import torch
import logging

logger = logging.getLogger(__name__)

class QwenVLModel(BaseVideoModel):
    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)

        if self.use_vllm:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"video": 1},
                tensor_parallel_size=self.args.tensor_parallel_size,
            )
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            logger.info(f"Loading Qwen2.5-VL model directly from {self.model_name}")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self.model.eval()
            logger.info("Qwen2.5-VL model loaded directly (no vLLM)")

    def mllm_response(self, text, video_inputs, max_new_tokens=512, size_list=None, fps=None):
        if self.use_vllm:
            return self._vllm_response(text, video_inputs, max_new_tokens)
        else:
            return self._direct_response(text, video_inputs, max_new_tokens, fps)

    # ------------------------------------------------------------------ vLLM
    def _vllm_response(self, text, video_inputs, max_new_tokens):
        from vllm import SamplingParams, TextPrompt

        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)

        current_video = video_inputs
        if isinstance(video_inputs, list):
            current_video = video_inputs[0]

        tokenizer = self.model.get_tokenizer()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video"},
                    {"type": "text", "text": text},
                ],
            }
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_input = TextPrompt(
            prompt=prompt_text, multi_modal_data={"video": current_video}
        )
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # -------------------------------------------------------------- Direct
    def _direct_response(self, text, video_inputs, max_new_tokens, fps=None):
        from qwen_vl_utils import process_vision_info

        current_video = video_inputs
        if isinstance(video_inputs, list):
            current_video = video_inputs[0]

        # Build the message with video content
        # If the input is a tensor, we convert to a list of PIL images for the processor
        if isinstance(current_video, torch.Tensor):
            from torchvision import transforms
            to_pil = transforms.ToPILImage()
            frames = [to_pil(current_video[i]) for i in range(current_video.shape[0])]
            # Use multi-image approach for tensors
            content = [{"type": "image", "image": img} for img in frames]
            content.append({"type": "text", "text": text})
        elif current_video is None:
            # Text-only
            content = [{"type": "text", "text": text}]
        else:
            # Assume it's a video path or similar
            video_entry = {"type": "video", "video": current_video}
            if fps is not None:
                video_entry["fps"] = fps
            content = [video_entry, {"type": "text", "text": text}]

        messages = [{"role": "user", "content": content}]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs_proc = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt_text],
            images=image_inputs,
            videos=video_inputs_proc,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

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
                messages = [{"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": text},
                ]}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt_inputs.append(
                    TextPrompt(prompt=prompt_text, multi_modal_data={"video": current_video})
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