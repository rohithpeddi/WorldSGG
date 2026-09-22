from .base_model import BaseVideoModel
import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

class LongVUModel(BaseVideoModel):
    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)

        if self.use_vllm:
            from vllm import LLM
            self.model = LLM(
                model=self.model_name,
                trust_remote_code=True,
                limit_mm_per_prompt={"video": 1},
                tensor_parallel_size=self.args.tensor_parallel_size,
                **self._vllm_engine_kwargs(),
            )
        else:
            # LongVU uses a custom 'cambrian_qwen' architecture that is NOT in
            # standard HuggingFace transformers.  It must be loaded via the
            # longvu package (pip install from the Vision-CAIR/LongVU repo).
            try:
                from longvu.builder import load_pretrained_model
                from longvu.mm_datautils import process_images
            except ImportError:
                raise ImportError(
                    "Direct (non-vLLM) loading of LongVU requires the 'longvu' "
                    "package. Install it by cloning https://github.com/Vision-CAIR/LongVU "
                    "and running 'pip install -e .' inside the repo, or use "
                    "--use_vllm instead."
                )

            logger.info(f"Loading LongVU model directly from {self.model_name}")
            self.tokenizer, self.model, self.image_processor, self._context_len = (
                load_pretrained_model(self.model_name, None, "cambrian_qwen")
            )
            self.model.eval()
            # Store process_images on instance for use in _direct_response
            self._process_images = process_images
            logger.info("LongVU model loaded directly (no vLLM)")

    def mllm_response(self, text, video_inputs, max_new_tokens=512, size_list=None, fps=None):
        if video_inputs is None:
            if self.use_vllm:
                return self._vllm_text_only(text, max_new_tokens)
            logger.warning("LongVU direct mode requires video input; returning empty.")
            return ""
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
    def _direct_response(self, text, video_inputs, max_new_tokens):
        """Direct inference using the official LongVU pipeline."""
        from longvu.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from longvu.conversation import conv_templates, SeparatorStyle
        from longvu.mm_datautils import KeywordsStoppingCriteria, tokenizer_image_token

        current_video = video_inputs
        if isinstance(video_inputs, list):
            current_video = video_inputs[0]

        # --- Convert input to numpy (H, W, C) array stack ---
        if isinstance(current_video, torch.Tensor):
            # Expect (T, C, H, W) float tensor -> (T, H, W, C) uint8 numpy
            video_np = current_video.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        elif isinstance(current_video, np.ndarray):
            video_np = current_video
        else:
            logger.warning("Unsupported video input type for direct LongVU inference")
            return ""

        image_sizes = [video_np[0].shape[:2]]

        # Process images using LongVU's custom processor
        video_tensor = self._process_images(video_np, self.image_processor, self.model.config)
        video_tensor = [item.unsqueeze(0) for item in video_tensor]

        # Build prompt using LongVU conversation template
        qs = DEFAULT_IMAGE_TOKEN + "\n" + text
        conv = conv_templates["qwen"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(self.model.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0.2,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        pred = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        return pred