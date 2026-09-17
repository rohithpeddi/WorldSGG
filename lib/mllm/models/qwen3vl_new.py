"""Qwen3-VL model wrapper – Qwen/Qwen3-VL-{8B,30B-A3B}-Instruct.

vLLM-only implementation.  Uses ``qwen_vl_utils.process_vision_info``
to prepare video data in the exact format the Qwen3VLProcessor expects.
"""
from .base_model import BaseVideoModel
import torch
import logging

logger = logging.getLogger(__name__)


class Qwen3VLModel(BaseVideoModel):

    def load_model(self):
        self.use_vllm = getattr(self.args, 'use_vllm', True)
        if not self.use_vllm:
            raise NotImplementedError(
                "Qwen3-VL direct (non-vLLM) inference is not implemented. "
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

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _tensor_to_pil_list(video_input):
        """Convert a (T, C, H, W) tensor to a list of PIL Images."""
        from PIL import Image
        tensor = video_input[0] if isinstance(video_input, list) else video_input
        if not isinstance(tensor, torch.Tensor):
            return tensor, False
        frames = []
        for i in range(tensor.shape[0]):
            frame_np = tensor[i].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            frames.append(Image.fromarray(frame_np))
        return frames, True

    @staticmethod
    def _pil_frames_to_video_payload(pil_frames):
        """Convert a list of PIL frames into (video_np, metadata) for vLLM.

        Performs the same work as ``process_vision_info`` but avoids re-
        decoding PIL images through the Qwen utility, which is expensive.
        """
        import numpy as np
        # Stack PIL frames → (T, H, W, C) uint8 numpy
        video_np = np.stack(
            [np.asarray(f.convert("RGB")) for f in pil_frames], axis=0
        )
        T = video_np.shape[0]
        metadata = {
            "total_num_frames": T,
            "fps": 1.0,
            "frames_indices": list(range(T)),
        }
        return video_np, metadata

    def _prepare_video_prompt(self, text, video_inputs, tokenizer=None,
                              video_payload=None):
        """Build a vLLM TextPrompt.

        Parameters
        ----------
        tokenizer : optional
            Pre-fetched tokenizer to avoid repeated get_tokenizer() calls.
        video_payload : optional
            Pre-computed ``(video_np, metadata)`` tuple.  When supplied the
            expensive ``_tensor_to_pil_list`` / ``process_vision_info``
            pipeline is skipped entirely.
        """
        from vllm import TextPrompt

        if tokenizer is None:
            tokenizer = self.model.get_tokenizer()

        if video_inputs is None and video_payload is None:
            messages = [{"role": "user", "content": [
                {"type": "text", "text": text},
            ]}]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return TextPrompt(prompt=prompt_text)

        if video_payload is not None:
            # Re-use the pre-computed payload
            video_np, metadata = video_payload
        else:
            pil_frames, was_tensor = self._tensor_to_pil_list(video_inputs)
            if was_tensor:
                video_np, metadata = self._pil_frames_to_video_payload(pil_frames)
            else:
                # Non-tensor input (e.g. file path) — fall through
                cv = pil_frames
                messages = [{"role": "user", "content": [
                    {"type": "video"},
                    {"type": "text", "text": text},
                ]}]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                return TextPrompt(
                    prompt=prompt_text,
                    multi_modal_data={"video": cv},
                )

        # Build prompt with video placeholder
        messages = [{"role": "user", "content": [
            {"type": "video", "video": [None] * video_np.shape[0], "fps": 1.0},
            {"type": "text", "text": text},
        ]}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return TextPrompt(
            prompt=prompt_text,
            multi_modal_data={"video": (video_np, metadata)},
        )

    # ------------------------------------------------------------- Single
    def mllm_response(self, text, video_inputs, max_new_tokens=512,
                      size_list=None, fps=None):
        if video_inputs is None:
            return self._vllm_text_only(text, max_new_tokens)
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        prompt_input = self._prepare_video_prompt(text, video_inputs)
        outputs = self.model.generate(prompt_input, sampling_params=sampling_params)
        return outputs[0].outputs[0].text

    # --------------------------------------------------------- Batch vLLM
    def mllm_batch_response(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams
        tokenizer = self.model.get_tokenizer()

        # Pre-compute video payload once per unique tensor
        payload_cache = {}
        prompt_inputs, max_tokens_list = [], []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            max_tokens_list.append(p.get("max_new_tokens", 512))
            if video_inputs is not None:
                raw = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                vid_key = id(raw)
                if vid_key not in payload_cache:
                    pil_frames, was_tensor = self._tensor_to_pil_list(video_inputs)
                    if was_tensor:
                        payload_cache[vid_key] = self._pil_frames_to_video_payload(pil_frames)
                    else:
                        payload_cache[vid_key] = None  # non-tensor, fallback
                cached = payload_cache[vid_key]
                if cached is not None:
                    prompt_inputs.append(
                        self._prepare_video_prompt(
                            text, None, tokenizer=tokenizer, video_payload=cached,
                        )
                    )
                else:
                    prompt_inputs.append(
                        self._prepare_video_prompt(
                            text, video_inputs, tokenizer=tokenizer,
                        )
                    )
            else:
                prompt_inputs.append(
                    self._prepare_video_prompt(
                        text, None, tokenizer=tokenizer,
                    )
                )
        sp = SamplingParams(temperature=getattr(self.args, 'temperature', 0.2),
                            top_p=getattr(self.args, 'top_p', 1.0),
                            seed=getattr(self.args, 'seed', None),
                            max_tokens=max(max_tokens_list))
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [o.outputs[0].text for o in outputs]

    def mllm_yes_no_batch(self, prompts):
        if not prompts:
            return []
        from vllm import SamplingParams
        tokenizer = self.model.get_tokenizer()

        payload_cache = {}
        prompt_inputs = []
        for p in prompts:
            text, video_inputs = p["text"], p.get("video_inputs")
            if video_inputs is not None:
                raw = video_inputs[0] if isinstance(video_inputs, list) else video_inputs
                vid_key = id(raw)
                if vid_key not in payload_cache:
                    pil_frames, was_tensor = self._tensor_to_pil_list(video_inputs)
                    if was_tensor:
                        payload_cache[vid_key] = self._pil_frames_to_video_payload(pil_frames)
                    else:
                        payload_cache[vid_key] = None
                cached = payload_cache[vid_key]
                if cached is not None:
                    prompt_inputs.append(
                        self._prepare_video_prompt(
                            text, None, tokenizer=tokenizer, video_payload=cached,
                        )
                    )
                else:
                    prompt_inputs.append(
                        self._prepare_video_prompt(
                            text, video_inputs, tokenizer=tokenizer,
                        )
                    )
            else:
                prompt_inputs.append(
                    self._prepare_video_prompt(
                        text, None, tokenizer=tokenizer,
                    )
                )
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outputs = self._vllm_batch_generate(prompt_inputs, sp)
        return [self._extract_yes_prob(o) for o in outputs]
