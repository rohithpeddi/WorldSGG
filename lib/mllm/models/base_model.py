from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import math
import numpy as np
from .utils import fetch_video, resize_video
import logging

logger = logging.getLogger(__name__)

class BaseVideoModel(ABC):
    def __init__(self, model_name, args=None):
        self.model_name = model_name
        self.args = args
        self.tokenizer = None
        self.model = None
        self.processor = None
        self.image_processor = None
        self.load_model()

    def _vllm_engine_kwargs(self) -> dict:
        """Build common ``vllm.LLM`` constructor kwargs from ``self.args``.

        Model wrappers should ``**self._vllm_engine_kwargs()`` into their
        ``LLM(...)`` call to inherit all config-driven engine settings
        (gpu_memory_utilization, max_model_len, dtype, etc.).

        Only includes keys whose values are set on the args namespace;
        missing attributes are silently skipped so older callers still work.
        """
        mapping = {
            "gpu_memory_utilization": "gpu_memory_utilization",
            "max_model_len": "max_model_len",
            "max_num_seqs": "max_num_seqs",
            "enable_chunked_prefill": "enable_chunked_prefill",
            "dtype": "dtype",
            # NOTE: swap_space is deprecated and ignored in vLLM ≥ 0.8.
            # See vllm/entrypoints/llm.py — it is explicitly popped from
            # kwargs with a DeprecationWarning.  Do NOT pass it.
        }
        kwargs: dict = {}
        for llm_key, attr_name in mapping.items():
            val = getattr(self.args, attr_name, None)
            if val is not None:
                kwargs[llm_key] = val
        return kwargs

    @abstractmethod
    def load_model(self):
        """
        Load the model, tokenizer, processor, and image_processor.
        Set self.model, self.tokenizer, self.processor, self.image_processor.
        """
        pass

    def load_video(self, video_path, args=None):
        """
        Common video loading logic.
        """
        if args is None:
            args = self.args
        
        # Use standard fetch_video for consistency
        raw_video, frame_idx, fps = fetch_video({"video": video_path, "fps": args.fps}, resize=False)
        
        # Resize video to avoid OOM
        num_chunks = max(1, int(round(np.ceil(len(raw_video) / args.chunk_size))))
        if num_chunks > 1:
            logger.info(f"Video {video_path} is too long, resizing to {num_chunks} chunks.")
        target_pixels = args.total_pixels * num_chunks * 28 * 28
        
        video, fps = resize_video(raw_video, fps, total_pixels=target_pixels)
        
        # Return structure: raw_video, tokenizer, processor, frame_idx, fps, video_inputs, size_list
        # Note: In the original code, load_video returns:
        # [raw_video], None, None, frame_idx, fps, [video], None
        # We will maintain this return signature for compatibility with Vgent
        return [raw_video], None, None, frame_idx, fps, [video], None

    @abstractmethod
    def mllm_response(self, text, video_inputs, max_new_tokens=512, size_list=None, fps=None):
        """
        Generate response from the MLLM.
        """
        pass

    # ------------------------------------------------------------------
    # Shared vLLM helpers
    # ------------------------------------------------------------------

    def _vllm_text_only(self, text, max_new_tokens=512):
        """Text-only vLLM inference (no multi_modal_data).

        Used when ``video_inputs is None`` — e.g. reasoning, refinement,
        and keyword-extraction prompts issued by Vgent.

        Uses plain string content format which is universally compatible
        with all chat template families (Qwen, InternVL, LLaVA, Ovis).
        """
        from vllm import SamplingParams, TextPrompt

        sampling_params = SamplingParams(temperature=0.2, max_tokens=max_new_tokens)
        tokenizer = self.model.get_tokenizer()
        messages = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.model.generate(
            TextPrompt(prompt=prompt_text), sampling_params=sampling_params
        )
        return outputs[0].outputs[0].text

    def _vllm_batch_generate(self, prompt_inputs, sampling_params):
        """Run vLLM generate and sort outputs by request_id to restore
        input order (vLLM may reorder for scheduling efficiency)."""
        outputs = self.model.generate(prompt_inputs, sampling_params=sampling_params)
        try:
            outputs = sorted(outputs, key=lambda o: int(o.request_id))
        except (ValueError, TypeError):
            outputs = sorted(outputs, key=lambda o: o.request_id)
        return outputs

    @staticmethod
    def _extract_yes_prob(output):
        """Extract P(Yes) from a single vLLM output with logprobs.

        Expects the output to have been generated with
        ``SamplingParams(max_tokens=1, logprobs=20)``.
        """
        completion = output.outputs[0]
        answer_text = completion.text.strip()

        if completion.logprobs and len(completion.logprobs) > 0:
            token_logprobs = completion.logprobs[0]  # dict{token_id: Logprob}
            yes_lp = None
            no_lp = None
            for _tok_id, logprob_obj in token_logprobs.items():
                decoded = logprob_obj.decoded_token.strip().lower()
                if decoded == "yes" and yes_lp is None:
                    yes_lp = logprob_obj.logprob
                elif decoded == "no" and no_lp is None:
                    no_lp = logprob_obj.logprob

            if yes_lp is not None and no_lp is not None:
                yes_p = math.exp(yes_lp)
                no_p = math.exp(no_lp)
                total = yes_p + no_p
                yes_prob = yes_p / total if total > 0 else 0.5
            elif yes_lp is not None:
                yes_prob = math.exp(yes_lp)
            elif no_lp is not None:
                yes_prob = 1.0 - math.exp(no_lp)
            else:
                yes_prob = 1.0 if answer_text.lower().startswith("yes") else 0.0

            chosen_lp = yes_lp if yes_lp is not None else (no_lp if no_lp is not None else 0.0)
        else:
            yes_prob = 1.0 if answer_text.lower().startswith("yes") else 0.0
            chosen_lp = 0.0

        return {
            "answer": "Yes" if answer_text.lower().startswith("yes") else "No",
            "yes_prob": round(yes_prob, 6),
            "logprob": round(chosen_lp, 6) if chosen_lp is not None else 0.0,
        }

    @staticmethod
    def _tensor_to_pil_frames(video):
        """Convert a video tensor (T, C, H, W) or numpy array to list[PIL.Image].

        Handles:
          - float tensors in [0, 255] range (from load_video_clip)
          - float tensors in [0, 1] range
          - uint8 tensors
          - numpy arrays (T, C, H, W) or (T, H, W, C)
        """
        import torch
        from PIL import Image

        if isinstance(video, torch.Tensor):
            arr = video.cpu()
            if arr.dtype in (torch.float32, torch.float16, torch.bfloat16):
                if arr.max() <= 1.0:
                    arr = (arr * 255).clamp(0, 255)
                arr = arr.clamp(0, 255).to(torch.uint8)
            arr = arr.permute(0, 2, 3, 1).numpy()
            return [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        elif isinstance(video, np.ndarray):
            if video.ndim == 4 and video.shape[1] in (1, 3):
                video = np.transpose(video, (0, 2, 3, 1))
            return [Image.fromarray(video[i].astype(np.uint8))
                    for i in range(video.shape[0])]
        else:
            return []

    # ------------------------------------------------------------------
    # Default batch implementations (sequential fallback)
    # ------------------------------------------------------------------

    def mllm_batch_response(
        self,
        prompts: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Batch inference: process multiple prompts in a single call.

        Each element of *prompts* is a dict::

            {
                "text": str,                       # required
                "video_inputs": tensor | None,     # optional
                "max_new_tokens": int,             # optional, default 512
            }

        The default implementation falls back to sequential
        ``mllm_response`` calls.  Model-specific subclasses override
        this with true batched vLLM ``generate`` when available.
        """
        results: List[str] = []
        for p in prompts:
            text = p["text"]
            video_inputs = p.get("video_inputs", None)
            max_new_tokens = p.get("max_new_tokens", 512)
            results.append(
                self.mllm_response(text, video_inputs, max_new_tokens=max_new_tokens)
            )
        return results

    def mllm_yes_no_batch(
        self,
        prompts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Batched Yes/No classification with logprob extraction.

        Each element of *prompts* is a dict::

            {
                "text": str,                       # required — should end with
                                                   #   "Answer only Yes or No."
                "video_inputs": tensor | None,     # optional
            }

        Returns a list of dicts, one per prompt::

            {
                "answer": "Yes" | "No",
                "yes_prob": float,   # P(Yes) in [0, 1]
                "logprob": float,    # raw log-probability of the answer token
            }

        The default implementation falls back to sequential
        ``mllm_response`` calls with manual "Yes"/"No" parsing.
        Model-specific subclasses override this to use vLLM
        ``logprobs`` for calibrated probability extraction.
        """
        results: List[Dict[str, Any]] = []
        for p in prompts:
            text = p["text"]
            video_inputs = p.get("video_inputs", None)
            try:
                resp = self.mllm_response(
                    text, video_inputs, max_new_tokens=1,
                )
                resp_clean = resp.strip().lower() if resp else ""
                is_yes = resp_clean.startswith("yes")
                results.append({
                    "answer": "Yes" if is_yes else "No",
                    "yes_prob": 1.0 if is_yes else 0.0,
                    "logprob": 0.0,  # no logprob available in fallback
                })
            except Exception as e:
                logger.error(f"mllm_yes_no_batch fallback error: {e}")
                results.append({
                    "answer": "No",
                    "yes_prob": 0.0,
                    "logprob": float("-inf"),
                })
        return results
