from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from backend.pseudo.models.utils import fetch_video, resize_video
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
