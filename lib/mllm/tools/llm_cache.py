"""
Content-hashed LLM response cache (B4/B5/B6): every VLM call of the tracks goes
through :class:`CachedVLM` so that re-runs after an annotation revision only pay
for prompts whose content changed.

Key = sha256 of ``{"model", "text", "images": [sha256(png bytes)], "params"}``;
value file ``<llm_cache>/<hh>/<hash>.json`` = ``{"response", "model", "params",
"created", "n_images", "text_head"}``.  Annotation-INDEPENDENT by construction
(the prompt content is the key; the object list is inside the prompt).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

logger = logging.getLogger(__name__)


def image_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def image_hash(img: Image.Image) -> str:
    return hashlib.sha256(image_bytes(img)).hexdigest()


class LLMCache:
    def __init__(self, cache_dir: str):
        self.root = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def key(self, model: str, text: str, image_hashes: Sequence[str], params: Dict[str, Any]) -> str:
        blob = json.dumps({"model": model, "text": text, "images": list(image_hashes),
                           "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Optional[str]:
        p = self.path(key)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self.hits += 1
                    return json.load(f)["response"]
            except Exception:
                return None
        return None

    def put(self, key: str, response: str, model: str, text: str, n_images: int, params: Dict[str, Any]) -> None:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"response": response, "model": model, "params": params, "n_images": n_images,
                       "text_head": text[:200], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        os.replace(tmp, p)
        self.misses += 1


class CachedVLM:
    """Batch interface over a vendored model wrapper with the cache in front.

    ``prompts``: list of ``{"text": str, "images": [PIL.Image], "max_new_tokens": int}``.
    Images are passed to the wrapper as a single "video" tensor (T, C, H, W) --
    the vendored wrappers accept exactly that -- so multi-image prompts (frames +
    BEV) work with every registered model.
    """

    def __init__(self, vgent_model, model_key: str, cache: LLMCache, temperature: float = 0.0,
                 strip_thinking: bool = True):
        self.model = vgent_model
        self.model_key = model_key
        self.cache = cache
        self.temperature = temperature
        self.strip_thinking = strip_thinking

    @staticmethod
    def _to_tensor(images: Sequence[Image.Image]):
        import numpy as np
        import torch
        if not images:
            return None
        w = max(im.width for im in images)
        h = max(im.height for im in images)
        arr = []
        for im in images:
            im = im.convert("RGB")
            if im.size != (w, h):
                canvas = Image.new("RGB", (w, h), (0, 0, 0))
                canvas.paste(im, (0, 0))
                im = canvas
            arr.append(np.asarray(im))
        t = torch.from_numpy(np.stack(arr, 0)).permute(0, 3, 1, 2).float()
        return [t]

    def generate(self, prompts: List[Dict[str, Any]]) -> List[str]:
        params = {"temperature": self.temperature}
        keys, todo, out = [], [], [None] * len(prompts)
        for i, p in enumerate(prompts):
            hs = [image_hash(im) for im in p.get("images", [])]
            k = self.cache.key(self.model_key, p["text"], hs, {**params, "max_new_tokens": p.get("max_new_tokens", 1024)})
            keys.append(k)
            r = self.cache.get(k)
            if r is None:
                todo.append(i)
            else:
                out[i] = r
        if todo:
            multi = bool(getattr(self.model, "supports_images", False))
            batch = []
            for i in todo:
                ims = [im.convert("RGB") for im in prompts[i].get("images", [])]
                item = {"text": prompts[i]["text"], "max_new_tokens": prompts[i].get("max_new_tokens", 1024)}
                if multi:
                    item["images"] = ims
                    item["video_inputs"] = None
                else:
                    item["video_inputs"] = self._to_tensor(ims)
                batch.append(item)
            resps = self.model.mllm_batch_response(batch)
            for i, r in zip(todo, resps):
                r = r if isinstance(r, str) else ""
                out[i] = r
                self.cache.put(keys[i], r, self.model_key, prompts[i]["text"], len(prompts[i].get("images", [])),
                               {**params, "max_new_tokens": prompts[i].get("max_new_tokens", 1024)})
        if self.strip_thinking:
            from lib.mllm.core.prompts import strip_thinking_tags
            out = [strip_thinking_tags(r or "") for r in out]
        return [r or "" for r in out]
