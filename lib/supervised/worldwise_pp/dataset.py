"""
WorldAG + the WorldWise++ grid cache.

``WorldAGGrid.__getitem__`` returns the WorldAG item plus

    grid_dino : (T, Hp, Wp, 256) float16 — DINOv3 L24n on the Pi3 grid (PCA-256)
    grid_pi3  : (T, Hp, Wp, 256) float16 — Pi3 G14 (PCA-256)
    image_hw  : (H, W) of the Pi3 image the grids (and bboxes_2d) live on
    grid_hw   : (Hp, Wp)

read from ``<grid_cache_root>/<phase>/<video>.npz`` (members ``frames`` (T,) str,
``dino``, ``pi3``, ``grid_hw``, ``target_size`` = (H, W)); only the rows for the
item's ``frame_names`` are taken, in that order.  A missing cache file or frame
raises — the model must never see silent zeros.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from dataloader.world_ag_dataset import WorldAG

logger = logging.getLogger(__name__)


class WorldAGGrid(WorldAG):
    def __init__(self, *args, grid_cache_root: str, allow_missing_grids: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._grid_dir = Path(grid_cache_root) / self._phase
        if not self._grid_dir.is_dir():
            raise FileNotFoundError(
                f"[WorldAGGrid] grid cache directory not found: {self._grid_dir} "
                f"(build it with datasets/preprocess/tokens/build_pp_grid_cache.py)")
        missing = [v for v in self.video_list if not self._grid_path(v).exists()]
        if missing:
            msg = (f"[WorldAGGrid][{self._phase}] {len(missing)}/{len(self.video_list)} videos have no grid "
                   f"cache under {self._grid_dir} (e.g. {missing[:5]})")
            if not allow_missing_grids:
                raise FileNotFoundError(msg + " — set allow_missing_grids to drop them")
            logger.warning(msg + " — dropped")
            drop = set(missing)
            self.video_list = [v for v in self.video_list if v not in drop]
        logger.info(f"[WorldAGGrid][{self._phase}] grids from {self._grid_dir} for {len(self.video_list)} videos")

    def _grid_path(self, video_id: str) -> Path:
        return self._grid_dir / f"{video_id}.npz"

    def load_grids(self, video_id: str, frame_names) -> Dict[str, Any]:
        path = self._grid_path(video_id)
        if not path.exists():
            raise FileNotFoundError(f"[WorldAGGrid] grid cache missing for {video_id}: {path}")
        with np.load(path) as z:
            frames = [str(f) for f in z["frames"].tolist()]
            row = {f: i for i, f in enumerate(frames)}
            try:
                idx = np.array([row[f] for f in frame_names], dtype=np.int64)
            except KeyError as e:
                raise KeyError(f"[WorldAGGrid] frame {e} of {video_id} not in grid cache {path} "
                               f"(cache has {len(frames)} frames, e.g. {frames[:3]})") from None
            dino = np.ascontiguousarray(z["dino"][idx])
            pi3 = np.ascontiguousarray(z["pi3"][idx])
            grid_hw = tuple(int(v) for v in z["grid_hw"].tolist())
            target = tuple(int(v) for v in z["target_size"].tolist())
        if dino.shape != pi3.shape or dino.ndim != 4:
            raise ValueError(f"[WorldAGGrid] bad grid shapes for {video_id}: dino {dino.shape} pi3 {pi3.shape}")
        if tuple(dino.shape[1:3]) != grid_hw:
            raise ValueError(f"[WorldAGGrid] grid_hw {grid_hw} != array grid {dino.shape[1:3]} for {video_id}")
        return {
            "grid_dino": torch.from_numpy(dino.astype(np.float16, copy=False)),
            "grid_pi3": torch.from_numpy(pi3.astype(np.float16, copy=False)),
            "image_hw": target,
            "grid_hw": grid_hw,
        }

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = super().__getitem__(index)
        item.update(self.load_grids(item["video_id"], item["frame_names"]))
        return item
