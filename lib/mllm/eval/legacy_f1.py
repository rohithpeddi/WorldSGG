"""
Legacy U-WSGG P/R/F1 (box-free) restricted to the locked worldbbox split (B2).

Wraps the vendored ``lib/mllm/eval/legacy/evaluate_relationships.py`` (the
numbers of the MLLM-baseline paper section) with the augmented GT of
``/data/rohith/ag/wsg_2d_augmentations`` (1,750 videos) filtered to the 1,511
worldbbox test videos.  The filtered GT directory is materialised once as a
directory of symlinks under ``/data3/rohith/ag/cache/mllm/legacy_gt_<split>/``
so the vendored evaluator (which globs ``*.pkl``) runs unchanged.

    from lib.mllm.eval.legacy_f1 import legacy_f1
    res = legacy_f1(predictions_dir, mode, model_name)   # dict of metric groups
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def restricted_gt_dir(cfg: Optional[dict] = None, video_ids=None) -> str:
    from lib.mllm.core.config_loader import get_path, load_config
    from lib.mllm.data.worldbbox import WorldBBoxTestSet
    cfg = cfg or load_config()
    src = Path(get_path(cfg, "outputs.wsg_2d_augmentations"))
    ts = WorldBBoxTestSet(cfg)
    ids = set(video_ids or ts.video_ids)
    import hashlib
    sub = "" if video_ids is None else "_" + hashlib.md5(",".join(sorted(ids)).encode()).hexdigest()[:8]
    dst = Path(get_path(cfg, "cache_root") or "/data3/rohith/ag") / "cache" / "mllm" / \
        f"legacy_gt_{ts.split_file.stem}{sub}"
    dst.mkdir(parents=True, exist_ok=True)
    n_new = n_missing = 0
    for vid in sorted(ids):
        s = src / f"{vid}.mp4.pkl"
        d = dst / f"{vid}.mp4.pkl"
        if not s.exists():
            n_missing += 1
            continue
        if not d.exists():
            os.symlink(s, d)
            n_new += 1
    have = len([p for p in dst.glob("*.pkl")])
    if n_missing:
        logger.warning(f"legacy GT: {n_missing} split videos have no wsg_2d_augmentations PKL")
    logger.info(f"legacy GT dir {dst}: {have} videos ({n_new} newly linked)")
    return str(dst)


def legacy_f1(predictions_dir: str, mode: str, model_name: str,
              cfg: Optional[dict] = None, video_ids=None) -> Dict[str, Any]:
    """Run the vendored evaluator; returns its metric dict (see
    ``evaluate_relationships.evaluate``) plus ``n_videos_evaluated``."""
    from lib.mllm.eval.legacy.evaluate_relationships import evaluate
    gt_dir = restricted_gt_dir(cfg, video_ids)
    res = evaluate(augmented_dir=gt_dir, predictions_dir=predictions_dir,
                   mode=mode, model_name=model_name)
    return res or {}


def summarize_legacy(res: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view: per slice micro P/R/F1, macro F1, per-head F1, n_pairs; object P/R/F1."""
    out: Dict[str, Any] = {k: res[k] for k in ("n_videos_evaluated", "n_videos_skipped") if k in res}
    for key in ("gt_plus_corrections", "corrections_only", "matched_objects_only"):
        b = res.get(key)
        if not isinstance(b, dict):
            continue
        mi, ma = b.get("micro_avg", {}), b.get("macro_avg", {})
        out[key] = {
            "micro_P": mi.get("precision"), "micro_R": mi.get("recall"), "micro_F1": mi.get("f1"),
            "macro_F1": ma.get("f1"), "n_pairs": b.get("n_pairs"),
            "per_head_F1": {h: b.get(h, {}).get("f1") for h in ("attention", "spatial", "contacting")},
        }
    od = res.get("object_detection")
    if isinstance(od, dict):
        out["object_detection"] = {k: od.get(k) for k in ("precision", "recall", "f1", "n_frames")}
    return out
