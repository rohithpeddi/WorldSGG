"""Centralized configuration loader for the pseudo-label pipeline.

Loads ``configs/mllm/server.yaml`` from the WorldSGG repository and provides helpers that all
scripts and modules use to obtain paths, model definitions, and defaults.

Discovery order:
    1. Explicit ``config_path`` argument to ``load_config()``.
    2. ``--config <path>`` on the command line (scanned from ``sys.argv``).
    3. ``$WSGG_MLLM_CONFIG`` if set, else ``configs/mllm/server.yaml`` at the
       repository root (auto-discovered relative to this module).
"""
from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Where config_utd.yaml lives by default (backend/pseudo/config_utd.yaml)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent          # lib/mllm/core/
_REPO_ROOT = _THIS_DIR.parents[2]                    # WorldSGG repo root
# Default: configs/mllm/server.yaml; override with $WSGG_MLLM_CONFIG or --config.
_DEFAULT_CONFIG_PATH = Path(
    os.environ.get("WSGG_MLLM_CONFIG", _REPO_ROOT / "configs" / "mllm" / "server.yaml")
)

# Module-level cache so the file is only read once per process.
_cached_config: Optional[dict] = None


def _config_path_from_argv() -> Optional[str]:
    """Extract ``--config <path>`` from sys.argv without consuming it.

    This is called *before* argparse runs so that ``load_config()`` can be
    used at module-import time (e.g. to set argparse defaults).
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(config_path: Optional[str] = None) -> dict:
    """Load and cache the YAML configuration.

    Parameters
    ----------
    config_path : str, optional
        Explicit path to a YAML config file.  If *None*, the loader checks
        ``--config`` on the command line, then falls back to
        ``backend/pseudo/config_utd.yaml`` relative to this module.

    Returns
    -------
    dict
        The parsed configuration dictionary.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config

    if config_path is None:
        config_path = _config_path_from_argv()

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.is_file():
        logger.warning(
            f"Config file not found at {path}. Using built-in defaults."
        )
        _cached_config = _builtin_defaults()
        return _cached_config

    with open(path, "r", encoding="utf-8") as fh:
        _cached_config = yaml.safe_load(fh)
    logger.info(f"Loaded configuration from {path}")
    return _cached_config


def get_path(cfg: dict, key: str) -> str:
    """Return a path from ``cfg['paths']``.

    Supports dot-notation for nested keys, e.g.
    ``get_path(cfg, 'outputs.rag')`` → ``cfg['paths']['outputs']['rag']``.
    """
    obj = cfg.get("paths", {})
    for part in key.split("."):
        obj = obj.get(part, "")
    return str(obj)


def get_model_registry(cfg: dict) -> Dict[str, Dict[str, str]]:
    """Return the raw model registry dict from config."""
    return cfg.get("models", {})


def get_model_hf_ids(cfg: dict) -> Dict[str, str]:
    """Return ``{model_key: hf_id}`` for all registered models."""
    return {
        key: entry["hf_id"]
        for key, entry in get_model_registry(cfg).items()
    }


def resolve_model_path(cfg: dict, model_key: str) -> str:
    """Return the loadable model path for *model_key*.

    If a local weights directory is configured and the subdirectory
    ``<model_weights>/<model_key>/`` exists, return that path.
    Otherwise return the HuggingFace model ID.
    """
    registry = get_model_registry(cfg)
    entry = registry.get(model_key)
    if entry is None:
        raise ValueError(
            f"Model '{model_key}' not in config. "
            f"Available: {list(registry.keys())}"
        )
    hf_id = entry["hf_id"]

    weights_dir = get_path(cfg, "model_weights")
    if weights_dir:
        local = os.path.join(weights_dir, model_key)
        if os.path.isdir(local):
            logger.info(f"Using local weights: {local}")
            return local
        else:
            logger.debug(
                f"Local weights not found at {local}, using HF ID: {hf_id}"
            )
    return hf_id


def import_model_class(cfg: dict, model_key: str):
    """Dynamically import and return the model class for *model_key*.

    Uses the ``module`` and ``class`` fields from the config to avoid
    hardcoded imports at the top of ``vgent.py``.
    """
    registry = get_model_registry(cfg)
    entry = registry.get(model_key)
    if entry is None:
        raise ValueError(
            f"Model '{model_key}' not in config. "
            f"Available: {list(registry.keys())}"
        )
    module_path = entry['module']
    cls_name = entry["class"]
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def get_inference_defaults(cfg: dict) -> dict:
    """Return the ``inference`` section with safe fallbacks."""
    return cfg.get("inference", {
        "default_model": "qwen25vl_7b",
        "use_vllm": True,
        "fps": 1.0,
        "chunk_size": 128,
        "total_pixels": 128000,
    })


def get_vllm_engine_settings(cfg: dict) -> dict:
    """Return the ``vllm`` engine-tuning section with safe fallbacks.

    These values are forwarded to ``vllm.LLM(...)`` by each model wrapper.
    """
    defaults = {
        "gpu_memory_utilization": 0.90,
        "max_model_len": 32768,
        "max_num_seqs": 32,
        "enable_chunked_prefill": True,
        "dtype": "bfloat16",
        # NOTE: swap_space is deprecated in vLLM >= 0.8 and ignored.
    }
    vllm_cfg = cfg.get("vllm", {})
    merged = {**defaults, **vllm_cfg}
    return merged


def get_box_folder_ids(cfg: dict) -> Dict[str, str]:
    """Return the ``box_folders`` mapping ``{name: box_folder_id}``."""
    return cfg.get("box_folders", {})


def get_embedding_model_registry(cfg: dict) -> Dict[str, Dict[str, str]]:
    """Return the raw embedding-model registry dict from config."""
    return cfg.get("embedding_models", {})


def get_embedding_model_hf_ids(cfg: dict) -> Dict[str, str]:
    """Return ``{model_key: hf_id}`` for all registered embedding models."""
    return {
        key: entry["hf_id"]
        for key, entry in get_embedding_model_registry(cfg).items()
    }


def resolve_embedding_model_path(cfg: dict, model_key: str) -> str:
    """Return the loadable path for an embedding model.

    If a local weights directory is configured and the subdirectory
    ``<model_weights>/<model_key>/`` exists, return that path.
    Otherwise return the HuggingFace model ID.
    """
    registry = get_embedding_model_registry(cfg)
    entry = registry.get(model_key)
    if entry is None:
        raise ValueError(
            f"Embedding model '{model_key}' not in config. "
            f"Available: {list(registry.keys())}"
        )
    hf_id = entry["hf_id"]

    weights_dir = get_path(cfg, "model_weights")
    if weights_dir:
        local = os.path.join(weights_dir, model_key)
        if os.path.isdir(local):
            logger.info(f"Using local embedding weights: {local}")
            return local
        else:
            logger.debug(
                f"Local embedding weights not found at {local}, "
                f"using HF ID: {hf_id}"
            )
    return hf_id


# ---------------------------------------------------------------------------
# Fallback defaults (used when config_utd.yaml is missing)
# ---------------------------------------------------------------------------

def _builtin_defaults() -> dict:
    """Minimal defaults when no config file is found."""
    return {
        "paths": {
            "ag_root": "/data/rohith/ag",
            "dynamic_scenes": "/data2/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
            "model_weights": "",
            "graphs": "/data/rohith/ag/graphs/",
            "outputs": {
                "graphs": "/data/rohith/ag/graphs/",
                "rag": "/data/rohith/ag/mllms/rag_results/",
                "rag_all": "/data/rohith/ag/mllms/rag_all_objects_results/",
                "caption": "/data/rohith/ag/mllms/caption_results/",
                "caption_all": "/data/rohith/ag/mllms/caption_all_objects_results/",
                "zero_shot": "/data/rohith/ag/mllms/zero_shot_results/",
                "wsg_agent": "/data/rohith/ag/mllms/wsg_agent_results/",
                "visualizations": "/data/rohith/ag/mllms/visualizations/",
                "wsg_corrections": "/data/rohith/ag/wsg_corrections/",
                "wsg_2d_augmentations": "/data/rohith/ag/wsg_2d_augmentations/",
                "predictions": "/data/rohith/ag/mllms/predictions/",
            },
        },
        "models": {},
        "inference": {
            "default_model": "qwen25vl_7b",
            "use_vllm": True,
            "fps": 1.0,
            "chunk_size": 128,
            "total_pixels": 128000,
        },
        "vllm": {
            "gpu_memory_utilization": 0.90,
            "max_model_len": 32768,
            "max_num_seqs": 32,
            "enable_chunked_prefill": True,
            "dtype": "bfloat16",
        },
    }
