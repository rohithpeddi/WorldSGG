#!/usr/bin/env python3
"""
Monocular3D Training Entry Point — YAML Config + CLI Override

Usage:
    # Default config
    python -m lib.detector.monocular3d.train --config configs/default.yaml

    # Debug (fast iteration)
    python -m lib.detector.monocular3d.train --config configs/debug.yaml

    # Override any YAML field via CLI
    python -m lib.detector.monocular3d.train --config configs/default.yaml --lr 5e-5 --batch_size 32

    # Resume from checkpoint
    python -m lib.detector.monocular3d.train --config configs/default.yaml --ckpt checkpoint_10
"""

import argparse
import os
import sys
import yaml
from dataclasses import fields as dataclass_fields

# Ensure the package root is importable when run as a script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from lib.detector.monocular3d.trainer import TrainConfig, DinoAGTrainer3D


def load_config(config_path: str) -> dict:
    """Load a YAML config file and return as dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser with --config plus every TrainConfig field as an override."""
    parser = argparse.ArgumentParser(
        description="Train DINOv2 AG Monocular 3D detector (YAML + CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g. configs/default.yaml)",
    )

    # Dynamically add every TrainConfig field as an optional CLI argument
    for f in dataclass_fields(TrainConfig):
        arg_name = f"--{f.name}"
        field_type = f.type

        # Handle Optional types (e.g., Optional[int] shows as typing.Optional[int])
        origin = getattr(field_type, "__origin__", None)
        type_args = getattr(field_type, "__args__", ())

        # Check if it's Optional[X] (Union[X, None])
        is_optional = False
        inner_type = field_type
        if origin is not None and type(None) in type_args:
            is_optional = True
            # Get the non-None type
            inner_type = next((t for t in type_args if t is not type(None)), str)

        if inner_type is bool:
            # Booleans: support --flag / --no-flag
            parser.add_argument(arg_name, type=_str_to_bool, default=None,
                                help=f"(bool) Override '{f.name}'")
        elif inner_type is int or inner_type == "int":
            parser.add_argument(arg_name, type=lambda v: None if v.lower() == "null" else int(v),
                                default=None,
                                help=f"(int{', optional' if is_optional else ''}) Override '{f.name}'")
        elif inner_type is float or inner_type == "float":
            parser.add_argument(arg_name, type=float, default=None,
                                help=f"(float) Override '{f.name}'")
        else:
            parser.add_argument(arg_name, type=str, default=None,
                                help=f"(str) Override '{f.name}'")
    return parser


def _str_to_bool(v: str) -> bool:
    """Parse a boolean from CLI string."""
    if v.lower() in ("true", "1", "yes"):
        return True
    elif v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'")


def merge_config(yaml_cfg: dict, cli_args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI overrides. CLI takes precedence."""
    merged = dict(yaml_cfg)
    for key, val in vars(cli_args).items():
        if key == "config":
            continue  # skip the --config arg itself
        if val is not None:
            merged[key] = val
    return merged


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve config path relative to script dir if not absolute
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_SCRIPT_DIR, config_path)

    if not os.path.isfile(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    yaml_cfg = load_config(config_path)

    # Merge with CLI overrides
    merged = merge_config(yaml_cfg, args)

    # Filter to only TrainConfig fields
    valid_fields = {f.name for f in dataclass_fields(TrainConfig)}
    filtered = {k: v for k, v in merged.items() if k in valid_fields}

    # Handle null -> None for Optional fields
    for k, v in filtered.items():
        if v == "null" or v is None:
            filtered[k] = None

    print(f"Config: {filtered}")

    cfg = TrainConfig(**filtered)
    trainer = DinoAGTrainer3D(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
