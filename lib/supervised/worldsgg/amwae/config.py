"""
AMWAE Configuration
====================

Dataclass + argparse config for the Associative Masked World Auto-Encoder.
"""

from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class AMWAEConfig:
    """Hyperparameters for AMWAE training and model architecture."""

    # ---- Paths ----
    data_path: str = "/data/rohith/ag"
    save_path: str = "/data/rohith/ag/checkpoints"
    results_path: str = "results"
    method_name: str = "amwae"
    task_name: str = "worldsgg"
    detector_ckpt: str = ""
    detector_model: str = "v3l"
    ckpt: str = ""

    # ---- Model Architecture ----
    d_model: int = 256
    d_struct: int = 256
    d_visual: int = 256
    d_detector_roi: int = 1024
    n_self_attn_layers: int = 4
    n_cross_attn_layers: int = 2
    n_heads: int = 4
    d_feedforward: int = 512
    max_objects: int = 64
    dropout: float = 0.1

    # ---- Episodic Memory ----
    memory_bank_size: int = 32  # Number of past frames to store

    # ---- Training ----
    mode: str = "predcls"
    nepoch: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 5.0
    bce_loss: bool = True
    optimizer: str = "adamw"

    # ---- Masking & Loss Weights ----
    p_mask_visible: float = 0.5  # Prob of masking visible features during training
    lambda_masked: float = 2.0   # Weight for masked-edge scene graph loss
    lambda_recon: float = 1.0    # Feature reconstruction loss weight
    lambda_contrastive: float = 0.5  # InfoNCE contrastive loss weight
    temperature: float = 0.07       # InfoNCE temperature

    # ---- Data ----
    datasize: str = "large"
    world_sg_dir: str = ""
    include_invisible: bool = True

    # ---- Logging / Misc ----
    use_wandb: bool = False
    use_amp: bool = True
    log_every: int = 100

    @classmethod
    def from_args(cls) -> "AMWAEConfig":
        """Parse command-line arguments into an AMWAEConfig."""
        parser = cls.setup_parser()
        args = vars(parser.parse_args())
        return cls(**args)

    @staticmethod
    def setup_parser() -> ArgumentParser:
        parser = ArgumentParser(description="AMWAE training")

        # Paths
        parser.add_argument("--data_path", default="/data/rohith/ag", type=str)
        parser.add_argument("--save_path", default="/data/rohith/ag/checkpoints", type=str)
        parser.add_argument("--results_path", default="results", type=str)
        parser.add_argument("--method_name", default="amwae", type=str)
        parser.add_argument("--task_name", default="worldsgg", type=str)
        parser.add_argument("--detector_ckpt", default="", type=str)
        parser.add_argument("--detector_model", default="v3l", type=str)
        parser.add_argument("--ckpt", default="", type=str)

        # Model
        parser.add_argument("--d_model", default=256, type=int)
        parser.add_argument("--d_struct", default=256, type=int)
        parser.add_argument("--d_visual", default=256, type=int)
        parser.add_argument("--d_detector_roi", default=1024, type=int)
        parser.add_argument("--n_self_attn_layers", default=4, type=int)
        parser.add_argument("--n_cross_attn_layers", default=2, type=int)
        parser.add_argument("--n_heads", default=4, type=int)
        parser.add_argument("--d_feedforward", default=512, type=int)
        parser.add_argument("--max_objects", default=64, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

        # Memory
        parser.add_argument("--memory_bank_size", default=32, type=int,
                            help="FIFO queue: number of past frames to store")

        # Training
        parser.add_argument("--mode", default="predcls", type=str,
                            choices=["predcls", "sgcls", "sgdet"])
        parser.add_argument("--nepoch", default=20, type=int)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        parser.add_argument("--warmup_steps", default=500, type=int)
        parser.add_argument("--grad_clip", default=5.0, type=float)
        parser.add_argument("--bce_loss", action="store_true")
        parser.add_argument("--optimizer", default="adamw", type=str)

        # Masking & Loss
        parser.add_argument("--p_mask_visible", default=0.5, type=float,
                            help="Prob of masking visible features during training")
        parser.add_argument("--lambda_masked", default=2.0, type=float,
                            help="Weight for masked-edge SG loss")
        parser.add_argument("--lambda_recon", default=1.0, type=float,
                            help="Feature reconstruction loss weight")
        parser.add_argument("--lambda_contrastive", default=0.5, type=float,
                            help="InfoNCE contrastive loss weight")
        parser.add_argument("--temperature", default=0.07, type=float,
                            help="InfoNCE temperature")

        # Data
        parser.add_argument("--datasize", default="large", type=str)
        parser.add_argument("--world_sg_dir", default="", type=str)
        parser.add_argument("--include_invisible", action="store_true")

        # Logging
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--use_amp", action="store_true")
        parser.add_argument("--log_every", default=100, type=int)

        return parser
