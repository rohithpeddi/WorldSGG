"""
Amnesic Geometric GNN Configuration
=====================================

Simpler config — no memory, no masking curriculum, no contrastive loss.
Pure stateless feed-forward baseline.
"""

from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class AmnesicGNNConfig:
    """Hyperparameters for the Amnesic Geometric GNN baseline."""

    # ---- Paths ----
    data_path: str = "/data/rohith/ag"
    save_path: str = "/data/rohith/ag/checkpoints"
    results_path: str = "results"
    method_name: str = "amnesic_gnn"
    task_name: str = "worldsgg"
    detector_ckpt: str = ""
    detector_model: str = "v3l"
    ckpt: str = ""

    # ---- Model Architecture ----
    d_model: int = 256
    d_struct: int = 256
    d_visual: int = 256
    d_detector_roi: int = 1024
    n_gnn_layers: int = 4
    n_heads: int = 4
    d_feedforward: int = 512
    max_objects: int = 64
    dropout: float = 0.1

    # ---- Training ----
    mode: str = "predcls"
    nepoch: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    batch_size: int = 16  # Frames per batch (frame-shuffled)
    bce_loss: bool = True
    optimizer: str = "adamw"

    # ---- Data ----
    datasize: str = "large"
    world_sg_dir: str = ""
    include_invisible: bool = True

    # ---- Logging / Misc ----
    use_wandb: bool = False
    use_amp: bool = True
    log_every: int = 200

    @classmethod
    def from_args(cls) -> "AmnesicGNNConfig":
        parser = cls.setup_parser()
        args = vars(parser.parse_args())
        return cls(**args)

    @staticmethod
    def setup_parser() -> ArgumentParser:
        parser = ArgumentParser(description="Amnesic Geometric GNN training")

        # Paths
        parser.add_argument("--data_path", default="/data/rohith/ag", type=str)
        parser.add_argument("--save_path", default="/data/rohith/ag/checkpoints", type=str)
        parser.add_argument("--results_path", default="results", type=str)
        parser.add_argument("--method_name", default="amnesic_gnn", type=str)
        parser.add_argument("--task_name", default="worldsgg", type=str)
        parser.add_argument("--detector_ckpt", default="", type=str)
        parser.add_argument("--detector_model", default="v3l", type=str)
        parser.add_argument("--ckpt", default="", type=str)

        # Model
        parser.add_argument("--d_model", default=256, type=int)
        parser.add_argument("--d_struct", default=256, type=int)
        parser.add_argument("--d_visual", default=256, type=int)
        parser.add_argument("--d_detector_roi", default=1024, type=int)
        parser.add_argument("--n_gnn_layers", default=4, type=int)
        parser.add_argument("--n_heads", default=4, type=int)
        parser.add_argument("--d_feedforward", default=512, type=int)
        parser.add_argument("--max_objects", default=64, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

        # Training
        parser.add_argument("--mode", default="predcls", type=str,
                            choices=["predcls", "sgcls", "sgdet"])
        parser.add_argument("--nepoch", default=20, type=int)
        parser.add_argument("--lr", default=1e-4, type=float)
        parser.add_argument("--weight_decay", default=1e-4, type=float)
        parser.add_argument("--grad_clip", default=5.0, type=float)
        parser.add_argument("--batch_size", default=16, type=int,
                            help="Frames per batch (frame-shuffled)")
        parser.add_argument("--bce_loss", action="store_true")
        parser.add_argument("--optimizer", default="adamw", type=str)

        # Data
        parser.add_argument("--datasize", default="large", type=str)
        parser.add_argument("--world_sg_dir", default="", type=str)
        parser.add_argument("--include_invisible", action="store_true")

        # Logging
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--use_amp", action="store_true")
        parser.add_argument("--log_every", default=200, type=int)

        return parser
