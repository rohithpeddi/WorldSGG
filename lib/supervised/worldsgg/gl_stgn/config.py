"""
GL-STGN Configuration
=====================

Dataclass + argparse config for the Global-Local Spatio-Temporal Graph Network.
"""

from argparse import ArgumentParser
from dataclasses import dataclass, field


@dataclass
class GLSTGNConfig:
    """Hyperparameters for GL-STGN training and model architecture."""

    # ---- Paths ----
    data_path: str = "/data/rohith/ag"
    save_path: str = "/data/rohith/ag/checkpoints"
    results_path: str = "results"
    method_name: str = "gl_stgn"
    task_name: str = "worldsgg"
    detector_ckpt: str = ""
    detector_model: str = "v3l"
    ckpt: str = ""

    # ---- Model Architecture ----
    d_memory: int = 256
    d_struct: int = 256
    d_visual: int = 256
    d_detector_roi: int = 1024  # Frozen DINO ROI feature dim
    n_graph_layers: int = 3
    n_heads: int = 4
    max_objects: int = 64
    dropout: float = 0.1

    # ---- Training ----
    mode: str = "predcls"  # predcls / sgcls / sgdet
    nepoch: int = 20
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 500
    grad_clip: float = 5.0
    bce_loss: bool = True
    optimizer: str = "adamw"

    # ---- GL-STGN Specific ----
    chunk_length: int = 16
    lambda_unseen: float = 2.0
    p_mask_visual: float = 0.3
    teacher_forcing_epochs: int = 5

    # ---- Data ----
    datasize: str = "large"
    world_sg_dir: str = ""
    include_invisible: bool = True

    # ---- Logging / Misc ----
    use_wandb: bool = False
    use_amp: bool = True
    log_every: int = 100

    @classmethod
    def from_args(cls) -> "GLSTGNConfig":
        """Parse command-line arguments into a GLSTGNConfig."""
        parser = cls.setup_parser()
        args = vars(parser.parse_args())
        return cls(**args)

    @staticmethod
    def setup_parser() -> ArgumentParser:
        parser = ArgumentParser(description="GL-STGN training")

        # Paths
        parser.add_argument("--data_path", default="/data/rohith/ag", type=str)
        parser.add_argument("--save_path", default="/data/rohith/ag/checkpoints", type=str)
        parser.add_argument("--results_path", default="results", type=str)
        parser.add_argument("--method_name", default="gl_stgn", type=str)
        parser.add_argument("--task_name", default="worldsgg", type=str)
        parser.add_argument("--detector_ckpt", default="", type=str,
                            help="Path to frozen DINO detector checkpoint")
        parser.add_argument("--detector_model", default="v3l", type=str,
                            help="DINO model variant (v2, v2l, v3l, etc.)")
        parser.add_argument("--ckpt", default="", type=str,
                            help="GL-STGN checkpoint to resume from")

        # Model
        parser.add_argument("--d_memory", default=256, type=int)
        parser.add_argument("--d_struct", default=256, type=int)
        parser.add_argument("--d_visual", default=256, type=int)
        parser.add_argument("--d_detector_roi", default=1024, type=int)
        parser.add_argument("--n_graph_layers", default=3, type=int)
        parser.add_argument("--n_heads", default=4, type=int)
        parser.add_argument("--max_objects", default=64, type=int)
        parser.add_argument("--dropout", default=0.1, type=float)

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

        # GL-STGN specific
        parser.add_argument("--chunk_length", default=16, type=int,
                            help="BPTT temporal chunk length")
        parser.add_argument("--lambda_unseen", default=2.0, type=float,
                            help="Weight for unseen-object loss")
        parser.add_argument("--p_mask_visual", default=0.3, type=float,
                            help="Prob of masking visual features during training")
        parser.add_argument("--teacher_forcing_epochs", default=5, type=int,
                            help="Epochs using GT tracking IDs before switching to predicted")

        # Data
        parser.add_argument("--datasize", default="large", type=str)
        parser.add_argument("--world_sg_dir", default="", type=str,
                            help="Directory with world scene graph PKLs")
        parser.add_argument("--include_invisible", action="store_true",
                            help="Include RAG-predicted invisible objects")

        # Logging
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--use_amp", action="store_true",
                            help="Use automatic mixed precision")
        parser.add_argument("--log_every", default=100, type=int)

        return parser
