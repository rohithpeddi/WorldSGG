"""
AMWAE++: Associative Masked World Auto-Encoder with Energy Transformer
=======================================================================

Subclass of AMWAE that replaces Module 6 (ContextualDiffusion) with
EnergyDiffusion — a weight-tied recurrent transformer that iterates
a single shared layer until the representation converges.

All other modules (scaffold tokenizer, memory, retriever, predictor,
temporal edge attention) are inherited unchanged from AMWAE.
"""

from .amwae import AMWAE
from .energy_diffusion import EnergyDiffusion


class AMWAEPP(AMWAE):
    """
    AMWAE++ with Energy Transformer diffusion.

    Differences from AMWAE:
      - Module 6 is EnergyDiffusion (weight-tied, convergence-based)
      - 75% fewer parameters in diffusion module
      - Adaptive inference compute (dynamic stopping when converged)
      - Returns h_prev for attractor stability loss

    Args:
        Same as AMWAE.
    """

    def __init__(
        self,
        config,
        num_object_classes: int = 37,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
    ):
        super().__init__(
            config=config,
            num_object_classes=num_object_classes,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
        )

        # Override Module 6: Replace ContextualDiffusion with EnergyDiffusion
        self.diffusion = EnergyDiffusion(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
            train_iters=getattr(config, 'energy_train_iters', 4),
            eval_iters=getattr(config, 'energy_eval_iters', 15),
            epsilon=getattr(config, 'energy_epsilon', 1e-3),
        )
