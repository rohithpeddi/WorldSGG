"""V2: WorldWise+-3D — WorldWise-3D whose appearance projectors consume cached foundation tokens.

WorldWise+ (lib/supervised/worldwise_plus/model.py) is WorldWise with the scaffold tokenizer's visual projector (and
its EMA target copy) and the relation head's union projector replaced by a GatedFusionProjector over the cached
DINOv3 / Pi3 tokens. WorldWise+-3D applies exactly that swap to WorldWise-3D, so every V2 rule (no box input for
hidden / masked slots, [GEO-MASK], inferred + injected g_hat, decoded box, 3-D loss on masked items only) is
WorldWise-3D's, unchanged.
"""
from __future__ import annotations

import copy

from lib.supervised.v2.worldwise_3d import WorldWise3D
from lib.supervised.worldwise_plus.model import GatedFusionProjector, WorldWisePlus


class WorldWisePlus3D(WorldWise3D):
    def __init__(self, config, num_object_classes: int = 37, attention_class_num: int = 3,
                 spatial_class_num: int = 6, contact_class_num: int = 17):
        super().__init__(config, num_object_classes, attention_class_num, spatial_class_num, contact_class_num)
        # Same swap as WorldWisePlus.__init__
        streams = [int(d) for d in getattr(config, "token_streams", [config.d_detector_roi])]
        fusion = getattr(config, "fusion", "gated")
        if sum(streams) != config.d_detector_roi:
            raise ValueError(f"token_streams {streams} must sum to d_detector_roi={config.d_detector_roi}")
        if config.d_union_roi != config.d_detector_roi:
            raise ValueError("WorldWise+ expects d_union_roi == d_detector_roi (same token layout for union boxes)")
        tok = self.scaffold_tokenizer
        tok.visual_projector = GatedFusionProjector(streams, tok.d_visual, fusion)
        if getattr(tok, "use_ema_target", False):
            tok.target_projector = copy.deepcopy(tok.visual_projector)
            for p in tok.target_projector.parameters():
                p.requires_grad_(False)
        rp = self.rel_predictor
        rp.union_proj = GatedFusionProjector(streams, rp.d_rel // 4, fusion)
        self.token_streams = streams
        self.fusion = fusion

    gate_summary = WorldWisePlus.gate_summary
