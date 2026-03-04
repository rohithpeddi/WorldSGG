"""
WSGG Testing Methods (Padded Tensor API)
==========================================

Per-method testing classes. Each overrides:
  - init_model()                → create model
  - is_temporal()               → sequential or frame-shuffled
  - process_test_video(batch)   → inference

The dataset returns a single dict with (T, N_max, ...) and (T, K_max, ...)
pre-padded tensors per video. No per-frame loops needed.

Usage:
  python test_wsgg_methods.py --config configs/methods/predcls/gl_stgn_predcls_dinov2b.yaml --ckpt path/to/ckpt.tar
"""

import logging

import torch

from wsgg_base import load_wsgg_config
from test_wsgg_base import TestWSGGBase

logger = logging.getLogger(__name__)


def _to_device(batch, device):
    """Move all tensor values in batch dict to device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ============================================================================
# GL-STGN (Temporal Transformer)
# ============================================================================

class TestGLSTGN(TestWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.gl_stgn.gl_stgn import GLSTGN

        self._model = GLSTGN(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)

    def is_temporal(self) -> bool:
        return True

    def process_test_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            camera_pose_seq=b.get("camera_poses"),
            union_features_seq=b.get("union_features"),
        )

        # Return last-frame predictions for evaluation
        T = b["visual_features"].shape[0]
        if T > 0:
            return {
                "attention_distribution": pred["attention_distribution"][-1],
                "spatial_distribution": pred["spatial_distribution"][-1],
                "contacting_distribution": pred["contacting_distribution"][-1],
            }
        return None


# ============================================================================
# AMWAE (Associative Masked World Auto-Encoder)
# ============================================================================

class TestAMWAE(TestWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.amwae.amwae import AMWAE

        self._model = AMWAE(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)

    def is_temporal(self) -> bool:
        return True

    def process_test_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            camera_pose_seq=b.get("camera_poses"),
        )

        T = b["visual_features"].shape[0]
        if T > 0:
            return {
                "attention_distribution": pred["attention_distribution"][-1],
                "spatial_distribution": pred["spatial_distribution"][-1],
                "contacting_distribution": pred["contacting_distribution"][-1],
            }
        return None


# ============================================================================
# LKS Buffer (Passive Memory Baseline)
# ============================================================================

class TestLKSGNN(TestWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.lks_buffer.lks_gnn import LKSGNN

        self._model = LKSGNN(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)

    def is_temporal(self) -> bool:
        return True

    def process_test_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
        )

        T = b["visual_features"].shape[0]
        if T > 0:
            return {
                "attention_distribution": pred["attention_distribution"][-1],
                "spatial_distribution": pred["spatial_distribution"][-1],
                "contacting_distribution": pred["contacting_distribution"][-1],
            }
        return None


# ============================================================================
# AMWAE++ (Energy Transformer variant)
# ============================================================================

class TestAMWAEPP(TestAMWAE):
    """AMWAE++ tester — same batched API as AMWAE, only model differs."""

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.amwae.amwae_pp import AMWAEPP

        self._model = AMWAEPP(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# Entry Point
# ============================================================================

METHOD_MAP = {
    "gl_stgn": TestGLSTGN,
    "amwae": TestAMWAE,
    "amwae_pp": TestAMWAEPP,
    "lks_buffer": TestLKSGNN,
}


def main():
    conf = load_wsgg_config()
    method_name = conf.method_name

    if method_name not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {list(METHOD_MAP.keys())}")

    tester_cls = METHOD_MAP[method_name]
    tester = tester_cls(conf)
    tester.init_method_evaluation()


if __name__ == "__main__":
    main()
