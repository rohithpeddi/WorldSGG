"""
WSGG Testing Methods
=====================

Per-method testing classes. Each overrides:
  - init_model()                → create model
  - is_temporal()               → sequential or frame-shuffled
  - process_test_video(batch)   → inference

Usage:
  python test_wsgg_methods.py --config configs/methods/predcls/gl_stgn_predcls.yaml --ckpt path/to/ckpt.tar
  python test_wsgg_methods.py --config configs/methods/predcls/amwae_predcls.yaml --ckpt path/to/ckpt.tar
  python test_wsgg_methods.py --config configs/methods/predcls/amwae_pp_predcls.yaml --ckpt path/to/ckpt.tar
  python test_wsgg_methods.py --config configs/methods/predcls/lks_buffer_predcls.yaml --ckpt path/to/ckpt.tar
"""

import torch

from wsgg_base import load_wsgg_config
from test_wsgg_base import TestWSGGBase


# ============================================================================
# GL-STGN
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
        tensors = batch
        self._model.reset_memory(self._device)
        all_preds = []

        T = len(tensors) if isinstance(tensors, list) else 1
        for t in range(T):
            frame = tensors[t] if isinstance(tensors, list) else tensors
            pred = self._model.forward_frame(
                visual_features=frame["visual_features"].to(self._device),
                corners=frame["corners"].to(self._device),
                valid_mask=frame["valid_mask"].to(self._device),
                visibility_mask=frame["visibility_mask"].to(self._device),
                person_idx=frame["person_idx"].to(self._device),
                object_idx=frame["object_idx"].to(self._device),
            )
            all_preds.append(pred)

        return all_preds[-1] if all_preds else None


# ============================================================================
# AMWAE
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
        tensors = batch
        self._model.reset_memory()
        all_preds = []

        T = len(tensors) if isinstance(tensors, list) else 1
        for t in range(T):
            frame = tensors[t] if isinstance(tensors, list) else tensors
            pred = self._model.forward_frame(
                visual_features=frame["visual_features"].to(self._device),
                corners=frame["corners"].to(self._device),
                valid_mask=frame["valid_mask"].to(self._device),
                visibility_mask=frame["visibility_mask"].to(self._device),
                person_idx=frame["person_idx"].to(self._device),
                object_idx=frame["object_idx"].to(self._device),
            )
            all_preds.append(pred)

        return all_preds[-1] if all_preds else None


# ============================================================================
# LKS Buffer
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
        tensors = batch
        self._model.reset_memory(self._device)
        all_preds = []

        T = len(tensors) if isinstance(tensors, list) else 1
        for t in range(T):
            frame = tensors[t] if isinstance(tensors, list) else tensors
            pred = self._model.forward_frame(
                visual_features=frame["visual_features"].to(self._device),
                corners=frame["corners"].to(self._device),
                valid_mask=frame["valid_mask"].to(self._device),
                visibility_mask=frame["visibility_mask"].to(self._device),
                person_idx=frame["person_idx"].to(self._device),
                object_idx=frame["object_idx"].to(self._device),
            )
            all_preds.append(pred)

        return all_preds[-1] if all_preds else None


# ============================================================================
# AMWAE++ (Energy Transformer variant)
# ============================================================================

class TestAMWAEPP(TestWSGGBase):

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

    def is_temporal(self) -> bool:
        return True

    def process_test_video(self, batch) -> dict:
        tensors = batch
        self._model.reset_memory()
        all_preds = []

        T = len(tensors) if isinstance(tensors, list) else 1
        for t in range(T):
            frame = tensors[t] if isinstance(tensors, list) else tensors
            pred = self._model.forward_frame(
                visual_features=frame["visual_features"].to(self._device),
                corners=frame["corners"].to(self._device),
                valid_mask=frame["valid_mask"].to(self._device),
                visibility_mask=frame["visibility_mask"].to(self._device),
                person_idx=frame["person_idx"].to(self._device),
                object_idx=frame["object_idx"].to(self._device),
            )
            all_preds.append(pred)

        return all_preds[-1] if all_preds else None


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
