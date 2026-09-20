"""
WSGG Testing Methods (Padded Tensor API)
==========================================

Per-method testing classes. Each overrides:
  - init_model()                → create model
  - is_temporal()               → sequential or frame-shuffled
  - process_test_video(batch)   → inference

The dataset returns a single dict with (T, N_max, ...) and (T, K_max, ...)
pre-padded tensors per video. No per-frame loops needed.

Methods:
  - w_sttran      : W-STTran      (world-adapted STTran, simplest baseline)
  - w_sttran_pp   : W-STTran++    (+ camera, motion, temporal edge attention)
  - w_dsgdetr     : W-DSGDetr     (+ temporal object encoder)
  - w_dsgdetr_pp  : W-DSGDetr++   (+ camera, motion, ego-motion)
  - worldwise     : WorldWise     (MWAE-based full proposed method)

Usage:
  python test_wsgg_methods.py --config configs/methods/predcls/worldwise_predcls_dinov2b.yaml --ckpt path/to/ckpt.tar
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
# W-STTran (World-adapted STTran — simplest baseline)
# ============================================================================

class TestWSTTran(TestWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.baselines.w_sttran.w_sttran import WSTTran

        self._model = WSTTran(
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
# W-STTran++ (Enhanced: + camera, motion, temporal edge attention)
# ============================================================================

class TestWSTTranPP(TestWSTTran):
    """W-STTran++ tester — same API, only model class differs."""

    def init_model(self):
        from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP

        self._model = WSTTranPP(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# W-DSGDetr (World-adapted DSGDetr — with temporal object encoder)
# ============================================================================

class TestWDSGDetr(TestWSTTran):
    """W-DSGDetr tester — same batched API as W-STTran, only model differs."""

    def init_model(self):
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr

        self._model = WDSGDetr(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# W-DSGDetr++ (Enhanced: + camera, motion, ego-motion)
# ============================================================================

class TestWDSGDetrPP(TestWDSGDetr):
    """W-DSGDetr++ tester — same API, only model class differs."""

    def init_model(self):
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP

        self._model = WDSGDetrPP(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# W-USG (External baseline: USG-Par relation machinery on the WSGG substrate)
# ============================================================================

class TestWUSG(TestWSTTran):
    """W-USG tester — same batched API as W-STTran, only model differs."""

    def init_model(self):
        from lib.supervised.baselines.w_usg.w_usg import WUSG

        self._model = WUSG(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# WorldWise (MWAE-based — full proposed method with ablation support)
# ============================================================================

class TestWorldWise(TestWSGGBase):
    """WorldWise tester — MWAE-based with config-flag ablation support."""

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldwise.worldwise import WorldWise

        self._model = WorldWise(
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
# Entry Point
# ============================================================================

class TestWorldWisePlus(TestWorldWise):
    """WorldWise+ (formerly WorldFormer C1)."""

    def init_model(self):
        from lib.supervised.worldwise_plus import WorldWisePlus

        self._model = WorldWisePlus(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)


TestWorldFormerC1 = TestWorldWisePlus


class TestWorldWisePP(TestWorldWisePlus):
    """WorldWise++: needs the token-grid cache (config ``grid_cache_root``)."""

    def _make_test_dataset(self):
        from lib.supervised.worldwise_pp.dataset import WorldAGGrid
        from wsgg_base import annot_dir_for
        return WorldAGGrid(
            phase="test",
            data_path=self._conf.data_path,
            mode=self._conf.mode,
            feature_model=getattr(self._conf, 'feature_model', 'dinov2b'),
            include_invisible=getattr(self._conf, 'include_invisible', True),
            max_objects=getattr(self._conf, 'max_objects', 64),
            annot_dir_name=annot_dir_for(self._conf, "test"),
            grid_cache_root=self._conf.grid_cache_root,
            allow_missing_grids=getattr(self._conf, 'grid_cache_allow_missing', False),
        )

    def init_model(self):
        from lib.supervised.worldwise_pp import WorldWisePP

        self._model = WorldWisePP(
            config=self._conf,
            num_object_classes=len(self._test_dataset.object_classes),
            attention_class_num=len(self._test_dataset.attention_relationships),
            spatial_class_num=len(self._test_dataset.spatial_relationships),
            contact_class_num=len(self._test_dataset.contacting_relationships),
        ).to(self._device)

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
            node_labels_seq=b.get("object_classes") if self._conf.mode == "predcls" else None,
            grid_dino_seq=b["grid_dino"],
            grid_pi3_seq=b["grid_pi3"],
            image_hw=b["image_hw"],
            bboxes_2d_seq=b.get("bboxes_2d"),
        )

        T = b["visual_features"].shape[0]
        if T > 0:
            return {
                "attention_distribution": pred["attention_distribution"][-1],
                "spatial_distribution": pred["spatial_distribution"][-1],
                "contacting_distribution": pred["contacting_distribution"][-1],
            }
        return None


METHOD_MAP = {
    # Baseline adaptations (FasterRCNN / ResNet50 backbone)
    "w_sttran": TestWSTTran,
    "w_sttran_pp": TestWSTTranPP,
    "w_dsgdetr": TestWDSGDetr,
    "w_dsgdetr_pp": TestWDSGDetrPP,
    # External baseline (beside the ladder): USG-Par-style relation decoding
    "w_usg": TestWUSG,
    # WorldWise (Dino backbones — ablation via config flags)
    "worldwise": TestWorldWise,
    "worldwise_plus": TestWorldWisePlus,
    "worldformer_c1": TestWorldWisePlus,   # legacy name
    "worldwise_pp": TestWorldWisePP,
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
