"""
WSGG Training Methods (Padded Tensor API)
==========================================

Per-method training classes. Each overrides:
  - init_model()               → create model
  - init_loss_fn()             → create loss module
  - is_temporal()              → sequential or frame-shuffled
  - process_train_video(batch) → forward + loss dict
  - process_test_video(batch)  → inference

The dataset returns a single dict with (T, N_max, ...) and (T, K_max, ...)
pre-padded tensors per video. No per-frame loops needed.

Usage:
  python train_wsgg_methods.py --config configs/methods/predcls/gl_stgn_predcls_dinov2b.yaml
"""

import logging

import torch

from wsgg_base import WSGGBase, load_wsgg_config
from train_wsgg_base import TrainWSGGBase

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

class TrainGLSTGN(TrainWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.gl_stgn.gl_stgn import GLSTGN

        self._model = GLSTGN(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldsgg.gl_stgn.loss import GLSTGNLoss
        self._loss_fn = GLSTGNLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],     # (T, N_max, D)
            corners_seq=b["corners"],                       # (T, N_max, 8, 3)
            valid_mask_seq=b["valid_mask"],                 # (T, N_max)
            visibility_mask_seq=b["visibility_mask"],       # (T, N_max)
            person_idx_seq=b["person_idx"],                 # (T, K_max)
            object_idx_seq=b["object_idx"],                 # (T, K_max)
            pair_valid=b["pair_valid"],                     # (T, K_max)
            camera_pose_seq=b.get("camera_poses"),
            union_features_seq=b.get("union_features"),
        )

        losses = self._loss_fn(
            predictions=pred,
            gt_attention=b["gt_attention"],
            gt_spatial=b["gt_spatial"],
            gt_contacting=b["gt_contacting"],
            pair_valid=b["pair_valid"],
            visibility_mask=b["visibility_mask"],
            person_idx=b["person_idx"],
            object_idx=b["object_idx"],
            valid_mask=b.get("valid_mask"),
            gt_node_labels=b.get("object_classes"),
        )

        return losses

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

class TrainAMWAE(TrainWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.amwae.amwae import AMWAE

        self._model = AMWAE(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldsgg.amwae.loss import AMWAELoss
        self._loss_fn = AMWAELoss(
            lambda_vlm=self._conf.lambda_vlm,
            lambda_recon=self._conf.lambda_reconstruction,
            lambda_recon_dominance=self._conf.lambda_recon_dominance,
            p_simulate_unseen=self._conf.p_simulate_unseen,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        b = _to_device(batch, self._device)

        pred = self._model.forward(
            visual_features_seq=b["visual_features"],
            corners_seq=b["corners"],
            valid_mask_seq=b["valid_mask"],
            visibility_mask_seq=b["visibility_mask"],
            person_idx_seq=b["person_idx"],
            object_idx_seq=b["object_idx"],
            pair_valid=b["pair_valid"],
            p_mask_visible=getattr(self._conf, 'p_mask_visible', 0.3),
            camera_pose_seq=b.get("camera_poses"),
        )

        losses = self._loss_fn(
            predictions=pred,
            gt_attention=b["gt_attention"],
            gt_spatial=b["gt_spatial"],
            gt_contacting=b["gt_contacting"],
            pair_valid=b["pair_valid"],
            visibility_mask=b["visibility_mask"],
            person_idx=b["person_idx"],
            object_idx=b["object_idx"],
            valid_mask=b.get("valid_mask"),
            corners=b.get("corners"),
            gt_node_labels=b.get("object_classes"),
        )

        return losses

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

class TrainLKSGNN(TrainWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.lks_buffer.lks_gnn import LKSGNN

        self._model = LKSGNN(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldsgg.lks_buffer.loss import LKSLoss
        self._loss_fn = LKSLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
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

        losses = self._loss_fn(
            predictions=pred,
            gt_attention=b["gt_attention"],
            gt_spatial=b["gt_spatial"],
            gt_contacting=b["gt_contacting"],
            pair_valid=b["pair_valid"],
            visibility_mask=b["visibility_mask"],
            person_idx=b["person_idx"],
            object_idx=b["object_idx"],
            valid_mask=b.get("valid_mask"),
            gt_node_labels=b.get("object_classes"),
        )

        return losses

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

class TrainAMWAEPP(TrainAMWAE):
    """AMWAE++ trainer — same batched API as AMWAE, only model differs."""

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.worldsgg.amwae.amwae_pp import AMWAEPP

        self._model = AMWAEPP(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldsgg.amwae.loss import AMWAELoss
        self._loss_fn = AMWAELoss(
            lambda_vlm=self._conf.lambda_vlm,
            lambda_recon=self._conf.lambda_reconstruction,
            lambda_recon_dominance=self._conf.lambda_recon_dominance,
            p_simulate_unseen=self._conf.p_simulate_unseen,
            label_smoothing=self._conf.label_smoothing_vlm,
            lambda_stability=self._conf.lambda_stability,
            mode=self._conf.mode,
        )


# ============================================================================
# W-STTran (World-adapted STTran — simplest baseline)
# ============================================================================

class TrainWSTTran(TrainWSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        from lib.supervised.baselines.w_sttran.w_sttran import WSTTran

        self._model = WSTTran(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.baselines.w_sttran.loss import WSTTranLoss
        self._loss_fn = WSTTranLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
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

        losses = self._loss_fn(
            predictions=pred,
            gt_attention=b["gt_attention"],
            gt_spatial=b["gt_spatial"],
            gt_contacting=b["gt_contacting"],
            pair_valid=b["pair_valid"],
            visibility_mask=b["visibility_mask"],
            person_idx=b["person_idx"],
            object_idx=b["object_idx"],
            valid_mask=b.get("valid_mask"),
            gt_node_labels=b.get("object_classes"),
        )

        return losses

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

class TrainWSTTranPP(TrainWSTTran):
    """W-STTran++ trainer — same API, only model class differs."""

    def init_model(self):
        from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP

        self._model = WSTTranPP(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# W-DSGDetr (World-adapted DSGDetr — with temporal object encoder)
# ============================================================================

class TrainWDSGDetr(TrainWSTTran):
    """W-DSGDetr trainer — same batched API as W-STTran, only model differs."""

    def init_model(self):
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr

        self._model = WDSGDetr(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.baselines.w_dsgdetr.loss import WDSGDetrLoss
        self._loss_fn = WDSGDetrLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )


# ============================================================================
# W-DSGDetr++ (Enhanced: + camera, motion, ego-motion)
# ============================================================================

class TrainWDSGDetrPP(TrainWDSGDetr):
    """W-DSGDetr++ trainer — same API, only model class differs."""

    def init_model(self):
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP

        self._model = WDSGDetrPP(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)


# ============================================================================
# WorldWise (MWAE-based — full proposed method with ablation support)
# ============================================================================

class TrainWorldWise(TrainAMWAE):
    """
    WorldWise trainer — MWAE-based with config-flag ablation support.

    Inherits from TrainAMWAE since WorldWise shares the same
    training/testing API (p_mask_visible, reconstruction loss, etc.).
    """

    def init_model(self):
        from lib.supervised.worldwise.worldwise import WorldWise

        self._model = WorldWise(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldwise.loss import WorldWiseLoss
        self._loss_fn = WorldWiseLoss(
            lambda_vlm=self._conf.lambda_vlm,
            lambda_recon=self._conf.lambda_reconstruction,
            lambda_recon_dominance=self._conf.lambda_recon_dominance,
            p_simulate_unseen=self._conf.p_simulate_unseen,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
        )


# ============================================================================
# Entry Point
# ============================================================================

METHOD_MAP = {
    # Existing WorldSGG methods
    "gl_stgn": TrainGLSTGN,
    "amwae": TrainAMWAE,
    "amwae_pp": TrainAMWAEPP,
    "lks_buffer": TrainLKSGNN,
    # Baseline adaptations (FasterRCNN backbone)
    "w_sttran": TrainWSTTran,
    "w_sttran_pp": TrainWSTTranPP,
    "w_dsgdetr": TrainWDSGDetr,
    "w_dsgdetr_pp": TrainWDSGDetrPP,
    # WorldWise (Dino backbones — same class, ablation via config flags)
    "worldwise": TrainWorldWise,
}


def main():
    conf = load_wsgg_config()
    method_name = conf.method_name

    if method_name not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {list(METHOD_MAP.keys())}")

    trainer_cls = METHOD_MAP[method_name]
    trainer = trainer_cls(conf)
    trainer.init_method_training()


if __name__ == "__main__":
    main()

