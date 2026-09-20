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

Methods form a strict nested capability ladder (each a superset of the prior):
  - w_sttran      : W-STTran      GSE + spatial transformer + temporal-edge attn
  - w_sttran_pp   : W-STTran++    + ObjectSpatialEncoder
  - w_dsgdetr     : W-DSGDetr     + TemporalObjectEncoder
  - w_dsgdetr_pp  : W-DSGDetr++   + ObjectMotionEncoder
  - worldwise     : WorldWise     + ego-motion + MWAE + tail-aware logit adjustment

Usage:
  python train_wsgg_methods.py --config configs/methods/predcls/worldwise_predcls_dinov2b.yaml
"""

import logging

import torch

from wsgg_base import load_wsgg_config
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


def _predcls_labels(conf, b):
    """GT node labels for the text pathway — task inputs in predcls only."""
    return b.get("object_classes") if conf.mode == "predcls" else None


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
            node_labels_seq=_predcls_labels(self._conf, b),
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
            node_labels_seq=_predcls_labels(self._conf, b),
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
# W-USG (External baseline: USG-Par relation machinery on the WSGG substrate)
# ============================================================================

class TrainWUSG(TrainWSTTran):
    """W-USG trainer — same batched API as W-STTran; model and loss differ.

    Not a ladder tier: USG-Par-style relation decoder + text-centric
    alignment replace the SpatialGNN + TemporalEdgeAttention stack.
    """

    def init_model(self):
        from lib.supervised.baselines.w_usg.w_usg import WUSG

        self._model = WUSG(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.baselines.w_usg.loss import WUSGLoss
        self._loss_fn = WUSGLoss(
            lambda_vlm=self._conf.lambda_vlm,
            label_smoothing=self._conf.label_smoothing_vlm,
            mode=self._conf.mode,
            lambda_align=getattr(self._conf, 'lambda_align', 0.1),
        )


# ============================================================================
# WorldWise (MWAE-based — full proposed method with ablation support)
# ============================================================================

class TrainWorldWise(TrainWSGGBase):
    """
    WorldWise trainer — MWAE-based with config-flag ablation support.

    Differs from the baselines in that the forward pass takes a masking
    probability (p_mask_visible) and the loss consumes a reconstruction
    target (corners), matching the masked world auto-encoder objective.
    """

    def __init__(self, conf):
        super().__init__(conf)

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
            # Tail-aware logit adjustment (WorldWise-exclusive)
            use_logit_adjustment=getattr(self._conf, 'use_logit_adjustment', False),
            logit_adjustment_tau=getattr(self._conf, 'logit_adjustment_tau', 1.0),
            predicate_priors_path=getattr(self._conf, 'predicate_priors_path', None),
            data_path=self._conf.data_path,
            # Plugin I-5: per-pair VLM confidence weighting (inert until the
            # dataset provides a vlm_confidence tensor)
            use_confidence_weighted_vlm=getattr(self._conf, 'use_confidence_weighted_vlm', False),
            # Plugin I-8: attractor stability term for energy refinement
            lambda_stability=getattr(self._conf, 'lambda_stability', 0.0),
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
            union_features_seq=b.get("union_features"),
            node_labels_seq=_predcls_labels(self._conf, b),
            # Plugin I-6: GT contacting labels drive the training-time EMA
            # prototype updates (unused unless use_predicate_prototypes)
            gt_contacting_seq=b.get("gt_contacting"),
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
            # Plugin I-5: per-pair VLM label confidence, when the dataset has it
            vlm_confidence=b.get("vlm_confidence"),
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
            node_labels_seq=_predcls_labels(self._conf, b),
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

class TrainWorldWisePlus(TrainWorldWise):
    """WorldWise+ (formerly "WorldFormer C1"): WorldWise whose appearance projectors
    consume cached foundation tokens (lib/supervised/worldwise_plus). Training loop
    and loss identical to WorldWise."""

    def init_model(self):
        from lib.supervised.worldwise_plus import WorldWisePlus

        self._model = WorldWisePlus(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)


TrainWorldFormerC1 = TrainWorldWisePlus  # backward-compatible name


def _pp_grid_kwargs(b):
    """Extra forward inputs of WorldWise++ (from WorldAGGrid items)."""
    return dict(
        grid_dino_seq=b["grid_dino"],
        grid_pi3_seq=b["grid_pi3"],
        image_hw=b["image_hw"],
        bboxes_2d_seq=b.get("bboxes_2d"),
    )


class TrainWorldWisePP(TrainWorldWisePlus):
    """WorldWise++ (lib/supervised/worldwise_pp): entity decoder over fused DINOv3 /
    Pi3 token grids with joint free-query detection and an EGTR-style relation
    readout. Needs the grid cache (config ``grid_cache_root``)."""

    def _make_dataset(self, phase: str):
        from lib.supervised.worldwise_pp.dataset import WorldAGGrid
        from wsgg_base import annot_dir_for
        return WorldAGGrid(
            phase=phase,
            data_path=self._conf.data_path,
            mode=self._conf.mode,
            feature_model=getattr(self._conf, 'feature_model', 'dinov2b'),
            include_invisible=getattr(self._conf, 'include_invisible', True),
            max_objects=getattr(self._conf, 'max_objects', 64),
            annot_dir_name=annot_dir_for(self._conf, phase),
            grid_cache_root=self._conf.grid_cache_root,
            allow_missing_grids=getattr(self._conf, 'grid_cache_allow_missing', False),
        )

    def init_model(self):
        from lib.supervised.worldwise_pp import WorldWisePP

        self._model = WorldWisePP(
            config=self._conf,
            num_object_classes=len(self._object_classes),
            attention_class_num=len(self._train_dataset.attention_relationships),
            spatial_class_num=len(self._train_dataset.spatial_relationships),
            contact_class_num=len(self._train_dataset.contacting_relationships),
        ).to(self._device)

    def init_loss_fn(self):
        from lib.supervised.worldwise_pp import WorldWisePPLoss
        c = self._conf
        self._loss_fn = WorldWisePPLoss(
            num_object_classes=len(self._object_classes),
            lambda_det=getattr(c, 'lambda_det', 1.0),
            lambda_slot_box=getattr(c, 'lambda_slot_box', 1.0),
            det_one_to_many_iou=getattr(c, 'det_one_to_many_iou', 0.5),
            no_object_weight=getattr(c, 'det_no_object_weight', 0.1),
            # WorldWiseLoss settings (identical to TrainWorldWise.init_loss_fn)
            lambda_vlm=c.lambda_vlm,
            lambda_recon=c.lambda_reconstruction,
            lambda_recon_dominance=c.lambda_recon_dominance,
            p_simulate_unseen=c.p_simulate_unseen,
            label_smoothing=c.label_smoothing_vlm,
            mode=c.mode,
            use_logit_adjustment=getattr(c, 'use_logit_adjustment', False),
            logit_adjustment_tau=getattr(c, 'logit_adjustment_tau', 1.0),
            predicate_priors_path=getattr(c, 'predicate_priors_path', None),
            data_path=c.data_path,
            use_confidence_weighted_vlm=getattr(c, 'use_confidence_weighted_vlm', False),
            lambda_stability=getattr(c, 'lambda_stability', 0.0),
        ).to(self._device)

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
            node_labels_seq=_predcls_labels(self._conf, b),
            gt_contacting_seq=b.get("gt_contacting"),
            **_pp_grid_kwargs(b),
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
            vlm_confidence=b.get("vlm_confidence"),
            # joint detection targets
            gt_bboxes_2d=b.get("gt_bboxes_2d"),
            gt_corners=b.get("gt_corners"),
            camera_poses=b.get("camera_poses"),
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
            node_labels_seq=_predcls_labels(self._conf, b),
            **_pp_grid_kwargs(b),
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
    # Nested baseline ladder (any backbone via feature_model)
    "w_sttran": TrainWSTTran,
    "w_sttran_pp": TrainWSTTranPP,
    "w_dsgdetr": TrainWDSGDetr,
    "w_dsgdetr_pp": TrainWDSGDetrPP,
    # External baseline (beside the ladder): USG-Par-style relation decoding
    "w_usg": TrainWUSG,
    # WorldWise (full proposed method — MWAE + tail-aware loss)
    "worldwise": TrainWorldWise,
    # WorldWise+: WorldWise over cached DINOv3 / Pi3 tokens (gated fusion)
    "worldwise_plus": TrainWorldWisePlus,
    "worldformer_c1": TrainWorldWisePlus,   # legacy name (running jobs / old configs)
    # WorldWise++: entity decoder over fused token grids + joint detection
    "worldwise_pp": TrainWorldWisePP,
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
