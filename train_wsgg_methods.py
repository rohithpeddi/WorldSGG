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
            lambda_smooth=self._conf.lambda_smooth,
            movement_thresh=self._conf.movement_thresh,
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
            lambda_contrastive=self._conf.lambda_contrastive,
            p_simulate_unseen=self._conf.p_simulate_unseen,
            label_smoothing=self._conf.label_smoothing_vlm,
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

class TrainAMWAEPP(TrainWSGGBase):

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
            lambda_contrastive=self._conf.lambda_contrastive,
            p_simulate_unseen=self._conf.p_simulate_unseen,
            label_smoothing=self._conf.label_smoothing_vlm,
            lambda_stability=self._conf.lambda_stability,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        from tqdm import tqdm
        b = _to_device(batch, self._device)
        self._model.reset_memory()
        T = b["visual_features"].shape[0]

        # AMWAE++ processes frame-by-frame (recurrent memory)
        # but we still pass the full batch for loss after all frames
        all_preds = []
        for t in tqdm(range(T), desc="AMWAE++ frames", leave=False):
            frame_pred = self._model.forward_frame(
                visual_features=b["visual_features"][t],
                corners=b["corners"][t],
                valid_mask=b["valid_mask"][t],
                visibility_mask=b["visibility_mask"][t],
                person_idx=b["person_idx"][t],
                object_idx=b["object_idx"][t],
                pair_valid=b["pair_valid"][t],
            )
            all_preds.append(frame_pred)

        # Stack per-frame predictions into (T, ...) for batched loss
        stacked_pred = {}
        for key in ["attention_distribution", "spatial_distribution", "contacting_distribution"]:
            if key in all_preds[0]:
                stacked_pred[key] = torch.stack([p[key] for p in all_preds])
        # Pass through non-stackable items
        for key in all_preds[0]:
            if key not in stacked_pred:
                stacked_pred[key] = [p.get(key) for p in all_preds]

        losses = self._loss_fn(
            predictions=stacked_pred,
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
        from tqdm import tqdm
        b = _to_device(batch, self._device)
        self._model.reset_memory()
        T = b["visual_features"].shape[0]

        last_pred = None
        for t in tqdm(range(T), desc="AMWAE++ eval", leave=False):
            last_pred = self._model.forward_frame(
                visual_features=b["visual_features"][t],
                corners=b["corners"][t],
                valid_mask=b["valid_mask"][t],
                visibility_mask=b["visibility_mask"][t],
                person_idx=b["person_idx"][t],
                object_idx=b["object_idx"][t],
                pair_valid=b["pair_valid"][t],
            )

        return last_pred


# ============================================================================
# Entry Point
# ============================================================================

METHOD_MAP = {
    "gl_stgn": TrainGLSTGN,
    "amwae": TrainAMWAE,
    "amwae_pp": TrainAMWAEPP,
    "lks_buffer": TrainLKSGNN,
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
