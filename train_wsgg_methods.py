"""
WSGG Training Methods
======================

Per-method training classes. Each overrides:
  - init_model()               → create model
  - init_loss_fn()             → create loss module
  - is_temporal()              → sequential or frame-shuffled
  - process_train_video(batch) → forward + loss dict
  - process_test_video(batch)  → inference

Usage:
  python train_wsgg_methods.py --config configs/methods/predcls/gl_stgn_predcls.yaml
  python train_wsgg_methods.py --config configs/methods/predcls/amwae_predcls.yaml
  python train_wsgg_methods.py --config configs/methods/predcls/amwae_pp_predcls.yaml
  python train_wsgg_methods.py --config configs/methods/predcls/lks_buffer_predcls.yaml
"""

import torch

from wsgg_base import WSGGBase, load_wsgg_config
from train_wsgg_base import TrainWSGGBase


# ============================================================================
# GL-STGN (Recurrent Temporal Memory)
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
            use_physics_veto=self._conf.use_physics_veto,
            physics_veto_thresh=self._conf.physics_veto_dist_thresh,
            lambda_smooth=self._conf.lambda_smooth,
            movement_thresh=self._conf.movement_thresh,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        tensors = batch
        total_losses = {}
        chunk_size = self._conf.bptt_chunk_size
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

            frame_losses = self._loss_fn(pred, frame, self._device)
            for k, v in frame_losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

            if (t + 1) % chunk_size == 0 and t < T - 1:
                loss = sum(total_losses.values())
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                total_losses = {}

        remaining = T % chunk_size or chunk_size
        return {k: v / remaining for k, v in total_losses.items()} if total_losses else {"loss": torch.tensor(0.0)}

    def process_test_video(self, batch) -> dict:
        tensors = batch
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
            use_physics_veto=self._conf.use_physics_veto,
            physics_veto_thresh=self._conf.physics_veto_dist_thresh,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        tensors = batch
        self._model.reset_memory()

        total_losses = {}
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

            frame_losses = self._loss_fn(pred, frame, self._device)
            for k, v in frame_losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

        return {k: v / T for k, v in total_losses.items()} if total_losses else {"loss": torch.tensor(0.0)}

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
            use_physics_veto=self._conf.use_physics_veto,
            physics_veto_thresh=self._conf.physics_veto_dist_thresh,
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        tensors = batch

        T = len(tensors) if isinstance(tensors, list) else 1
        frames = tensors if isinstance(tensors, list) else [tensors]

        pred = self._model.forward(
            visual_features_seq=[f["visual_features"].to(self._device) for f in frames],
            corners_seq=[f["corners"].to(self._device) for f in frames],
            valid_mask_seq=[f["valid_mask"].to(self._device) for f in frames],
            visibility_mask_seq=[f["visibility_mask"].to(self._device) for f in frames],
            person_idx_seq=[f["person_idx"].to(self._device) for f in frames],
            object_idx_seq=[f["object_idx"].to(self._device) for f in frames],
        )

        total_losses = {}
        for t in range(T):
            frame = frames[t]
            frame_pred = {
                "node_logits": pred["node_logits"][t],
                "attention_distribution": pred["attention_distribution"][t],
                "spatial_distribution": pred["spatial_distribution"][t],
                "contacting_distribution": pred["contacting_distribution"][t],
            }
            frame_losses = self._loss_fn(frame_pred, frame, self._device)
            for k, v in frame_losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

        return {k: v / T for k, v in total_losses.items()} if total_losses else {"loss": torch.tensor(0.0)}

    def process_test_video(self, batch) -> dict:
        tensors = batch

        T = len(tensors) if isinstance(tensors, list) else 1
        frames = tensors if isinstance(tensors, list) else [tensors]

        pred = self._model.forward(
            visual_features_seq=[f["visual_features"].to(self._device) for f in frames],
            corners_seq=[f["corners"].to(self._device) for f in frames],
            valid_mask_seq=[f["valid_mask"].to(self._device) for f in frames],
            visibility_mask_seq=[f["visibility_mask"].to(self._device) for f in frames],
            person_idx_seq=[f["person_idx"].to(self._device) for f in frames],
            object_idx_seq=[f["object_idx"].to(self._device) for f in frames],
        )

        if T > 0:
            return {
                "node_logits": pred["node_logits"][-1],
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
        tensors = batch
        self._model.reset_memory()

        total_losses = {}
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

            frame_losses = self._loss_fn(pred, frame, self._device)
            for k, v in frame_losses.items():
                total_losses[k] = total_losses.get(k, 0.0) + v

        return {k: v / T for k, v in total_losses.items()} if total_losses else {"loss": torch.tensor(0.0)}

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
