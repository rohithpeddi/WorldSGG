"""
WSGG Training Methods
======================

Thin per-method training classes. Each overrides:
  - init_model()               → create model
  - init_loss_fn()             → create loss module
  - is_temporal()              → sequential or frame-shuffled
  - process_train_video(batch) → forward + loss dict
  - process_test_video(batch)  → inference

Usage:
  python train_wsgg_methods.py --config configs/wsgg.yaml --method_name gl_stgn
"""

import torch

from wsgg_base import WSGGBase, load_wsgg_config
from train_wsgg_base import TrainWSGGBase


# ============================================================================
# GL-STGN (Method 1: Recurrent Temporal Memory)
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
            lambda_vlm=getattr(self._conf, 'lambda_vlm', 0.2),
            label_smoothing=getattr(self._conf, 'label_smoothing_vlm', 0.2),
            use_physics_veto=getattr(self._conf, 'use_physics_veto', True),
            physics_veto_thresh=getattr(self._conf, 'physics_veto_dist_thresh', 2.0),
            lambda_smooth=getattr(self._conf, 'lambda_smooth', 0.1),
            movement_thresh=getattr(self._conf, 'movement_thresh', 0.3),
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        tensors = batch
        self._model.reset_memory(self._device)

        total_losses = {}
        chunk_size = getattr(self._conf, 'bptt_chunk_size', 5)
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

            # BPTT: truncate gradients every chunk_size frames
            if (t + 1) % chunk_size == 0 and t < T - 1:
                loss = sum(total_losses.values())
                loss.backward()
                self._optimizer.step()
                self._optimizer.zero_grad()
                self._model.detach_memory()
                total_losses = {}

        remaining = T % chunk_size or chunk_size
        return {k: v / remaining for k, v in total_losses.items()} if total_losses else {"loss": torch.tensor(0.0)}

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
# AMWAE (Method 2: Associative Masked World Auto-Encoder)
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
            lambda_vlm=getattr(self._conf, 'lambda_vlm', 0.2),
            lambda_recon=getattr(self._conf, 'lambda_reconstruction', 1.0),
            lambda_recon_dominance=getattr(self._conf, 'lambda_recon_dominance', 5.0),
            lambda_contrastive=getattr(self._conf, 'lambda_contrastive', 0.5),
            p_simulate_unseen=getattr(self._conf, 'p_simulate_unseen', 0.25),
            label_smoothing=getattr(self._conf, 'label_smoothing_vlm', 0.2),
            use_physics_veto=getattr(self._conf, 'use_physics_veto', True),
            physics_veto_thresh=getattr(self._conf, 'physics_veto_dist_thresh', 2.0),
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
# LKS Buffer (Baseline 1: Passive Memory)
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
            lambda_vlm=getattr(self._conf, 'lambda_vlm', 0.2),
            label_smoothing=getattr(self._conf, 'label_smoothing_vlm', 0.2),
            use_physics_veto=getattr(self._conf, 'use_physics_veto', True),
            physics_veto_thresh=getattr(self._conf, 'physics_veto_dist_thresh', 2.0),
        )

    def is_temporal(self) -> bool:
        return True

    def process_train_video(self, batch) -> dict:
        tensors = batch
        self._model.reset_memory(self._device)

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
# Entry Point
# ============================================================================

METHOD_MAP = {
    "gl_stgn": TrainGLSTGN,
    "amwae": TrainAMWAE,
    "lks_buffer": TrainLKSGNN,
}


def main():
    conf = load_wsgg_config()
    method_name = getattr(conf, 'method_name', 'gl_stgn')

    if method_name not in METHOD_MAP:
        raise ValueError(f"Unknown method: {method_name}. Choose from: {list(METHOD_MAP.keys())}")

    trainer_cls = METHOD_MAP[method_name]
    trainer = trainer_cls(conf)
    trainer.init_method_training()


if __name__ == "__main__":
    main()
