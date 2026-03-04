"""
WSGG Testing Base
==================

Common testing loop for all WSGG methods.
Analogous to test_sgg_base.py in the SGG pipeline.

Handles:
  - Dataset loading (WorldAG test)
  - Testing loop
  - Stratified evaluation (Vis-Vis, Vis-Unseen, Unseen-Unseen)
  - init_method_evaluation() orchestration
"""

import logging
import time
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsgg_base import WSGGBase

logger = logging.getLogger(__name__)


class TestWSGGBase(WSGGBase):
    """
    Common testing loop for all WSGG methods.

    Subclasses override:
      - init_model()                → instantiate method-specific model
      - is_temporal()               → True = sequential, False = frame-shuffled
      - process_test_video(batch)   → inference for one video/frame
    """

    def __init__(self, conf):
        super().__init__(conf)
        self._dataloader_test = None

        # Stratified evaluation accumulators
        self._stratified_results = {
            "vis_vis": {"correct": 0, "total": 0},
            "vis_unseen": {"correct": 0, "total": 0},
            "unseen_unseen": {"correct": 0, "total": 0},
        }

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    def _init_dataset(self):
        """Initialize WorldAG test dataset and dataloader."""
        from dataloader.world_ag_dataset import WorldAG, world_collate_fn

        logger.info("Initializing WorldAG test dataset...")

        self._test_dataset = WorldAG(
            phase="test",
            data_path=self._conf.data_path,
            mode=self._conf.mode,
            feature_model=getattr(self._conf, 'feature_model', 'dinov2b'),
            include_invisible=getattr(self._conf, 'include_invisible', True),
            max_objects=getattr(self._conf, 'max_objects', 64),
        )

        self._dataloader_test = DataLoader(
            self._test_dataset, batch_size=1, shuffle=False, num_workers=0,
            collate_fn=world_collate_fn,
        )

        logger.info(f"  Test: {len(self._test_dataset)} items")

    # ------------------------------------------------------------------
    # Testing Loop
    # ------------------------------------------------------------------
    def _test_model(self):
        """Main testing loop with stratified evaluation."""
        from lib.supervised.evaluation_recall import evaluate_wsgg_video

        start_time = time.time()
        logger.info('-------------------------------------------------------------------')
        logger.info(f"Testing: {self._conf.method_name} | Mode: {self._conf.mode}")
        logger.info('-------------------------------------------------------------------')

        test_iter = iter(self._dataloader_test)
        self._model.eval()

        with torch.no_grad():
            for num_video in tqdm(range(len(self._dataloader_test)), desc="Testing"):
                batch = next(test_iter)

                # Method-specific inference
                prediction = self.process_test_video(batch)

                # Standard evaluation via WSGG adapter
                if prediction is not None and self._evaluator is not None:
                    T = batch["T"]
                    last = T - 1
                    pred_pkl = {
                        "video_id": batch["video_id"],
                        # Model predictions (last frame)
                        "attention_distribution": prediction["attention_distribution"].cpu().numpy(),
                        "spatial_distribution": prediction["spatial_distribution"].cpu().numpy(),
                        "contacting_distribution": prediction["contacting_distribution"].cpu().numpy(),
                        # GT labels (last frame)
                        "gt_attention": batch["gt_attention"][last].numpy(),
                        "gt_spatial": batch["gt_spatial"][last].numpy(),
                        "gt_contacting": batch["gt_contacting"][last].numpy(),
                        # Pair metadata (last frame)
                        "pair_valid": batch["pair_valid"][last].numpy(),
                        "person_idx": batch["person_idx"][last].numpy(),
                        "object_idx": batch["object_idx"][last].numpy(),
                        # Object metadata (last frame)
                        "object_classes": batch["object_classes"][last].numpy(),
                        "bboxes_2d": batch["bboxes_2d"][last].numpy(),
                        "valid_mask": batch["valid_mask"][last].numpy(),
                    }
                    # SGDet: add detector-predicted labels and corners
                    if self._conf.mode == "sgdet":
                        pred_pkl["pred_labels"] = batch["object_classes"][last].numpy()
                        pred_pkl["pred_scores"] = np.ones(
                            batch["object_classes"][last].shape[0], dtype=np.float32)
                        # Real GT annotation boxes/corners for proper IoU evaluation
                        pred_pkl["gt_bboxes_2d"] = batch["gt_bboxes_2d"][last].numpy()
                        pred_pkl["gt_corners"] = batch["gt_corners"][last].numpy()

                        # Transform detector 3D corners: camera space → FINAL space
                        corners_raw = batch.get("corners")
                        cam_pose = batch.get("camera_poses")
                        if corners_raw is not None:
                            corners_cam = corners_raw[last].numpy()  # (N_max, 8, 3)
                            if cam_pose is not None:
                                T = cam_pose[last].numpy()  # (4, 4) cam-to-FINAL
                                R, t = T[:3, :3], T[:3, 3]
                                corners_cam = np.einsum('ij,nkj->nki', R, corners_cam) + t
                            pred_pkl["bboxes_3d"] = corners_cam

                    # Feed both evaluators with the same predictions
                    evaluate_wsgg_video(
                        pred_pkl, self._evaluator,
                        mode=self._conf.mode, verbose=False,
                    )
                    evaluate_wsgg_video(
                        pred_pkl, self._evaluator_nc,
                        mode=self._conf.mode, verbose=False,
                    )

                # Stratified evaluation
                if prediction is not None:
                    self._update_stratified_metrics(batch, prediction)

        elapsed = time.time() - start_time
        logger.info(f"\nTesting complete: {elapsed:.1f}s")
        logger.info('-------------------------------------------------------------------')

    def _update_stratified_metrics(self, batch, prediction):
        """
        Update stratified evaluation buckets based on visibility.

        For each edge prediction, classify into:
          - Vis-Vis: both subject and object were visible
          - Vis-Unseen: one visible, one unseen
          - Unseen-Unseen: both unseen
        """
        # Subclasses can override this with actual visibility info
        pass

    def _print_stratified_results(self):
        """Print stratified evaluation results."""
        logger.info("\n===== Stratified Evaluation Results =====")
        for bucket_name, stats in self._stratified_results.items():
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0.0
            logger.info(f"  {bucket_name}: {correct}/{total} = {acc:.4f}")
        logger.info("=========================================\n")

    # ------------------------------------------------------------------
    # Evaluation Publishing
    # ------------------------------------------------------------------
    def _publish_evaluation_results(self):
        """Publish evaluation stats and stratified results."""
        if self._evaluator is not None:
            logger.info("--- With-Constraint ---")
            self._evaluator.print_stats()
            logger.info("--- No-Constraint ---")
            self._evaluator_nc.print_stats()

        self._print_stratified_results()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def init_method_evaluation(self):
        """Full initialization → testing pipeline."""
        # 0. Config
        self._init_config(is_train=False)

        # 1. Dataset
        self._init_dataset()

        # 2. Evaluators
        self._init_evaluators()

        # 3. Model
        self.init_model()
        self._load_checkpoint()

        # 4. Test
        self._test_model()

        # 5. Results
        self._publish_evaluation_results()

    # ------------------------------------------------------------------
    # Abstract Methods
    # ------------------------------------------------------------------
    @abstractmethod
    def is_temporal(self) -> bool:
        """True for sequential, False for frame-shuffled."""
        pass

    @abstractmethod
    def process_test_video(self, batch) -> dict:
        """Method-specific inference."""
        pass
