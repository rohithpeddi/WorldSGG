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

import copy
import time
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from wsgg_base import WSGGBase


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
        from dataloader.supervised.generation.world_ag.world_ag_dataset import WorldAG

        print("Initializing WorldAG test dataset...")

        self._test_dataset = WorldAG(
            phase="test",
            data_path=self._conf.data_path,
            datasize=getattr(self._conf, 'datasize', 'large'),
            world_sg_dir=getattr(self._conf, 'world_sg_dir', ''),
            include_invisible=getattr(self._conf, 'include_invisible', True),
        )

        self._dataloader_test = DataLoader(
            self._test_dataset, batch_size=1, shuffle=False, num_workers=0,
        )

        print(f"  Test: {len(self._test_dataset)} items")

    # ------------------------------------------------------------------
    # Testing Loop
    # ------------------------------------------------------------------
    def _test_model(self):
        """Main testing loop with stratified evaluation."""
        start_time = time.time()
        print('-------------------------------------------------------------------')
        print(f"Testing: {self._conf.method_name} | Mode: {self._conf.mode}")
        print('-------------------------------------------------------------------')

        test_iter = iter(self._dataloader_test)
        self._model.eval()

        with torch.no_grad():
            for num_video in tqdm(range(len(self._dataloader_test)), desc="Testing"):
                batch = next(test_iter)

                # Method-specific inference
                prediction = self.process_test_video(batch)

                # Standard evaluation
                if prediction is not None and self._evaluator is not None:
                    self._evaluator.evaluate_scene_graph(batch, prediction)

                # Stratified evaluation
                if prediction is not None:
                    self._update_stratified_metrics(batch, prediction)

        elapsed = time.time() - start_time
        print(f"\nTesting complete: {elapsed:.1f}s")
        print('-------------------------------------------------------------------')

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
        print("\n===== Stratified Evaluation Results =====")
        for bucket_name, stats in self._stratified_results.items():
            total = stats["total"]
            correct = stats["correct"]
            acc = correct / total if total > 0 else 0.0
            print(f"  {bucket_name}: {correct}/{total} = {acc:.4f}")
        print("=========================================\n")

    # ------------------------------------------------------------------
    # Evaluation Publishing
    # ------------------------------------------------------------------
    def _publish_evaluation_results(self):
        """Publish evaluation stats and stratified results."""
        if self._evaluator is not None:
            self._evaluator.print_stats()

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
