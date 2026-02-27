"""
GL-STGN: Global-Local Spatio-Temporal Graph Network
=====================================================

Main model that wires together:
  1. GlobalStructuralEncoder — encodes 3D bbox layout ("wireframe")
  2. PersistentWorldMemoryBank — GRU-based temporal memory for all objects
  3. RelationalGraphTransformer — spatial-aware context propagation
  4. NodePredictor + EdgePredictor — scene graph prediction heads

Processes a video's frames sequentially, maintaining the memory bank state,
and outputs per-frame scene graph predictions for ALL objects (visible + unseen).
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .memory_bank import PersistentWorldMemoryBank
from .graph_transformer import RelationalGraphTransformer
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor,
)


class GLSTGN(nn.Module):
    """
    Global-Local Spatio-Temporal Graph Network.

    Processes a sequence of frames from a video, maintaining persistent memory
    for all objects, and predicts the world scene graph at each timestep.

    The model expects pre-extracted DINO visual features (from the frozen detector)
    and world-frame 3D bounding boxes from the dataset.

    Args:
        config: GLSTGNConfig with architecture hyperparameters.
        num_object_classes: Number of object categories (incl. background).
        attention_class_num: Number of attention relationship classes.
        spatial_class_num: Number of spatial relationship classes.
        contact_class_num: Number of contacting relationship classes.
    """

    def __init__(
        self,
        config: GLSTGNConfig,
        num_object_classes: int = 37,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
    ):
        super().__init__()
        self.config = config
        self.num_object_classes = num_object_classes

        # Module 2: Global Structural Encoder
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 3: Persistent World Memory Bank
        self.memory_bank = PersistentWorldMemoryBank(
            d_memory=config.d_memory,
            d_visual=config.d_visual,
            d_struct=config.d_struct,
            d_detector_roi=config.d_detector_roi,
            n_heads=config.n_heads,
            dropout=config.dropout,
        )

        # Module 4: Relational Graph Transformer
        self.graph_transformer = RelationalGraphTransformer(
            d_model=config.d_memory,
            n_layers=config.n_graph_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_memory * 2,
            dropout=config.dropout,
        )

        # Module 5: Prediction Heads
        self.node_predictor = NodePredictor(
            d_memory=config.d_memory,
            num_classes=num_object_classes,
        )

        self.edge_predictor = EdgePredictor(
            d_memory=config.d_memory,
            attention_class_num=attention_class_num,
            spatial_class_num=spatial_class_num,
            contact_class_num=contact_class_num,
        )

    def forward(
        self,
        visual_features_seq: List[torch.Tensor],
        corners_seq: List[torch.Tensor],
        valid_mask_seq: List[torch.Tensor],
        visibility_mask_seq: List[torch.Tensor],
        person_idx_seq: List[torch.Tensor],
        object_idx_seq: List[torch.Tensor],
        p_mask_visual: float = 0.0,
    ) -> Dict[str, List]:
        """
        Process a sequence of frames and predict scene graphs.

        All lists are indexed by frame (timestep t = 0, 1, ..., T-1).

        Args:
            visual_features_seq: List of (N_t, d_detector_roi) tensors.
                Per-object DINO ROI features. Zeros for unseen/undetected objects.
            corners_seq: List of (N_t, 8, 3) tensors.
                World-frame 3D bbox corners for ALL objects (the "wireframe").
            valid_mask_seq: List of (N_t,) bool tensors.
                True for real objects, False for padding.
            visibility_mask_seq: List of (N_t,) bool tensors.
                True if the object is in the camera FOV at this frame.
            person_idx_seq: List of (K_t,) long tensors.
                Indices of person nodes in relationship pairs.
            object_idx_seq: List of (K_t,) long tensors.
                Indices of object nodes in relationship pairs.
            p_mask_visual: Visual feature masking probability (training only).

        Returns:
            dict with lists (one per frame):
                node_logits: List of (N_t, num_classes) — object class logits
                attention_distribution: List of (K_t, 3) — attention rel logits
                spatial_distribution: List of (K_t, 6) — spatial rel probs
                contacting_distribution: List of (K_t, 17) — contacting rel probs
                memory_states: List of (N_t, d_memory) — for analysis/debugging
        """
        T = len(corners_seq)
        device = corners_seq[0].device

        outputs = {
            "node_logits": [],
            "attention_distribution": [],
            "spatial_distribution": [],
            "contacting_distribution": [],
            "memory_states": [],
        }

        memory = None

        for t in range(T):
            corners_t = corners_seq[t]          # (N_t, 8, 3)
            valid_t = valid_mask_seq[t]           # (N_t,)
            vis_t = visibility_mask_seq[t]        # (N_t,)
            visual_t = visual_features_seq[t]     # (N_t, d_detector_roi)
            person_idx_t = person_idx_seq[t]      # (K_t,)
            object_idx_t = object_idx_seq[t]      # (K_t,)

            N_t = corners_t.shape[0]

            # --- Step 1: Encode global structure ---
            # Add batch dim for structural encoder
            struct_tokens, global_token = self.structural_encoder(
                corners_t.unsqueeze(0),  # (1, N_t, 8, 3)
                valid_t.unsqueeze(0),    # (1, N_t)
            )
            struct_tokens = struct_tokens.squeeze(0)  # (N_t, d_struct)
            global_token = global_token.squeeze(0)    # (d_struct,)

            # --- Step 2: Update memory bank ---
            if memory is None:
                # First frame: initialize memory
                memory = self.memory_bank.initialize_memory(
                    visual_features=visual_t,
                    struct_tokens=struct_tokens,
                    valid_mask=valid_t,
                )
            else:
                # Handle size changes (new objects appearing)
                N_prev = memory.shape[0]
                if N_t > N_prev:
                    # Pad memory with zeros for new objects
                    new_mem = torch.zeros(
                        N_t - N_prev, self.config.d_memory,
                        device=device,
                    )
                    memory = torch.cat([memory, new_mem], dim=0)
                elif N_t < N_prev:
                    # Truncate (shouldn't normally happen with proper tracking)
                    memory = memory[:N_t]

                # Update memory
                memory = self.memory_bank.step(
                    memory=memory,
                    visual_features=visual_t,
                    struct_tokens=struct_tokens,
                    global_struct_tokens=struct_tokens,  # all objects' tokens
                    visibility_mask=vis_t,
                    valid_mask=valid_t,
                    p_mask_visual=p_mask_visual if self.training else 0.0,
                )

            # --- Step 3: Relational reasoning via Graph Transformer ---
            enriched = self.graph_transformer(
                memory_states=memory,
                corners=corners_t,
                valid_mask=valid_t,
            )  # (N_t, d_memory)

            # --- Step 4: Memory Shielding ---
            # For visible pairs: enriched flows gradients to GRU (normal)
            # For unseen pairs: enriched.detach() → only MLP trains, GRU protected
            enriched_detached = enriched.detach()

            # --- Step 5: Predict scene graph ---
            # Node classification (uses full enriched)
            node_logits = self.node_predictor(enriched)  # (N_t, num_classes)

            # Edge prediction — use detached for unseen pairs
            # The loss module handles which edges are "unseen" based on visibility_mask
            # But we also produce detached edge predictions to enable the loss to
            # apply memory shielding selectively
            edge_out = self.edge_predictor(
                enriched_states=enriched,
                person_idx=person_idx_t,
                object_idx=object_idx_t,
                corners=corners_t,
            )

            # Also produce detached predictions for unseen edges
            edge_out_detached = self.edge_predictor(
                enriched_states=enriched_detached,
                person_idx=person_idx_t,
                object_idx=object_idx_t,
                corners=corners_t,
            )

            # Collect outputs
            outputs["node_logits"].append(node_logits)
            outputs["attention_distribution"].append(edge_out["attention_distribution"])
            outputs["spatial_distribution"].append(edge_out["spatial_distribution"])
            outputs["contacting_distribution"].append(edge_out["contacting_distribution"])
            outputs["memory_states"].append(memory.detach().clone())

            # Detached predictions for memory-shielded unseen loss
            if "attention_distribution_detached" not in outputs:
                outputs["attention_distribution_detached"] = []
                outputs["spatial_distribution_detached"] = []
                outputs["contacting_distribution_detached"] = []
            outputs["attention_distribution_detached"].append(edge_out_detached["attention_distribution"])
            outputs["spatial_distribution_detached"].append(edge_out_detached["spatial_distribution"])
            outputs["contacting_distribution_detached"].append(edge_out_detached["contacting_distribution"])

        return outputs

    def reset_memory(self):
        """Reset memory bank state (call between videos)."""
        # Memory is created fresh in forward() when memory=None
        pass

    def detach_memory(self):
        """Detach memory for BPTT truncation (called between chunks)."""
        # Memory is local to forward(), so this is a no-op.
        # BPTT truncation is handled by the training loop.
        pass

