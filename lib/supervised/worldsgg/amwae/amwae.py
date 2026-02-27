"""
AMWAE: Associative Masked World Auto-Encoder
==============================================

Main model that wires together:
  1. GlobalStructuralEncoder — encodes wireframe into geometry tokens (reused from GL-STGN)
  2. ScaffoldTokenizer — binds visual evidence or [MASK] to geometry tokens
  3. EpisodicMemoryBank — FIFO queue of past visible tokens
  4. AssociativeRetriever — cross-attention retrieval to auto-complete masked tokens
  5. ContextualDiffusion — self-attention for context propagation
  6. NodePredictor + EdgePredictor — scene graph prediction (reused from GL-STGN)

The model initializes the graph TOP-DOWN from the complete wireframe scaffold,
then auto-completes missing visual evidence via associative memory retrieval.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional

from .scaffold_tokenizer import ScaffoldTokenizer
from .episodic_memory import EpisodicMemoryBank
from .associative_retriever import AssociativeRetriever
from .contextual_diffusion import ContextualDiffusion

# Reuse from GL-STGN
from lib.supervised.worldsgg.worldsgg_base import (
    GlobalStructuralEncoder, NodePredictor, EdgePredictor,
)


class AMWAE(nn.Module):
    """
    Associative Masked World Auto-Encoder.

    Processes frames sequentially, but each frame is an independent
    masked auto-encoding step (no RNN/GRU recurrence). The episodic
    memory bank provides temporal context through cross-attention.

    Args:
        config: AMWAEConfig with architecture hyperparameters.
        num_object_classes: Number of object categories.
        attention_class_num: Attention relationship classes.
        spatial_class_num: Spatial relationship classes.
        contact_class_num: Contacting relationship classes.
    """

    def __init__(
        self,
        config,
        num_object_classes: int = 37,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
    ):
        super().__init__()
        self.config = config

        # Module 1: Global Structural Encoder (reused from GL-STGN)
        self.structural_encoder = GlobalStructuralEncoder(
            d_struct=config.d_struct,
            d_hidden=config.d_struct // 2,
        )

        # Module 2: Scaffold Tokenizer (evidence binding + masking)
        self.scaffold_tokenizer = ScaffoldTokenizer(
            d_struct=config.d_struct,
            d_visual=config.d_visual,
            d_model=config.d_model,
            d_detector_roi=config.d_detector_roi,
        )

        # Module 3: Episodic Memory Bank (FIFO)
        memory_capacity = config.memory_bank_size * config.max_objects
        self.memory_bank = EpisodicMemoryBank(
            capacity=memory_capacity,
            d_model=config.d_model,
        )

        # Module 4: Associative Retriever (cross-attention)
        self.retriever = AssociativeRetriever(
            d_model=config.d_model,
            n_layers=config.n_cross_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 5: Contextual Diffusion (self-attention)
        self.diffusion = ContextualDiffusion(
            d_model=config.d_model,
            n_layers=config.n_self_attn_layers,
            n_heads=config.n_heads,
            d_feedforward=config.d_feedforward,
            dropout=config.dropout,
        )

        # Module 6: Prediction Heads (reused from GL-STGN)
        self.node_predictor = NodePredictor(
            d_memory=config.d_model,
            num_classes=num_object_classes,
        )

        self.edge_predictor = EdgePredictor(
            d_memory=config.d_model,
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
        p_mask_visible: float = 0.0,
    ) -> Dict[str, List]:
        """
        Process a sequence of frames via masked auto-encoding.

        Args:
            visual_features_seq: List of (N_t, d_detector_roi) — DINO features.
            corners_seq: List of (N_t, 8, 3) — world-frame 3D corners.
            valid_mask_seq: List of (N_t,) bool — valid objects.
            visibility_mask_seq: List of (N_t,) bool — visible in FOV.
            person_idx_seq: List of (K_t,) — person pair indices.
            object_idx_seq: List of (K_t,) — object pair indices.
            p_mask_visible: Training masking probability for visible objects.

        Returns:
            dict with per-frame lists:
                node_logits: List of (N_t, num_classes)
                attention_distribution: List of (K_t, 3)
                spatial_distribution: List of (K_t, 6)
                contacting_distribution: List of (K_t, 17)
                is_masked: List of (N_t,) bool — which tokens were masked
                original_visual: List of (N_t, d_visual) — for reconstruction loss
                retrieved_tokens: List of (N_t, d_model) — post-retrieval tokens
                attn_weights: List of (N_t, M) — cross-attn weights for InfoNCE
                memory_object_ids: List of (M,) — object IDs in memory
        """
        T = len(corners_seq)
        device = corners_seq[0].device

        outputs = {
            "node_logits": [],
            "attention_distribution": [],
            "spatial_distribution": [],
            "contacting_distribution": [],
            "is_masked": [],
            "original_visual": [],
            "retrieved_tokens": [],
            "attn_weights": [],
            "memory_object_ids": [],
            # Simulated-unseen clean fine-tuning outputs
            "simulated_unseen_predictions": [],
            "simulated_unseen_gt": [],
        }

        for t in range(T):
            corners_t = corners_seq[t]          # (N_t, 8, 3)
            valid_t = valid_mask_seq[t]           # (N_t,)
            vis_t = visibility_mask_seq[t]        # (N_t,)
            visual_t = visual_features_seq[t]     # (N_t, d_detector_roi)
            person_idx_t = person_idx_seq[t]
            object_idx_t = object_idx_seq[t]

            N_t = corners_t.shape[0]

            # --- Step 1: Encode wireframe geometry ---
            struct_tokens, _ = self.structural_encoder(
                corners_t.unsqueeze(0),
                valid_t.unsqueeze(0),
            )
            struct_tokens = struct_tokens.squeeze(0)  # (N_t, d_struct)

            # --- Step 2: Scaffold tokenization (bind evidence / apply mask) ---
            hybrid_tokens, is_masked_t, original_visual_t = self.scaffold_tokenizer(
                geometry_tokens=struct_tokens,
                visual_features=visual_t,
                visibility_mask=vis_t,
                valid_mask=valid_t,
                p_mask_visible=p_mask_visible if self.training else 0.0,
            )

            # --- Step 3: Store visible (unmasked) tokens in memory ---
            with torch.no_grad():
                unmasked_tokens = self.scaffold_tokenizer.fusion_proj(
                    torch.cat([struct_tokens, self.scaffold_tokenizer.visual_projector(visual_t)], dim=-1)
                )
            object_slot_ids = torch.arange(N_t, device=device)

            self.memory_bank.store(
                tokens=unmasked_tokens,
                object_slot_ids=object_slot_ids,
                visibility_mask=vis_t,
                valid_mask=valid_t,
                frame_id=t,
            )

            # --- Step 4: Associative retrieval from memory ---
            mem_tokens, mem_obj_ids, mem_frame_ids, mem_valid = self.memory_bank.retrieve()

            completed_tokens, attn_weights_t = self.retriever(
                tokens=hybrid_tokens,
                memory_tokens=mem_tokens,
                memory_valid=mem_valid,
            )

            # --- Step 5: Contextual diffusion (self-attention) ---
            enriched = self.diffusion(
                tokens=completed_tokens,
                corners=corners_t,
                valid_mask=valid_t,
            )

            # --- Step 6: Scene graph prediction ---
            node_logits = self.node_predictor(enriched)

            edge_out = self.edge_predictor(
                enriched_states=enriched,
                person_idx=person_idx_t,
                object_idx=object_idx_t,
                corners=corners_t,
            )

            # Collect outputs
            outputs["node_logits"].append(node_logits)
            outputs["attention_distribution"].append(edge_out["attention_distribution"])
            outputs["spatial_distribution"].append(edge_out["spatial_distribution"])
            outputs["contacting_distribution"].append(edge_out["contacting_distribution"])
            outputs["is_masked"].append(is_masked_t)
            outputs["original_visual"].append(original_visual_t)
            outputs["retrieved_tokens"].append(completed_tokens)
            outputs["attn_weights"].append(attn_weights_t)
            outputs["memory_object_ids"].append(mem_obj_ids)

            # --- Step 7: Simulated-Unseen (training only) ---
            # Artificially mask p_simulate_unseen of VISIBLE objects
            # Force retrieval → predict SG → loss against perfect manual GT
            if self.training and self.config.p_simulate_unseen > 0:
                p_sim = getattr(self.config, 'p_simulate_unseen', 0.25)
                n_visible = vis_t.sum().item()
                n_to_mask = max(1, int(n_visible * p_sim))

                if n_visible > 1:
                    visible_indices = torch.where(vis_t & valid_t)[0]
                    perm = torch.randperm(len(visible_indices), device=device)
                    sim_mask_indices = visible_indices[perm[:n_to_mask]]

                    # Create simulated visibility mask (hide selected visible objects)
                    sim_vis_t = vis_t.clone()
                    sim_vis_t[sim_mask_indices] = False

                    # Re-tokenize with simulated mask
                    sim_tokens, sim_is_masked, _ = self.scaffold_tokenizer(
                        geometry_tokens=struct_tokens,
                        visual_features=visual_t,
                        visibility_mask=sim_vis_t,
                        valid_mask=valid_t,
                        p_mask_visible=0.0,  # No additional random masking
                    )

                    # Retrieve from memory
                    sim_completed, _ = self.retriever(
                        tokens=sim_tokens,
                        memory_tokens=mem_tokens,
                        memory_valid=mem_valid,
                    )

                    # Diffuse
                    sim_enriched = self.diffusion(
                        tokens=sim_completed,
                        corners=corners_t,
                        valid_mask=valid_t,
                    )

                    # Predict — only for the simulated-masked object pairs
                    sim_edge_out = self.edge_predictor(
                        enriched_states=sim_enriched,
                        person_idx=person_idx_t,
                        object_idx=object_idx_t,
                        corners=corners_t,
                    )

                    outputs["simulated_unseen_predictions"].append(sim_edge_out)
                    outputs["simulated_unseen_gt"].append(None)  # GT filled by loss module
                else:
                    outputs["simulated_unseen_predictions"].append(None)
                    outputs["simulated_unseen_gt"].append(None)
            else:
                outputs["simulated_unseen_predictions"].append(None)
                outputs["simulated_unseen_gt"].append(None)

        return outputs

    def reset_memory(self):
        """Reset episodic memory bank (call between videos)."""
        self.memory_bank.reset()

