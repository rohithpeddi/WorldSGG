"""
World Scene Graph Prediction Heads
===================================

Node predictor (object class) and Edge predictor (relationship triplets)
for the GL-STGN architecture. Outputs follow the Action Genome taxonomy:
  - Attention: 3 classes (softmax / CE)
  - Spatial: 6 classes (sigmoid / BCE)
  - Contacting: 17 classes (sigmoid / BCE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodePredictor(nn.Module):
    """
    Predicts object class from memory state.

    Input:  (N, d_memory)
    Output: (N, num_classes) logits
    """

    def __init__(self, d_memory: int, num_classes: int, d_hidden: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_memory, d_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, num_classes),
        )

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory_states: (N, d_memory)
        Returns:
            class_logits: (N, num_classes)
        """
        return self.mlp(memory_states)


class EdgePredictor(nn.Module):
    """
    Predicts relationship distributions for person-object pairs.

    For each pair (person_i, object_j):
      - Concatenate enriched representations + relative spatial features
      - MLP → (attention_logits, spatial_logits, contacting_logits)

    Args:
        d_memory: Memory / enriched representation dimension.
        attention_class_num: Number of attention relationship classes (3).
        spatial_class_num: Number of spatial relationship classes (6).
        contact_class_num: Number of contacting relationship classes (17).
        d_hidden: Hidden dimension in prediction MLPs.
    """

    def __init__(
        self,
        d_memory: int,
        attention_class_num: int = 3,
        spatial_class_num: int = 6,
        contact_class_num: int = 17,
        d_hidden: int = 256,
    ):
        super().__init__()
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num

        # Spatial geometry encoder for pair: relative center(3) + dist(1) + vol_ratio(1) = 5
        self.spatial_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, d_hidden),
        )

        # Input: person_repr + object_repr + spatial = d_memory * 2 + d_hidden
        pair_input_dim = d_memory * 2 + d_hidden

        # Shared pair feature extractor
        self.pair_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.LayerNorm(d_hidden),
            nn.Dropout(0.1),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(inplace=True),
        )

        # Relationship classifiers
        self.a_rel_compress = nn.Linear(d_hidden, attention_class_num)
        self.s_rel_compress = nn.Linear(d_hidden, spatial_class_num)
        self.c_rel_compress = nn.Linear(d_hidden, contact_class_num)

    def compute_pair_spatial(
        self,
        person_corners: torch.Tensor,
        object_corners: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute spatial features for person-object pairs.

        Args:
            person_corners: (K, 8, 3) person 3D bbox corners.
            object_corners: (K, 8, 3) object 3D bbox corners.

        Returns:
            spatial_feats: (K, 5) — [rel_center(3), distance(1), log_vol_ratio(1)]
        """
        # Centers
        p_center = person_corners.mean(dim=1)  # (K, 3)
        o_center = object_corners.mean(dim=1)  # (K, 3)

        # Relative center
        rel_center = o_center - p_center  # (K, 3)

        # Distance
        dist = rel_center.norm(dim=-1, keepdim=True)  # (K, 1)

        # Volume ratio
        p_mins, _ = person_corners.min(dim=1)
        p_maxs, _ = person_corners.max(dim=1)
        p_vol = (p_maxs - p_mins).clamp(min=1e-6).prod(dim=-1)  # (K,)

        o_mins, _ = object_corners.min(dim=1)
        o_maxs, _ = object_corners.max(dim=1)
        o_vol = (o_maxs - o_mins).clamp(min=1e-6).prod(dim=-1)  # (K,)

        log_vol_ratio = (torch.log(o_vol + 1e-6) - torch.log(p_vol + 1e-6)).unsqueeze(-1)  # (K, 1)

        return torch.cat([rel_center, dist, log_vol_ratio], dim=-1)  # (K, 5)

    def forward(
        self,
        enriched_states: torch.Tensor,
        person_idx: torch.Tensor,
        object_idx: torch.Tensor,
        corners: torch.Tensor,
    ) -> dict:
        """
        Args:
            enriched_states: (N, d_memory) context-enriched representations.
            person_idx: (K,) indices of person nodes in each pair.
            object_idx: (K,) indices of object nodes in each pair.
            corners: (N, 8, 3) 3D bbox corners for spatial features.

        Returns:
            dict with:
                attention_distribution: (K, 3) — raw logits (softmax applied externally)
                spatial_distribution: (K, 6) — sigmoid probabilities
                contacting_distribution: (K, 17) — sigmoid probabilities
        """
        K = person_idx.shape[0]
        if K == 0:
            device = enriched_states.device
            return {
                "attention_distribution": torch.zeros(0, self.attention_class_num, device=device),
                "spatial_distribution": torch.zeros(0, self.spatial_class_num, device=device),
                "contacting_distribution": torch.zeros(0, self.contact_class_num, device=device),
            }

        # Gather person and object representations
        person_repr = enriched_states[person_idx]  # (K, d_memory)
        object_repr = enriched_states[object_idx]  # (K, d_memory)

        # Compute spatial features
        person_corners = corners[person_idx]  # (K, 8, 3)
        object_corners = corners[object_idx]  # (K, 8, 3)
        spatial_feats = self.compute_pair_spatial(person_corners, object_corners)  # (K, 5)
        spatial_encoded = self.spatial_encoder(spatial_feats)  # (K, d_hidden)

        # Concatenate and predict
        pair_input = torch.cat([person_repr, object_repr, spatial_encoded], dim=-1)
        pair_features = self.pair_mlp(pair_input)  # (K, d_hidden)

        # Relationship distributions
        attention_logits = self.a_rel_compress(pair_features)  # (K, 3)
        spatial_logits = self.s_rel_compress(pair_features)  # (K, 6)
        contacting_logits = self.c_rel_compress(pair_features)  # (K, 17)

        return {
            "attention_distribution": attention_logits,  # CE loss: no activation here
            "spatial_distribution": torch.sigmoid(spatial_logits),  # BCE loss
            "contacting_distribution": torch.sigmoid(contacting_logits),  # BCE loss
        }
