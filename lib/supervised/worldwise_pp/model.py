"""
WorldWise++ — entity-query world scene graphs over frozen token grids.

Lineage: WorldWise -> WorldWise+ (token swap) -> WorldWise++ (this file).

WorldWise++ keeps everything WorldWise+ has *before* and *after* the object-token
core unchanged — geometry / camera / ego / motion encoders, the ScaffoldTokenizer
(DINOv3 Tier-1 ROI token through the GatedFusionProjector, [MASK] for invisible
slots + artificial masking, EMA reconstruction target), the visibility embedding,
NodePredictor, reconstruction head, RelationshipPredictor and TemporalEdgeAttention —
and returns the same output dict as WorldWise (plus a ``det`` sub-dict).  What changes:

1. Grid memory.  The DINOv3 (L24n, PCA-256) and Pi3 (G14, PCA-256) token grids of
   each frame — already on the same Pi3 patch grid — are fused per cell by
   ``TokenGridFusion`` (per-stream LayerNorm->Linear, 2-way softmax gate, LayerNorm,
   2-D sine positional encoding) into a memory of shape (T, Hp*Wp, d_model).
2. Entity decoder replaces AssociativeRetriever + InterObjectTransformer.  Per
   frame the queries are ``n_free_queries`` learnable free (DETR) queries plus the N
   world-slot tokens from the scaffold tokenizer (slot positional embedding =
   Linear(structural token)).  Each of ``n_decoder_layers`` layers runs
   (a) temporal self-attention per slot across the T frames (slots only, learned
       frame-position embedding) — an invisible / masked slot reads its own visible
       frames, the retriever's job;
   (b) spatial self-attention over all queries within a frame, weights kept;
   (c) cross-attention to the frame's grid memory;
   (d) FFN — pre-norm residuals, padded slots zeroed after every layer.
3. Heads.  Slots -> visibility embedding -> NodePredictor / reconstruction_proj /
   relation pathway, plus a zero-initialised box + corner refinement head (residual
   on the input 2-D box and 3-D corners).  Free queries -> class (C+1, no-object),
   2-D box (cxcywh, sigmoid), 3-D OBB corners via the factorised parameterisation of
   the monocular detector (``_compute_3d_corners``: dims / yaw / depth / offset).
4. EGTR-style relation readout.  For every (person, object) pair the spatial
   self-attention weights in both directions from every decoder layer
   (2 * L * heads values) and q_p * q_o are projected by a small MLP to d_rel // 4
   and fed to ``RelationshipPredictor.batched_form_and_attend`` in place of the
   union-box features (``union_proj`` is replaced accordingly; the PKL union tokens
   are not used).

Config keys (configs/methods/*/worldwise_pp_*.yaml):
    d_grid 256 | n_free_queries 30 | n_decoder_layers 4 | n_decoder_heads 8
    d_decoder_ff (default 4 * d_model) | max_frames 1024 | grid_cache_root
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.supervised.worldwise_plus.model import WorldWisePlus


def _compute_3d_corners(dims, rot_sin_cos, depth, center_offset, bbox_2d, focal_lengths, principal_point):
    """Reconstruct 8 corners from factorised parameters via pinhole back-projection.

    Verbatim copy of ``lib.detector.monocular3d.models.dino_mono_3d._compute_3d_corners``
    (the detector module imports torchvision / wandb at import time, which the training
    and scoring environments do not all have)."""
    cx_2d = (bbox_2d[:, 0] + bbox_2d[:, 2]) / 2.0
    cy_2d = (bbox_2d[:, 1] + bbox_2d[:, 3]) / 2.0
    u_final = cx_2d + center_offset[:, 0]
    v_final = cy_2d + center_offset[:, 1]

    l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
    x_corners = torch.stack([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2], dim=1)
    y_corners = torch.stack([w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2], dim=1)
    z_corners = torch.stack([h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)

    sin, cos = rot_sin_cos[:, 0:1], rot_sin_cos[:, 1:2]
    x_rot = x_corners * cos - y_corners * sin
    y_rot = x_corners * sin + y_corners * cos

    z_c = depth
    px, py = principal_point[:, 0:1], principal_point[:, 1:2]
    fx, fy = focal_lengths[:, 0:1], focal_lengths[:, 1:2]
    fx = fx.clamp(min=1.0)
    fy = fy.clamp(min=1.0)
    # Divide by focal length first to keep intermediates within float16 range.
    x_c = ((u_final.unsqueeze(1) - px) / fx) * z_c
    y_c = ((v_final.unsqueeze(1) - py) / fy) * z_c

    return torch.stack([x_rot + x_c, y_rot + y_c, z_corners + z_c], dim=-1)  # (N, 8, 3)


# ---------------------------------------------------------------------------
# Grid memory
# ---------------------------------------------------------------------------
def sine_pos_2d(h: int, w: int, d: int, device, dtype=torch.float32) -> torch.Tensor:
    """(h*w, d) 2-D sine/cosine positional encoding (x-sin, x-cos, y-sin, y-cos)."""
    d4 = d // 4
    y = torch.arange(h, device=device, dtype=torch.float32)[:, None].expand(h, w).reshape(-1)
    x = torch.arange(w, device=device, dtype=torch.float32)[None, :].expand(h, w).reshape(-1)
    omega = 1.0 / (10000 ** (torch.arange(d4, device=device, dtype=torch.float32) / d4))
    pe = torch.cat([torch.sin(x[:, None] * omega), torch.cos(x[:, None] * omega),
                    torch.sin(y[:, None] * omega), torch.cos(y[:, None] * omega)], dim=1)
    if pe.shape[1] < d:
        pe = F.pad(pe, (0, d - pe.shape[1]))
    return pe.to(dtype)


class TokenGridFusion(nn.Module):
    """Fuse the DINOv3 and Pi3 grids (same Hp x Wp grid, both ``d_grid``-d) into a
    (T, Hp*Wp, d_model) memory: per-stream LayerNorm -> Linear, per-cell 2-way softmax
    gate, LayerNorm, + 2-D sine positional encoding for the grid size at hand."""

    def __init__(self, d_grid: int = 256, d_model: int = 256):
        super().__init__()
        self.p_dino = nn.Sequential(nn.LayerNorm(d_grid), nn.Linear(d_grid, d_model))
        self.p_pi3 = nn.Sequential(nn.LayerNorm(d_grid), nn.Linear(d_grid, d_model))
        self.gate = nn.Linear(2 * d_model, 2)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model
        self.register_buffer("last_gate_mean", torch.zeros(2), persistent=False)
        self._pe_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}

    def _pe(self, h: int, w: int, device, dtype) -> torch.Tensor:
        key = (h, w, str(device))
        pe = self._pe_cache.get(key)
        if pe is None:
            pe = sine_pos_2d(h, w, self.d_model, device)
            self._pe_cache[key] = pe
        return pe.to(dtype)

    def forward(self, grid_dino: torch.Tensor, grid_pi3: torch.Tensor) -> torch.Tensor:
        """grid_dino, grid_pi3: (T, Hp, Wp, d_grid) -> memory (T, Hp*Wp, d_model)."""
        if grid_dino.shape != grid_pi3.shape:
            raise ValueError(f"grid shapes differ: dino {tuple(grid_dino.shape)} vs pi3 {tuple(grid_pi3.shape)}")
        T, Hp, Wp, _ = grid_pi3.shape
        a = self.p_dino(grid_dino.float()).reshape(T, Hp * Wp, -1)
        b = self.p_pi3(grid_pi3.float()).reshape(T, Hp * Wp, -1)
        g = torch.softmax(self.gate(torch.cat([a, b], dim=-1)), dim=-1)        # (T, HW, 2)
        with torch.no_grad():
            self.last_gate_mean = g.detach().float().reshape(-1, 2).mean(0)
        mem = self.norm(g[..., :1] * a + g[..., 1:] * b)
        return mem + self._pe(Hp, Wp, mem.device, mem.dtype)[None]


# ---------------------------------------------------------------------------
# Entity decoder layer
# ---------------------------------------------------------------------------
def _unmask_all_padded(mask: torch.Tensor) -> torch.Tensor:
    """Key-padding failsafe: a row with every key masked yields NaN attention;
    unmask its first key (the row's output is zeroed afterwards anyway)."""
    all_pad = mask.all(dim=-1)
    if all_pad.any():
        mask = mask.clone()
        mask[all_pad, 0] = False
    return mask


class EntityDecoderLayer(nn.Module):
    """(a) temporal self-attn per slot over T, (b) spatial self-attn over the frame's
    queries (weights returned), (c) cross-attn to the grid memory, (d) FFN."""

    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.spatial_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                nn.Linear(d_ff, d))
        self.n_t, self.n_s, self.n_c, self.n_f = (nn.LayerNorm(d), nn.LayerNorm(d),
                                                  nn.LayerNorm(d), nn.LayerNorm(d))
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, q_pos: torch.Tensor, pad: torch.Tensor,
                n_free: int, slot_valid: torch.Tensor, frame_pos: torch.Tensor):
        """q, q_pos (T, Q+N, d); mem (T, HW, d); pad (T, Q+N) True = padding;
        slot_valid (T, N); frame_pos (T, d).  Returns (q, attn (T, heads, Q+N, Q+N))."""
        T = q.shape[0]
        # (a) temporal self-attention, per slot, across frames (slots only)
        if q.shape[1] > n_free:
            s = q[:, n_free:]                                                   # (T, N, d)
            h = self.n_t(s)
            qk = (h + q_pos[:, n_free:] + frame_pos[:, None, :]).transpose(0, 1)   # (N, T, d)
            kpm = _unmask_all_padded((~slot_valid).transpose(0, 1))             # (N, T)
            out, _ = self.temporal_attn(qk, qk, h.transpose(0, 1), key_padding_mask=kpm,
                                        need_weights=False)
            s = s + self.drop(out.transpose(0, 1))
            q = torch.cat([q[:, :n_free], s], dim=1)
        # (b) spatial self-attention over all queries of the frame
        h = self.n_s(q)
        sa, w = self.spatial_attn(h + q_pos, h + q_pos, h, key_padding_mask=_unmask_all_padded(pad),
                                  need_weights=True, average_attn_weights=False)
        q = q + self.drop(sa)
        # (c) cross-attention to the grid memory
        h = self.n_c(q)
        ca, _ = self.cross_attn(h + q_pos, mem, mem, need_weights=False)
        q = q + self.drop(ca)
        # (d) FFN
        q = q + self.drop(self.ff(self.n_f(q)))
        q = q.masked_fill(pad.unsqueeze(-1), 0.0)
        return q, w


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WorldWisePP(WorldWisePlus):
    """WorldWise+ whose object-token core is an entity decoder over fused token grids,
    with joint free-query detection and an EGTR-style relation readout."""

    def __init__(self, config, num_object_classes: int = 37, attention_class_num: int = 3,
                 spatial_class_num: int = 6, contact_class_num: int = 17):
        super().__init__(config, num_object_classes, attention_class_num,
                         spatial_class_num, contact_class_num)
        if self.use_energy_refinement or self.use_geometric_attn_bias:
            raise ValueError("WorldWise++ replaces the inter-object transformer; "
                             "use_energy_refinement / use_geometric_attn_bias must be off")
        d = config.d_model
        self.num_object_classes = num_object_classes
        self.d_grid = int(getattr(config, "d_grid", 256))
        self.n_free_queries = int(getattr(config, "n_free_queries", 30))
        self.n_decoder_layers = int(getattr(config, "n_decoder_layers", 4))
        self.n_decoder_heads = int(getattr(config, "n_decoder_heads", 8))
        d_ff = int(getattr(config, "d_decoder_ff", 4 * d))
        self.max_frames = int(getattr(config, "max_frames", 1024))
        self.box_ref = float(getattr(config, "box_reference_size", 1000.0))
        dropout = config.dropout

        # The retriever and the inter-object transformer are replaced by the decoder.
        del self.retriever
        del self.inter_object_encoder

        # 1. grid memory
        self.grid_fusion = TokenGridFusion(self.d_grid, d)

        # 2. entity decoder
        if self.n_free_queries > 0:
            self.free_queries = nn.Parameter(torch.randn(self.n_free_queries, d) * 0.02)
            self.free_pos = nn.Parameter(torch.randn(self.n_free_queries, d) * 0.02)
        self.slot_pos = nn.Linear(config.d_struct, d)
        self.frame_pos = nn.Embedding(self.max_frames, d)
        nn.init.normal_(self.frame_pos.weight, std=0.02)
        self.decoder = nn.ModuleList([EntityDecoderLayer(d, self.n_decoder_heads, d_ff, dropout)
                                      for _ in range(self.n_decoder_layers)])
        self.out_norm = nn.LayerNorm(d)

        # 3a. free-query detection heads (class incl. no-object, 2-D box, factorised 3-D OBB)
        self.det_cls_head = nn.Linear(d, num_object_classes + 1)
        self.det_box_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 4))
        self.p3d = nn.Sequential(nn.Linear(d + 4, 512), nn.ReLU(inplace=True))
        self.dim_pred, self.rot_pred = nn.Linear(512, 3), nn.Linear(512, 2)
        self.depth_pred, self.off_pred = nn.Linear(512, 1), nn.Linear(512, 2)
        nn.init.normal_(self.depth_pred.weight, std=1e-3)
        nn.init.constant_(self.depth_pred.bias, 1.0)
        nn.init.normal_(self.dim_pred.weight, std=1e-3)
        nn.init.zeros_(self.dim_pred.bias)

        # 3b. slot box / corner refinement (residual; zero-initialised so the
        #     untrained model returns the input boxes)
        self.slot_box_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 4))
        self.slot_corner_head = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, 24))
        for head in (self.slot_box_head, self.slot_corner_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        # 4. EGTR-style pair readout replaces the union-box projector: input =
        #    [attn p->o and o->p from every layer (2*L*heads) ; q_p * q_o (d)]
        rp = self.rel_predictor
        self.d_pair_attn = 2 * self.n_decoder_layers * self.n_decoder_heads
        rp.union_proj = nn.Sequential(
            nn.Linear(self.d_pair_attn + d, rp.d_rel // 4),
            nn.ReLU(inplace=True),
            nn.LayerNorm(rp.d_rel // 4),
        )

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def cxcywh_to_xyxy(b: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = float(image_hw[0]), float(image_hw[1])
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], dim=-1)

    def _free_corners(self, x: torch.Tensor, boxes_xyxy: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        """x (M, d) query features, boxes (M, 4) pixels -> (M, 8, 3) camera-frame corners."""
        H, W = float(image_hw[0]), float(image_hw[1])
        h = self.p3d(torch.cat([x, boxes_xyxy / self.box_ref], dim=-1))
        dims = F.softplus(self.dim_pred(h)) + 1e-4
        rot = self.rot_pred(h)
        rot = rot / torch.sqrt((rot ** 2).sum(-1, keepdim=True) + 1e-6)
        depth = F.softplus(self.depth_pred(h)) + 1e-4
        off = self.off_pred(h)
        f = max(H, W)
        fl = torch.full((x.shape[0], 2), f, device=x.device, dtype=h.dtype)
        pp = torch.tensor([W / 2.0, H / 2.0], device=x.device, dtype=h.dtype).expand(x.shape[0], 2)
        return _compute_3d_corners(dims, rot, depth, off, boxes_xyxy.to(h.dtype), fl, pp)

    def _pair_attention_features(self, attn_maps: List[torch.Tensor], person_idx: torch.Tensor,
                                 object_idx: torch.Tensor) -> torch.Tensor:
        """Gather, per (person, object) pair, the spatial self-attention weights in both
        directions from every decoder layer -> (T, K, 2 * L * heads)."""
        A = torch.stack(attn_maps, dim=1)                                   # (T, L, heads, Qn, Qn)
        T, K = person_idx.shape
        t_idx = torch.arange(T, device=A.device)[:, None].expand(T, K)
        p = person_idx + self.n_free_queries
        o = object_idx + self.n_free_queries
        a_po = A[t_idx, :, :, p, o]                                         # (T, K, L, heads)
        a_op = A[t_idx, :, :, o, p]
        return torch.cat([a_po, a_op], dim=-1).flatten(2)                   # (T, K, 2*L*heads)

    # -- forward ---------------------------------------------------------------
    def forward(
        self,
        visual_features_seq: torch.Tensor,
        corners_seq: torch.Tensor,
        valid_mask_seq: torch.Tensor,
        visibility_mask_seq: torch.Tensor,
        person_idx_seq: torch.Tensor,
        object_idx_seq: torch.Tensor,
        pair_valid: torch.Tensor,
        p_mask_visible: float = 0.0,
        camera_pose_seq: Optional[torch.Tensor] = None,
        union_features_seq: Optional[torch.Tensor] = None,
        node_labels_seq: Optional[torch.Tensor] = None,
        gt_contacting_seq: Optional[torch.Tensor] = None,
        grid_dino_seq: Optional[torch.Tensor] = None,
        grid_pi3_seq: Optional[torch.Tensor] = None,
        image_hw: Optional[Tuple[int, int]] = None,
        bboxes_2d_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Same contract as WorldWise.forward plus the grid inputs:

            grid_dino_seq / grid_pi3_seq: (T, Hp, Wp, d_grid) fp16/fp32 token grids
            image_hw:                      (H, W) of the Pi3 image the grids cover
            bboxes_2d_seq:                 (T, N, 4) xyxy input boxes in that pixel space

        ``union_features_seq`` is accepted for API compatibility and ignored.
        Returns the WorldWise output dict plus ``det`` with the free-query detections
        (``logits`` (T,Q,C+1), ``boxes`` cxcywh in [0,1], ``boxes_xyxy`` pixels,
        ``corners`` (T,Q,8,3) camera frame) and the slot refinements (``slot_boxes``
        xyxy normalised by (W,H), ``slot_corners`` (T,N,8,3) in the input frame).
        """
        if grid_dino_seq is None or grid_pi3_seq is None or image_hw is None:
            raise ValueError("WorldWisePP needs grid_dino_seq, grid_pi3_seq and image_hw "
                             "(load the video through lib.supervised.worldwise_pp.dataset.WorldAGGrid)")
        T, N = corners_seq.shape[:2]
        device = corners_seq.device
        if T > self.max_frames:
            raise ValueError(f"video has {T} frames > max_frames={self.max_frames}")

        # ==================== Steps 1-4: identical to WorldWise ====================
        struct_all, _ = self.global_structural_encoder(corners_seq, valid_mask_seq)

        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            if self.use_object_spatial_encoder:
                _, cam_all = self.object_spatial_encoder(camera_pose_seq, corners_seq, valid_mask_seq)
            if self.use_camera_temporal:
                ego_tokens_all = self.camera_temporal_encoder(camera_pose_seq)

        motion_all = None
        if self.use_object_motion_encoder:
            centers_all = corners_seq.mean(dim=2)
            velocity_all = torch.zeros_like(centers_all)
            accel_all = torch.zeros_like(centers_all)
            valid_vel = valid_mask_seq[1:] & valid_mask_seq[:-1]
            velocity_all[1:] = torch.where(valid_vel.unsqueeze(-1), centers_all[1:] - centers_all[:-1],
                                           torch.zeros_like(centers_all[1:]))
            valid_acc = valid_mask_seq[2:] & valid_mask_seq[1:-1] & valid_mask_seq[:-2]
            accel_all[2:] = torch.where(valid_acc.unsqueeze(-1), velocity_all[2:] - velocity_all[1:-1],
                                        torch.zeros_like(velocity_all[2:]))
            camera_R_all = camera_pose_seq[:, :3, :3] if camera_pose_seq is not None else None
            motion_all = self.object_motion_encoder(velocity=velocity_all, acceleration=accel_all,
                                                    camera_R=camera_R_all, valid_mask=valid_mask_seq)

        hybrid_tokens, is_masked_all, original_visual_all, artificially_masked_all = self.scaffold_tokenizer(
            geometry_tokens=struct_all,
            visual_features=visual_features_seq,
            visibility_mask=visibility_mask_seq,
            valid_mask=valid_mask_seq,
            p_mask_visible=p_mask_visible if self.training else 0.0,
            cam_feats=cam_all,
            motion_feats=motion_all,
            ego_tokens=ego_tokens_all,
        )

        # ==================== Step 5: grid memory ====================
        mem = self.grid_fusion(grid_dino_seq, grid_pi3_seq)                      # (T, HW, d)
        mem = mem.to(hybrid_tokens.dtype)

        # ==================== Step 6: entity decoder ====================
        Q = self.n_free_queries
        slot_pos = self.slot_pos(struct_all).masked_fill(~valid_mask_seq.unsqueeze(-1), 0.0)
        if Q > 0:
            q = torch.cat([self.free_queries[None].expand(T, -1, -1).to(hybrid_tokens.dtype), hybrid_tokens], dim=1)
            q_pos = torch.cat([self.free_pos[None].expand(T, -1, -1).to(slot_pos.dtype), slot_pos], dim=1)
            pad = torch.cat([torch.zeros(T, Q, dtype=torch.bool, device=device), ~valid_mask_seq], dim=1)
        else:
            q, q_pos, pad = hybrid_tokens, slot_pos, ~valid_mask_seq
        frame_pos = self.frame_pos(torch.arange(T, device=device)).to(q.dtype)      # (T, d)

        attn_maps = []
        for layer in self.decoder:
            q, w = layer(q, mem, q_pos, pad, Q, valid_mask_seq, frame_pos)
            attn_maps.append(w)
        q = self.out_norm(q).masked_fill(pad.unsqueeze(-1), 0.0)
        free_q, slots = q[:, :Q], q[:, Q:]

        # ==================== Step 7: visibility embedding ====================
        effective_visible = visibility_mask_seq & (~is_masked_all)
        completed_tokens = slots + self.visibility_emb(effective_visible.long())
        completed_tokens = completed_tokens * valid_mask_seq.unsqueeze(-1).to(completed_tokens.dtype)

        # ==================== Steps 8-9: node prediction + reconstruction ====================
        node_logits_all = self.node_predictor(completed_tokens)
        recon_pred_all = self.reconstruction_proj(completed_tokens)

        # ==================== Step 10: EGTR pair readout + edge prediction ====================
        K_max = person_idx_seq.shape[1] if person_idx_seq.dim() > 1 else 0
        pair_feat = None
        if K_max > 0:
            p_safe = person_idx_seq.clamp(0, N - 1)
            o_safe = object_idx_seq.clamp(0, N - 1)
            attn_feat = self._pair_attention_features(attn_maps, p_safe, o_safe).to(slots.dtype)
            dexp = slots.shape[-1]
            q_p = torch.gather(slots, 1, p_safe.unsqueeze(-1).expand(T, K_max, dexp))
            q_o = torch.gather(slots, 1, o_safe.unsqueeze(-1).expand(T, K_max, dexp))
            pair_feat = torch.cat([attn_feat, q_p * q_o], dim=-1)                # (T, K, 2LH + d)

        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            completed_tokens, node_logits_all, person_idx_seq, object_idx_seq,
            pair_valid, pair_feat,
            node_class_override=node_labels_seq,
            corners=corners_seq,
        )

        if self.use_temporal_edge_attn:
            enriched_rel = self.temporal_edge_attn(rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq)
        else:
            enriched_rel = rel_tokens

        con_tokens = None
        if self.use_predicate_prototypes:
            if self.training and gt_contacting_seq is not None:
                self.prototype_memory.update(enriched_rel, gt_contacting_seq, pair_valid_out)
            con_tokens = self.prototype_memory(enriched_rel, pair_valid_out)

        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out, rel_tokens_contacting=con_tokens)

        # ==================== Step 11: detection heads ====================
        H, W = int(image_hw[0]), int(image_hw[1])
        if Q > 0:
            det_logits = self.det_cls_head(free_q)                                      # (T, Q, C+1)
            det_boxes = torch.sigmoid(self.det_box_head(free_q).float())                # cxcywh in [0, 1]
            det_xyxy = self.cxcywh_to_xyxy(det_boxes, (H, W))
            det_corners = self._free_corners(free_q.reshape(T * Q, -1), det_xyxy.reshape(T * Q, 4), (H, W))
            det_corners = det_corners.view(T, Q, 8, 3)
        else:
            det_logits = slots.new_zeros(T, 0, self.num_object_classes + 1)
            det_boxes = slots.new_zeros(T, 0, 4, dtype=torch.float32)
            det_xyxy = det_boxes
            det_corners = slots.new_zeros(T, 0, 8, 3, dtype=torch.float32)

        if bboxes_2d_seq is None:
            bboxes_2d_seq = corners_seq.new_zeros(T, N, 4)
        scale = torch.tensor([W, H, W, H], device=device, dtype=torch.float32)
        slot_boxes = bboxes_2d_seq.float() / scale + self.slot_box_head(slots).float()   # xyxy, normalised
        slot_corners = corners_seq.float() + self.slot_corner_head(slots).float().view(T, N, 8, 3)

        out = {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
            "is_masked": is_masked_all,
            "artificially_masked": artificially_masked_all,
            "original_visual": original_visual_all,
            "reconstruction_predictions": recon_pred_all,
            "reconstruction_targets": original_visual_all,
            "det": {
                "logits": det_logits,
                "boxes": det_boxes,
                "boxes_xyxy": det_xyxy,
                "corners": det_corners,
                "slot_boxes": slot_boxes,
                "slot_corners": slot_corners,
                "image_hw": (H, W),
            },
        }
        return out

    def gate_summary(self) -> Optional[dict]:
        out = super().gate_summary() or {}
        out["grid"] = self.grid_fusion.last_gate_mean.tolist()
        return out
