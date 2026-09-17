"""
WorldFormer C2 — entity-query joint detection + scene-graph scaffold (ICLR WS3, C2).

Scaffold only (forward/backward smoke-tested in tests/test_worldformer_c2_smoke.py);
training is wired later.  What it implements, and what it reuses:

  Input      Tier-2 token grids from the caches (datasets/preprocess/tokens):
             DINOv3 grid (T, 24, 42, 1024) on the /16-padded 672x384 image and the
             Pi3 grid (T, 27, 48, 1024) on the 672x378 Pi3 image.  Both are projected
             to d_model, the DINOv3 grid is resampled onto the Pi3 grid over the
             unpadded region (both streams share Pi3-space pixel coordinates), and the
             two are combined by a per-cell softmax gate -> fused memory (T, 1296, d).
  Queries    Q free entity queries (DETR) + N persistent world-slot queries built from
             the world-graph geometry (GlobalStructuralEncoder over corners, reused
             from lib.supervised.components).  Slots that are not visible at t get the
             learnable [MASK] token added — MWAE at the query level: the decoder must
             recover their appearance from the image + other slots.
  Decoder    L custom decoder layers (self-attn over all queries with returned weights,
             cross-attn to the fused grid, FFN).  The last layer's self-attention map is
             the EGTR-style relation by-product.
  Heads      class (C+1), 2D box (cxcywh, sigmoid), 3D OBB via the factorized
             parameterisation of dino_mono_3d (_compute_3d_corners: dims / yaw sin-cos /
             depth / centre offset, pinhole with f=max(H,W)), relation head over pairs
             (query features + attention by-product -> 26 predicate logits +
             connectivity), reconstruction head for masked slots (predicts the target
             projector's embedding of the slot's cached ROI token).
  Matching   Hungarian (scipy) on class + L1 box + L1 corners; optional Hydra-SGG-style
             one-to-many aux assignment (IoU > thr) for faster convergence.
  Loss       CE(class, no-object down-weighted) + L1 box + L1 corners (matched)
             + BCE relations over matched GT pairs + MSE reconstruction (masked slots).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.supervised.components import GlobalStructuralEncoder


def _load_compute_3d_corners():
    """The factorised OBB decoder of dino_mono_3d.  Imported by file path so the
    detector package __init__ (which pulls in wandb / the trainer) is not needed."""
    try:
        from lib.detector.monocular3d.models.dino_mono_3d import _compute_3d_corners
        return _compute_3d_corners
    except ImportError:
        import importlib.util
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "detector",
                            "monocular3d", "models", "dino_mono_3d.py")
        spec = importlib.util.spec_from_file_location("_dino_mono_3d_standalone", os.path.normpath(path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._compute_3d_corners


_compute_3d_corners = _load_compute_3d_corners()

N_PREDICATES = 3 + 6 + 17


# ---------------------------------------------------------------------------
# Grid fusion
# ---------------------------------------------------------------------------
def sine_pos_2d(h: int, w: int, d: int, device) -> torch.Tensor:
    """(h*w, d) 2-D sine/cosine positional encoding."""
    d4 = d // 4
    y = torch.arange(h, device=device).float()[:, None].expand(h, w).reshape(-1)
    x = torch.arange(w, device=device).float()[None, :].expand(h, w).reshape(-1)
    omega = 1.0 / (10000 ** (torch.arange(d4, device=device).float() / d4))
    pe = torch.cat([torch.sin(x[:, None] * omega), torch.cos(x[:, None] * omega),
                    torch.sin(y[:, None] * omega), torch.cos(y[:, None] * omega)], dim=1)
    if pe.shape[1] < d:
        pe = F.pad(pe, (0, d - pe.shape[1]))
    return pe


class TokenGridFusion(nn.Module):
    def __init__(self, d_dino: int = 1024, d_pi3: int = 1024, d_model: int = 256):
        super().__init__()
        self.p_dino = nn.Sequential(nn.LayerNorm(d_dino), nn.Linear(d_dino, d_model))
        self.p_pi3 = nn.Sequential(nn.LayerNorm(d_pi3), nn.Linear(d_pi3, d_model))
        self.gate = nn.Linear(2 * d_model, 2)
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, dino_grid: torch.Tensor, pi3_grid: torch.Tensor,
                dino_padded_hw: Tuple[int, int], image_hw: Tuple[int, int]) -> torch.Tensor:
        """dino_grid (T,Hd,Wd,Cd) on the padded image; pi3_grid (T,Hp,Wp,Cp) on the
        image -> fused memory (T, Hp*Wp, d) with positional encoding added."""
        T, Hp, Wp, _ = pi3_grid.shape
        a = self.p_dino(dino_grid).permute(0, 3, 1, 2)                      # (T,d,Hd,Wd)
        # crop the padded region (bottom/right zero pad) then resample to the Pi3 grid
        Hd, Wd = a.shape[-2:]
        fh = image_hw[0] / dino_padded_hw[0]
        fw = image_hw[1] / dino_padded_hw[1]
        a = a[:, :, : max(1, round(Hd * fh)), : max(1, round(Wd * fw))]
        a = F.interpolate(a, size=(Hp, Wp), mode="bilinear", align_corners=False)
        a = a.permute(0, 2, 3, 1).reshape(T, Hp * Wp, -1)
        b = self.p_pi3(pi3_grid).reshape(T, Hp * Wp, -1)
        g = torch.softmax(self.gate(torch.cat([a, b], dim=-1)), dim=-1)
        mem = g[..., :1] * a + g[..., 1:] * b
        return self.norm(mem) + sine_pos_2d(Hp, Wp, self.d_model, mem.device)[None]


# ---------------------------------------------------------------------------
# Decoder with exposed self-attention (EGTR by-product)
# ---------------------------------------------------------------------------
class DecoderLayer(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, d_ff), nn.ReLU(inplace=True), nn.Dropout(dropout),
                                nn.Linear(d_ff, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, q_pos: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None):
        h = self.n1(q)
        sa, w = self.self_attn(h + q_pos, h + q_pos, h, key_padding_mask=key_padding_mask,
                               need_weights=True, average_attn_weights=False)  # w: (T,heads,Q,Q)
        q = q + self.drop(sa)
        h = self.n2(q)
        ca, _ = self.cross_attn(h + q_pos, mem, mem, need_weights=False)
        q = q + self.drop(ca)
        q = q + self.drop(self.ff(self.n3(q)))
        return q, w


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class WorldFormerC2(nn.Module):
    def __init__(self, num_classes: int = 37, d_model: int = 256, n_heads: int = 8, d_ff: int = 1024,
                 n_layers: int = 4, n_queries: int = 50, max_slots: int = 64, d_struct: int = 256,
                 d_dino: int = 1024, d_pi3: int = 1024, dropout: float = 0.1,
                 input_reference_size: float = 1000.0):
        super().__init__()
        self.num_classes = num_classes
        self.n_queries = n_queries
        self.max_slots = max_slots
        self.d_model = d_model
        self.n_heads = n_heads
        self.ref = input_reference_size

        self.fusion = TokenGridFusion(d_dino, d_pi3, d_model)
        self.free_queries = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        self.free_pos = nn.Parameter(torch.randn(n_queries, d_model) * 0.02)
        # persistent world slots: geometry -> query; [MASK] for invisible slots
        self.geo_encoder = GlobalStructuralEncoder(d_struct=d_struct, d_hidden=d_struct // 2)
        self.slot_proj = nn.Linear(d_struct, d_model)
        self.slot_pos = nn.Linear(d_struct, d_model)
        self.mask_token = nn.Parameter(torch.randn(d_model) * 0.02)
        self.slot_visual_proj = nn.Linear(d_dino + d_pi3, d_model)   # visible slots: cached ROI token
        self.target_proj = nn.Linear(d_dino + d_pi3, d_model)        # reconstruction target (frozen copy)
        for p in self.target_proj.parameters():
            p.requires_grad_(False)

        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(d_model)

        self.cls_head = nn.Linear(d_model, num_classes + 1)          # +1 no-object
        self.box_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, 4))
        self.p3d = nn.Sequential(nn.Linear(d_model + 4, 512), nn.ReLU(inplace=True))
        self.dim_pred, self.rot_pred = nn.Linear(512, 3), nn.Linear(512, 2)
        self.depth_pred, self.off_pred = nn.Linear(512, 1), nn.Linear(512, 2)
        nn.init.normal_(self.depth_pred.weight, std=1e-3)
        nn.init.constant_(self.depth_pred.bias, 1.0)
        nn.init.normal_(self.dim_pred.weight, std=1e-3)
        nn.init.zeros_(self.dim_pred.bias)

        self.rel_head = nn.Sequential(nn.Linear(3 * d_model + n_heads, d_model), nn.ReLU(inplace=True),
                                      nn.Linear(d_model, N_PREDICATES + 1))   # +1 connectivity
        self.recon_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                        nn.Linear(d_model, d_model))

    # -- helpers ---------------------------------------------------------------
    def _corners(self, x: torch.Tensor, boxes_xyxy: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        """x (M,d) query feats, boxes (M,4) pixels -> (M,8,3) camera-frame corners."""
        H, W = image_hw
        h = self.p3d(torch.cat([x, boxes_xyxy / self.ref], dim=-1))
        dims = F.softplus(self.dim_pred(h)) + 1e-4
        rot = self.rot_pred(h)
        rot = rot / torch.sqrt((rot ** 2).sum(-1, keepdim=True) + 1e-6)
        depth = F.softplus(self.depth_pred(h)) + 1e-4
        off = self.off_pred(h)
        f = float(max(H, W))
        fl = torch.full((x.shape[0], 2), f, device=x.device)
        pp = torch.tensor([W / 2.0, H / 2.0], device=x.device).expand(x.shape[0], 2)
        return _compute_3d_corners(dims, rot, depth, off, boxes_xyxy, fl, pp)

    @staticmethod
    def cxcywh_to_xyxy(b: torch.Tensor, image_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = image_hw
        cx, cy, w, h = b.unbind(-1)
        return torch.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], dim=-1)

    # -- forward ---------------------------------------------------------------
    def forward(self, dino_grid: torch.Tensor, pi3_grid: torch.Tensor, dino_padded_hw: Tuple[int, int],
                image_hw: Tuple[int, int], slot_corners: Optional[torch.Tensor] = None,
                slot_valid: Optional[torch.Tensor] = None, slot_visible: Optional[torch.Tensor] = None,
                slot_tokens: Optional[torch.Tensor] = None, p_mask_visible: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        dino_grid (T,Hd,Wd,Cd) fp16/32, pi3_grid (T,Hp,Wp,Cp); slot_* describe the persistent
        world slots (N = max_slots): corners (T,N,8,3), valid/visible (T,N) bool, tokens
        (T,N,Cd+Cp) cached ROI tokens (reconstruction targets / visible-slot appearance).
        """
        T = pi3_grid.shape[0]
        dev = pi3_grid.device
        mem = self.fusion(dino_grid.float(), pi3_grid.float(), dino_padded_hw, image_hw)

        q = self.free_queries[None].expand(T, -1, -1)
        q_pos = self.free_pos[None].expand(T, -1, -1)
        pad = torch.zeros(T, self.n_queries, dtype=torch.bool, device=dev)
        is_masked = None
        if slot_corners is not None:
            struct, _ = self.geo_encoder(slot_corners, slot_valid)               # (T,N,d_struct)
            s_pos = self.slot_pos(struct)
            s_q = self.slot_proj(struct)
            is_masked = ~slot_visible
            if self.training and p_mask_visible > 0:
                is_masked = is_masked | ((torch.rand(T, slot_valid.shape[1], device=dev) < p_mask_visible)
                                         & slot_visible)
            is_masked = is_masked & slot_valid
            vis = self.slot_visual_proj(slot_tokens.float()) if slot_tokens is not None else torch.zeros_like(s_q)
            app = torch.where(is_masked[..., None], self.mask_token.expand_as(s_q), vis)
            q = torch.cat([q, s_q + app], dim=1)
            q_pos = torch.cat([q_pos, s_pos], dim=1)
            pad = torch.cat([pad, ~slot_valid], dim=1)

        attn_last = None
        for layer in self.layers:
            q, attn_last = layer(q, mem, q_pos, key_padding_mask=pad)
        q = self.out_norm(q)

        Qf = q[:, : self.n_queries]                                                   # free queries
        logits = self.cls_head(Qf)
        boxes = torch.sigmoid(self.box_head(Qf))                                      # cxcywh in [0,1]
        boxes_xyxy = self.cxcywh_to_xyxy(boxes, image_hw)
        corners = self._corners(Qf.reshape(-1, self.d_model), boxes_xyxy.reshape(-1, 4), image_hw)
        corners = corners.view(T, self.n_queries, 8, 3)

        # EGTR-style relation extraction over all query pairs (free + slots)
        A = attn_last.permute(0, 2, 3, 1)                                             # (T,Q,Q,heads)
        Qn = q.shape[1]
        qi = q[:, :, None, :].expand(T, Qn, Qn, -1)
        qj = q[:, None, :, :].expand(T, Qn, Qn, -1)
        rel = self.rel_head(torch.cat([qi, qj, qi * qj, A], dim=-1))                  # (T,Q,Q,27)

        out = dict(logits=logits, boxes=boxes, boxes_xyxy=boxes_xyxy, corners=corners,
                   rel_logits=rel[..., :N_PREDICATES], connectivity=rel[..., N_PREDICATES],
                   queries=q, attn=A)
        if slot_corners is not None:
            S = q[:, self.n_queries:]
            out["slot_queries"] = S
            out["is_masked"] = is_masked
            out["reconstruction_predictions"] = self.recon_head(S)
            with torch.no_grad():
                out["reconstruction_targets"] = (self.target_proj(slot_tokens.float())
                                                 if slot_tokens is not None else torch.zeros_like(S))
        return out


# ---------------------------------------------------------------------------
# Matching + loss
# ---------------------------------------------------------------------------
def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-6)


@torch.no_grad()
def hungarian_match(logits: torch.Tensor, boxes: torch.Tensor, corners: torch.Tensor,
                    gt_labels: torch.Tensor, gt_boxes: torch.Tensor, gt_corners: Optional[torch.Tensor],
                    w_cls: float = 1.0, w_box: float = 5.0, w_3d: float = 1.0,
                    one_to_many_iou: float = 0.0, boxes_xyxy: Optional[torch.Tensor] = None,
                    gt_boxes_xyxy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """One frame. Returns (query_idx, gt_idx) long tensors; with ``one_to_many_iou`` > 0 also
    adds every unmatched query whose box IoU with a GT exceeds the threshold (Hydra-SGG-style
    hybrid assignment, train-time only)."""
    from scipy.optimize import linear_sum_assignment
    if gt_labels.numel() == 0:
        e = torch.zeros(0, dtype=torch.long, device=logits.device)
        return e, e
    prob = logits.softmax(-1)[:, gt_labels]                                 # (Q,G)
    cost = -w_cls * prob + w_box * torch.cdist(boxes, gt_boxes, p=1)
    if gt_corners is not None:
        cost = cost + w_3d * torch.cdist(corners.flatten(1), gt_corners.flatten(1), p=1) / 24.0
    qi, gi = linear_sum_assignment(cost.cpu().numpy())
    qi = torch.as_tensor(qi, device=logits.device, dtype=torch.long)
    gi = torch.as_tensor(gi, device=logits.device, dtype=torch.long)
    if one_to_many_iou > 0 and boxes_xyxy is not None and gt_boxes_xyxy is not None:
        iou = box_iou_xyxy(boxes_xyxy, gt_boxes_xyxy)
        iou[qi] = -1
        best, arg = iou.max(1)
        extra = torch.nonzero(best > one_to_many_iou).squeeze(1)
        qi = torch.cat([qi, extra])
        gi = torch.cat([gi, arg[extra]])
    return qi, gi


class WorldFormerC2Loss(nn.Module):
    def __init__(self, num_classes: int, no_object_weight: float = 0.1, w_cls: float = 1.0, w_box: float = 5.0,
                 w_3d: float = 1.0, w_rel: float = 1.0, w_recon: float = 0.5, one_to_many_iou: float = 0.0):
        super().__init__()
        w = torch.ones(num_classes + 1)
        w[num_classes] = no_object_weight
        self.register_buffer("cls_weight", w)
        self.num_classes = num_classes
        self.w = dict(cls=w_cls, box=w_box, p3d=w_3d, rel=w_rel, recon=w_recon)
        self.one_to_many_iou = one_to_many_iou

    def forward(self, out: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]],
                image_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """targets[t]: labels (G,) long, boxes (G,4) cxcywh in [0,1], corners (G,8,3) or None,
        rel_pairs (R,2) long indices into the frame's GT objects, rel_labels (R,26) float."""
        T = out["logits"].shape[0]
        Qf = out["logits"].shape[1]
        losses = {k: out["logits"].new_zeros(()) for k in ("cls", "box", "p3d", "rel", "recon")}
        n_matched = 0
        for t in range(T):
            tg = targets[t]
            gxy = WorldFormerC2.cxcywh_to_xyxy(tg["boxes"], image_hw) if tg["labels"].numel() else None
            qi, gi = hungarian_match(out["logits"][t], out["boxes"][t], out["corners"][t], tg["labels"],
                                     tg["boxes"], tg.get("corners"), one_to_many_iou=self.one_to_many_iou,
                                     boxes_xyxy=out["boxes_xyxy"][t], gt_boxes_xyxy=gxy)
            tgt_cls = torch.full((Qf,), self.num_classes, dtype=torch.long, device=out["logits"].device)
            tgt_cls[qi] = tg["labels"][gi]
            losses["cls"] = losses["cls"] + F.cross_entropy(out["logits"][t], tgt_cls, weight=self.cls_weight)
            if qi.numel():
                losses["box"] = losses["box"] + F.l1_loss(out["boxes"][t][qi], tg["boxes"][gi])
                if tg.get("corners") is not None:
                    losses["p3d"] = losses["p3d"] + F.l1_loss(out["corners"][t][qi], tg["corners"][gi])
                n_matched += int(qi.numel())
                # relations between matched GT objects (EGTR: by-product of the free queries)
                rp = tg.get("rel_pairs")
                if rp is not None and rp.numel():
                    g2q = torch.full((int(tg["labels"].numel()),), -1, dtype=torch.long, device=qi.device)
                    g2q[gi] = qi
                    s, o = g2q[rp[:, 0]], g2q[rp[:, 1]]
                    ok = (s >= 0) & (o >= 0)
                    if ok.any():
                        pred = out["rel_logits"][t][s[ok], o[ok]]
                        losses["rel"] = losses["rel"] + F.binary_cross_entropy_with_logits(pred, tg["rel_labels"][ok])
        if "reconstruction_predictions" in out:
            m = out["is_masked"]
            if m.any():
                losses["recon"] = F.mse_loss(out["reconstruction_predictions"][m], out["reconstruction_targets"][m])
        total = sum(self.w[k] * v for k, v in losses.items()) / max(T, 1)
        losses = {k: v / max(T, 1) for k, v in losses.items()}
        losses["total"] = total
        losses["n_matched"] = torch.tensor(float(n_matched))
        return losses
