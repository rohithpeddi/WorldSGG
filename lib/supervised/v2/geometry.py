"""V2 (round 3): explicit geometry token for hidden and masked slots.

A hidden slot (unobserved object) and a masked item (a visible slot the model masks on purpose during training) enter
with their class only: appearance is [MASK] (scaffold tokenizer, as in V1) and geometry is the learned [GEO-MASK]
embedding below. After completion, ``infer`` maps the completed slot token to g_hat, a token in the geometry
encoder's output space; g_hat is injected back into the slot token, and ``decode`` turns it into a Z-up oriented 3-D
box in the video's world frame (corners 0-3 bottom, 4-7 top, like the annotation).

Losses (``geometry_losses``) apply to masked items only: g_hat towards the geometry encoding of the item's own
annotated box (stop-gradient target) and the decoded box towards that box (centre L1 + symmetric corner Chamfer, both
divided by the video's scale). Unobserved objects never get a 3-D loss.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_SX = torch.tensor([1., 1., -1., -1., 1., 1., -1., -1.]) * 0.5
_SY = torch.tensor([1., -1., -1., 1., 1., -1., -1., 1.]) * 0.5
_SZ = torch.tensor([-1., -1., -1., -1., 1., 1., 1., 1.]) * 0.5


def sample_masked_items(vis_valid: torch.Tensor, p_point: float, p_span: float, max_span: int) -> torch.Tensor:
    """(T, N) bool mask of masked items among visible, valid slots: each visible slot-frame with probability
    ``p_point``, plus, for each slot with probability ``p_span``, a run of k ~ U{1..max_span} consecutive frames
    from a random start (only its visible frames inside the run are masked)."""
    T, N = vis_valid.shape
    dev = vis_valid.device
    m = (torch.rand(T, N, device=dev) < p_point) & vis_valid
    if p_span > 0 and T > 0 and N > 0:
        pick = torch.rand(N, device=dev) < p_span
        k = torch.randint(1, max(1, int(max_span)) + 1, (N,), device=dev)
        t0 = torch.randint(0, T, (N,), device=dev)
        t = torch.arange(T, device=dev).unsqueeze(1)
        span = (t >= t0.unsqueeze(0)) & (t < (t0 + k).unsqueeze(0)) & pick.unsqueeze(0)
        m = m | (span & vis_valid)
    return m


def to_world(corners: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """(T, N, 8, 3) camera-frame corners -> world with (T, 4, 4) camera->world poses."""
    R, t = pose[:, :3, :3].to(corners.dtype), pose[:, :3, 3].to(corners.dtype)
    return (torch.einsum("tij,tnkj->tnki", R, corners) + t[:, None, None, :]).contiguous()


def to_camera(corners_w: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """(T, N, 8, 3) world corners -> camera frame (R^T (w - t))."""
    R, t = pose[:, :3, :3].to(corners_w.dtype), pose[:, :3, 3].to(corners_w.dtype)
    return torch.einsum("tji,tnkj->tnki", R, corners_w - t[:, None, None, :]).contiguous()


def box_corners(centre: torch.Tensor, size: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Z-up oriented box -> (..., 8, 3) corners, bottom 4 then top 4."""
    sx, sy, sz = (v.to(centre.device, centre.dtype) for v in (_SX, _SY, _SZ))
    x = sx * size[..., 0:1]
    y = sy * size[..., 1:2]
    z = sz * size[..., 2:3]
    c, s = torch.cos(yaw).unsqueeze(-1), torch.sin(yaw).unsqueeze(-1)
    return torch.stack([c * x - s * y, s * x + c * y, z], dim=-1) + centre.unsqueeze(-2)


class GeometryV2(nn.Module):
    def __init__(self, d_model: int, d_struct: int, num_classes: int, d_hidden: int = 256):
        super().__init__()
        self.geo_mask = nn.Parameter(torch.randn(d_struct) * 0.02)          # [GEO-MASK]
        self.infer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_hidden), nn.GELU(),
                                   nn.Linear(d_hidden, d_struct))
        self.inject = nn.Linear(d_struct, d_model)
        nn.init.zeros_(self.inject.weight)
        nn.init.zeros_(self.inject.bias)
        self.decoder = nn.Sequential(nn.LayerNorm(d_struct), nn.Linear(d_struct, d_hidden), nn.GELU(),
                                     nn.Linear(d_hidden, 8))
        self.log_size_prior = nn.Embedding(num_classes + 1, 3)
        nn.init.zeros_(self.log_size_prior.weight)
        self.num_classes = num_classes

    @staticmethod
    def video_scale(corners_world: torch.Tensor, ok: torch.Tensor) -> torch.Tensor:
        """Median box diagonal of the boxes the model may see (visible, unmasked) — the per-video length unit."""
        d = (corners_world.amax(dim=2) - corners_world.amin(dim=2)).norm(dim=-1)
        v = d[ok & (d > 0)]
        if v.numel() == 0:
            return torch.ones((), device=corners_world.device, dtype=corners_world.dtype)
        return v.median().clamp(min=1e-3).detach()

    def decode(self, g: torch.Tensor, cam_centre: torch.Tensor, scale: torch.Tensor, classes: torch.Tensor):
        p = self.decoder(g).float()
        centre = cam_centre[:, None, :].float() + scale.float() * p[..., :3]
        cls = classes.clamp(min=0, max=self.num_classes).long()
        size = scale.float() * torch.exp((p[..., 3:6] + self.log_size_prior(cls)).clamp(-6.0, 3.0))
        yaw = torch.atan2(p[..., 6], p[..., 7])
        return box_corners(centre, size, yaw), centre


def geometry_losses(g_hat, g_target, corners_pred_w, centre_pred_w, corners_gt_w, mask, scale):
    """Masked items only (``mask``): latent 1 - cos(g_hat, sg(g_target)) and box centre L1 + Chamfer, / scale."""
    zero = g_hat.sum() * 0.0
    if mask is None or not bool(mask.any()):
        return zero, zero
    cos = F.cosine_similarity(g_hat[mask].float(), g_target[mask].detach().float(), dim=-1)
    latent = (1.0 - cos).mean()
    gt = corners_gt_w[mask].float()
    pr = corners_pred_w[mask].float()
    s = scale.float().clamp(min=1e-3)
    centre_l1 = (centre_pred_w[mask].float() - gt.mean(dim=1)).abs().sum(-1).mean() / s
    dist = torch.cdist(pr, gt)
    chamfer = (dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)).mean() / s
    return latent, centre_l1 + chamfer


def prepare_masks(valid: torch.Tensor, vis: torch.Tensor, training: bool, p_point: float, p_span: float,
                  max_span: int):
    """H (hidden), M (masked items, training only), G = H | M, valid_geo = valid & ~G, vis_eff = vis & ~M."""
    valid = valid.bool()
    vis = vis.bool()
    H = valid & ~vis
    M = sample_masked_items(valid & vis, p_point, p_span, max_span) if training else torch.zeros_like(H)
    G = H | M
    return H, M, G, valid & ~G, vis & ~M


def masked_motion(corners_in: torch.Tensor, valid_geo: torch.Tensor):
    """Velocity / acceleration of box centres from frames whose box the model may see (V1 formula)."""
    centers = corners_in.mean(dim=2)
    vel = torch.zeros_like(centers)
    acc = torch.zeros_like(centers)
    vv = valid_geo[1:] & valid_geo[:-1]
    vel[1:] = torch.where(vv.unsqueeze(-1), centers[1:] - centers[:-1], torch.zeros_like(centers[1:]))
    va = valid_geo[2:] & valid_geo[1:-1] & valid_geo[:-2]
    acc[2:] = torch.where(va.unsqueeze(-1), vel[2:] - vel[1:-1], torch.zeros_like(vel[2:]))
    return vel, acc


def zero_masked_union(union_features_seq: Optional[torch.Tensor], G: torch.Tensor, person_idx: torch.Tensor,
                      object_idx: torch.Tensor):
    """Union features of pairs whose person or object slot is hidden / masked are zeroed (no box-derived evidence)."""
    if union_features_seq is None:
        return None
    N = G.shape[1]
    og = torch.gather(G, 1, object_idx.clamp(0, N - 1))
    pg = torch.gather(G, 1, person_idx.clamp(0, N - 1))
    return union_features_seq * (~(og | pg)).unsqueeze(-1).to(union_features_seq.dtype)
