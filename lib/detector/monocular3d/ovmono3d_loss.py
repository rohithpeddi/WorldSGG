"""
OVMono3D-style 3D detection loss: uncertainty-weighted L3D with
geometry-level disentangled attribute losses and Chamfer (holistic) loss.

L = sqrt(2) * exp(-mu) * L3D + mu
L3D = sum_a L3D^(a) + L3D^all

Disentangled (geometry-level): For each attribute group a (xy, z, dims, r),
  - Take prediction for attribute a and ground truth for all other attributes.
  - Reconstruct a full 3D box (8 corners) from this mixed set.
  - L3D^(a) = Chamfer(reconstructed_box, GT_box).

L3D^all: Chamfer(pred_corners, gt_corners).

We use oriented box representation: center (3), dims (l, w, h) in object frame,
yaw r in xy-plane. Attributes are derived from corners via PCA in xy for
rotation and projected extents for dimensions (robust to corner ordering).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _wrap_angle(r: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return torch.atan2(torch.sin(r), torch.cos(r))


def corners_to_attributes(corners: torch.Tensor) -> dict:
    """
    Derive oriented 3D attributes from 8 corners in a robust way.
    corners: (N, 8, 3). Returns center (N,3), dims (N,3) as (length, width, height)
    in object frame, r (N,) yaw in [-pi, pi].

    Uses PCA in xy to get principal direction (yaw) and projected extents for
    oriented dimensions, so corner ordering does not need to be consistent.
    """
    N = corners.shape[0]
    device = corners.device
    center = corners.mean(dim=1)  # (N, 3)
    centered = corners - center.unsqueeze(1)  # (N, 8, 3)
    xy = centered[..., :2]  # (N, 8, 2)

    # Covariance of xy per sample: (N, 2, 2)
    cov = torch.bmm(xy.transpose(1, 2), xy) / 8.0
    # Eigen decomposition for 2x2: main eigenvector gives yaw
    # For 2x2 sym: [[a,b],[b,c]], largest eval = (a+c)/2 + sqrt((a-c)^2/4 + b^2)
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    delta = (a - c) * (a - c) / 4 + b * b
    delta = torch.clamp(delta, min=1e-12)
    sqrt_d = torch.sqrt(delta)
    # Eigenvector for larger eigenvalue (a+c)/2 + sqrt_d: (b, lambda - a) with lambda = (a+c)/2 + sqrt_d
    lam = (a + c) / 2 + sqrt_d
    ev_x = b
    ev_y = lam - a
    # Normalize
    nrm = torch.sqrt(ev_x * ev_x + ev_y * ev_y).clamp(min=1e-8)
    ev_x = ev_x / nrm
    ev_y = ev_y / nrm
    r = torch.atan2(ev_y, ev_x)  # (N,) in [-pi, pi]

    # Project corners onto local frame: x' = cos(r)*x + sin(r)*y, y' = -sin(r)*x + cos(r)*y
    cos_r = torch.cos(r).unsqueeze(1).unsqueeze(2)  # (N, 1, 1)
    sin_r = torch.sin(r).unsqueeze(1).unsqueeze(2)
    x = centered[..., 0:1]
    y = centered[..., 1:2]
    z = centered[..., 2:3]
    local_x = cos_r * x + sin_r * y   # (N, 8, 1)
    local_y = -sin_r * x + cos_r * y
    local_z = z

    # Oriented extents in local frame
    lx = local_x.squeeze(-1)  # (N, 8)
    ly = local_y.squeeze(-1)
    lz = local_z.squeeze(-1)
    length = lx.max(dim=1)[0] - lx.min(dim=1)[0]
    width = ly.max(dim=1)[0] - ly.min(dim=1)[0]
    height = lz.max(dim=1)[0] - lz.min(dim=1)[0]
    dims = torch.stack([length, width, height], dim=1)  # (N, 3)
    dims = dims.clamp(min=1e-4)  # avoid degenerate boxes

    return {"center": center, "dims": dims, "r": r}


def attributes_to_corners(center: torch.Tensor, dims: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Build 8 corners from (center, dims, yaw). Convention: dims = (length, width, height);
    in local frame corners are at (±l/2, ±w/2, ±h/2); local x = length, local y = width, z = height.
    Rotation r in xy: world_x = cos(r)*local_x - sin(r)*local_y, world_y = sin(r)*local_x + cos(r)*local_y.

    center: (N, 3), dims: (N, 3), r: (N,) -> (N, 8, 3)
    """
    N = center.shape[0]
    device = center.device
    l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
    half = 0.5
    # 8 corners in local: (±l/2, ±w/2, ±h/2) in order: bottom then top
    # (0,1,2,3) bottom, (4,5,6,7) top
    corners_local = torch.stack([
        -l * half, -w * half, -h * half,
        l * half, -w * half, -h * half,
        l * half, w * half, -h * half,
        -l * half, w * half, -h * half,
        -l * half, -w * half, h * half,
        l * half, -w * half, h * half,
        l * half, w * half, h * half,
        -l * half, w * half, h * half,
    ], dim=1).view(N, 8, 3)  # (N, 8, 3)

    cos_r = torch.cos(r).unsqueeze(1)
    sin_r = torch.sin(r).unsqueeze(1)
    lx = corners_local[..., 0]
    ly = corners_local[..., 1]
    lz = corners_local[..., 2]
    wx = cos_r * lx - sin_r * ly
    wy = sin_r * lx + cos_r * ly
    corners_world = torch.stack([wx, wy, lz], dim=-1)
    corners_world = corners_world + center.unsqueeze(1)
    return corners_world


def chamfer_loss_8corners(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Chamfer loss between two sets of 8 corners.
    pred: (N, 8, 3), gt: (N, 8, 3).
    For each box: (1/8) sum_i min_j ||p_i - q_j||^2 + (1/8) sum_j min_i ||p_i - q_j||^2.
    """
    diff = pred.unsqueeze(3) - gt.unsqueeze(2)
    dist_sq = (diff ** 2).sum(dim=-1)
    min_pred_to_gt, _ = dist_sq.min(dim=2)
    min_gt_to_pred, _ = dist_sq.min(dim=1)
    per_box = min_pred_to_gt.mean(dim=1) + min_gt_to_pred.mean(dim=1)
    if reduction == "mean":
        return per_box.mean()
    elif reduction == "sum":
        return per_box.sum()
    return per_box


def ovmono3d_loss(
    pred_corners_flat: torch.Tensor,
    gt_corners: torch.Tensor,
    pred_mu: torch.Tensor,
    use_smooth_l1: bool = True,
) -> tuple:
    """
    OVMono3D full loss with geometry-level disentangled supervision.

    L = sqrt(2)*exp(-mu)*L3D + mu
    L3D = L_xy + L_z + L_whl + L_r + L_all

    For each attribute a in {xy, z, dims, r}: build box B_a from (pred_a, gt_rest),
    then L_a = Chamfer(B_a, gt_corners). L_all = Chamfer(pred_corners, gt_corners).

    pred_corners_flat: (N, 24), gt_corners: (N, 8, 3), pred_mu: (N,) or (N, 1).

    Returns:
        loss_total, loss_3d (scalar for logging), loss_chamfer (L_all for logging).
    """
    N = gt_corners.shape[0]
    device = gt_corners.device

    pred_corners = pred_corners_flat.view(N, 8, 3)

    gt_flat = gt_corners.view(N, -1)
    valid = (gt_flat.abs().sum(dim=1) > 1e-6)
    if valid.sum() == 0:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        )

    pred_c = pred_corners[valid]
    gt_c = gt_corners[valid]
    n_valid = pred_c.shape[0]

    pred_attr = corners_to_attributes(pred_c)
    gt_attr = corners_to_attributes(gt_c)

    # Geometry-level disentangled losses: mixed box per attribute, then Chamfer to GT
    # L_xy: box from (pred_xy, gt_z, gt_dims, gt_r)
    center_xy = torch.cat([pred_attr["center"][:, :2], gt_attr["center"][:, 2:3]], dim=1)
    box_xy = attributes_to_corners(center_xy, gt_attr["dims"], gt_attr["r"])
    L_xy = chamfer_loss_8corners(box_xy, gt_c, reduction="mean")

    # L_z: box from (gt_xy, pred_z, gt_dims, gt_r)
    center_z = torch.cat([gt_attr["center"][:, :2], pred_attr["center"][:, 2:3]], dim=1)
    box_z = attributes_to_corners(center_z, gt_attr["dims"], gt_attr["r"])
    L_z = chamfer_loss_8corners(box_z, gt_c, reduction="mean")

    # L_whl: box from (gt_xy, gt_z, pred_dims, gt_r)
    box_whl = attributes_to_corners(gt_attr["center"], pred_attr["dims"], gt_attr["r"])
    L_whl = chamfer_loss_8corners(box_whl, gt_c, reduction="mean")

    # L_r: box from (gt_xy, gt_z, gt_dims, pred_r). Use wrapped angle for consistency.
    pred_r = _wrap_angle(pred_attr["r"])
    box_r = attributes_to_corners(gt_attr["center"], gt_attr["dims"], pred_r)
    L_r = chamfer_loss_8corners(box_r, gt_c, reduction="mean")

    # Holistic Chamfer
    L_all = chamfer_loss_8corners(pred_c, gt_c, reduction="mean")

    L3D = L_xy + L_z + L_whl + L_r + L_all

    mu = pred_mu[valid].float().mean()
    mu = torch.clamp(mu, -5.0, 10.0)
    loss_total = (2.0 ** 0.5) * torch.exp(-mu) * L3D + mu

    return loss_total, L3D.detach(), L_all.detach()
