"""
OVMono3D-style 3D detection loss: uncertainty-weighted L3D with
geometry-level disentangled attribute losses and Chamfer (holistic) loss.

L_i = sqrt(2) * exp(-mu_i) * L3D_i + mu_i   (per-sample, then averaged)
L3D_i = sum_a L3D_i^(a) + L3D_i^all

Disentangled (geometry-level): For each attribute group a (xy, z, dims, r),
  - Take prediction for attribute a and ground truth for all other attributes.
  - Reconstruct a full 3D box (8 corners) from this mixed set.
  - L3D^(a) = Chamfer_SL1(reconstructed_box, GT_box).

L3D^all: Chamfer_SL1(pred_corners, gt_corners).

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
    a, b, c = cov[:, 0, 0], cov[:, 0, 1], cov[:, 1, 1]
    delta = (a - c) * (a - c) / 4 + b * b
    delta = torch.clamp(delta, min=1e-12)
    sqrt_d = torch.sqrt(delta)
    # Eigenvector for larger eigenvalue
    lam = (a + c) / 2 + sqrt_d
    ev_x = b
    ev_y = lam - a
    # Normalize
    nrm = torch.sqrt(ev_x * ev_x + ev_y * ev_y).clamp(min=1e-8)
    ev_x = ev_x / nrm
    ev_y = ev_y / nrm
    r = torch.atan2(ev_y, ev_x)  # (N,) in [-pi, pi]

    # Project corners onto local frame
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

    # Canonicalize: enforce length >= width to remove 90° PCA ambiguity.
    # When length ≈ width, PCA's principal axis is ill-defined and can flip
    # unpredictably. By always assigning the longer extent to "length" (and
    # rotating yaw by π/2 to compensate), we get a stable decomposition.
    swap = width > length
    if swap.any():
        new_length = torch.where(swap, width, length)
        new_width = torch.where(swap, length, width)
        length, width = new_length, new_width
        r = torch.where(swap, _wrap_angle(r + (3.14159265358979 / 2.0)), r)

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
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    return _corners_from_precomputed(center, dims, cos_r, sin_r)


def _corners_from_precomputed(
    center: torch.Tensor, dims: torch.Tensor,
    cos_r: torch.Tensor, sin_r: torch.Tensor,
) -> torch.Tensor:
    """Build 8 corners from pre-computed cos/sin of yaw (avoids redundant trig)."""
    N = center.shape[0]
    l, w, h = dims[:, 0], dims[:, 1], dims[:, 2]
    half = 0.5
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

    cr = cos_r.unsqueeze(1)
    sr = sin_r.unsqueeze(1)
    lx = corners_local[..., 0]
    ly = corners_local[..., 1]
    lz = corners_local[..., 2]
    wx = cr * lx - sr * ly
    wy = sr * lx + cr * ly
    corners_world = torch.stack([wx, wy, lz], dim=-1)
    corners_world = corners_world + center.unsqueeze(1)
    return corners_world


def _chamfer_pairwise_dist(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute per-box Chamfer distance using smooth L1 (Huber). pred, gt: (N, 8, 3) -> (N,).
    
    Uses smooth L1 instead of L2² for stability: linear growth for large errors
    prevents catastrophic loss values when predictions are far from GT (training start).
    """
    diff = pred.unsqueeze(2) - gt.unsqueeze(1)  # (N, 8, 8, 3)
    # Smooth L1 per-coordinate, then sum over xyz → per-pair distance
    dist = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction='none').sum(dim=-1)  # (N, 8, 8)
    return dist.min(dim=2).values.mean(dim=1) + dist.min(dim=1).values.mean(dim=1)


def chamfer_loss_8corners(pred: torch.Tensor, gt: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Chamfer loss between two sets of 8 corners using smooth L1 (Huber) distance.
    pred: (N, 8, 3), gt: (N, 8, 3).
    For each box: (1/8) sum_i min_j SL1(p_i, q_j) + (1/8) sum_j min_i SL1(p_i, q_j).
    """
    per_box = _chamfer_pairwise_dist(pred, gt)
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

    L_i = sqrt(2)*exp(-mu_i)*L3D_i + mu_i   (per-sample, then averaged)
    L3D_i = L_xy_i + L_z_i + L_whl_i + L_r_i + L_all_i

    Chamfer distance uses smooth L1 (Huber) instead of L2² for stability.
    GT attributes and trig values are computed ONCE and reused across all
    4 disentangled attribute losses. All 5 Chamfer computations are batched
    into a single fused call.

    pred_corners_flat: (N, 24), gt_corners: (N, 8, 3), pred_mu: (N,) or (N, 1).

    Returns:
        loss_total, loss_3d (scalar for logging), loss_chamfer (L_all for logging).
    """
    N = gt_corners.shape[0]
    device = gt_corners.device

    pred_corners = pred_corners_flat.view(N, 8, 3)

    # Filter out invalid GT boxes: zero-padding, NaN/Inf, or extreme magnitudes
    gt_flat = gt_corners.view(N, -1)
    non_zero = gt_flat.abs().sum(dim=1) > 1e-6
    finite = torch.isfinite(gt_flat).all(dim=1)
    reasonable = gt_flat.abs().max(dim=1).values < 1e6
    valid = non_zero & finite & reasonable
    if valid.sum() == 0:
        return (
            torch.tensor(0.0, device=device, requires_grad=True),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        )

    pred_c = pred_corners[valid]
    gt_c = gt_corners[valid]
    n_valid = pred_c.shape[0]

    # Decompose corners → (center, dims, yaw) for both pred and GT
    pred_attr = corners_to_attributes(pred_c)
    gt_attr = corners_to_attributes(gt_c)

    # Pre-compute GT trig values ONCE (reused across 3 of 4 disentangled losses)
    gt_cos_r = torch.cos(gt_attr["r"])
    gt_sin_r = torch.sin(gt_attr["r"])

    # Build all 5 disentangled boxes in one go, then batch Chamfer
    # L_xy: (pred_xy, gt_z, gt_dims, gt_r)
    center_xy = torch.cat([pred_attr["center"][:, :2], gt_attr["center"][:, 2:3]], dim=1)
    box_xy = _corners_from_precomputed(center_xy, gt_attr["dims"], gt_cos_r, gt_sin_r)

    # L_z: (gt_xy, pred_z, gt_dims, gt_r)
    center_z = torch.cat([gt_attr["center"][:, :2], pred_attr["center"][:, 2:3]], dim=1)
    box_z = _corners_from_precomputed(center_z, gt_attr["dims"], gt_cos_r, gt_sin_r)

    # L_whl: (gt_center, pred_dims, gt_r)
    box_whl = _corners_from_precomputed(gt_attr["center"], pred_attr["dims"], gt_cos_r, gt_sin_r)

    # L_r: (gt_center, gt_dims, pred_r)  — only this one needs new trig
    pred_r = _wrap_angle(pred_attr["r"])
    pred_cos_r = torch.cos(pred_r)
    pred_sin_r = torch.sin(pred_r)
    box_r = _corners_from_precomputed(gt_attr["center"], gt_attr["dims"], pred_cos_r, pred_sin_r)

    # Batch all 5 Chamfer computations into one: stack (5*n_valid, 8, 3)
    all_pred = torch.cat([box_xy, box_z, box_whl, box_r, pred_c], dim=0)   # (5*n, 8, 3)
    all_gt = gt_c.repeat(5, 1, 1)                                           # (5*n, 8, 3)
    all_chamfer = _chamfer_pairwise_dist(all_pred, all_gt)                  # (5*n,)

    # Normalize by GT box diagonal² to make loss scale-invariant.
    # Clamp min=0.25 (diagonal≥0.5m) prevents small objects (phone, doorknob)
    # from amplifying the loss 30-100× via tiny denominators.
    gt_diag_sq = ((gt_c.max(dim=1).values - gt_c.min(dim=1).values) ** 2).sum(dim=1).clamp(min=0.25)
    gt_diag_sq_rep = gt_diag_sq.repeat(5)  # (5*n,)
    all_chamfer = all_chamfer / gt_diag_sq_rep

    # Split back into 5 per-sample loss components (each is (n_valid,))
    L_xy, L_z, L_whl, L_r, L_all = all_chamfer.split(n_valid)

    # Per-sample L3D before any reduction
    L3D_per_sample = L_xy + L_z + L_whl + L_r + L_all  # (n_valid,)

    # Per-sample uncertainty weighting: L_i = sqrt(2) * exp(-mu_i) * L3D_i + mu_i
    # Each box's mu_i individually modulates its own loss, so the network can
    # learn to down-weight noisy/hard samples rather than using a global knob.
    mu = pred_mu[valid].float().view(-1)  # (n_valid,)
    mu = torch.clamp(mu, -5.0, 10.0)
    loss_per_sample = (2.0 ** 0.5) * torch.exp(-mu) * L3D_per_sample + mu
    loss_total = loss_per_sample.mean()

    # Scalar summaries for logging
    L3D = L3D_per_sample.mean()

    return loss_total, L3D.detach(), L_all.mean().detach()

