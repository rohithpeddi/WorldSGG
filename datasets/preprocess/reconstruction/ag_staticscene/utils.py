import json
import math
from pathlib import Path
from typing import List, Tuple, Dict

import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam

# ---------------------------------------
# Helpers
# ---------------------------------------

def set_torch_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def write_ply(points_xyz: np.ndarray, colors: np.ndarray, out_path: str):
    assert points_xyz.shape[1] == 3
    if colors is None:
        colors = np.zeros_like(points_xyz, dtype=np.uint8)
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points_xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(out_path, "w") as f:
        f.write(header)
        for (x, y, z), (r, g, b) in zip(points_xyz, colors):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def to_torch(arr, device, dtype=torch.float32):
    return torch.as_tensor(arr, device=device, dtype=dtype)


def make_grid_points(width: int, height: int, n_points: int) -> np.ndarray:
    """Uniform grid of (x,y) image points."""
    xs = np.linspace(16, width - 16, int(math.sqrt(n_points)))
    ys = np.linspace(16, height - 16, int(math.sqrt(n_points)))
    pts = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    if len(pts) > n_points:
        pts = pts[:n_points]
    return pts


def se3_project(K, R, t, X):
    """Project 3D points with intrinsics K (B,3,3), R (B,3,3), t (B,3,1), X (P,3)."""
    # Expand to batch
    B = R.shape[0]
    P = X.shape[0]
    X_h = torch.cat([X, torch.ones(P, 1, device=X.device, dtype=X.dtype)], dim=-1)  # (P,4)
    RT = torch.cat([R, t], dim=-1)  # (B,3,4)
    PX = (K @ (RT @ X_h.T).transpose(1, 2))  # (B,3,P)
    uv = PX[:, :2, :] / PX[:, 2:3, :].clamp(min=1e-6)
    return uv.transpose(1, 2)  # (B,P,2)


def axis_angle_to_R(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues for (B,3) -> (B,3,3)."""
    theta = torch.norm(axis_angle + 1e-9, dim=-1, keepdim=True)
    k = axis_angle / theta
    k = torch.nan_to_num(k)
    Kx = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    Kx[:, 0, 1] = -k[:, 2];
    Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2];
    Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1];
    Kx[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).repeat(axis_angle.shape[0], 1, 1)
    sin = torch.sin(theta).unsqueeze(-1)
    cos = torch.cos(theta).unsqueeze(-1)
    R = I + sin * Kx + (1 - cos) * (Kx @ Kx)
    return R