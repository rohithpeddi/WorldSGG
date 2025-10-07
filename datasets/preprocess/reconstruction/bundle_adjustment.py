from typing import Optional

import numpy as np
import torch

from datasets.preprocess.reconstruction.utils import axis_angle_to_R, se3_project, log_so3


class TorchBA(torch.nn.Module):

    def __init__(
            self,
            Ks: torch.Tensor,  # (N,3,3)
            Tcw_init: torch.Tensor,  # (N,4,4) camera-from-world
            tracks_uv: torch.Tensor,  # (N,P,2)
            vis_mask: torch.Tensor,  # (N,P) bool
            X_init: Optional[torch.Tensor] = None,  # (P,3) initial 3D
    ):
        super().__init__()
        device = Ks.device
        self.register_buffer("Ks", Ks)
        self.register_buffer("vis", vis_mask.float())
        self.N, self.P = tracks_uv.shape[:2]

        # Initialize rotations/translations from Tcw
        R0 = Tcw_init[:, :3, :3]
        t0 = Tcw_init[:, :3, 3]
        self.rot_aa = torch.nn.Parameter(torch.zeros(self.N, 3, device=device))
        self.t = torch.nn.Parameter(t0.clone())
        with torch.no_grad():
            for i in range(self.N):
                self.rot_aa[i] = log_so3(R0[i])
        # lock first camera
        self.rot_aa.requires_grad_(True)
        self.t.requires_grad_(True)

        # Initialize 3D points
        if X_init is None:
            # Linear triangulation from first & last visible views per point
            self.X = torch.nn.Parameter(self._triangulate_linear(R0, t0[:, :, None], tracks_uv, vis_mask))
        else:
            self.X = torch.nn.Parameter(X_init.clone().to(device))

        self.tracks_uv = torch.nn.Parameter(tracks_uv, requires_grad=False)

    def forward(self):
        R = axis_angle_to_R(self.rot_aa)  # (N,3,3)
        t = self.t[:, :, None]  # (N,3,1)
        uv_pred = se3_project(self.Ks, R, t, self.X)  # (N,P,2)
        diff = (uv_pred - self.tracks_uv) * self.vis[:, :, None]
        loss = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6).mean()
        return loss

    def _triangulate_linear(self, R, t, uv, vis):
        device = R.device
        N, P = uv.shape[:2]
        X = torch.zeros(P, 3, device=device, dtype=R.dtype)
        # pick first and last visible view per point
        vis_idx = vis.nonzero(as_tuple=False)  # (M,2)
        first = torch.full((P,), -1, dtype=torch.long, device=device)
        last = torch.full((P,), -1, dtype=torch.long, device=device)
        for n, p in vis_idx:
            if first[p] == -1:
                first[p] = n
            last[p] = n
        for p in range(P):
            i = int(first[p].item()) if first[p] >= 0 else 0
            j = int(last[p].item()) if last[p] >= 0 else min(1, N - 1)
            Ki = self.Ks[i]
            Kj = self.Ks[j]
            Pi = Ki @ torch.cat([R[i], t[i]], dim=-1)  # (3,4)
            Pj = Kj @ torch.cat([R[j], t[j]], dim=-1)
            xi, yi = uv[i, p]
            xj, yj = uv[j, p]
            A = torch.stack([
                xi * Pi[2] - Pi[0],
                yi * Pi[2] - Pi[1],
                xj * Pj[2] - Pj[0],
                yj * Pj[2] - Pj[1],
            ], dim=0)  # (4,4)
            _, _, V = torch.linalg.svd(A)
            Xh = V[-1]
            X[p] = Xh[:3] / torch.clamp(Xh[3], min=1e-9)
        return X


# ==========================
# Torch Bundle Adjustment (optional backend)
# ==========================


def run_torch_ba(
        Ks_np: np.ndarray,  # (N,3,3)
        Tcw_np: np.ndarray,  # (N,4,4)
        tracks_uv_np: np.ndarray,  # (N,P,2)
        vis_mask_np: np.ndarray,  # (N,P)
        X_init_np: Optional[np.ndarray] = None,
        iters: int = 200,
        lr: float = 1e-2,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = torch.from_numpy(Ks_np).to(device=device, dtype=torch.float32)
    Tcw = torch.from_numpy(Tcw_np).to(device=device, dtype=torch.float32)
    tracks_uv = torch.from_numpy(tracks_uv_np).to(device=device, dtype=torch.float32)
    vis_mask = torch.from_numpy(vis_mask_np.astype(bool)).to(device=device)
    X_init = torch.from_numpy(X_init_np).to(device=device, dtype=torch.float32) if X_init_np is not None else None

    ba = TorchBA(Ks, Tcw, tracks_uv, vis_mask, X_init=X_init).to(device)
    opt = torch.optim.Adam([p for p in ba.parameters() if p.requires_grad], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = ba()
        loss.backward()
        opt.step()
    with torch.no_grad():
        R = axis_angle_to_R(ba.rot_aa)
        t = ba.t[:, :, None]
        Tcw_new = torch.eye(4, device=device).unsqueeze(0).repeat(Ks.shape[0], 1, 1)
        Tcw_new[:, :3, :3] = R
        Tcw_new[:, :3, 3:4] = t
        X_new = ba.X
    return Tcw_new.detach().cpu().numpy(), X_new.detach().cpu().numpy()