import os
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

# VGGT imports
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


# ---------------------------------------
# Helpers
# ---------------------------------------

def set_torch_flags():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def sample_video_frames(
        video_path: str,
        max_frames: int = 1000,
        target_count: int = None,
        stride: int = None
) -> List[np.ndarray]:
    """Return a list of RGB frames sampled from the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        raise RuntimeError(f"Could not read frames from {video_path}")

    # Choose stride automatically if not given
    if target_count is not None:
        stride = max(1, total // min(max_frames, target_count))
    if stride is None:
        stride = max(1, total // min(max_frames, total))

    frames = []
    idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    return frames


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


# ---------------------------------------
# VGGT runner with windowing
# ---------------------------------------

class VGGTStaticScene:
    def __init__(
            self,
            device: str = "cuda",
            dtype_amp: torch.dtype = None,
            model_id: str = "facebook/VGGT-1B",
            image_long_side: int = 512,
            window: int = 96,
            overlap: int = 24,
    ):
        self.device = device
        if dtype_amp is None:
            major = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
            dtype_amp = torch.bfloat16 if major >= 8 else torch.float16
        self.dtype_amp = dtype_amp
        self.image_long_side = image_long_side
        self.window = window
        self.overlap = overlap

        self.model = VGGT.from_pretrained(model_id).to(device)
        self.model.eval()

    def _prep_paths(self, frames: List[np.ndarray], tmp_dir: Path) -> List[str]:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, im in enumerate(frames):
            p = tmp_dir / f"frame_{i:06d}.png"
            cv2.imwrite(str(p), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            paths.append(str(p))
        return paths

    @torch.no_grad()
    def _run_chunk(self, image_paths: List[str]):
        # VGGT expects a tensor of (T,C,H,W), after load/preprocess
        images = load_and_preprocess_images(
            image_paths,
            long_side=self.image_long_side
        ).to(self.device)

        with torch.cuda.amp.autocast(dtype=self.dtype_amp):
            # Get shared tokens
            agg_tokens, ps_idx = self.model.aggregator(images[None])  # add batch dim
            # Cameras
            pose_enc = self.model.camera_head(agg_tokens)[-1]
            extri, intra = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
            # Depths (optional but handy for dense fusion)
            depth_map, depth_conf = self.model.depth_head(agg_tokens, images, ps_idx)

        # Squeeze batch
        extri = extri.squeeze(0)  # (T, 4, 4) or (T,3,4) depending on implementation
        intra = intra.squeeze(0)  # (T, 3, 3)
        depth_map = depth_map.squeeze(0)  # (T, H, W)
        depth_conf = depth_conf.squeeze(0)
        return {
            "extri": extri, "intra": intra,
            "depth": depth_map, "depth_conf": depth_conf,
            "image_hw": images.shape[-2:]
        }

    @torch.no_grad()
    def infer_long_video(
            self,
            frame_paths: List[str],
            want_dense_points: bool = True,
            dense_stride: int = 8
    ):
        N = len(frame_paths)
        cams_extri = []
        cams_intra = []
        depths = []
        H = W = None

        # sliding windows
        win = self.window
        ov = self.overlap
        starts = list(range(0, N, win - ov))
        if starts and starts[-1] + win < N:
            starts.append(N - win)

        for s in tqdm(starts, desc="VGGT windows"):
            e = min(N, s + win)
            chunk_paths = frame_paths[s:e]
            preds = self._run_chunk(chunk_paths)

            if H is None:
                H, W = preds["image_hw"]
            # Accumulate; on overlaps keep the earliest estimate
            for i in range(s, e):
                j = i - s
                if i >= len(cams_extri):
                    cams_extri.append(preds["extri"][j].detach().cpu())
                    cams_intra.append(preds["intra"][j].detach().cpu())
                    if want_dense_points:
                        depths.append(preds["depth"][j].detach().cpu())
                # else: keep earlier estimate

        cams_extri = torch.stack(cams_extri, dim=0)  # (N, 4,4) or (N,3,4)
        cams_intra = torch.stack(cams_intra, dim=0)  # (N, 3,3)
        if want_dense_points:
            depths = torch.stack(depths, dim=0)  # (N, H, W)
        else:
            depths = None

        # Build a fused static point cloud from depth unprojection
        fused_xyz = []
        fused_rgb = []
        if want_dense_points and depths is not None:
            for i in tqdm(range(N), desc="Unproject depth"):
                K = cams_intra[i].numpy()
                ext = cams_extri[i].numpy()
                d = depths[i].numpy()
                # Downsample
                d_ds = d[::dense_stride, ::dense_stride]
                # Unproject to 3D points (camera->world) via provided util
                pts_cam = unproject_depth_map_to_point_map(
                    torch.from_numpy(d_ds).unsqueeze(0),  # (1,h,w)
                    torch.from_numpy(cams_extri[i:i + 1]),
                    torch.from_numpy(cams_intra[i:i + 1])
                ).squeeze(0).numpy().reshape(-1, 3)
                fused_xyz.append(pts_cam)

        if fused_xyz:
            fused_xyz = np.concatenate(fused_xyz, axis=0)
        else:
            fused_xyz = np.zeros((0, 3), dtype=np.float32)

        return {
            "K": cams_intra,  # (N,3,3)
            "Tcw": cams_extri,  # (N,4,4) [OpenCV: camera-from-world]
            "depths": depths,  # (N,H,W) or None
            "fused_xyz": fused_xyz,  # (M,3)
            "image_size": (int(H), int(W))
        }


# ---------------------------------------
# Minimal Torch Bundle Adjustment
# ---------------------------------------

class TorchBA(nn.Module):
    """
    Optimize camera extrinsics (axis-angle + translation) and 3D points to
    minimize reprojection error, with the first camera fixed to gauge the system.
    """

    def __init__(self, Ks: torch.Tensor, R_init: torch.Tensor, t_init: torch.Tensor,
                 tracks_uv: torch.Tensor, vis_mask: torch.Tensor):
        """
        Ks: (N,3,3)
        R_init: (N,3,3)
        t_init: (N,3,1)
        tracks_uv: (N,P,2) pixel coordinates
        vis_mask: (N,P) boolean
        """
        super().__init__()
        device = Ks.device
        self.register_buffer("Ks", Ks)
        self.register_buffer("vis", vis_mask.float())
        self.N, self.P = tracks_uv.shape[:2]

        # Initialize 3D from linear triangulation (DLT) using first two well-visible views
        self.X = nn.Parameter(self._triangulate_dlt(Ks, R_init, t_init, tracks_uv, vis_mask))  # (P,3)

        # Parameterize rotations (except first) via axis-angle, translations as free
        self.rot_aa = nn.Parameter(torch.zeros(self.N, 3, device=device))
        self.t = nn.Parameter(t_init.squeeze(-1).clone())  # (N,3)

        # Lock first camera
        self.rot_aa.data[0] = torch.zeros(3, device=device)
        self.t.data[0] = t_init[0].squeeze(-1)

        # Set initial rotations from R_init
        with torch.no_grad():
            # Convert R_init to axis-angle approximately via log(R)
            for i in range(self.N):
                R = R_init[i]
                aa = self._log_so3(R)
                self.rot_aa.data[i] = aa

        self.tracks_uv = nn.Parameter(tracks_uv, requires_grad=False)

    def _log_so3(self, R: torch.Tensor) -> torch.Tensor:
        """Matrix log for rotation -> axis angle (approx, stable for small angles)."""
        cos = (R.trace() - 1) / 2
        cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)
        theta = torch.acos(cos)
        w = torch.tensor([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1]
        ], device=R.device) / (2 * torch.sin(theta) + 1e-6)
        return w * theta

    def forward(self):
        B = self.N
        # Build R from axis-angle
        R = axis_angle_to_R(self.rot_aa)  # (N,3,3)
        t = self.t.unsqueeze(-1)  # (N,3,1)
        uv_pred = se3_project(self.Ks, R, t, self.X)  # (N,P,2)
        # Reprojection error
        diff = (uv_pred - self.tracks_uv) * self.vis.unsqueeze(-1)
        # Charbonnier robust loss
        loss = torch.sqrt(diff.pow(2).sum(dim=-1) + 1e-6).mean()
        return loss

    def _triangulate_dlt(self, Ks, R, t, uv, vis):
        """Simple linear triangulation from first two strongest views per point."""
        device = Ks.device
        N, P, _ = uv.shape
        X = torch.zeros(P, 3, device=device)

        # pick two views with visibility
        vis_idx = vis.nonzero(as_tuple=False)  # (M,2): [n, p]
        # naive: use first and last visible frame for each point
        first = torch.full((P,), -1, dtype=torch.long, device=device)
        last = torch.full((P,), -1, dtype=torch.long, device=device)
        for n, p in vis_idx:
            if first[p] == -1:
                first[p] = n
            last[p] = n

        for p in range(P):
            i = int(first[p].item()) if first[p] >= 0 else 0
            j = int(last[p].item()) if last[p] >= 0 else min(1, N - 1)
            Pi = Ks[i] @ torch.cat([R[i], t[i]], dim=-1)
            Pj = Ks[j] @ torch.cat([R[j], t[j]], dim=-1)
            xi, yi = uv[i, p]
            xj, yj = uv[j, p]
            A = torch.stack([
                xi * Pi[2] - Pi[0],
                yi * Pi[2] - Pi[1],
                xj * Pj[2] - Pj[0],
                yj * Pj[2] - Pj[1],
            ], dim=0)  # (4,4)
            # Solve AX=0 via SVD
            _, _, V = torch.linalg.svd(A)
            Xh = V[-1]
            X[p] = (Xh[:3] / (Xh[3] + 1e-8))
        return X


def run_torch_ba(
        Ks: torch.Tensor, Tcw: torch.Tensor,
        key_uv: torch.Tensor, vis_mask: torch.Tensor,
        iters: int = 200, lr: float = 1e-2
):
    """
    Ks: (N,3,3)
    Tcw: (N,4,4) camera-from-world
    key_uv: (N,P,2)
    vis_mask: (N,P) bool
    """
    device = Ks.device
    # Extract initial R,t from Tcw
    R_init = Tcw[:, :3, :3]
    t_init = Tcw[:, :3, 3:4]
    ba = TorchBA(Ks, R_init, t_init, key_uv, vis_mask).to(device)
    opt = Adam([p for p in ba.parameters() if p.requires_grad], lr=lr)
    pbar = tqdm(range(iters), desc="Torch BA")
    for _ in pbar:
        opt.zero_grad()
        loss = ba()
        loss.backward()
        opt.step()
        pbar.set_postfix({"loss": float(loss.item())})
    # Rebuild Tcw
    with torch.no_grad():
        R = axis_angle_to_R(ba.rot_aa)
        t = ba.t.unsqueeze(-1)
        Tcw_new = torch.eye(4, device=device).unsqueeze(0).repeat(Ks.shape[0], 1, 1)
        Tcw_new[:, :3, :3] = R
        Tcw_new[:, :3, 3:4] = t
        X = ba.X
    return Tcw_new, X


# ---------------------------------------
# Pipeline entry
# ---------------------------------------

def reconstruct_static_scene(
        video_path: str,
        out_dir: str = "vggt_static_out",
        max_frames: int = 1000,
        target_frames: int = 480,
        image_long_side: int = 512,
        window: int = 96,
        overlap: int = 24,
        tracks_per_window: int = 1024,  # for BA
        do_bundle_adjustment: bool = True
):
    set_torch_flags()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Info] Using device={device}")

    out = Path(out_dir);
    out.mkdir(parents=True, exist_ok=True)
    frames = sample_video_frames(video_path, max_frames=max_frames, target_count=target_frames)
    print(f"[Info] Sampled {len(frames)} frames")
    tmp_images = out / "frames_raw"
    # Prep on-disk PNGs (VGGT loader expects paths)
    runner = VGGTStaticScene(device=device, image_long_side=image_long_side, window=window, overlap=overlap)
    frame_paths = runner._prep_paths(frames, tmp_images)

    # VGGT predictions with windowing
    preds = runner.infer_long_video(frame_paths, want_dense_points=True, dense_stride=8)
    K = preds["K"].to(device=device, dtype=torch.float32)
    Tcw = preds["Tcw"].to(device=device, dtype=torch.float32)
    fused_xyz = preds["fused_xyz"]
    H, W = preds["image_size"]

    # Save initial cameras
    (out / "cameras_init.json").write_text(json.dumps({
        "image_size": [H, W],
        "K": K.cpu().numpy().tolist(),
        "Tcw": Tcw.cpu().numpy().tolist()
    }))

    # Optional: Bundle Adjustment using VGGT tracks on a grid
    if do_bundle_adjustment:
        # Create a grid of query points in the FIRST window extent
        npts = min(tracks_per_window, (H // 8) * (W // 8))
        query_points = make_grid_points(W, H, npts)  # (P,2), xy order

        # Re-run aggregator and track head ONCE across all frames if feasible,
        # else do it chunkwise and stitch (kept simple here: single pass; for >~256 frames,
        # window size ensures memory okay on A40 40GB).
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=runner.dtype_amp):
            images = load_and_preprocess_images(frame_paths, long_side=image_long_side).to(device)
            agg_tokens, ps_idx = runner.model.aggregator(images[None])
            q = torch.from_numpy(query_points).to(device=device, dtype=torch.float32)[None]  # (1,P,2)
            track_list, vis_score, conf_score = runner.model.track_head(agg_tokens, images, ps_idx, query_points=q)

        # track_list: list per stage; use last scale
        uv_all = track_list[-1].squeeze(0)  # (N,P,2)
        vis = (vis_score.squeeze(0) > 0.5) & (conf_score.squeeze(0) > 0.5)  # (N,P)

        # Run a light BA
        Tcw_ba, X_ba = run_torch_ba(K, Tcw, uv_all, vis, iters=200, lr=1e-2)
        Tcw = Tcw_ba

        # Optionally augment fused cloud with BA points
        fused_xyz = np.concatenate([fused_xyz, X_ba.detach().cpu().numpy()], axis=0)

    # Simple voxel downsample (no Open3D)
    if fused_xyz.shape[0] > 0:
        # Remove NaNs/Infs and big outliers
        fused_xyz = fused_xyz[~np.isnan(fused_xyz).any(1)]
        q = np.quantile(np.linalg.norm(fused_xyz, axis=1), 0.99)
        fused_xyz = fused_xyz[np.linalg.norm(fused_xyz, axis=1) < (q * 1.5)]
        # Save PLY
        write_ply(fused_xyz.astype(np.float32), None, str(out / "static_scene_fused.ply"))

    # Save refined cameras
    (out / "cameras_refined.json").write_text(json.dumps({
        "image_size": [H, W],
        "K": K.cpu().numpy().tolist(),
        "Tcw": Tcw.detach().cpu().numpy().tolist()
    }))

    print(f"[Done] Outputs in: {out.resolve()}")
    print(" - static_scene_fused.ply")
    print(" - cameras_init.json / cameras_refined.json")
    print(" - frames_raw/")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="vggt_static_out")
    ap.add_argument("--max_frames", type=int, default=1000)
    ap.add_argument("--target_frames", type=int, default=480)
    ap.add_argument("--long_side", type=int, default=512)
    ap.add_argument("--window", type=int, default=96)
    ap.add_argument("--overlap", type=int, default=24)
    ap.add_argument("--tracks_per_window", type=int, default=1024)
    ap.add_argument("--no_ba", action="store_true", help="Disable bundle adjustment")
    args = ap.parse_args()

    reconstruct_static_scene(
        video_path=args.video,
        out_dir=args.out,
        max_frames=args.max_frames,
        target_frames=args.target_frames,
        image_long_side=args.long_side,
        window=args.window,
        overlap=args.overlap,
        tracks_per_window=args.tracks_per_window,
        do_bundle_adjustment=(not args.no_ba)
    )
