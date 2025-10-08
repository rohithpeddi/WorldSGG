import math

import numpy as np
import pycolmap
import torch
from vggt.dependency.np_to_pycolmap import _build_pycolmap_intri


# ---------------------------------------
# Helpers
# ---------------------------------------

def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N = len(extrinsics)
    P = len(points3d)

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            image_id=fidx + 1, name=f"image_{fidx + 1}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} does not have any points")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction



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
    theta = torch.norm(axis_angle + 1e-12, dim=-1, keepdim=True)  # (B,1)
    k = axis_angle / torch.clamp(theta, min=1e-12)
    k = torch.nan_to_num(k)
    B = axis_angle.shape[0]
    Kx = torch.zeros(B, 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    Kx[:, 0, 1] = -k[:, 2]
    Kx[:, 0, 2] = k[:, 1]
    Kx[:, 1, 0] = k[:, 2]
    Kx[:, 1, 2] = -k[:, 0]
    Kx[:, 2, 0] = -k[:, 1]
    Kx[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)[None].repeat(B, 1, 1)
    sin = torch.sin(theta)[:, None]
    cos = torch.cos(theta)[:, None]
    R = I + sin * Kx + (1 - cos) * (Kx @ Kx)
    return R


def log_so3(R: torch.Tensor) -> torch.Tensor:
    """Matrix log for rotation -> axis angle (approx, robust). (3,3)->(3,)"""
    tr = torch.trace(R)
    cos = torch.clamp((tr - 1) / 2, -1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(cos)
    if theta < 1e-8:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    w = torch.tensor([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], device=R.device, dtype=R.dtype) / (2 * torch.sin(theta))
    return w * theta
