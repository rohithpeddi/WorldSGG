import os
from typing import Optional, Tuple, List, Dict, Callable

import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk
import trimesh
from scipy.spatial.transform import Rotation

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline


def _faces_u32(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"`faces` must be (F,3), got {faces.shape}")
    return faces.astype(np.uint32, copy=False)


def _color_from_id(tid: int) -> Tuple[int, int, int]:
    """Deterministic vivid-ish color per track id."""
    rng = np.random.RandomState(tid * 9973 + 123)
    return tuple(rng.randint(60, 235, size=3).tolist())


def _pinhole_from_fov(W: int, H: int, fov_y_rad: float) -> Tuple[float, float, float, float]:
    """Compute fx, fy, cx, cy from vertical FOV (in radians) and resolution."""
    fy = (H * 0.5) / np.tan(0.5 * fov_y_rad)
    fx = fy * (W / H)
    cx = W * 0.5
    cy = H * 0.5
    return fx, fy, cx, cy


def predictions_to_glb_with_static(
        predictions: dict,
        *,
        conf_min: float = 0.5,           # 0..1 threshold on predictions["conf"]
        filter_by_frames: str = "all",   # e.g. "12:..." to use only frame index 12
) -> tuple[trimesh.Scene, np.ndarray, np.ndarray]:
    """
    Build a GLB-ready trimesh.Scene from VGGT-style predictions AND return a background
    point cloud (xyz,rgb) in the original world frame.

    Inputs (expected prediction keys):
      - points:  (S, H, W, 3) float32   world-space points
      - conf:    (S, H, W)    float32   confidence per point in [0,1] (fallback: ones)
      - images:  (S, H, W, 3) or (S, 3, H, W) float32 in [0,1] for colors
      - camera_poses: (S, 3, 4) or (S,4,4); kept for signature parity (not visualized here)

    Returns:
      scene_3d      : trimesh.Scene with a point cloud (rotated for nicer viewing)
      static_points : (N,3) float32 background points in ORIGINAL world frame
      static_colors : (N,3) uint8   RGB colors aligned with static_points
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # ------------ Frame selection ------------
    selected_frame_idx = None
    if filter_by_frames not in ("all", "All"):
        try:
            selected_frame_idx = int(str(filter_by_frames).split(":")[0])
        except (ValueError, IndexError):
            selected_frame_idx = None

    pts = predictions["points"]
    conf = predictions.get("conf", np.ones_like(pts[..., 0], dtype=np.float32))
    imgs = predictions["images"]
    cam_poses = predictions.get("camera_poses", None)

    if selected_frame_idx is not None:
        pts = pts[selected_frame_idx][None]
        conf = conf[selected_frame_idx][None]
        imgs = imgs[selected_frame_idx][None]
        cam_poses = cam_poses[selected_frame_idx][None] if cam_poses is not None else None

    # ------------ Color layout handling ------------
    if imgs.ndim == 4 and imgs.shape[1] == 3:  # NCHW -> NHWC
        imgs_nhwc = np.transpose(imgs, (0, 2, 3, 1))
    else:
        imgs_nhwc = imgs
    colors_rgb = (imgs_nhwc.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    # ------------ Confidence filtering ------------
    verts = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1).astype(np.float32)

    thr = float(conf_min) if conf_min is not None else 0.1
    mask = (conf_flat >= thr) & (conf_flat > 1e-5)

    verts = verts[mask]
    colors_rgb = colors_rgb[mask]

    if verts.size == 0:
        # robust fallback
        verts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors_rgb = np.array([[255, 255, 255]], dtype=np.uint8)

    # Keep an ORIGINAL copy for the background return (world frame)
    static_points = verts.astype(np.float32, copy=True)
    static_colors = colors_rgb.astype(np.uint8, copy=True)

    # ------------ Build Scene (with visualization alignment rotation) ------------
    scene_3d = trimesh.Scene()
    point_cloud = trimesh.points.PointCloud(vertices=verts, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud)

    # Nice-view rotation: R = R_y(100°) @ R_x(155°)
    align_R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = align_R
    scene_3d.apply_transform(T)

    # (Optional) If you later want to visualize camera frustums/axes, add them here
    # using cam_poses if available. Currently `show_cam` is a no-op for signature parity.
    R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    R = R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    static_points = static_points @ R.T  # undo rotation

    return scene_3d, static_points, static_colors

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

# ---------- Umeyama Sim(3) ----------
def umeyama_sim3(A, B, allow_reflection=False):
    """
    Solve B ≈ s * R * A + t, with A,B of shape (N,3).
    Returns (s, R, t) with R (3,3), t (3,).
    """
    assert A.ndim == 2 and B.ndim == 2 and A.shape == B.shape and A.shape[1] == 3
    N = A.shape[0]
    muA, muB = A.mean(0), B.mean(0)
    A0, B0 = A - muA, B - muB
    Sigma = (B0.T @ A0) / N
    U, D, Vt = np.linalg.svd(Sigma)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        D[-1] *= -1
        R = U @ Vt
    varA = (A0 ** 2).sum() / N
    s = D.sum() / varA
    t = muB - s * (R @ muA)
    return float(s), R.astype(np.float64), t.astype(np.float64)

# ---------- Camera center helpers ----------
def centers_from_extrinsics_wc(Rwc, Twc):
    """
    camera→world extrinsics: x_w = Rwc x_c + Twc  =>  center is Twc.
    Rwc: (...,3,3), Twc: (...,3)
    """
    C = Twc.reshape(-1, 3).copy()
    return C

def centers_from_extrinsics_cw(Rcw, Tcw):
    """
    world→camera extrinsics: x_c = Rcw x_w + Tcw  =>  center is -Rcw^T Tcw.
    Rcw: (...,3,3), Tcw: (...,3)
    """
    Rcw = Rcw.reshape(-1, 3, 3)
    Tcw = Tcw.reshape(-1, 3)
    C = -(np.transpose(Rcw, (0, 2, 1)) @ Tcw[..., None])[..., 0]
    return C

def _split_R_t_from_3x4(E):
    """E: (S,3,4) -> R( S,3,3 ), t( S,3 )"""
    R = E[..., :3]
    t = E[..., 3]
    return R, t

def _coerce_R_t_stack(cam_poses):
    """
    Accepts (S,3,4) or (S,4,4) or dict with R/t.
    Returns a list of candidate (centers, tag), where tag ∈ {'wc','cw'}.
    """
    S = None
    candidates = []

    if isinstance(cam_poses, dict):
        # try common key patterns
        if {'Rwc', 'Twc'} <= cam_poses.keys():
            C_wc = centers_from_extrinsics_wc(np.asarray(cam_poses['Rwc']),
                                              np.asarray(cam_poses['Twc']))
            candidates.append((C_wc, 'wc'))
        if {'Rcw', 'Tcw'} <= cam_poses.keys():
            C_cw = centers_from_extrinsics_cw(np.asarray(cam_poses['Rcw']),
                                              np.asarray(cam_poses['Tcw']))
            candidates.append((C_cw, 'cw'))
        return candidates

    E = np.asarray(cam_poses)
    if E.ndim == 3 and E.shape[1:] == (3, 4):
        R, t = _split_R_t_from_3x4(E)
        # Treat as BOTH possibilities; we'll choose by residuals
        C_wc = centers_from_extrinsics_wc(R, t)   # assume camera→world
        C_cw = centers_from_extrinsics_cw(R, t)   # assume world→camera
        candidates.extend([(C_wc, 'wc'), (C_cw, 'cw')])
    elif E.ndim == 3 and E.shape[1:] == (4, 4):
        R = E[:, :3, :3]
        t = E[:, :3, 3]
        C_wc = centers_from_extrinsics_wc(R, t)
        C_cw = centers_from_extrinsics_cw(R, t)
        candidates.extend([(C_wc, 'wc'), (C_cw, 'cw')])
    else:
        raise ValueError(f"Unsupported camera_poses shape: {E.shape}")

    return candidates

def _select_frames_with_baseline(C, min_pair_dist=0.05, max_frames=200):
    """
    Keep frames that contribute baseline; helps robustness.
    min_pair_dist in scene units; tune if your units are meters.
    """
    C = np.asarray(C)
    if len(C) <= 3:
        return np.arange(len(C))
    keep = [0]
    last = 0
    for i in range(1, len(C)):
        if np.linalg.norm(C[i] - C[last]) >= min_pair_dist:
            keep.append(i)
            last = i
        if len(keep) >= max_frames:
            break
    if len(keep) < max(4, min(10, len(C)//10)):  # fall back if too few
        return np.arange(min(len(C), max_frames))
    return np.asarray(keep)

# ---------- Build Sim(3) from cameras ----------
def sim3_from_cameras(predictions, results, frame_map=None, min_pair_dist=0.05):
    """
    Compute Sim(3) that maps HUMAN world -> STATIC world.

    Args:
      predictions: dict with predictions["camera_poses"] as (S,3,4)/(S,4,4) or dict (Rwc/Twc or Rcw/Tcw)
      results:     dict with results['camera_world']['Rwc'], ['Twc'] (camera→world)
      frame_map:   Optional np.array of indices mapping human frames to static frames.
                   If None, assumes 1:1 up to min length.

    Returns: (s, R, t, chosen_tag), where tag ∈ {'wc','cw'} for static pose convention tried.
    """
    # Human camera centers
    CW = results.get('camera_world', None)
    if CW is None or 'Rwc' not in CW or 'Twc' not in CW:
        raise KeyError("results['camera_world'] must contain 'Rwc' and 'Twc'")
    C_human = centers_from_extrinsics_wc(np.asarray(CW['Rwc']), np.asarray(CW['Twc']))

    # Static camera centers (try both conventions and pick the best)
    static_candidates = _coerce_R_t_stack(predictions.get('camera_poses'))
    if len(static_candidates) == 0:
        raise KeyError("predictions['camera_poses'] missing or unsupported format")

    best = None
    for C_static, tag in static_candidates:
        # Pair frames
        if frame_map is None:
            L = min(len(C_static), len(C_human))
            idx_s = np.arange(L)
            idx_h = np.arange(L)
        else:
            idx_h = np.asarray([i for i, j in frame_map if i is not None and j is not None])
            idx_s = np.asarray([j for i, j in frame_map if i is not None and j is not None])

        A = np.asarray(C_human)[idx_h]
        B = np.asarray(C_static)[idx_s]

        # Select informative frames
        keepA = _select_frames_with_baseline(A, min_pair_dist=min_pair_dist)
        keepB = _select_frames_with_baseline(B, min_pair_dist=min_pair_dist)
        keep = np.intersect1d(keepA, keepB)
        if keep.size < 4:
            keep = np.arange(min(len(A), 200))

        A_use, B_use = A[keep], B[keep]
        s, R, t = umeyama_sim3(A_use, B_use)

        # Compute residual
        A2B = (s * (A_use @ R.T)) + t
        rmse = float(np.sqrt(np.mean(np.sum((A2B - B_use) ** 2, axis=1))))

        if (best is None) or (rmse < best['rmse']):
            best = dict(s=s, R=R, t=t, rmse=rmse, tag=tag)

    return best['s'], best['R'], best['t'], best['tag']

# ---------- Apply Sim(3) to the human pipeline ----------
def apply_sim3_to_results(results, s, R, t, rotate_global_orient=True):
    """
    Mutates 'results' (human pipeline) in-place to align to the static frame.
    Updates:
      - people[*]['smplx_world']['trans']
      - people[*]['smplx_world']['pose'] (first 3 dims as global orient, axis-angle), if requested
      - camera_world ('Rwc','Twc')
    """
    # Humans
    for pid, track in results.get('people', {}).items():
        W = track.get('smplx_world', None)
        if W is None:
            continue
        if 'trans' in W:
            trans = np.asarray(W['trans'])
            W['trans'] = (s * (trans @ R.T)) + t  # (T,3)

        if rotate_global_orient and 'pose' in W:
            pose = np.asarray(W['pose'])
            if pose.ndim == 2 and pose.shape[1] >= 3:
                glob_aa = pose[:, :3]
                Rg = SciRot.from_rotvec(glob_aa)
                Rfix = SciRot.from_matrix(R)
                Rg_new = Rfix * Rg  # left-multiply in world
                pose[:, :3] = Rg_new.as_rotvec()
                W['pose'] = pose

    # Cameras (camera→world)
    CW = results.get('camera_world', None)
    if CW is not None and 'Rwc' in CW and 'Twc' in CW:
        Rwc = np.asarray(CW['Rwc']).reshape(-1, 3, 3)
        Twc = np.asarray(CW['Twc']).reshape(-1, 3)
        CW['Rwc'] = (R @ Rwc.reshape(-1, 3, 3)).reshape(Rwc.shape)
        CW['Twc'] = (s * (Twc @ R.T)) + t

    return results

def fuse_humans_into_static_frame(predictions, results, frame_map=None, min_pair_dist=0.05):
    """
    Compute Sim(3) from cameras and apply it so that human pipeline lives in the static scene frame.
    Returns (results_aligned, (s,R,t), chosen_tag, diagnostics_dict)
    """
    s, R, t, tag = sim3_from_cameras(predictions, results, frame_map=frame_map, min_pair_dist=min_pair_dist)

    # Diagnostics: residual on the paired (kept) frames
    CW = results['camera_world']
    C_human = centers_from_extrinsics_wc(np.asarray(CW['Rwc']), np.asarray(CW['Twc']))

    # Pick the same static candidate used
    static_candidates = _coerce_R_t_stack(predictions.get('camera_poses'))
    C_static = None
    for C, tg in static_candidates:
        if tg == tag:
            C_static = C
            break

    L = min(len(C_human), len(C_static))
    A = C_human[:L]
    B = C_static[:L]
    A2B = (s * (A @ R.T)) + t
    rmse_all = float(np.sqrt(np.mean(np.sum((A2B - B) ** 2, axis=1))))

    # Mutate results in-place (return anyway for convenience)
    results_aligned = apply_sim3_to_results(results, s, R, t, rotate_global_orient=True)

    diags = dict(rmse_all=rmse_all, frames_compared=L, static_pose_convention=tag)
    return results_aligned, (s, R, t), tag, diags


# Add near the top of your file (if not already present)
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

# Helper to build a wireframe frustum in the *camera's local frame* (RUB; camera looks along -Z).
def _make_frustum_lines(W, H, fx, fy, cx, cy, near=0.12, far=0.45):
    def rect_at(depth):
        z = -float(depth)  # forward along -Z in R-U-B
        x0 = (0   - cx) * (z / fx)
        x1 = (W-1 - cx) * (z / fx)
        y0 = (0   - cy) * (z / fy)
        y1 = (H-1 - cy) * (z / fy)
        return np.array([[x0, y0, z],
                         [x1, y0, z],
                         [x1, y1, z],
                         [x0, y1, z]], dtype=np.float32)

    n = rect_at(near)
    f = rect_at(far)
    strips = [
        np.vstack([n, n[0]]),           # near loop
        np.vstack([f, f[0]]),           # far loop
        np.vstack([n[0], f[0]]),        # connect near/far
        np.vstack([n[1], f[1]]),
        np.vstack([n[2], f[2]]),
        np.vstack([n[3], f[3]]),
        np.vstack([np.zeros(3, np.float32), n[0]]),  # rays from center
        np.vstack([np.zeros(3, np.float32), n[1]]),
        np.vstack([np.zeros(3, np.float32), n[2]]),
        np.vstack([np.zeros(3, np.float32), n[3]]),
    ]
    return strips



def visualize_camera_poses_mismatch(
        images: List[Optional[np.ndarray]],
        world4d: List[Dict],
        results: dict,
        pipeline: AgPipeline,
        faces: np.ndarray,
        init_fps: float = 25.0,
        floor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        img_maxsize: int = 320,
        app_id: str = "World4D",
        *,
        # NEW: control how the image for a given timestep i is chosen.
        # Option A: map timeline index -> source image index (e.g., {10: 0} uses images[0] at t=10)
        image_frame_map: Optional[Dict[int, int]] = None,
        # Option B: a callback to transform/replace the image for display at time i.
        # Signature: fn(i, base_image) -> np.ndarray | None
        image_fn: Optional[Callable[[int, Optional[np.ndarray]], Optional[np.ndarray]]] = None,
        # NEW: reuse stable entity paths so old frames don't linger on screen.
        reuse_paths: bool = True,
) -> None:
    faces_u32 = _faces_u32(faces)
    # Start a fresh recording & spawn the viewer.
    rr.init(app_id, spawn=True)
    # Set world axes: Right (+X), Up (+Y), Back (+Z)
    try:
        rr.log("/", rr.ViewCoordinates.RUB)
    except Exception:
        pass  # older rerun versions may not have this enum; safe to skip

    # A timeline named "frame" – the viewer exposes a scrubber for this.
    rr.set_time_seconds("frame_fps", 1.0 / max(1e-6, float(init_fps)))
    num_frames = len(world4d)

    # Log the (optional) static floor once (no timeline).
    if floor is not None:
        fv, ff = floor
        fv = np.asarray(fv, dtype=np.float32)
        ff = _faces_u32(np.asarray(ff))
        rr.log(
            "floor",
            rr.Mesh3D(
                vertex_positions=fv,
                triangle_indices=ff,
                albedo_factor=...,  # keep user's original style (inferred material)
            ),
        )

    def _frustum_path(i: int) -> str:
        return f"{BASE}/frustum" if reuse_paths else f"{BASE}/frames/t{i}/frustum"

    # Helper: select/prepare image for frame i, with remap + transform
    def _get_image_for_time(i: int) -> Optional[np.ndarray]:
        src_idx = i
        if image_frame_map and i in image_frame_map:
            src_idx = image_frame_map[i]
        base_img = None
        if images is not None and 0 <= src_idx < len(images):
            base_img = images[src_idx]
        if image_fn is not None:
            base_img = image_fn(i, base_img)

        if base_img is None:
            return None

        img = base_img
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    video_save_dir = os.path.join("/data2/rohith/ag/ag4D/static_scenes/pi3", f"0DJ6R_10")
    prediction_save_path = os.path.join(video_save_dir, "predictions.npz")
    if os.path.exists(prediction_save_path):
        predictions = np.load(prediction_save_path, allow_pickle=True)
        predictions = {k: predictions[k] for k in predictions.files}
        print(f"Loaded existing predictions for video from {prediction_save_path}")
    else:
        raise NotImplementedError("Prediction generation not implemented in this snippet.")
    BASE = "world4d_check_camera_mismatch"

    # TODO: Log each frame's camera frustum + image from the predictions.
    cam_poses = predictions.get("camera_poses", None)

    for i in range(num_frames):
        rr.set_time_sequence("frame", i)

        # -------------------------------------------------------------------------

        R_i = cam_poses[i][:3, :3]
        t_i = cam_poses[i][:3, 3]
        quat_xyzw = SciRot.from_matrix(R_i).as_quat().astype(np.float32)
        rr.log(
            f"{BASE}/predicted_camera/frame_{i}",
            rr.Transform3D(
                translation=t_i.astype(np.float32),
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            )
        )

        # -------------------------------------------------------------------------

        # Camera frustum + image.
        cam_3x4 = np.asarray(world4d[i]["camera"], dtype=np.float32)
        R_wc = cam_3x4[:3, :3]        # (3,3)
        t_wc = cam_3x4[:3, 3]         # (3,)

        image = _get_image_for_time(i)
        if image is not None:
            H, W = image.shape[:2]
            aspect = W / float(H)
        else:
            # No image; pick a reasonable resolution from a default aspect & maxsize.
            aspect = 16.0 / 9.0
            H, W = img_maxsize, int(img_maxsize * aspect)

        fov_y = 0.96  # mirrors your viser default
        fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)

        # SciPy returns quats in (x, y, z, w) order
        quat_xyzw = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)

        # Log transform at a STABLE path (so previous frames don't linger)
        frus_path = _frustum_path(i)
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=t_wc.astype(np.float32),
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            )
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(
                focal_length=(fx, fy),
                principal_point=(cx, cy),
                resolution=(W, H)
            ),
        )
        if image is not None:
            rr.log(f"{frus_path}/image", rr.Image(image))

        # Small local axes at the frustum (visual aid).
        axes_len = 0.3
        rr.log(
            f"{frus_path}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.asarray(
                    [[axes_len, 0, 0], [0, axes_len, 0], [0, 0, axes_len]],
                    dtype=np.float32,
                ),
                colors=np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            ),
        )


def rerun_vis_world4d(
    images: List[Optional[np.ndarray]],
    world4d: List[Dict],
    results: dict,
    pipeline: AgPipeline,
    faces: np.ndarray,
    init_fps: float = 25.0,
    floor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    img_maxsize: int = 320,
    app_id: str = "World4D",
    *,
    # NEW: control how the image for a given timestep i is chosen.
    # Option A: map timeline index -> source image index (e.g., {10: 0} uses images[0] at t=10)
    image_frame_map: Optional[Dict[int, int]] = None,
    # Option B: a callback to transform/replace the image for display at time i.
    # Signature: fn(i, base_image) -> np.ndarray | None
    image_fn: Optional[Callable[[int, Optional[np.ndarray]], Optional[np.ndarray]]] = None,
    # NEW: reuse stable entity paths so old frames don't linger on screen.
    reuse_paths: bool = True,
):
    faces_u32 = _faces_u32(faces)
    # Start a fresh recording & spawn the viewer.
    rr.init(app_id, spawn=True)
    # Set world axes: Right (+X), Up (+Y), Back (+Z)
    try:
        rr.log("/", rr.ViewCoordinates.RUB)
    except Exception:
        pass  # older rerun versions may not have this enum; safe to skip

    # A timeline named "frame" – the viewer exposes a scrubber for this.
    rr.set_time_seconds("frame_fps", 1.0 / max(1e-6, float(init_fps)))
    num_frames = len(world4d)

    # Log the (optional) static floor once (no timeline).
    if floor is not None:
        fv, ff = floor
        fv = np.asarray(fv, dtype=np.float32)
        ff = _faces_u32(np.asarray(ff))
        rr.log(
            "floor",
            rr.Mesh3D(
                vertex_positions=fv,
                triangle_indices=ff,
                albedo_factor=...,  # keep user's original style (inferred material)
            ),
        )

    # Helper: select/prepare image for frame i, with remap + transform
    def _get_image_for_time(i: int) -> Optional[np.ndarray]:
        src_idx = i
        if image_frame_map and i in image_frame_map:
            src_idx = image_frame_map[i]
        base_img = None
        if images is not None and 0 <= src_idx < len(images):
            base_img = images[src_idx]
        if image_fn is not None:
            base_img = image_fn(i, base_img)

        if base_img is None:
            return None

        img = base_img
        H, W = img.shape[:2]
        if max(H, W) > img_maxsize:
            scale = float(img_maxsize) / float(max(H, W))
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img

    # entity path helpers now live under BASE
    def _human_path(tid: int, i: int) -> str:
        return f"{BASE}/humans/h{tid}" if reuse_paths else f"{BASE}/frames/t{i}/human_{tid}"

    def _frustum_path(i: int) -> str:
        return f"{BASE}/frustum" if reuse_paths else f"{BASE}/frames/t{i}/frustum"

    video_save_dir = os.path.join("/data2/rohith/ag/ag4D/static_scenes/pi3", f"0DJ6R_10")
    static_mesh_path = os.path.join(video_save_dir, "0DJ6R.glb")
    prediction_save_path = os.path.join(video_save_dir, "predictions.npz")
    if os.path.exists(prediction_save_path):
        predictions = np.load(prediction_save_path, allow_pickle=True)
        predictions = {k: predictions[k] for k in predictions.files}
        print(f"Loaded existing predictions for video from {prediction_save_path}")
    else:
        raise NotImplementedError("Prediction generation not implemented in this snippet.")

    # Pre-extract static points from all humans across time.
    scene_3d, static_points, static_colors = predictions_to_glb_with_static(predictions, conf_min=0.1)

    results, (s, R, t), tag, diags = fuse_humans_into_static_frame(predictions, results)

    print("Chosen static convention:", tag)  # 'wc' or 'cw'
    print("Global Sim(3): scale", s, "\nR=\n", R, "\nt=", t)
    print("Camera-center RMSE (all):", diags['rmse_all'])

    world4d = pipeline.create_world4d(results=results, step=1, total=1500)  # or your instance method call

    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)

    # fix floor (and move it under the same space)
    if floor is not None:
        fv, ff = floor
        fv = np.asarray(fv, dtype=np.float32)
        ff = _faces_u32(np.asarray(ff))
        rr.log(
            f"{BASE}/floor",
            rr.Mesh3D(vertex_positions=fv, triangle_indices=ff),  # drop the ellipsis
        )

    if static_points.size > 0:
        rr.log(
            f"{BASE}/static",
            rr.Points3D(
                positions=static_points.astype(np.float32),
                colors=static_colors.astype(np.uint8),
                radii=0.01,  # <-- small but visible point size (tweak if needed)
            ),
        )

        print("[static] count:", len(static_points), "finite:", np.isfinite(static_points).all(),
              "min:", np.nanmin(static_points, axis=0), "max:", np.nanmax(static_points, axis=0))

    # Sequence logging.
    for i in range(num_frames):
        rr.set_time_sequence("frame", i)

        # Per-frame human meshes (if any).
        track_ids = world4d[i].get("track_id", [])
        verts_list = world4d[i].get("vertices", [])
        if len(track_ids) > 0 and len(verts_list) == len(track_ids):
            for tid, verts in zip(track_ids, verts_list):
                verts = np.asarray(verts, dtype=np.float32)
                rr.log(
                    _human_path(int(tid), i),
                    rr.Mesh3D(
                        vertex_positions=verts,
                        triangle_indices=faces_u32,
                        albedo_factor=_color_from_id(int(tid)),
                    ),
                )

        # Camera frustum + image.
        cam_3x4 = np.asarray(world4d[i]["camera"], dtype=np.float32)
        R_wc = cam_3x4[:3, :3]        # (3,3)
        t_wc = cam_3x4[:3, 3]         # (3,)

        image = _get_image_for_time(i)
        if image is not None:
            H, W = image.shape[:2]
            aspect = W / float(H)
        else:
            # No image; pick a reasonable resolution from a default aspect & maxsize.
            aspect = 16.0 / 9.0
            H, W = img_maxsize, int(img_maxsize * aspect)

        fov_y = 0.96  # mirrors your viser default
        fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)

        # SciPy returns quats in (x, y, z, w) order
        quat_xyzw = SciRot.from_matrix(R_wc).as_quat().astype(np.float32)

        # Log transform at a STABLE path (so previous frames don't linger)
        frus_path = _frustum_path(i)
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=t_wc.astype(np.float32),
                rotation=rr.Quaternion(xyzw=quat_xyzw),
            )
        )
        rr.log(
            f"{frus_path}/camera",
            rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W, H)),
        )
        if image is not None:
            rr.log(f"{frus_path}/image", rr.Image(image))

        # Small local axes at the frustum (visual aid).
        axes_len = 0.3
        rr.log(
            f"{frus_path}/axes",
            rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.asarray(
                    [[axes_len, 0, 0], [0, axes_len, 0], [0, 0, axes_len]],
                    dtype=np.float32,
                ),
                colors=np.asarray([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8),
            ),
        )

    print(
        "Rerun: use the 'frame' timeline in the viewer to play/pause/scrub. "
        "Meshes and frustum reuse stable paths so only the current frame is shown. "
        "Use `image_frame_map` or `image_fn` to change which image is displayed per frame."
    )
