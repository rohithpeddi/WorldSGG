import os
from typing import Optional, Tuple, List, Dict, Callable

import cv2
import matplotlib
import numpy as np
import rerun as rr  # pip install rerun-sdk
from scipy.spatial.transform import Rotation as R
import trimesh
from scipy.spatial.transform import Rotation


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


def rerun_vis_world4d(
    images: List[Optional[np.ndarray]],
    world4d: List[Dict],
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
    """
    Visualize a 4D sequence in Rerun.

    Args
    ----
    images: list of HxWx{3,4} uint8 frames (or None); one per timestep (unless remapped).
    world4d: list (len T) of dicts with keys:
        - 'track_id': Iterable[int] (can be empty)
        - 'vertices': Iterable[np.ndarray] (per-human vertices as (V,3), world coords)
        - 'camera': (3x4) camera pose matrix; rotation & translation for the frustum in world
    faces: (F,3) triangle indices for the human meshes.
    init_fps: initial playback FPS shown in the Rerun time panel.
    floor: optional (vertices(FV,3), faces(FF,3)) for a static floor mesh.
    img_maxsize: if a frame is larger, it will be resized to this longest edge before logging.
    app_id: rerun recording id/title.
    image_frame_map: (optional) map time index -> source image index.
    image_fn: (optional) per-frame function to transform/replace the image before logging.
    reuse_paths: if True (default), reuse entity paths per track so previous frames don't remain visible.

    Notes
    -----
    * Use the Rerun Viewer’s time panel (timeline: "frame") to play/pause/scrub.
    * Coordinate frame is set to Right-Up-Back (RUB), so +Y is up.
    * To quickly switch which image is shown at any frame, provide `image_frame_map` or `image_fn`.
    """
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
        quat_xyzw = R.from_matrix(R_wc).as_quat().astype(np.float32)

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
