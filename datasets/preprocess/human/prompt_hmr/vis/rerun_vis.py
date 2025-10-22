import copy
import time
from typing import Optional, Tuple, List, Dict, Callable
from scipy.spatial.transform import Rotation as R
import cv2
import numpy as np
import rerun as rr  # pip install rerun-sdk


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

    # Entity path helpers:
    def _human_path(tid: int, i: int) -> str:
        # If reusing paths (recommended), keep one path per track id.
        return f"humans/h{tid}" if reuse_paths else f"frames/t{i}/human_{tid}"

    def _frustum_path(i: int) -> str:
        return "frustum" if reuse_paths else f"frames/t{i}/frustum"

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
            rr.Pinhole(
                focal_length=(fx, fy),
                principal_point=(cx, cy),
                resolution=(W, H),
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

    print(
        "Rerun: use the 'frame' timeline in the viewer to play/pause/scrub. "
        "Meshes and frustum reuse stable paths so only the current frame is shown. "
        "Use `image_frame_map` or `image_fn` to change which image is displayed per frame."
    )
