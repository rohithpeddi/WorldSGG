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


def predictions_to_glb(
        predictions,
        conf_thres=50.0,
        filter_by_frames="all",
        show_cam=True,
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        show_cam (bool): Include camera visualization (default: True)

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_poses = predictions["camera_poses"]

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_poses = camera_poses[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        # conf_threshold = np.percentile(conf, conf_thres)
        conf_threshold = conf_thres / 100

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_poses)

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()  # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155,
                                                                          degrees=True).as_matrix()  # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d

def build_static_background(
        predictions: dict,
        conf_min: float = 0.5,
        voxel_size: float = 0.03,
        min_frames: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Replacement that *uses predictions_to_glb* to generate a background point cloud,
    then extracts (xyz, rgb) from the returned trimesh.Scene.

    Notes / differences vs the old implementation:
      - Ignores true 'static across frames' logic (no per-voxel frame counting).
      - Approximates a static background by aggregating all points that survive
        predictions_to_glb's confidence filtering, then (optionally) voxel-downsampling.
      - Automatically *undoes* the visualization rotation applied inside predictions_to_glb,
        so outputs are back in the original world frame.
      - 'min_frames' is kept for signature compatibility but not used.

    Returns:
        static_points: (N,3) float32
        static_colors: (N,3) uint8
    """
    # 1) Call your existing predictions_to_glb with show_cam=False to avoid camera meshes.
    #    Map conf_min (0..1) -> conf_thres percentage (0..100).
    conf_thres_pct = float(conf_min) * 100.0
    scene = predictions_to_glb(
        predictions=predictions,
        conf_thres=conf_thres_pct,
        filter_by_frames="all",
        show_cam=False
    )

    # 2) Extract point cloud geometry from the scene (PointCloud only).
    pts_list, cols_list = [], []
    for geom in scene.geometry.values():
        # We only want point clouds, not camera meshes or other geometry
        if isinstance(geom, trimesh.points.PointCloud):
            v = np.asarray(geom.vertices, dtype=np.float32)
            c = np.asarray(geom.colors)
            # colors can be (N,3) or (N,4); drop alpha if present
            if c.ndim == 2 and c.shape[1] >= 3:
                c = c[:, :3]
            else:
                # If no colors, default to white
                c = np.full((v.shape[0], 3), 255, dtype=np.uint8)
            # Ensure types
            v = v.astype(np.float32, copy=False)
            c = c.astype(np.uint8, copy=False)
            if v.size:
                pts_list.append(v)
                cols_list.append(c)

    if not pts_list:
        # Fallback: empty scene protection
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

    points = np.concatenate(pts_list, axis=0)
    colors = np.concatenate(cols_list, axis=0)

    # 3) Undo predictions_to_glb's visualization alignment (Y=100°, then X=155°).
    #    In predictions_to_glb:
    #        R = R_y(100) @ R_x(155)
    #    We apply R^T to revert.
    R = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    R = R @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    points = points @ R.T  # undo rotation

    # # 4) Optional voxel downsample to keep the cloud compact (averaging xyz & rgb per voxel).
    # if voxel_size and voxel_size > 0:
    #     points, colors = _voxel_downsample_mean(points, colors, voxel_size)

    return points.astype(np.float32), colors.astype(np.uint8)


# --- NEW: helper to load a static mesh and expose arrays ----------------------
def load_static_mesh_as_arrays(
    path_or_mesh,
    *,
    default_color=(200, 200, 200)
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Loads a static triangle mesh and returns:
        vertices: (N, 3) float32
        faces:    (F, 3) uint32
        colors:   (N, 3) uint8 or None (if per-vertex colors unavailable)

    Accepts a file path (GLB/GLTF/PLY/OBJ) or a trimesh.Trimesh.
    If a file is a Scene, it is concatenated into a single mesh.
    """
    import trimesh

    if isinstance(path_or_mesh, str):
        # force='mesh' concatenates scene nodes w/ transforms into a single Trimesh
        mesh = trimesh.load(path_or_mesh, force='mesh', skip_materials=False)
    elif isinstance(path_or_mesh, trimesh.Trimesh):
        mesh = path_or_mesh
    else:
        raise TypeError("path_or_mesh must be a filepath or a trimesh.Trimesh")

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Failed to coerce input into a Trimesh")

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = _faces_u32(np.asarray(mesh.faces))

    C: Optional[np.ndarray] = None
    # Try vertex colors first
    try:
        vc = getattr(mesh.visual, "vertex_colors", None)
        if vc is not None and len(vc) == len(V):
            vc = np.asarray(vc)
            if vc.ndim == 2 and vc.shape[1] >= 3:
                C = vc[:, :3].astype(np.uint8, copy=False)
    except Exception:
        C = None

    # If vertex colors not present, ignore per-face materials and fall back to a flat color
    if C is None:
        C = None  # returning None signals to use albedo_factor

    # Small sanity: drop NaNs/Infs
    mask = np.isfinite(V).all(axis=1)
    if not mask.all():
        V = V[mask]
        # Faces that reference dropped vertices must be cleaned:
        # build a remap for surviving vertices
        remap = -np.ones(len(mask), dtype=np.int64)
        remap[np.flatnonzero(mask)] = np.arange(mask.sum(), dtype=np.int64)
        F = remap[F]
        F = F[(F >= 0).all(axis=1)]
        if C is not None:
            C = C[mask]

    if V.size == 0 or F.size == 0:
        raise ValueError("Static mesh is empty after cleaning.")

    return V.astype(np.float32, copy=False), F.astype(np.uint32, copy=False), (C if C is None else C.astype(np.uint8, copy=False))


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
    # prediction_save_path = os.path.join(video_save_dir, "predictions.npz")
    # if os.path.exists(prediction_save_path):
    #     predictions = np.load(prediction_save_path, allow_pickle=True)
    #     predictions = {k: predictions[k] for k in predictions.files}
    #     print(f"Loaded existing predictions for video from {prediction_save_path}")
    # else:
    #     raise NotImplementedError("Prediction generation not implemented in this snippet.")

    # # Pre-extract static points from all humans across time.
    # static_points, static_colors = build_static_background(
    #     predictions,
    #     conf_min=0.1,  # high-confidence for background
    #     voxel_size=0.01,  # 3cm voxels
    #     min_frames=3  # seen in >= 3 frames -> static
    # )
    BASE = "world"
    rr.log(BASE, rr.ViewCoordinates.RUB, timeless=True)  # single, consistent space
    # (remove rr.log("/", rr.ViewCoordinates.RUB) and the later RDF line)

    # fix floor (and move it under the same space)
    if floor is not None:
        fv, ff = floor
        fv = np.asarray(fv, dtype=np.float32)
        ff = _faces_u32(np.asarray(ff))
        rr.log(
            f"{BASE}/floor",
            rr.Mesh3D(vertex_positions=fv, triangle_indices=ff),  # drop the ellipsis
        )

    # --- NEW: log static background as a mesh, not a point cloud -----------------
    # Provide either static_mesh_arrays=(V,F,C?) directly, or a static_mesh_path
    VFC: Optional[Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = None

    try:
        VFC = load_static_mesh_as_arrays(static_mesh_path)
    except Exception as e:
        print(f"[static-mesh] Failed to load mesh from '{static_mesh_path}': {e}")

    if VFC is not None:
        V, F, C = VFC
        # Note: keep everything in world coords (no extra rotations applied).
        mesh_kwargs = dict(vertex_positions=V, triangle_indices=F)
        if C is not None:
            mesh_kwargs["vertex_colors"] = C
        else:
            mesh_kwargs["albedo_factor"] = (200, 200, 200)  # neutral gray fallback

        rr.log(f"{BASE}/static", rr.Mesh3D(**mesh_kwargs))
        print("[static-mesh] V:", V.shape, "F:", F.shape, "colored:", C is not None)
    else:
        print("[static-mesh] No static mesh provided; skipping static background.")

    # if static_points.size > 0:
    #     rr.log(
    #         f"{BASE}/static",
    #         rr.Points3D(
    #             positions=static_points.astype(np.float32),
    #             colors=static_colors.astype(np.uint8),
    #             radii=0.01,  # <-- small but visible point size (tweak if needed)
    #         ),
    #         timeless=True,
    #     )
    #
    #     print("[static] count:", len(static_points), "finite:", np.isfinite(static_points).all(),
    #           "min:", np.nanmin(static_points, axis=0), "max:", np.nanmax(static_points, axis=0))

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
