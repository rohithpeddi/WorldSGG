import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
import numpy as np
import rerun as rr
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as Rot


# ---------------------------
# Split logic (yours)
# ---------------------------

def get_video_belongs_to_split(video_id: str) -> Optional[str]:
    """
    Get the split that the video belongs to based on its ID.
    Accepts either a bare ID (e.g., '0DJ6R') or a filename (e.g., '0DJ6R.mp4').
    """
    stem = Path(video_id).stem
    if not stem:
        return None
    first_letter = stem[0]
    if first_letter.isdigit() and int(first_letter) < 5:
        return "04"
    elif first_letter.isdigit() and int(first_letter) >= 5:
        return "59"
    elif first_letter in "ABCD":
        return "AD"
    elif first_letter in "EFGH":
        return "EH"
    elif first_letter in "IJKL":
        return "IL"
    elif first_letter in "MNOP":
        return "MP"
    elif first_letter in "QRST":
        return "QT"
    elif first_letter in "UVWXYZ":
        return "UZ"
    return None


# -------------------------------------------------------------------------------------------------------------------

def predictions_to_glb(
        predictions,
        conf_thres=50.0,
        filter_by_frames="all",
        show_cam=True,
) -> trimesh.Scene:
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

    # Prepare 4x4 matrices for camera extrinsic
    num_cameras = len(camera_poses)

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            # integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.)  # fixed camera size

    # Rotate scene for better visualize
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()  # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()  # roll
    scene_3d.apply_transform(align_rotation)

    print("GLB Scene built")
    return scene_3d


def integrate_camera_into_scene(scene: trimesh.Scene, transform: np.ndarray, face_colors: tuple, scene_scale: float):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


# --------------------------------------------------------------------------------------------------------------


# ---------------------------
# Utils
# ---------------------------

def _ensure_nhwc(images: np.ndarray) -> np.ndarray:
    """Ensure images are NHWC in [0,1]."""
    if images.ndim != 4:
        raise ValueError("images must be 4D")
    if images.shape[1] == 3:  # NCHW -> NHWC
        images = np.transpose(images, (0, 2, 3, 1))
    return np.clip(images, 0.0, 1.0)


def _flatten_points_colors_frames(
        points_wh: np.ndarray,
        colors_wh: np.ndarray,
        conf_wh: np.ndarray,
        conf_min: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flatten (S,H,W,3) points/colors and (S,H,W[,1]) conf to arrays:
      P: (N,3) points, C: (N,3) uint8 colors, F: (N,) frame idx
    Filter by conf >= conf_min and conf > 1e-5; drop NaNs/Infs.
    """
    S = points_wh.shape[0]
    if conf_wh.ndim == 4 and conf_wh.shape[-1] == 1:
        conf_wh = conf_wh[..., 0]

    mask = (conf_wh >= conf_min) & (conf_wh > 1e-5)

    P = points_wh[mask]  # (N,3)
    C = (colors_wh[mask] * 255.0).astype(np.uint8)  # (N,3)
    F = np.repeat(np.arange(S), repeats=points_wh.shape[1] * points_wh.shape[2])[mask.ravel()]

    good = np.isfinite(P).all(axis=1)
    return P[good], C[good], F[good]


def _voxel_ids(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Integer voxel coordinates for each point."""
    return np.floor(points / voxel_size).astype(np.int64)


def _camera_R_t_from_4x4(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract rotation (3x3) and translation (3,) from a 4x4 or 3x4 transform."""
    if T.shape == (3, 4):
        Rm, t = T[:, :3], T[:, 3]
    elif T.shape == (4, 4):
        Rm, t = T[:3, :3], T[:3, 3]
    else:
        raise ValueError("camera pose must be (3,4) or (4,4)")
    return Rm.astype(np.float32), t.astype(np.float32)


def _pinhole_from_fov(W: int, H: int, fov_y_rad: float) -> Tuple[float, float, float, float]:
    """Compute fx, fy, cx, cy from vertical FOV (in radians) and resolution."""
    fy = (H * 0.5) / np.tan(0.5 * fov_y_rad)
    fx = fy * (W / H)
    cx = W * 0.5
    cy = H * 0.5
    return fx, fy, cx, cy


def _frustum_path(i: int) -> str:
    return f"world/frames/t{i}/frustum"


# ---------------------------
# Static scene builder
# ---------------------------

def predictions_to_colors(
        predictions: Dict,
        conf_min: float = 0.5,                  # 0..1 threshold on predictions["conf"]
        filter_by_frames: str = "all",          # e.g. "12:..." to use only frame index 12
        filter_low_conf_black: bool = False,    # enable extra filtering
        *,
        black_rgb_max: int = 8,                 # 0..255; <= this per-channel is considered "black"
        black_conf_max: float = 1.0,           # 0..1; if conf < this AND pixel is black -> drop
):
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # ------------ Frame selection ------------
    selected_frame_idx = None
    if filter_by_frames not in ("all", "All"):
        try:
            selected_frame_idx = int(str(filter_by_frames).split(":")[0])
        except (ValueError, IndexError):
            selected_frame_idx = None

    print("Selected frame index for extraction:", selected_frame_idx)

    pts = predictions["points"]                             # (S,H,W,3)
    conf = predictions.get("conf", np.ones_like(pts[..., 0], dtype=np.float32))  # (S,H,W) or (S,H,W,1)
    imgs = predictions["images"]                            # (S,?,H,W) or (S,H,W,3/4/1)
    cam_poses = predictions.get("camera_poses", None)

    if selected_frame_idx is not None:
        pts = pts[selected_frame_idx][None]
        conf = conf[selected_frame_idx][None]
        imgs = imgs[selected_frame_idx][None]
        cam_poses = cam_poses[selected_frame_idx][None] if cam_poses is not None else None

    # ------------ Color layout handling ------------
    # Normalize to NHWC; keep only the first 3 channels (RGB); expand grayscale to RGB.
    if imgs.ndim == 4 and imgs.shape[1] in (1, 3, 4):            # NCHW
        imgs_nhwc = np.transpose(imgs, (0, 2, 3, 1))
    elif imgs.ndim == 4 and imgs.shape[-1] in (1, 3, 4):         # already NHWC
        imgs_nhwc = imgs
    else:
        raise ValueError(f"`images` must be 4D with channels in {{1,3,4}}, got shape {imgs.shape}")

    C = imgs_nhwc.shape[-1]
    if C == 1:
        imgs_nhwc = np.repeat(imgs_nhwc, 3, axis=-1)
    elif C >= 3:
        imgs_nhwc = imgs_nhwc[..., :3]  # drop alpha if present

    # Convert to uint8 RGB for color output
    colors_rgb = (imgs_nhwc.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    # ------------ Confidence filtering ------------
    verts = pts.reshape(-1, 3)
    conf_flat = conf.reshape(-1).astype(np.float32)

    thr = float(conf_min) if conf_min is not None else 0.1
    base_mask = (conf_flat >= thr) & (conf_flat > 1e-5)

    # Optional: drop pixels that are BOTH near-black AND have low confidence (below black_conf_max)
    if filter_low_conf_black:
        # Per-channel black check (inclusive): 0..black_rgb_max
        is_black = (
            (colors_rgb[:, 0] <= black_rgb_max) &
            (colors_rgb[:, 1] <= black_rgb_max) &
            (colors_rgb[:, 2] <= black_rgb_max)
        )
        low_conf_black = is_black & (conf_flat < float(black_conf_max))
        mask = base_mask & (~low_conf_black)
    else:
        mask = base_mask

    # Apply mask
    verts = verts[mask]
    colors_rgb = colors_rgb[mask]

    if verts.size == 0:
        # robust fallback
        verts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        colors_rgb = np.array([[255, 255, 255]], dtype=np.uint8)

    # Keep an ORIGINAL copy for the background return (world frame)
    static_points = verts.astype(np.float32, copy=True)
    static_colors = colors_rgb.astype(np.uint8, copy=True)

    return static_points, static_colors, verts, colors_rgb


def glb_to_points(glb_scene_path:  str) -> Tuple[np.ndarray, np.ndarray]:
    scene = trimesh.load(glb_scene_path, force='scene')
    all_points = []
    all_colors = []

    for geom_name, geom in scene.geometry.items():
        if isinstance(geom, trimesh.points.PointCloud):
            all_points.append(geom.vertices.astype(np.float32))
            if geom.colors is not None:
                all_colors.append(geom.colors.astype(np.uint8))
            else:
                # Default to white if no colors
                all_colors.append(np.ones((geom.vertices.shape[0], 3), dtype=np.uint8) * 255)

    if len(all_points) == 0:
        raise ValueError(f"No point clouds found in GLB file: {glb_scene_path}")

    points = np.vstack(all_points)
    colors = np.vstack(all_colors)

    # Ensure colors are uint8 and in the correct range and points in float32
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    points = points.astype(np.float32)

    align_R = Rot.from_euler("y", 100, degrees=True).as_matrix()
    align_R = align_R @ Rot.from_euler("x", 155, degrees=True).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = align_R

    points = (points @ align_R.T).astype(np.float32)

    return points, colors


def predictions_to_glb_with_static(
        predictions: Dict,
        *,
        conf_min: float = 0.5,  # 0..1 threshold on predictions["conf"]
        filter_by_frames: str = "all",  # e.g. "12:..." to use only frame index 12
        show_cam: bool = True,
) -> Tuple[trimesh.Scene, np.ndarray, np.ndarray]:
    """
    Build a GLB-ready trimesh.Scene from VGGT-style predictions AND return a background
    point cloud (xyz,rgb) in the original world frame.

    Inputs (expected prediction keys):
      - points:        (S, H, W, 3) float32   world-space points
      - conf:          (S, H, W)    float32   confidence per point in [0,1] (fallback: ones)
      - images:        (S, H, W, 3) or (S, 3, H, W) float32 in [0,1] for colors
      - camera_poses:  (S, 3, 4) or (S,4,4); kept for signature parity (not visualized here)

    Returns:
      scene_3d      : trimesh.Scene with a point cloud (rotated for nicer viewing)
      static_points : (N,3) float32 background points in ORIGINAL world frame
      static_colors : (N,3) uint8   RGB colors aligned with static_points
    """
    print("Estimating static scene with a confidence mask threshold of:", conf_min)

    static_points, static_colors, verts, colors_rgb = predictions_to_colors(
        predictions, conf_min=conf_min, filter_by_frames=filter_by_frames, filter_low_conf_black=True
    )

    # ------------ Build Scene (with visualization alignment rotation) ------------
    scene_3d = trimesh.Scene()
    point_cloud = trimesh.points.PointCloud(vertices=verts, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud)

    # Prepare 4x4 matrices for camera extrinsic
    camera_poses = predictions["camera_poses"]
    num_cameras = len(camera_poses)
    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])
            # integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.)  # fixed camera size

    # Nice-view rotation: R = R_y(100°) @ R_x(155°)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()  # plane rotate
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()  # roll
    scene_3d.apply_transform(align_rotation)

    # (Optional) If you later want to visualize camera frustums/axes, add them here
    # using cam_poses if available. Currently, `show_cam` is a no-op for signature parity.
    # Undo the visualization rotation for the returned static points (stay in world frame)
    # static_points = static_points @ align_R.T
    return scene_3d, static_points, static_colors

def _conf_to_hw(conf_f: np.ndarray, H: int, W: int) -> np.ndarray:
    """
    Normalize a per-frame confidence array to shape (H, W), handling common variants:
      - (H, W)
      - (H, W, 1)
      - (H,)  -> repeat across W
      - (W,)  -> repeat across H
      - (H*W,) -> reshape to (H, W)
    Falls back to ones(H,W) if shape is unexpected.
    """
    c = np.asarray(conf_f)
    if c.ndim == 2 and c.shape == (H, W):
        return c
    if c.ndim == 3 and c.shape[2] == 1 and c.shape[:2] == (H, W):
        return c[..., 0]
    if c.ndim == 1:
        if c.shape[0] == H * W:
            return c.reshape(H, W)
        if c.shape[0] == H:
            return np.repeat(c[:, None], W, axis=1)
        if c.shape[0] == W:
            return np.repeat(c[None, :], H, axis=0)
    # Fallback: uniform confidence
    return np.ones((H, W), dtype=np.float64)


def ground_dynamic_scene_to_static_scene(
        predictions: Dict,
        static_points: np.ndarray,
        static_colors: np.ndarray,  # kept for API parity; unused here
        frame_idx: int,
        conf_min: float = 0.1,
        dedup_voxel: Optional[float] = 0.02,  # used in fallback paths only
        *,
        # ICP knobs (safe defaults)
        icp_max_iters: int = 100,
        icp_tol: float = 1e-5,
        trim_frac: float = 0.8,  # keep this fraction of closest pairs each iter
        max_corr_dist: Optional[float] = None,  # meters; None disables
        src_sample_max: int = 50000,  # NOTE: ignored for ICP (we use full dynamic), kept for API compat
        tgt_sample_max: int = 100000,  # we may still subsample static for speed
        dynamic_voxel: Optional[float] = 0.01,
        merge_voxel: Optional[float] = None,
        camera_pose: Optional[np.ndarray] = None,   # pose for this frame
        pose_convention: str = "c2w",               # "c2w" or "w2c"
        return_pose: bool = True,                   # return updated pose
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    # ---------- Helpers ----------
    def _weighted_rigid_fit(A: np.ndarray, B: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Weighted Kabsch: find R,t minimizing sum_i w_i || R A_i + t - B_i ||^2."""
        if w is None:
            w = np.ones((A.shape[0],), dtype=np.float64)
        w = w.astype(np.float64)
        w_sum = np.sum(w) + 1e-12
        mu_A = (A * w[:, None]).sum(axis=0) / w_sum
        mu_B = (B * w[:, None]).sum(axis=0) / w_sum
        AA = A - mu_A
        BB = B - mu_B
        H = (AA * w[:, None]).T @ BB
        U, S, Vt = np.linalg.svd(H, full_matrices=True)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_B - R @ mu_A
        return R.astype(np.float64), t.astype(np.float64)

    def _apply(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        return (R @ X.T).T + t[None, :]

    def _to_4x4(T: np.ndarray) -> np.ndarray:
        if T.shape == (4, 4):
            return T.astype(np.float64)
        if T.shape == (3, 4):
            T4 = np.eye(4, dtype=np.float64)
            T4[:3, :4] = T
            return T4
        raise ValueError(f"camera_pose must be (4,4) or (3,4), got {T.shape}")

    def _to_like(T_ref: Optional[np.ndarray], T4: np.ndarray) -> np.ndarray:
        if T_ref is None:
            return T4
        return T4[:3, :4] if T_ref.shape == (3, 4) else T4

    # ---------- Prepare frame data ----------
    images = _ensure_nhwc(predictions["images"])   # (S,H,W,3)
    points = predictions["points"]                 # (S,H,W,3)
    conf    = predictions.get("conf", None)

    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    H, W = pts_f.shape[:2]

    # Try to fetch camera pose if not supplied
    pose_in = camera_pose
    if pose_in is None:
        cam_poses = predictions.get("camera_poses", None)
        if cam_poses is not None:
            pose_in = cam_poses[frame_idx]

    # Normalize confidence to (H,W)
    if conf is None:
        conf_hw = np.ones((H, W), dtype=np.float64)
        orig_conf_shape = None
    else:
        conf_f = conf[frame_idx]
        conf_hw = _conf_to_hw(conf_f, H, W).astype(np.float64)
        orig_conf_shape = getattr(conf_f, "shape", None)

    # Clean confidences
    conf_hw = np.nan_to_num(conf_hw, nan=0.0, posinf=0.0, neginf=0.0)
    conf_hw = np.maximum(conf_hw, 0.0)

    # Flatten
    pts_flat = pts_f.reshape(-1, 3).astype(np.float64)                    # (H*W,3)
    col_flat = (img_f.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_vec = conf_hw.reshape(-1)                                        # (H*W,)

    # Sanity check
    if conf_vec.shape[0] != pts_flat.shape[0]:
        raise ValueError(
            f"conf has {conf_vec.shape[0]} elems but points have {pts_flat.shape[0]} rows. "
            f"Original conf frame shape: {orig_conf_shape} normalized to {(H, W)}"
        )

    # Base validity (finite + confidence)
    good = np.isfinite(pts_flat).all(axis=1) & (conf_vec >= conf_min) & (conf_vec > 1e-5)

    # Dynamic sets (full frame)
    src_full = pts_flat[good]
    col_full = col_flat[good]
    w_full   = conf_vec[good].astype(np.float64)

    # Early outs
    if src_full.shape[0] == 0:
        updated_pose = None
        if pose_in is not None:
            updated_pose = _to_like(pose_in, _to_4x4(pose_in))  # identity
        return src_full.astype(np.float32), col_full.astype(np.uint8), (updated_pose if return_pose else None)

    # If no static target, we cannot register; just (optionally) voxel-reduce and return raw dynamic
    if static_points is None or static_points.shape[0] == 0:
        dyn_P = src_full.copy()
        dyn_C = col_full.copy()
        # Prefer dynamic_voxel; if None but dedup_voxel is set, use it as fallback
        vox_sz = dynamic_voxel if dynamic_voxel is not None else dedup_voxel
        if vox_sz is not None and dyn_P.size:
            vox = _voxel_ids(dyn_P, vox_sz)
            vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
            uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
            G = uniq_keys.shape[0]
            counts = np.bincount(inv, minlength=G).astype(np.float32); counts[counts == 0] = 1.0
            sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
            sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
            sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
            sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
            sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
            sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
            dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        if merge_voxel is not None and dyn_P.size:
            vox = _voxel_ids(dyn_P, merge_voxel)
            vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
            uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
            G = uniq_keys.shape[0]
            counts = np.bincount(inv, minlength=G).astype(np.float32); counts[counts == 0] = 1.0
            sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
            sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
            sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
            sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
            sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
            sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
            dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

        # Pose unchanged if we didn't ICP
        updated_pose = None
        if pose_in is not None:
            updated_pose = _to_like(pose_in, _to_4x4(pose_in))  # identity
        return dyn_P.astype(np.float32), dyn_C.astype(np.uint8), (updated_pose if return_pose else None)

    # ---------- Build static target (optionally subsampled for speed) ----------
    if static_points.shape[0] > tgt_sample_max:
        rng = np.random.default_rng(1337 + frame_idx)
        jdx = rng.choice(static_points.shape[0], size=tgt_sample_max, replace=False)
        tgt = static_points[jdx].astype(np.float64)
    else:
        tgt = static_points.astype(np.float64)

    # ---------- ICP using the FULL dynamic cloud ----------
    tree = cKDTree(tgt)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_err = np.inf
    src_iter = src_full.copy()

    for it in range(max(1, icp_max_iters)):
        dists, nn = tree.query(src_iter, k=1, workers=-1)
        valid = np.isfinite(dists)
        if max_corr_dist is not None:
            valid &= (dists <= max_corr_dist)
        if not np.any(valid):
            break

        if trim_frac < 1.0:
            cutoff = np.percentile(dists[valid], trim_frac * 100.0)
            valid &= (dists <= cutoff)

        A = src_iter[valid]
        B = tgt[nn[valid]]
        w = w_full[valid]
        if A.shape[0] < 10:
            break

        R_inc, t_inc = _weighted_rigid_fit(A, B, w)

        # compose transforms
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc
        src_iter = _apply(R_inc, t_inc, src_iter)

        err = float(np.mean((A - B) ** 2))
        if abs(prev_err - err) < icp_tol:
            break
        prev_err = err

    # ---------- Apply final transform to ALL dynamic points of this frame ----------
    dyn_P = _apply(R_total, t_total, src_full).astype(np.float64)
    dyn_C = col_full

    # ---------- Optional voxel reductions on the grounded dynamic cloud ----------
    if dynamic_voxel is not None and dyn_P.size:
        vox = _voxel_ids(dyn_P, dynamic_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32); counts[counts == 0] = 1.0
        sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
        dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    if merge_voxel is not None and dyn_P.size:
        vox = _voxel_ids(dyn_P, merge_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32); counts[counts == 0] = 1.0
        sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
        dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    # ---------- Build ICP transform as 4x4 and update camera pose ----------
    T_icp = np.eye(4, dtype=np.float64)
    T_icp[:3, :3] = R_total
    T_icp[:3, 3]  = t_total

    updated_pose = None
    if pose_in is not None:
        T_in4 = _to_4x4(np.asarray(pose_in))
        if pose_convention.lower() == "c2w":
            # points: X'_W = T_icp X_W  =>  pose: T'_cw = T_icp T_cw
            T_out4 = T_icp @ T_in4
        elif pose_convention.lower() == "w2c":
            # points: X'_W = T_icp X_W  =>  extrinsic: E'_wc = E_wc T_icp^{-1}
            T_out4 = T_in4 @ np.linalg.inv(T_icp)
        else:
            raise ValueError("pose_convention must be 'c2w' or 'w2c'")
        updated_pose = _to_like(np.asarray(pose_in), T_out4).astype(np.float64)

    dyn_P = dyn_P.astype(np.float32)
    dyn_C = dyn_C.astype(np.uint8)

    return dyn_P, dyn_C, updated_pose


def merge_static_with_frame(
        predictions: Dict,
        static_points: np.ndarray,
        static_colors: np.ndarray,
        interaction_masks: np.ndarray,
        frame_idx: int,
        conf_min: float = 0.1,
        dedup_voxel: Optional[float] = 0.02,
        *,
        # ICP knobs (safe defaults)
        icp_max_iters: int = 100,
        icp_tol: float = 1e-5,
        trim_frac: float = 0.8,  # keep this fraction of closest pairs each iter
        max_corr_dist: Optional[float] = None,  # meters; None disables
        src_sample_max: int = 50000,  # NOTE: ignored for ICP (we use full dynamic), kept for API compat
        tgt_sample_max: int = 100000,  # we may still subsample static for speed
        dynamic_voxel: Optional[float] = 0.01,
        merge_voxel: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align per-frame points to the static background via ICP and return the merged cloud.

    Changes vs previous:
      (1) ICP uses the full dynamic frame (no source subsampling).
      (2) Only dynamic points within the frame interaction mask are merged with static.
    Returns:
        mean_xyz (N,3) float32, mean_rgb (N,3) uint8
    """
    # ---------- Helpers ----------
    def _weighted_rigid_fit(A: np.ndarray, B: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Find R,t minimizing sum_i w_i || R A_i + t - B_i ||^2 via weighted Kabsch."""
        if w is None:
            w = np.ones((A.shape[0],), dtype=np.float64)
        w = w.astype(np.float64)
        w_sum = np.sum(w) + 1e-12
        mu_A = (A * w[:, None]).sum(axis=0) / w_sum
        mu_B = (B * w[:, None]).sum(axis=0) / w_sum
        AA = A - mu_A
        BB = B - mu_B
        H = (AA * w[:, None]).T @ BB  # 3x3
        U, S, Vt = np.linalg.svd(H, full_matrices=True)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = mu_B - R @ mu_A
        return R.astype(np.float64), t.astype(np.float64)

    def _apply(R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
        return (R @ X.T).T + t[None, :]

    # ---------- Prepare frame data ----------
    images = _ensure_nhwc(predictions["images"])  # (S,H,W,3)
    points = predictions["points"]                # (S,H,W,3)
    conf    = predictions.get("conf", None)

    pts_f = points[frame_idx]  # (H,W,3)
    img_f = images[frame_idx]  # (H,W,3)
    mask_f = interaction_masks[frame_idx]  # (H,W) boolean/0-1

    H, W = pts_f.shape[:2]

    # Normalize confidence to (H,W)
    if conf is None:
        conf_hw = np.ones((H, W), dtype=np.float64)
    else:
        conf_f = conf[frame_idx]
        conf_hw = _conf_to_hw(conf_f, H, W).astype(np.float64)

    # Clean confidences
    conf_hw = np.nan_to_num(conf_hw, nan=0.0, posinf=0.0, neginf=0.0)
    conf_hw = np.maximum(conf_hw, 0.0)

    # Flatten
    pts_flat  = pts_f.reshape(-1, 3).astype(np.float64)                # (H*W,3)
    col_flat  = (img_f.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    conf_vec  = conf_hw.reshape(-1)                                    # (H*W,)
    mask_vec  = (mask_f.reshape(-1) > 0)                               # (H*W,) bool

    # Sanity check
    if conf_vec.shape[0] != pts_flat.shape[0]:
        raise ValueError(
            f"conf has {conf_vec.shape[0]} elems but points have {pts_flat.shape[0]} rows. "
            f"Original conf frame shape: {getattr(conf_f, 'shape', None)} normalized to {(H,W)}"
        )
    if mask_vec.shape[0] != pts_flat.shape[0]:
        raise ValueError(f"interaction mask shape {mask_f.shape} incompatible with points frame {(H,W)}")

    # Base validity (finite + confidence)
    good_all = np.isfinite(pts_flat).all(axis=1) & (conf_vec >= conf_min) & (conf_vec > 1e-5)

    # Split into: full dynamic for ICP vs masked dynamic for merging
    sel_all   = good_all
    sel_mask  = good_all & mask_vec   # merge only these later

    # Build dynamic sets
    src_full        = pts_flat[sel_all]        # used for ICP (full)
    col_full        = col_flat[sel_all]
    w_full          = conf_vec[sel_all].astype(np.float64)

    src_masked_raw  = pts_flat[sel_mask]       # used for MERGE (subset)
    col_masked_raw  = col_flat[sel_mask]

    # Early exits if empty
    if static_points.shape[0] == 0 and src_masked_raw.shape[0] == 0:
        return np.empty((0,3), np.float32), np.empty((0,3), np.uint8)
    if src_full.shape[0] == 0 or static_points.shape[0] == 0:
        # Nothing to register; just merge masked dynamic (if any) with static
        if src_masked_raw.shape[0] == 0 or static_points.shape[0] == 0:
            merged_P = static_points
            merged_C = static_colors
        else:
            dyn_P, dyn_C = src_masked_raw.astype(np.float64), col_masked_raw
            if dynamic_voxel is not None and dyn_P.size:
                vox = _voxel_ids(dyn_P, dynamic_voxel)
                vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
                uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
                G = uniq_keys.shape[0]
                counts = np.bincount(inv, minlength=G).astype(np.float32)
                sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
                sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
                sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
                sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
                sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
                sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
                counts[counts == 0] = 1.0
                dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
                dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
            merged_P = np.concatenate([static_points.astype(np.float64), dyn_P], axis=0)
            merged_C = np.concatenate([static_colors, dyn_C], axis=0)

        # Optional final global voxel
        if dedup_voxel is not None and merged_P.size:
            vox = _voxel_ids(merged_P, dedup_voxel)
            vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
            uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
            G = uniq_keys.shape[0]
            counts = np.bincount(inv, minlength=G).astype(np.float32)
            sum_x = np.bincount(inv, weights=merged_P[:, 0], minlength=G)
            sum_y = np.bincount(inv, weights=merged_P[:, 1], minlength=G)
            sum_z = np.bincount(inv, weights=merged_P[:, 2], minlength=G)
            sum_r = np.bincount(inv, weights=merged_C[:, 0].astype(np.float32), minlength=G)
            sum_g = np.bincount(inv, weights=merged_C[:, 1].astype(np.float32), minlength=G)
            sum_b = np.bincount(inv, weights=merged_C[:, 2].astype(np.float32), minlength=G)
            counts[counts == 0] = 1.0
            merged_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            merged_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        if merge_voxel is not None and merged_P.size:
            vox = _voxel_ids(merged_P, merge_voxel)
            vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
            uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
            G = uniq_keys.shape[0]
            counts = np.bincount(inv, minlength=G).astype(np.float32)
            sum_x = np.bincount(inv, weights=merged_P[:, 0], minlength=G)
            sum_y = np.bincount(inv, weights=merged_P[:, 1], minlength=G)
            sum_z = np.bincount(inv, weights=merged_P[:, 2], minlength=G)
            sum_r = np.bincount(inv, weights=merged_C[:, 0].astype(np.float32), minlength=G)
            sum_g = np.bincount(inv, weights=merged_C[:, 1].astype(np.float32), minlength=G)
            sum_b = np.bincount(inv, weights=merged_C[:, 2].astype(np.float32), minlength=G)
            counts[counts == 0] = 1.0
            merged_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
            merged_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)
        return merged_P.astype(np.float32), merged_C.astype(np.uint8)

    # ---------- Build static target (optionally subsampled for speed) ----------
    if static_points.shape[0] > tgt_sample_max:
        rng = np.random.default_rng(1337 + frame_idx)
        jdx = rng.choice(static_points.shape[0], size=tgt_sample_max, replace=False)
        tgt = static_points[jdx].astype(np.float64)
    else:
        tgt = static_points.astype(np.float64)

    # ---------- ICP using the FULL dynamic cloud ----------
    tree = cKDTree(tgt)
    R_total = np.eye(3, dtype=np.float64)
    t_total = np.zeros(3, dtype=np.float64)
    prev_err = np.inf
    src_iter = src_full.copy()

    for it in range(max(1, icp_max_iters)):
        dists, nn = tree.query(src_iter, k=1, workers=-1)
        valid = np.isfinite(dists)
        if max_corr_dist is not None:
            valid &= (dists <= max_corr_dist)
        if not np.any(valid):
            break

        if trim_frac < 1.0:
            cutoff = np.percentile(dists[valid], trim_frac * 100.0)
            valid &= (dists <= cutoff)

        A = src_iter[valid]     # transformed source samples
        B = tgt[nn[valid]]      # matched static
        w = w_full[valid]
        if A.shape[0] < 10:
            break

        R_inc, t_inc = _weighted_rigid_fit(A, B, w)

        # compose transforms
        R_total = R_inc @ R_total
        t_total = R_inc @ t_total + t_inc
        src_iter = _apply(R_inc, t_inc, src_iter)

        err = float(np.mean((A - B) ** 2))
        if abs(prev_err - err) < icp_tol:
            break
        prev_err = err

    # ---------- Apply final transform to MASKED dynamic only (for merging) ----------
    dyn_P = _apply(R_total, t_total, src_masked_raw).astype(np.float64)
    dyn_C = col_masked_raw

    # Voxel-reduce the dynamic subset if requested
    if dynamic_voxel is not None and dyn_P.size:
        vox = _voxel_ids(dyn_P, dynamic_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32)
        sum_x = np.bincount(inv, weights=dyn_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=dyn_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=dyn_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=dyn_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=dyn_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=dyn_C[:, 2].astype(np.float32), minlength=G)
        counts[counts == 0] = 1.0
        dyn_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        dyn_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    # ---------- Merge with static (do NOT voxelize static by default) ----------
    if dyn_P.size:
        merged_P = np.concatenate([static_points.astype(np.float64), dyn_P], axis=0)
        merged_C = np.concatenate([static_colors,              dyn_C], axis=0)
    else:
        merged_P = static_points.astype(np.float64)
        merged_C = static_colors

    # Optional final global voxel (usually keep None to preserve static density)
    if merge_voxel is not None and merged_P.size:
        vox = _voxel_ids(merged_P, merge_voxel)
        vox_keys = np.ascontiguousarray(vox).view([('', vox.dtype)] * 3).ravel()
        uniq_keys, inv = np.unique(vox_keys, return_inverse=True)
        G = uniq_keys.shape[0]
        counts = np.bincount(inv, minlength=G).astype(np.float32)
        sum_x = np.bincount(inv, weights=merged_P[:, 0], minlength=G)
        sum_y = np.bincount(inv, weights=merged_P[:, 1], minlength=G)
        sum_z = np.bincount(inv, weights=merged_P[:, 2], minlength=G)
        sum_r = np.bincount(inv, weights=merged_C[:, 0].astype(np.float32), minlength=G)
        sum_g = np.bincount(inv, weights=merged_C[:, 1].astype(np.float32), minlength=G)
        sum_b = np.bincount(inv, weights=merged_C[:, 2].astype(np.float32), minlength=G)
        counts[counts == 0] = 1.0
        merged_P = np.stack([sum_x / counts, sum_y / counts, sum_z / counts], axis=1)
        merged_C = np.stack([sum_r / counts, sum_g / counts, sum_b / counts], axis=1).astype(np.uint8)

    return merged_P.astype(np.float32), merged_C.astype(np.uint8)


# ---------------------------
# Rerun visualization
# ---------------------------

# Helper to build a wireframe frustum in the *camera's local frame* (RUB; camera looks along -Z).
def _make_frustum_lines(W, H, fx, fy, cx, cy, near=0.12, far=0.45):
    def rect_at(depth):
        z = -float(depth)  # forward along -Z in R-U-B
        x0 = (0 - cx) * (z / fx)
        x1 = (W - 1 - cx) * (z / fx)
        y0 = (0 - cy) * (z / fy)
        y1 = (H - 1 - cy) * (z / fy)
        return np.array([[x0, y0, z],
                         [x1, y0, z],
                         [x1, y1, z],
                         [x0, y1, z]], dtype=np.float32)

    n = rect_at(near)
    f = rect_at(far)
    strips = [
        np.vstack([n, n[0]]),  # near loop
        np.vstack([f, f[0]]),  # far loop
        np.vstack([n[0], f[0]]),  # connect near/far
        np.vstack([n[1], f[1]]),
        np.vstack([n[2], f[2]]),
        np.vstack([n[3], f[3]]),
        np.vstack([np.zeros(3, np.float32), n[0]]),  # rays from center
        np.vstack([np.zeros(3, np.float32), n[1]]),
        np.vstack([np.zeros(3, np.float32), n[2]]),
        np.vstack([np.zeros(3, np.float32), n[3]]),
    ]
    return strips


def _log_cameras(
        predictions: Dict,
        fov_y: float,
        W: int,
        H: int,
        type: str,
        color
) -> None:
    if "camera_poses" not in predictions:
        return
    cam_poses = predictions["camera_poses"]  # (S, 4, 4) or (S, 3, 4)
    fx, fy, cx, cy = _pinhole_from_fov(W, H, fov_y)

    for i, Tcw in enumerate(cam_poses):
        Rcw, tcw = _camera_R_t_from_4x4(Tcw)
        q_xyzw = Rot.from_matrix(Rcw).as_quat().astype(np.float32)  # [x, y, z, w]

        frus_path = _frustum_path(i)
        rr.log(
            frus_path,
            rr.Transform3D(
                translation=tcw.astype(np.float32),
                rotation=rr.Quaternion(xyzw=q_xyzw),
            ),
        )

        frustum_strips = _make_frustum_lines(W, H, fx, fy, cx, cy, near=0.12, far=0.45)

        pred_path = f"world/{type}/frames/t{i}/predicted/frustum"
        # Use same intrinsics for visualization; swap to predicted intrinsics if you have them.
        rr.log(
            f"{pred_path}/camera",
            rr.Pinhole(focal_length=(fx, fy), principal_point=(cx, cy), resolution=(W, H)),
        )
        # predicted frustum wire + center dot (ORANGE)
        rr.log(
            f"{pred_path}/frustum_wire",
            rr.LineStrips3D(
                frustum_strips,  # <-- positional instead of line_strips=
                colors=[color] * len(frustum_strips),
                radii=0.003,
            ),
        )
        col = np.asarray(color, dtype=np.uint8).reshape(1, 3)  # (1,3), 0–255
        rr.log(
            f"{pred_path}/center",
            rr.Points3D(
                positions=np.zeros((1, 3), dtype=np.float32),
                colors=col,
                radii=0.01,  # or np.array([0.01], dtype=np.float32)
            ),
        )




# ------------------------------------------------------------------------------------------------------------
# ---------------------------------- VERIFICATION UTILS ---------------------------------------
# ------------------------------------------------------------------------------------------------------------
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from typing import Dict, Tuple
from zipfile import ZipFile

# ---------- Ultra-lightweight header reader (no full array load) ----------
def _peek_npy_shape(fileobj) -> Tuple[int, ...]:
    """Read only the .npy header to get the shape; avoids loading the array."""
    major_minor = read_magic(fileobj)  # (major, minor)
    if major_minor == (1, 0):
        shape, _, _ = read_array_header_1_0(fileobj)
        return shape
    elif major_minor == (2, 0):
        shape, _, _ = read_array_header_2_0(fileobj)
        return shape
    else:
        # Rare (npy 3.0 for >4GB headers). Fallback to full np.load for this file only.
        return None

def _get_array_shape_from_npz(npz_path: str, key: str = "points") -> Tuple[int, ...]:
    """Get array shape from an .npz by reading the embedded .npy header only."""
    key_npy = key if key.endswith(".npy") else f"{key}.npy"
    with ZipFile(npz_path) as zf:
        if key_npy not in zf.namelist():
            raise KeyError(f"Key '{key}' not found in {npz_path}")
        with zf.open(key_npy, "r") as fobj:
            shape = _peek_npy_shape(fobj)
            if shape is not None:
                return shape

    # Fallback for unusual headers: load only this array
    with np.load(npz_path, allow_pickle=True, mmap_mode=None) as npz:
        return npz[key].shape

# ---------- Worker run in a separate process ----------
def _verify_worker(
        video_id: str,
        static_scene_dir_path: str,
        dynamic_scene_dir_path: str,
        conf_tag: int = 10,
        key: str = "points",
) -> Dict:
    """
    Verify a single video in isolation (safe for multiprocessing).
    Returns a small dict with status and optional warning/error text.
    """
    try:
        base = video_id[:-4] if len(video_id) > 4 and video_id[-4] == "." else os.path.splitext(video_id)[0]
        static_scene_pred_path = os.path.join(static_scene_dir_path, f"{base}_{conf_tag}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(dynamic_scene_dir_path, f"{base}_{conf_tag}", "predictions.npz")

        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            return {
                "video_id": video_id,
                "ok": False,
                "warning": f"predictions.npz not found for {video_id}",
            }

        S_dyn, H_dyn, W_dyn = _get_array_shape_from_npz(dynamic_scene_pred_path, key)[:3]
        S_stat, H_stat, W_stat = _get_array_shape_from_npz(static_scene_pred_path, key)[:3]

        if (S_dyn, H_dyn, W_dyn) != (S_stat, H_stat, W_stat):
            warn = (
                "--------------------------------------------------------------------------------\n"
                f"[warn] Mismatched shapes for static and dynamic scenes in video {video_id}: "
                f"static (S={S_stat}, H={H_stat}, W={W_stat}), "
                f"dynamic (S={S_dyn}, H={H_dyn}, W={W_dyn}). Skipping inference.\n"
                "--------------------------------------------------------------------------------"
            )
            return {"video_id": video_id, "ok": False, "warning": warn}

        return {"video_id": video_id, "ok": True}

    except Exception as e:
        return {"video_id": video_id, "ok": False, "error": f"[error] Unexpected error for video {video_id}: {e}"}
