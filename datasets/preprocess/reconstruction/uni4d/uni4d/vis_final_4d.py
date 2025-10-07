# import numpy as np
# import rerun as rr
# import argparse
#
# def log_cam_dyn(R,t,k,rgb, dyn_coords, dyn_rgb):
#
#     h,w,_ = rgb.shape
#
#     rr.log(
#         f"/camera",
#         rr.Pinhole(
#             resolution=[w, h],
#             image_from_camera=k,
#         ),
#         rr.Transform3D(translation=t, mat3x3=R),
#     )
#
#     rr.log(f"/image/rgb", rr.Image(rgb))
#     rr.log(f"/dyn/pcd", rr.Points3D(dyn_coords, colors=dyn_rgb, radii=0.01))
#
# def log_pcd(control_points, colors):
#
#     rr.set_time_seconds("stable_time", 0)
#     mask = control_points[:,2] < 50        # Filter out points that might be too far away
#     rr.log(f"/static/pcd", rr.Points3D(control_points[mask], colors=colors[mask], radii=0.01))
#
# # Set up argument parser
# parser = argparse.ArgumentParser(description='Visualize 4D data from a npz file')
# parser.add_argument('--results_path', type=str, default=r"E:\DATA\MODELS\uni4D\fused_4d.npz",
#                     help='Path to the npz file containing results (default: fused_dyn_points.npz)')
# args = parser.parse_args()
#
# rr.init("vis")
# rr.spawn()
# rr.log("/", timeless=True)
#
# RESULTS_PATH = args.results_path
# results = np.load(RESULTS_PATH, allow_pickle=True)
#
# fused_points_coords = results["static_points"]
# fused_points_rgbs = results["static_rgbs"]
# dyn_points_coords = results["dyn_points"]
# dyn_points_rgbs = results["dyn_rgbs"]
# poses = results["c2w"]
# rgbs = results["rgbs"]
# Ks = results["Ks"]
#
# Rs = []
# ts = []
#
# log_pcd(fused_points_coords, fused_points_rgbs)
#
# for i, pose in enumerate(poses):
#
#     rr.set_time_seconds("stable_time", i)
#
#     R = pose[:3, :3]
#     t = pose[:3, 3]
#     rgb = rgbs[i]
#
#     log_cam_dyn(R,t,Ks[i],rgb, dyn_points_coords[i], dyn_points_rgbs[i])


import numpy as np
import rerun as rr
import argparse


# ─────────────────────────── Scene alignment helpers ──────────────────────────
def align_to_first_camera(static_pts, dyn_pts_list, poses):
    """
    Re-express every point and every camera pose in the coordinate frame of
    the first camera (pose 0).  After this transform:

        • camera 0 is located at the origin with R = I, t = 0
        • all other cameras and points are expressed in that same frame

    Returns
    -------
    static_c   : (N,3)  centered static points
    dyn_c      : list[(Mi,3)]  centered dynamic points
    poses_c    : (T,4,4)  camera-to-world poses in the new frame
    """
    R0 = poses[0][:3, :3]          # (3×3)
    t0 = poses[0][:3, 3]           # (3,)
    R0_T = R0.T

    # Points
    static_c = (static_pts - t0) @ R0_T
    dyn_c = [(p - t0) @ R0_T for p in dyn_pts_list]

    # Poses
    poses_c = poses.copy()
    poses_c[:, :3, 3] = (poses[:, :3, 3] - t0) @ R0_T        # translate then rotate
    poses_c[:, :3, :3] = R0_T @ poses[:, :3, :3]             # rotate orientations

    return static_c, dyn_c, poses_c


# ───────────────────────────── Rerun logging utils ────────────────────────────
def log_cam_dyn(R, t, K, rgb, dyn_coords, dyn_rgb):
    h, w, _ = rgb.shape
    rr.log(
        "/camera",
        rr.Pinhole(resolution=[w, h], image_from_camera=K),
        rr.Transform3D(translation=t, mat3x3=R),
    )
    rr.log("/image/rgb", rr.Image(rgb))
    rr.log("/dyn/pcd", rr.Points3D(dyn_coords, colors=dyn_rgb, radii=0.01))


def log_static_pcd(points, colors):
    rr.set_time_seconds("stable_time", 0)
    # Optionally drop very distant points for clarity
    mask = points[:, 2] < 50
    rr.log("/static/pcd", rr.Points3D(points[mask], colors=colors[mask], radii=0.01))


# ────────────────────────────────── Main ─────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize 4D data from an NPZ file")
    parser.add_argument(
        "--results_path",
        type=str,
        default=r"E:\DATA\MODELS\uni4D\fused_4d.npz",
        help="Path to the npz file containing results",
    )
    args = parser.parse_args()

    # Load data
    data = np.load(args.results_path, allow_pickle=True)
    static_pts   = data["static_points"]
    static_rgbs  = data["static_rgbs"]
    dyn_pts      = data["dyn_points"]      # shape (T, Mi, 3) or list
    dyn_rgbs     = data["dyn_rgbs"]
    poses        = data["c2w"]             # (T, 4, 4)
    rgbs         = data["rgbs"]
    Ks           = data["Ks"]

    # --- Align everything so the first camera becomes the world origin ---
    static_pts, dyn_pts, poses = align_to_first_camera(
        static_pts, list(dyn_pts), poses
    )

    # ------------------------- Rerun streaming ------------------------- #
    rr.init("vis")
    rr.spawn()
    rr.log("/", timeless=True)

    log_static_pcd(static_pts, static_rgbs)

    for i, pose in enumerate(poses):
        rr.set_time_seconds("stable_time", i)
        R, t = pose[:3, :3], pose[:3, 3]
        log_cam_dyn(R, t, Ks[i], rgbs[i], dyn_pts[i], dyn_rgbs[i])

