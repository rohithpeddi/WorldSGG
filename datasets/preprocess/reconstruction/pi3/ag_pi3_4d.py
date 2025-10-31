import argparse
import gc
import pickle
from typing import List
from typing import Optional, Tuple

import numpy as np
import rerun as rr
import torch
from PIL import Image
from scipy.spatial.transform import Rotation as Rot
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
from tqdm import tqdm

from datasets.preprocess.reconstruction.pi3.recon_utils import predictions_to_glb_with_static, _pinhole_from_fov, \
    _log_cameras, predictions_to_colors, glb_to_points, ground_dynamic_scene_to_static_scene, merge_static_with_frame, \
    get_video_belongs_to_split


# ---------------------------
# Main class
# ---------------------------

class AgPi3:

    def __init__(
            self,
            root_dir_path: str,
            dynamic_scene_dir_path: Optional[str] = None,
            static_scene_dir_path: Optional[str] = None,  # accepted for parity; not used here
            frame_annotated_dir_path: Optional[str] = None,  # accepted for parity; not used here
            grounded_dynamic_scene_dir_path: Optional[str] = None,  # accepted for parity; not used here
            masks_dir_path: Optional[str] = None,  # accepted for parity; not used here
    ):
        self.model = None
        self.root_dir_path = root_dir_path
        self.static_scene_dir_path = static_scene_dir_path
        self.dynamic_scene_dir_path = dynamic_scene_dir_path if dynamic_scene_dir_path is not None else root_dir_path
        self.frame_annotated_dir_path = frame_annotated_dir_path
        self.grounded_dynamic_scene_dir_path = grounded_dynamic_scene_dir_path
        self.masks_dir_path = masks_dir_path

        os.makedirs(self.dynamic_scene_dir_path, exist_ok=True)

        self.sampled_frames_idx_root_dir = "/data/rohith/ag/sampled_frames_idx"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------- Infer basic video visualization -----------------------------

    def voxel_merge(self, static_P, static_C, dyn_P, dyn_C, voxel_size=None):
        """Concatenate static & dynamic. If voxel_size is set, dedup by quantizing."""
        if dyn_P is None or dyn_P.size == 0:
            return static_P, static_C

        P = np.concatenate([static_P, dyn_P], axis=0)
        C = np.concatenate([static_C, dyn_C], axis=0)

        if voxel_size is None:
            return P, C

        # Quantize to voxel grid and keep first occurrence per voxel.
        q = np.floor(P / float(voxel_size)).astype(np.int64)
        # Pack rows for fast unique:
        q_view = q.view([('', q.dtype)] * q.shape[1]).reshape(q.shape[0])
        uniq, idx = np.unique(q_view, return_index=True)
        idx = np.sort(idx)
        return P[idx], C[idx]

    def infer_basic_video(
            self,
            video_id: str,
            *,
            conf_static: float = 0.10,
            conf_frame: float = 0.01,
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,
            spawn: bool = True,
            log_cameras: bool = True,
            apply_transform: bool = True
    ) -> None:
        # ---- Load predictions ----
        stem = video_id[:-4] if video_id.endswith(".mp4") else video_id
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{stem}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{stem}_{10}", "predictions.npz")
        if not (os.path.exists(static_scene_pred_path) and os.path.exists(dynamic_scene_pred_path)):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_npz = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_npz = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)
        static_pred = {k: static_npz[k] for k in static_npz.files}
        dynamic_pred = {k: dynamic_npz[k] for k in dynamic_npz.files}

        static_points_wh = static_pred["points"]  # (S,H,W,3)
        S_static, H, W = static_points_wh.shape[:3]
        # Prefer the dynamic sequence length for the loop if present:
        S_dynamic = dynamic_pred.get("points", static_points_wh).shape[0]
        S = min(S_static, S_dynamic)

        print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame}")

        if apply_transform:
            align_R = Rot.from_euler("y", 100, degrees=True).as_matrix()
            align_R = align_R @ Rot.from_euler("x", 155, degrees=True).as_matrix()
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = align_R

        # ---- Build static background (once) ----
        # Returns (scene_3d, static_P, static_C); we only need P,C here.
        _, static_P, static_C = predictions_to_glb_with_static(static_pred, conf_min=float(conf_static))
        static_P = static_P.astype(np.float32, copy=False)
        static_C = static_C.astype(np.uint8, copy=False)

        # ---- Rerun setup ----
        rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

        # Log static background timelessly (persists across all frames)
        if static_P.size > 0:
            if apply_transform:
                static_P = static_P @ align_R.T
            rr.log(
                "world/static",
                rr.Points3D(
                    positions=static_P,
                    colors=static_C,
                ),
            )

        # Cameras (do this once; use np.array for color to avoid list-indexing errors)
        if log_cameras:
            _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
            _log_cameras(static_pred, fov_y=fov_y, W=W, H=H, type="static",
                         color=np.array([255, 0, 0], dtype=np.uint8))  # RED
            _log_cameras(dynamic_pred, fov_y=fov_y, W=W, H=H, type="dynamic",
                         color=np.array([0, 255, 0], dtype=np.uint8))  # GREEN

        # ---- Stream frames ----
        for i in tqdm(range(S), desc=f"[viz] Streaming frames for {video_id}"):
            # Dynamic points/colors for frame i
            dyn_P, dyn_C, _, _ = predictions_to_colors(
                dynamic_pred, conf_min=float(conf_frame), filter_by_frames=f"{i}:"
            )
            dyn_P = np.asarray(dyn_P, dtype=np.float32)
            dyn_C = np.asarray(dyn_C, dtype=np.uint8)

            if apply_transform:
                dyn_P = dyn_P @ align_R.T

            rr.set_time_sequence("frame", i)

            # Merged = static ⊕ dynamic (with optional voxel dedup)
            merged_P, merged_C = self.voxel_merge(static_P, static_C, dyn_P, dyn_C, voxel_size=dedup_voxel)
            rr.log(
                "world/frame/points_merged",
                rr.Points3D(
                    positions=merged_P.astype(np.float32, copy=False),
                    colors=merged_C.astype(np.uint8, copy=False),
                    radii=0.01,
                ),
            )

        print("[viz] Done streaming frames to Rerun.")

        # Cleanup
        del static_pred, dynamic_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------- Infer grounded dynamic video visualization -----------------------------

    def infer_grounded_dynamic_video(
            self,
            video_id: str,
            *,
            conf_static: float = 0.1,  # confidence for static background build
            conf_frame: float = 0.01,  # confidence for per-frame points
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,  # radians; matches your earlier default
            spawn: bool = True,
            log_cameras: bool = True,
            load_from_glb: bool = False,
            vis: bool = False,
    ) -> None:
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        grounded_dynamic_scene_pred_path = os.path.join(
            self.grounded_dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions_grounded.pkl"
        )
        os.makedirs(os.path.dirname(grounded_dynamic_scene_pred_path), exist_ok=True)
        if os.path.exists(grounded_dynamic_scene_pred_path):
            print(f"[viz] Grounded dynamic scene predictions already exist at {grounded_dynamic_scene_pred_path}. Skipping inference.")
            return

        static_scene_arr = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_arr = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_predictions = {k: dynamic_scene_arr[k] for k in dynamic_scene_arr.files}
        S, H, W = dynamic_scene_predictions["points"].shape[:3]

        if load_from_glb:
            print(f"[viz] Loading static scene from GLB for {video_id}...")
            glb_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", f"{video_id[:-4]}.glb")
            static_P, static_C = glb_to_points(glb_path)
        else:
            print("Loading static scene from predictions...")
            static_scene_predictions = {k: static_scene_arr[k] for k in static_scene_arr.files}
            static_scene_points_wh = static_scene_predictions["points"]  # (S,H,W,3)
            S, H, W = static_scene_points_wh.shape[:3]
            print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame}")

            # ---- Build static background (once) ----
            scene_3d, static_P, static_C = predictions_to_glb_with_static(
                static_scene_predictions, conf_min=float(conf_static),
            )

        # ---- Rerun setup ----
        if vis:
            rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
            rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

            # Log static background timelessly
            if static_P.size > 0:
                rr.log(
                    "world/static",
                    rr.Points3D(
                        positions=static_P.astype(np.float32),
                        colors=static_C.astype(np.uint8),
                    )
                )

            # Cameras & frustums (timeless transforms, separate camera nodes per frame)
            if log_cameras:
                _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
                if not load_from_glb:
                    _log_cameras(static_scene_predictions, fov_y=fov_y, W=W, H=H, type="static",
                                 color=np.array([255, 0, 0], dtype=np.uint8))  # RED
                _log_cameras(dynamic_scene_predictions, fov_y=fov_y, W=W, H=H, type="dynamic",
                             color=np.array([0, 255, 0], dtype=np.uint8))  # GREEN

        # ---- Precompute per-frame MERGED with static ----
        grounded_P: List[np.ndarray] = []
        grounded_C: List[np.ndarray] = []
        for i in tqdm(range(S), desc=f"[viz] Merging frames for {video_id}"):
            Pi, Ci = ground_dynamic_scene_to_static_scene(
                dynamic_scene_predictions,
                static_P, static_C,
                frame_idx=i,
                conf_min=float(conf_frame),
                dedup_voxel=dedup_voxel,
            )
            grounded_P.append(Pi)
            grounded_C.append(Ci)

            rr.set_time_sequence("frame", i)

            if grounded_P[i].size and vis:
                rr.log(
                    "world/frame/points_merged",
                    rr.Points3D(
                        positions=grounded_P[i].astype(np.float32),
                        colors=grounded_C[i].astype(np.uint8),
                        radii=0.01,
                    ),
                )

        # Clone the dynamic scene predictions and add grounded points and grounded colors and store them as a new npz file
        dynamic_scene_predictions_grounded = dynamic_scene_predictions.copy()
        dynamic_scene_predictions_grounded['grounded_points'] = grounded_P
        dynamic_scene_predictions_grounded['grounded_colors'] = grounded_C

        with open(grounded_dynamic_scene_pred_path, 'wb') as f:
            pickle.dump(dynamic_scene_predictions_grounded, f)

        print(f"[viz] Saved grounded dynamic scene predictions to {grounded_dynamic_scene_pred_path}")

        # Cleanup
        del static_scene_predictions
        del dynamic_scene_predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----------------------------- Infer 4D reconstruction video -----------------------------

    def load_masks_for_video(
            self,
            video_id: str,
            target_hw: Optional[Tuple[int, int]] = None,  # (H_out, W_out); default: (H, W)
    ) -> np.ndarray:
        # --- Determine output spatial size ---
        H_out, W_out = int(target_hw[0]), int(target_hw[1])

        # --- Identify annotated frames and the sampled-frame mapping ---
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        annotated_frame_id_list = [f for f in os.listdir(video_frames_annotated_dir_path) if f.endswith('.png')]
        if not annotated_frame_id_list:
            raise FileNotFoundError(f"No annotated frames found in: {video_frames_annotated_dir_path}")
        # Ensure deterministic first frame
        annotated_first_frame_id = int(sorted(annotated_frame_id_list)[0][:-4])

        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list: List[int] = np.load(video_sampled_frames_npy_path).tolist()
        start_idx = video_sampled_frame_id_list.index(annotated_first_frame_id)
        video_sampled_frame_id_list = video_sampled_frame_id_list[start_idx:]

        # --- Load & per-frame resize (2D only) ---
        video_masks_dir_path = os.path.join(self.masks_dir_path, video_id) if self.masks_dir_path is not None else None
        assert video_masks_dir_path is not None, "self.masks_dir_path must be set."

        masks_out = []
        for frame_id in video_sampled_frame_id_list:
            mask_path = os.path.join(video_masks_dir_path, f"{frame_id:06d}.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask not found: {mask_path}")

            # Keep integer IDs crisp
            img = Image.open(mask_path).convert("L")
            img = img.resize((W_out, H_out), resample=Image.NEAREST)
            mask_np = np.array(img, dtype=np.uint8)
            masks_out.append(mask_np)

        masks = np.stack(masks_out, axis=0)  # (S, H_out, W_out)
        return masks

    def infer_4d_reconstruction_video(
            self,
            video_id: str,
            *,
            conf_static: float = 0.1,  # confidence for static background build
            conf_frame: float = 0.01,  # confidence for per-frame points
            dedup_voxel: Optional[float] = 0.02,  # meters; None to disable
            fov_y: float = 0.96,  # radians; matches your earlier default
            spawn: bool = True,
            log_cameras: bool = True,
            load_from_glb: bool = False,
    ) -> None:
        """
        Unified visualization that replaces the old `infer_video` and `infer_video_points_3d`.

        - Builds a static background once (using `conf_static`).
        - Optionally logs per-frame RAW points (like the old `infer_video_points_3d`).
        - Optionally logs per-frame MERGED (static + frame) points (like the old `infer_video`).
        """
        # ---- Load predictions once ----
        static_scene_pred_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        dynamic_scene_pred_path = os.path.join(self.dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions.npz")
        if not os.path.exists(static_scene_pred_path) or not os.path.exists(dynamic_scene_pred_path):
            raise FileNotFoundError(f"predictions.npz not found for {video_id}")

        static_scene_arr = np.load(static_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_arr = np.load(dynamic_scene_pred_path, allow_pickle=True, mmap_mode=None)
        dynamic_scene_predictions = {k: dynamic_scene_arr[k] for k in dynamic_scene_arr.files}
        S, H, W = dynamic_scene_predictions["points"].shape[:3]

        interaction_masks = self.load_masks_for_video(video_id, target_hw=(H, W))  # (S,H,W), uint8

        if load_from_glb:
            print(f"[viz] Loading static scene from GLB for {video_id}...")
            glb_path = os.path.join(self.static_scene_dir_path, f"{video_id[:-4]}_{10}", f"{video_id[:-4]}.glb")
            static_P, static_C = glb_to_points(glb_path)
        else:
            print("Loading static scene from predictions...")
            static_scene_predictions = {k: static_scene_arr[k] for k in static_scene_arr.files}
            static_scene_points_wh = static_scene_predictions["points"]  # (S,H,W,3)
            S, H, W = static_scene_points_wh.shape[:3]
            print(f"[viz] {video_id}: {S} frames | HxW={H}x{W} | conf_static={conf_static} | conf_frame={conf_frame}")

            # ---- Build static background (once) ----
            scene_3d, static_P, static_C = predictions_to_glb_with_static(
                static_scene_predictions, conf_min=float(conf_static),
            )

        # ---- Rerun setup ----
        rr.init(f"AG-Pi3: {video_id}", spawn=spawn)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.log("world", rr.ViewCoordinates.RDF, timeless=True)

        # Log static background timelessly
        if static_P.size > 0:
            rr.log(
                "world/static",
                rr.Points3D(
                    positions=static_P.astype(np.float32),
                    colors=static_C.astype(np.uint8),
                )
            )

        # Cameras & frustums (timeless transforms, separate camera nodes per frame)
        if log_cameras:
            _fx, _fy, _cx, _cy = _pinhole_from_fov(W, H, fov_y)
            if not load_from_glb:
                _log_cameras(static_scene_predictions, fov_y=fov_y, W=W, H=H, type="static",
                             color=np.array([255, 0, 0], dtype=np.uint8))  # RED
            _log_cameras(dynamic_scene_predictions, fov_y=fov_y, W=W, H=H, type="dynamic",
                         color=np.array([0, 255, 0], dtype=np.uint8))  # GREEN

        # ---- Precompute per-frame MERGED with static ----
        merged_P: List[np.ndarray] = []
        merged_C: List[np.ndarray] = []
        for i in tqdm(range(S), desc=f"[viz] Merging frames for {video_id}"):
            Pi, Ci = merge_static_with_frame(
                dynamic_scene_predictions,
                static_P, static_C,
                interaction_masks,
                frame_idx=i,
                conf_min=float(conf_frame),
                dedup_voxel=dedup_voxel,
            )
            merged_P.append(Pi)
            merged_C.append(Ci)

            rr.set_time_sequence("frame", i)

            if merged_P[i].size:
                rr.log(
                    "world/frame/points_merged",
                    rr.Points3D(
                        positions=merged_P[i].astype(np.float32),
                        colors=merged_C[i].astype(np.uint8),
                        radii=0.01,
                    ),
                )

        print("[viz] Done streaming frames to Rerun.")

        # Cleanup
        del static_scene_predictions
        del dynamic_scene_predictions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------- Batch over a split -------------------------------------------

    def infer_all_videos(
            self,
            split: str,
            *,
            mode: str = "both",
            allowlist: Optional[List[str]] = None,
            **kwargs,
    ) -> None:
        """Process all videos in a split (optionally restricted by an allowlist)."""
        video_id_list = sorted(os.listdir(self.root_dir_path))

        video_id_list = ["A015X.mp4"]

        # Filter by naming convention and split
        filtered: List[str] = []
        for vid in video_id_list:
            if allowlist is not None and vid not in allowlist:
                continue
            if get_video_belongs_to_split(vid) == split:
                filtered.append(vid)

        if not filtered:
            print(f"[warn] No videos matching split={split} under {self.root_dir_path}")
            return

        for video_id in tqdm(filtered, desc=f"Split {split}"):
            self.infer_grounded_dynamic_video(video_id, **kwargs)

def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize static + per-frame 3D points with Rerun (AG-Pi3 unified)."
    )
    # Paths
    parser.add_argument(
        "--root_dir_path",
        type=str,
        default="/data/rohith/ag/frames",
        help="Path whose entries include the video IDs to process.",
    )
    parser.add_argument(
        "--frames_annotated_dir_path",
        type=str,
        default="/data/rohith/ag/frames_annotated",
        help="Optional: directory containing annotated frames (unused here).",
    )
    parser.add_argument(
        "--mask_dir_path",
        type=str,
        default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_masks",
        help="Path to directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--output_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_full",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )
    parser.add_argument(
        "--static_scene_dir_path",
        type=str,
        default="/data2/rohith/ag/ag4D/static_scenes/pi3",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_full"
    )
    parser.add_argument(
        "--grounded_dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_grounded"
    )
    # Selection
    parser.add_argument(
        "--split",
        type=_parse_split,
        default="AD",
        help="Shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}.",
    )
    parser.add_argument(
        "--video_id",
        type=str,
        default=None,
        help="If set, run only this video ID (ignores --split filter).",
    )

    # Viz options
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["raw", "merged", "both"],
        help="What to log per frame: raw points, merged (static+frame), or both.",
    )
    parser.add_argument(
        "--conf_static",
        type=float,
        default=0.10,
        help="Confidence threshold for building static background (0..1).",
    )
    parser.add_argument(
        "--conf_frame",
        type=float,
        default=0.01,
        help="Confidence threshold for per-frame points (0..1).",
    )
    parser.add_argument(
        "--dedup_voxel",
        type=float,
        default=0.02,
        help="Voxel size (m) for de-duplication in merged clouds; set <=0 to disable.",
    )
    parser.add_argument(
        "--fov_y",
        type=float,
        default=0.96,
        help="Vertical field-of-view (radians) for Pinhole used in camera logging.",
    )
    parser.add_argument(
        "--no_spawn",
        action="store_true",
        help="Do not spawn the external Rerun viewer (use in-process).",
    )
    parser.add_argument(
        "--no_cam",
        action="store_true",
        help="Disable camera/frustum logging.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dedup = None if (args.dedup_voxel is not None and args.dedup_voxel <= 0) else args.dedup_voxel

    ag_pi3 = AgPi3(
        root_dir_path=args.root_dir_path,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        static_scene_dir_path=args.static_scene_dir_path,
        frame_annotated_dir_path=args.frames_annotated_dir_path,
        grounded_dynamic_scene_dir_path=args.grounded_dynamic_scene_dir_path,
        masks_dir_path=args.mask_dir_path,
    )

    ag_pi3.infer_all_videos(
        split=args.split,
        mode=args.mode,
        conf_static=args.conf_static,
        conf_frame=args.conf_frame,
        dedup_voxel=dedup,
        fov_y=args.fov_y,
        spawn=not args.no_spawn,
        log_cameras=not args.no_cam,
        vis=True,
    )


if __name__ == "__main__":
    main()
