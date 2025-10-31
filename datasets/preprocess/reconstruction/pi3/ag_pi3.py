import argparse
import gc
import os
import pickle
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from datasets.preprocess.reconstruction.pi3.recon_utils import get_video_belongs_to_split, \
    predictions_to_glb_with_static, ground_dynamic_scene_to_static_scene
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from pi3.utils.geometry import depth_edge
from concurrent.futures import ThreadPoolExecutor, as_completed


class AgPi3:

    def __init__(
            self,
            static_root_dir_path,
            dynamic_root_dir_path,
            static_output_dir_path,
            dynamic_output_dir_path,
            grounded_dynamic_output_dir_path,
            frame_annotated_dir_path=None,
    ):
        self.model = None
        self.static_root_dir_path = static_root_dir_path
        self.dynamic_root_dir_path = dynamic_root_dir_path

        self.static_output_dir_path = static_output_dir_path
        os.makedirs(self.static_output_dir_path, exist_ok=True)
        self.dynamic_output_dir_path = dynamic_output_dir_path
        os.makedirs(self.dynamic_output_dir_path, exist_ok=True)

        self.frame_annotated_dir_path = frame_annotated_dir_path
        self.grounded_dynamic_scene_dir_path = grounded_dynamic_output_dir_path
        os.makedirs(self.grounded_dynamic_scene_dir_path, exist_ok=True)
        self.sampled_frames_idx_root_dir = "/data/rohith/ag/sampled_frames_idx"

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.load_model()

    def load_model(self):
        self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()

    def preprocess_image_list(self, data_path, is_video=False, video_id=None, sample_idx=None):
        interval = 10 if is_video else 1
        print(f'Sampling interval: {interval}')
        imgs = load_images_as_tensor(
            data_path,
            interval=interval,
            video_id=video_id,
            sample_idx=sample_idx
        ).to(self.device)  # (N, 3, H, W)
        return imgs

    def fetch_predictions(self, imgs):
        print("Running model inference...")
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                predictions = self.model(imgs[None])

        # Process mask
        masks = torch.sigmoid(predictions['conf'][..., 0]) > 0.1
        non_edge = ~depth_edge(predictions['local_points'][..., 2], rtol=0.03)
        masks = torch.logical_and(masks, non_edge)[0]

        predictions['images'] = imgs[None].permute(0, 1, 3, 4, 2)
        predictions['conf'] = torch.sigmoid(predictions['conf'])
        edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
        predictions['conf'][edge] = 0.0
        # # transform to first camera coordinate
        # predictions['points'] = torch.einsum('bij, bnhwj -> bnhwi', se3_inverse(predictions['camera_poses'][:, 0]), homogenize_points(predictions['points']))[..., :3]
        # predictions['camera_poses'] = torch.einsum('bij, bnjk -> bnik', se3_inverse(predictions['camera_poses'][:, 0]), predictions['camera_poses'])

        # Convert tensors to numpy
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension

        return predictions

    def infer_static_scene(
            self,
            video_id,
            conf_thres=10.0,
            sample_idx=None,
            video_sampled_frame_id_list=None
    ):
        data_path = f'{self.static_root_dir_path}/{video_id}'
        video_save_dir = os.path.join(self.static_output_dir_path, f"{video_id[:-4]}_{int(conf_thres)}")
        if os.path.exists(os.path.join(video_save_dir, "predictions.npz")):
            print(f"Skipping video {video_id} static scene creation as output already exists.")
            return None
        os.makedirs(video_save_dir, exist_ok=True)

        imgs = self.preprocess_image_list(
            data_path,
            is_video=False,
            video_id=video_id,
            sample_idx=sample_idx
        )

        return imgs

    def infer_dynamic_scene(
            self,
            video_id,
            conf_thres=10.0,
            sample_idx=None,
            video_sampled_frame_id_list=None
    ):
        data_path = f'{self.dynamic_root_dir_path}/{video_id}'
        video_save_dir = os.path.join(self.dynamic_output_dir_path, f"{video_id[:-4]}_{int(conf_thres)}")

        if os.path.exists(os.path.join(video_save_dir, "predictions.npz")):
            print(f"Skipping video {video_id} as output already exists.")
            return
        os.makedirs(video_save_dir, exist_ok=True)

        # NOTE: Use this for full frames not the sampled frames (Comment it for the other case)
        sample_idx = [video_sampled_frame_id_list[i] for i in sample_idx]

        imgs = self.preprocess_image_list(
            data_path,
            is_video=False,
            video_id=video_id,
            sample_idx=sample_idx
        )

        return imgs

    def infer_video(self, video_id, conf_thres=10.0, conf_static=0.1, dedup_voxel=0.02, conf_min=0.01):
        video_frames_annotated_dir_path = os.path.join(self.frame_annotated_dir_path, video_id)
        grounded_dynamic_scene_pred_path = os.path.join(
            self.grounded_dynamic_scene_dir_path, f"{video_id[:-4]}_{10}", "predictions_grounded.pkl"
        )
        os.makedirs(os.path.dirname(grounded_dynamic_scene_pred_path), exist_ok=True)

        annotated_frame_id_list = os.listdir(video_frames_annotated_dir_path)
        annotated_frame_id_list = [f for f in annotated_frame_id_list if f.endswith('.png')]
        annotated_first_frame_id = int(annotated_frame_id_list[0][:-4])
        annotated_last_frame_id = int(annotated_frame_id_list[-1][:-4])

        # Get the mapping for sampled_frame_id and the actual frame id
        # Now start from the sampled frame which corresponds to the first annotated frame and keep the rest of the sampled frames
        video_sampled_frames_npy_path = os.path.join(self.sampled_frames_idx_root_dir, f"{video_id[:-4]}.npy")
        video_sampled_frame_id_list = np.load(video_sampled_frames_npy_path).tolist()  # Numbers only

        an_first_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_first_frame_id)
        an_last_id_in_vid_sam_frame_id_list = video_sampled_frame_id_list.index(annotated_last_frame_id)
        sample_idx = list(range(an_first_id_in_vid_sam_frame_id_list, an_last_id_in_vid_sam_frame_id_list + 1))

        # Infer static scene
        static_images = self.infer_static_scene(video_id, conf_thres=conf_thres, sample_idx=sample_idx, video_sampled_frame_id_list=video_sampled_frame_id_list)
        if static_images is not None:
            print("Fetching static scene predictions...")
            static_predictions = self.fetch_predictions(static_images)
            static_video_save_dir = os.path.join(self.static_output_dir_path, f"{video_id[:-4]}_{int(conf_thres)}")

            glbfile = os.path.join(static_video_save_dir, f"{video_id[:-4]}.glb")
            glbscene, static_P, static_C = predictions_to_glb_with_static(static_predictions, conf_min=float(conf_static))
            glbscene.export(file_obj=glbfile)

            static_predictions["static_points"] = static_P
            static_predictions["static_colors"] = static_C

            prediction_save_path = os.path.join(static_video_save_dir, "predictions.npz")
            np.savez(prediction_save_path, **static_predictions)
            print("--------------------------------------------------------------------------------------------")
        else:
            print(f"[{video_id}] Static scene for video already exists. Skipping static scene creation.")

        # Infer dynamic scene
        dynamic_images = self.infer_dynamic_scene(video_id, conf_thres=conf_thres, sample_idx=sample_idx, video_sampled_frame_id_list=video_sampled_frame_id_list)
        if dynamic_images is not None:
            print("Fetching dynamic scene predictions...")
            dynamic_predictions = self.fetch_predictions(dynamic_images)
            dynamic_video_save_dir = os.path.join(self.dynamic_output_dir_path, f"{video_id[:-4]}_{int(conf_thres)}")

            prediction_save_path = os.path.join(dynamic_video_save_dir, "predictions.npz")
            np.savez(prediction_save_path, **dynamic_predictions)
            print("--------------------------------------------------------------------------------------------")
        else:
            print(f"[{video_id}] Dynamic scene for video already exists. Skipping dynamic scene creation.")

        # Verify both predictions dimensions match
        if dynamic_images is not None and static_images is not None:
            assert dynamic_predictions['points'].shape == static_predictions['points'].shape, \
                (f"Predictions shape mismatch between static {static_predictions['points'].shape} "
                 f"and dynamic {dynamic_predictions['points'].shape} scenes for video {video_id}.")

            S, H, W = dynamic_predictions["points"].shape[:3]

            # ---- Precompute per-frame MERGED with static ----
            grounded_P: List[np.ndarray] = [None] * S
            grounded_C: List[np.ndarray] = [None] * S
            updated_poses: List[np.ndarray] = [None] * S
            def _merge_one(idx: int):
                # Read-only access; no mutation of shared arrays
                Pi, Ci, updated_pose = ground_dynamic_scene_to_static_scene(
                    dynamic_predictions,
                    static_P, static_C,
                    frame_idx=idx,
                    conf_min=conf_min,
                    dedup_voxel=dedup_voxel,
                )
                assert updated_pose is not None
                return idx, Pi, Ci, updated_pose

            # Reasonable default: leave 1 core free; cap to avoid oversubscription
            max_workers = min(max(1, (os.cpu_count() or 4) - 1), 16)
            desc = f"[viz] Merging frames for {video_id} (parallel x{max_workers})"

            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(_merge_one, i) for i in range(S)]
                for fut in tqdm(as_completed(futures), total=S, desc=desc):
                    idx, Pi, Ci, updated_pose = fut.result()
                    grounded_P[idx] = Pi
                    grounded_C[idx] = Ci
                    updated_poses[idx] = updated_pose

            # Clone the dynamic scene predictions and add grounded points and grounded colors and store them as a new npz file
            dynamic_scene_predictions_grounded = dynamic_predictions.copy()
            dynamic_scene_predictions_grounded['grounded_points'] = grounded_P
            dynamic_scene_predictions_grounded['grounded_colors'] = grounded_C
            dynamic_scene_predictions_grounded['updated_camera_poses'] = np.array(updated_poses)

            with open(grounded_dynamic_scene_pred_path, 'wb') as f:
                pickle.dump(dynamic_scene_predictions_grounded, f)

            print(f"[viz] Saved grounded dynamic scene predictions to {grounded_dynamic_scene_pred_path}")

            # Cleanup
            del static_predictions
            del dynamic_predictions

        gc.collect()
        torch.cuda.empty_cache()

    def infer_all_videos(self, split):
        # video_id_list = os.listdir(self.static_root_dir_path)
        video_id_list = ["0DJ6R.mp4"]
        for video_id in tqdm(video_id_list):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            try:
                self.infer_video(video_id)
            except Exception as e:
                print(f"[ERROR] Error processing video {video_id}: {e}")


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample frames from videos based on homography-overlap filtering."
    )

    # Note: Use this for in-painting of frames
    # parser.add_argument(
    #     "--root_dir_path", type=str, default="/data/rohith/ag/ag4D/static_frames",
    #     help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    # )

    parser.add_argument(
        "--static_root_dir_path", type=str, default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_frames",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    parser.add_argument(
        "--dynamic_root_dir_path", type=str, default="/data/rohith/ag/frames",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )

    parser.add_argument(
        "--frames_annotated_dir_path", type=str, default="/data/rohith/ag/frames_annotated",
        help="Path to directory containing annotated frames (with masks)."
    )

    parser.add_argument(
        "--static_output_dir_path", type=str, default="/data3/rohith/ag/ag4D/static_scenes/pi3_static",
        help="Path to output directory where results will be saved."
    )
    parser.add_argument(
        "--dynamic_output_dir_path", type=str, default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
        help="Path to output directory where results will be saved."
    )
    parser.add_argument(
        "--grounded_dynamic_output_dir_path",
        type=str,
        default="/data2/rohith/ag/ag4D/dynamic_scenes/pi3_grounded_dynamic"
    )

    parser.add_argument(
        "--split", type=_parse_split, default="04",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}. "
             "If omitted, processes all videos."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ag_pi3 = AgPi3(
        static_root_dir_path=args.static_root_dir_path,
        dynamic_root_dir_path=args.dynamic_root_dir_path,
        static_output_dir_path=args.static_output_dir_path,
        dynamic_output_dir_path=args.dynamic_output_dir_path,
        grounded_dynamic_output_dir_path=args.grounded_dynamic_output_dir_path,
        frame_annotated_dir_path=args.frames_annotated_dir_path,
    )
    ag_pi3.infer_all_videos(args.split)


if __name__ == "__main__":
    main()
