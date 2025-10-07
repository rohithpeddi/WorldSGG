import os
import pickle
import sys
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm

sys.path.insert(0, './co-tracker')
from cotracker.predictor import CoTrackerPredictor


class AgCoTracker:

    def __init__(
            self,
            ag_root_dir,
            interval,
            grid_size
    ):
        self.ag_root_dir = ag_root_dir
        self.interval = interval
        self.grid_size = grid_size
        self.model = CoTrackerPredictor(
            checkpoint=os.path.join(
                './preprocess/pretrained/scaled_offline.pth'
            )
        )
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.ag_frames_dir = os.path.join(ag_root_dir, "frames")
        self.ag_4D_dir = os.path.join(ag_root_dir, "ag4D")
        self.uni4D_dir = os.path.join(self.ag_4D_dir, "uni4D")
        self.cotracker_dir = os.path.join(self.uni4D_dir, "cotracker")
        os.makedirs(self.cotracker_dir, exist_ok=True)

        self.frames_path = os.path.join(self.ag_root_dir, "frames")
        self.annotations_path = os.path.join(self.ag_root_dir, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

        video_id_frame_id_list_pkl_file_path = os.path.join(self.ag_root_dir, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)
        else:
            assert False, f"Please generate {video_id_frame_id_list_pkl_file_path} first"

    def get_frames(self, video_name):
        frames = []
        video_frames_path = os.path.join(self.frames_path, video_name)
        frame_id_list = self.video_id_frame_id_list[video_name]
        frame_id_list = sorted(np.unique(frame_id_list))
        for frame_id in frame_id_list:
            frames.append(imageio.imread(os.path.join(video_frames_path, f"{frame_id:06d}.png")))
        frames = np.array(frames)  # F x H x W x C
        return frames

    def process_ag_video(self, video_name):
        video = self.get_frames(video_name)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
        if torch.cuda.is_available():
            video = video.cuda()
        _, F, _, H, W = video.shape
        QUERY_FRAMES = [x for x in range(0, video.shape[1], self.interval)]
        all_tracks = np.zeros((F, 0, 2))
        all_visibilities = np.zeros((F, 0))
        all_confidences = np.zeros((F, 0))
        all_init_frames = []
        for f in QUERY_FRAMES:
            pred_tracks, pred_visibility, pred_confidence = (
                self.model(
                    video,
                    grid_size=self.grid_size,
                    grid_query_frame=f,
                    backward_tracking=True)
            )
            frame_tracks = pred_tracks[0].cpu().numpy()
            frame_visibilities = pred_visibility[0].cpu().numpy()
            frame_confidences = pred_confidence[0].cpu().numpy()
            init_frames = np.repeat(f, frame_tracks.shape[1])
            all_tracks = np.concatenate((all_tracks, frame_tracks), axis=1)
            all_visibilities = np.concatenate((all_visibilities, frame_visibilities), axis=1)
            all_confidences = np.concatenate((all_confidences, frame_confidences), axis=1)
            all_init_frames.extend(init_frames)
        feature_track_file_path = os.path.join(self.cotracker_dir, f"{video_name.split('.')[0]}.npz")
        np.savez(
            feature_track_file_path,
            all_confidences=all_confidences,
            all_tracks=all_tracks,
            all_visibilities=all_visibilities,
            init_frames=all_init_frames,
            orig_shape=video.shape[-2:]
        )

        self.vis(all_tracks, all_visibilities, video, video_name)

    def vis(self, pred_tracks, pred_visibility, video, video_name):
        vis_output_dir = os.path.join(self.cotracker_dir, "vis", video_name.split('.')[0])
        # Create output directory if it doesn't exist
        os.makedirs(vis_output_dir, exist_ok=True)

        pred_tracks = pred_tracks  # F x N x 2
        pred_visibility = pred_visibility  # F x N
        video = np.array(video.cpu())[0].astype(np.uint8)  # F x C x H x W

        # Get video dimensions
        frame_height, frame_width = video.shape[2], video.shape[3]

        # Determine how many tracklets to sample (1% of total)
        num_tracklets = pred_tracks.shape[1]
        sample_size = max(1, int(num_tracklets * 0.01))  # Ensure at least 1 tracklet

        # Randomly sample tracklet indices (1% of all tracklets)
        sampled_indices = random.sample(range(num_tracklets), sample_size)

        # Generate a unique color for each sampled tracklet (consistent across frames)
        tracklet_colors = np.random.rand(sample_size, 3)  # RGB colors for each tracklet

        # Process each frame
        num_frames = video.shape[0]
        for frame_idx in range(num_frames):
            # Get current frame and convert from CxHxW to HxWxC
            frame = video[frame_idx].transpose(1, 2, 0)

            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(frame)

            # Get coordinates of all sampled points for this frame
            x_coords = pred_tracks[frame_idx, sampled_indices, 0]
            y_coords = pred_tracks[frame_idx, sampled_indices, 1]
            visibility = pred_visibility[frame_idx, sampled_indices]

            # Create in-bounds mask to filter out points that go outside the frame
            in_bounds_mask = (
                    (x_coords >= 0) &
                    (x_coords < frame_width) &
                    (y_coords >= 0) &
                    (y_coords < frame_height)
            )

            # Combine visibility and bounds check
            visible_mask = visibility > 0
            visible_and_in_bounds = visible_mask & in_bounds_mask
            invisible_and_in_bounds = (~visible_mask) & in_bounds_mask

            # Plot visible points (filled markers) that are in bounds
            if np.any(visible_and_in_bounds):
                indices_to_plot = np.where(visible_and_in_bounds)[0]
                ax.scatter(
                    x_coords[indices_to_plot],
                    y_coords[indices_to_plot],
                    c=tracklet_colors[indices_to_plot],
                    s=32,  # size
                    alpha=1.0
                )

            # Plot invisible points (hollow markers) that are in bounds
            if np.any(invisible_and_in_bounds):
                indices_to_plot = np.where(invisible_and_in_bounds)[0]
                for i in indices_to_plot:
                    ax.scatter(
                        x_coords[i],
                        y_coords[i],
                        facecolors='none',
                        edgecolors=[tracklet_colors[i]],  # Same color as when visible
                        s=32,  # size
                        linewidths=2
                    )

            # Add a legend or text indicating these are 1% of tracklets
            ax.text(10, 20, f"Showing {sample_size}/{num_tracklets} tracklets (1%)",
                    color='white', backgroundcolor='black', fontsize=10)

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_yticks([])

            # Save the
            output_path = os.path.join(vis_output_dir, f'frame_{frame_idx:04d}.jpg')
            plt.savefig(output_path, bbox_inches='tight', dpi=100)
            plt.close()

        print(f"Visualization complete. {num_frames} frames saved to {vis_output_dir}")
        print(f"Displayed {sample_size} tracklets out of {num_tracklets} total (1%)")


def main():
    parser = argparse.ArgumentParser(description='Running cotrackers')
    parser.add_argument('--workdir',
                        metavar='DIR',
                        help='path to dataset',
                        default="sintel")
    parser.add_argument('--interval',
                        help='interval for cotracker',
                        default=10,
                        type=int)
    parser.add_argument('--grid_size',
                        help='grid size for cotracker',
                        default=50,
                        type=int)
    args = parser.parse_args()

    ag_cotracker = AgCoTracker(
        ag_root_dir="/data/rohith/ag",
        interval=args.interval,
        grid_size=args.grid_size
    )
    video_name = "00T1E.mp4"
    ag_cotracker.process_ag_video(video_name)


if __name__ == "__main__":
    main()
