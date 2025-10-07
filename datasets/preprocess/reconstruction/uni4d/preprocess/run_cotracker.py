import torch
import os
# add to path
import sys
import imageio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm

sys.path.insert(0,'./co-tracker')
from cotracker.predictor import CoTrackerPredictor 


def get_frames(path):
    
    frames = []
    files = sorted([x for x in os.listdir(path) if x.endswith(".png") or x.endswith(".jpg")])
    for file in files:
        # res = np.load(os.path.join(path, file))
        frames.append(imageio.imread(os.path.join(path, file)))
    frames = np.array(frames) # F x H x W x C

    return frames

def vis(pred_tracks, pred_visibility, video, output_dir):
    """
    Visualize tracklets, highlighting neighbors as red and others as blue if neighbors is not None
    """

    pred_tracks = pred_tracks              # F x N x 2
    pred_visibility = pred_visibility      # F x N
    video = np.array(video.cpu())[0].astype(np.uint8)       # F x C x H x W
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        output_path = os.path.join(output_dir, 'vis', f'frame_{frame_idx:04d}.jpg')
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    
    print(f"Visualization complete. {num_frames} frames saved to {output_dir}")
    print(f"Displayed {sample_size} tracklets out of {num_tracklets} total (1%)")

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
BASE = args.workdir

INTERVAL = args.interval
GRID_SIZE = args.grid_size

videos = sorted(os.listdir(BASE))

for idx, name in tqdm(enumerate(videos)):

    print(f"Working on {name}")
    video = get_frames(os.path.join(BASE, f"{name}/rgb/"))
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()    

    grid_query_frame=0

    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './preprocess/pretrained/scaled_offline.pth'
        )
    )

    if torch.cuda.is_available():
        model = model.cuda()
        video_interp = video.cuda()

    _, F, _, H, W = video.shape

    QUERY_FRAMES = [x for x in range(0, video.shape[1], INTERVAL)]

    all_tracks = np.zeros((F, 0, 2))
    all_visibilities = np.zeros((F, 0))
    all_confidences = np.zeros((F, 0))
    all_init_frames = []

    for f in QUERY_FRAMES:

        pred_tracks, pred_visibility, pred_confidence = model(video_interp, grid_size=GRID_SIZE, grid_query_frame=f, backward_tracking=True)

        frame_tracks = pred_tracks[0].cpu().numpy()
        frame_visibilities = pred_visibility[0].cpu().numpy()
        frame_confidences = pred_confidence[0].cpu().numpy()

        init_frames = np.repeat(f, frame_tracks.shape[1])

        all_tracks = np.concatenate((all_tracks, frame_tracks), axis=1)
        all_visibilities = np.concatenate((all_visibilities, frame_visibilities), axis=1)
        all_confidences = np.concatenate((all_confidences, frame_confidences), axis=1)
        all_init_frames.extend(init_frames)

    feature_track_path = os.path.join(BASE, f"{name}/cotrackerv3_{INTERVAL}_{GRID_SIZE}")

    os.makedirs(feature_track_path, exist_ok=True)

    np.savez(os.path.join(feature_track_path, "results.npz"), all_confidences=all_confidences, all_tracks=all_tracks, all_visibilities=all_visibilities, init_frames=all_init_frames, orig_shape=video.shape[-2:])

    vis(all_tracks, all_visibilities, video, feature_track_path)