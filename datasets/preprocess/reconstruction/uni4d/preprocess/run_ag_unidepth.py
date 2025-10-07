import pickle

import torch
import imageio
import numpy as np
import os
from unidepth.models import UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole, BatchCamera
import argparse
from tqdm import tqdm
import time
import cv2


class AgUniDepth:
    def __init__(
            self,
            ag_root_dir,
            is_v2,
            use_gt_K
    ):
        self.ag_root_dir = ag_root_dir
        self.is_v2 = is_v2
        self.use_gt_K = use_gt_K
        self.batch_size = 1
        self.model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.ag_frames_dir = os.path.join(ag_root_dir, "frames")
        self.ag_4D_dir = os.path.join(ag_root_dir, "ag4D")
        self.uni4D_dir = os.path.join(self.ag_4D_dir, "uni4D")
        self.unidepth_dir = os.path.join(self.uni4D_dir, "unidepth")
        os.makedirs(self.unidepth_dir, exist_ok=True)

        self.frames_path = os.path.join(self.ag_root_dir, "frames")
        self.annotations_path = os.path.join(self.ag_root_dir, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        video_output_dir = os.path.join(self.unidepth_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        video = self.get_frames(video_name)
        video = torch.from_numpy(video)
        if torch.cuda.is_available():
            video = video.cuda()
        f, h, w, c = video.shape
        with torch.no_grad():
            frame_range = list(range(f))
            chuncks = [frame_range[i:i + self.batch_size] for i in range(0, len(frame_range), self.batch_size)]
            initial_intrinsics = []
            for chunk in chuncks:
                imgs = video[chunk, ...]
                imgs = torch.permute(imgs, (0, 3, 1, 2))
                preds = self.model.infer(imgs)
                initial_intrinsics.append(preds['intrinsics'])
        initial_intrinsics = torch.cat(initial_intrinsics, dim=0)
        initial_intrinsics = torch.mean(initial_intrinsics, dim=0)
        K = initial_intrinsics.detach().cpu()
        K[0][-1] = w / 2
        K[1][-1] = h / 2

        intrinsics_save_path = os.path.join(video_output_dir, 'intrinsics.npy')
        depth_save_path_disp = os.path.join(video_output_dir, 'depth.npy')
        np.save(intrinsics_save_path, K)

        if len(K.shape) == 2:
            K = K.unsqueeze(0).repeat(video.shape[0], 1, 1)

        depths = []
        with torch.no_grad():
            frame_range = list(range(f))
            chuncks = [frame_range[i:i + self.batch_size] for i in range(0, len(frame_range), self.batch_size)]
            for chunk in chuncks:
                imgs = video[chunk, ...]
                Ks = K[chunk, ...]
                if isinstance(self.model, (UniDepthV2)):
                    Ks = Pinhole(K=Ks)
                    cam = BatchCamera.from_camera(Ks)
                imgs = torch.permute(imgs, (0, 3, 1, 2))
                preds = self.model.infer(imgs, Ks)
                depth = preds['depth']  # B x 1 x H x W
                depths.append(depth)
            depths = torch.cat(depths, dim=0)
            depths = depths.detach().cpu().numpy()

        depth_vis = depths[:, 0, :, :]
        disp_vis = 1 / (depth_vis + 1e-12)
        disp_vis = cv2.normalize(disp_vis, disp_vis, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        for f in range(len(disp_vis)):
            disp = disp_vis[f]
            disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
            imageio.imwrite(os.path.join(video_output_dir, str(f).zfill(4) + ".png"), disp)
        np.save(depth_save_path_disp, depths)


def main():
    parser = argparse.ArgumentParser(description="Run UniDepth on AG4D dataset")
    parser.add_argument("--ag_root_dir", type=str, default="/data/rohith/ag/", help="Path to AG4D root directory")
    parser.add_argument("--is_v2", action="store_true", help="Use UniDepth V2 model")
    parser.add_argument("--use_gt_K", action="store_true", help="Use ground truth camera intrinsics")
    args = parser.parse_args()

    ag_unidepth = AgUniDepth(
        ag_root_dir=args.ag_root_dir,
        is_v2=args.is_v2,
        use_gt_K=args.use_gt_K
    )

    # for video_name in tqdm(ag_unidepth.video_list):
    #     ag_unidepth.process_ag_video(video_name)

    video_name = "00T1E.mp4"  # Replace with an actual video name from your dataset
    ag_unidepth.process_ag_video(video_name)


if __name__ == "__main__":
    main()
