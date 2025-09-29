import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from UniDepth.unidepth.models import UniDepthV2


class AgUniDepth:

    def __init__(self, datapath):
        self.datapath = datapath

        # ------- UniDepth parameters -------
        self._unidepth_root = os.path.join(self.datapath, 'ag4D', "mega_sam", "unidepth")
        self.LONG_DIM = 640

        # -------------------------- UNIDEPTH --------------------------

    def _load_unidepth_model(self, args):
        self._unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14",
                                                          revision="1d0d3c52f60b5164629d279bb9a7546458e6dcc4")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._unidepth_model = self._unidepth_model.to(device)
        self._unidepth_model.eval()

    def video_unidepth_estimation(self, video_id, img_paths):
        output_dir = os.path.join(self._unidepth_root, video_id)

        if os.path.exists(output_dir):
            if len(os.listdir(output_dir)) == len(img_paths):
                print(f"Depth estimation already completed for {video_id}. Skipping...")
                return
            else:
                # Remove the existing directory if it is not empty and not complete
                os.rmdir(output_dir)
                print(f"Removing incomplete directory for {video_id}...")

        os.makedirs(output_dir, exist_ok=True)

        fovs = []
        for img_path in tqdm(img_paths):
            rgb = np.array(Image.open(img_path))[..., :3]
            if rgb.shape[1] > rgb.shape[0]:
                final_w, final_h = self.LONG_DIM, int(
                    round(self.LONG_DIM * rgb.shape[0] / rgb.shape[1])
                )
            else:
                final_w, final_h = (
                    int(round(self.LONG_DIM * rgb.shape[1] / rgb.shape[0])),
                    self.LONG_DIM,
                )
            rgb = cv2.resize(rgb, (final_w, final_h), cv2.INTER_AREA)  # .transpose(2, 0, 1)

            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)
            # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
            # predict
            predictions = self._unidepth_model.infer(rgb_torch)
            fov_ = np.rad2deg(
                2
                * np.arctan(
                    predictions["depth"].shape[-1]
                    / (2 * predictions["intrinsics"][0, 0, 0].cpu().numpy())
                )
            )
            depth = predictions["depth"][0, 0].cpu().numpy()
            # print(fov_)
            fovs.append(fov_)
            # breakpoint()
            np.savez(
                os.path.join(output_dir, img_path.split("/")[-1][:-4] + ".npz"),
                depth=np.float32(depth),
                fov=fov_,
            )

    def run_ag_unidepth_estimation(self):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = self.video_id_frame_id_list[video_id]
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            self.video_unidepth_estimation(video_id, img_paths)

        print("Depth estimation completed for all videos.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/data/rohith/ag/")

    # # ------------------------------ DEPTH ANYTHING ----------------------------
    parser.add_argument('--encoder', type=str, default='vitl')
    parser.add_argument('--load_from', type=str,
                        default='/home/rxp190007/CODE/mega-sam/Depth_Anything/checkpoints/depth_anything_vitl14.pth')
    parser.add_argument('--localhub', dest='localhub', action='store_true', default=False)

    args = parser.parse_args()
    ag_unidepth = AgUniDepth(datapath=args.datapath)
    ag_unidepth._load_unidepth_model(args)
    ag_unidepth.run_ag_unidepth_estimation()


if __name__ == "__main__":
    main()