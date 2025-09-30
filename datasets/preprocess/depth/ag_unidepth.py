import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from UniDepth.unidepth.models import UniDepthV2
from utils import get_video_belongs_to_split


class AgUniDepth:

    def __init__(self, datapath):
        self.datapath = datapath
        self.frames_path = os.path.join(self.datapath, "sampled_frames")
        self.video_list = sorted(os.listdir(self.frames_path))

        self._unidepth_root = os.path.join(self.datapath, 'ag4D', "unidepth")
        os.makedirs(self._unidepth_root, exist_ok=True)
        self.LONG_DIM = 640

        # -------------------------- UNIDEPTH --------------------------

    def load_unidepth_model(self, args):
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

    def run_ag_unidepth_estimation(self, split):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = sorted([int(Path(p).stem) for p in
                                    glob.glob(os.path.join(video_frames_path, "*.png"))])
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            if get_video_belongs_to_split(video_id) == split:
                self.video_unidepth_estimation(video_id, img_paths)
        print("Depth estimation completed for all videos.")


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default="/data/rohith/ag/")

    parser.add_argument(
        "--split", type=_parse_split, default="04",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}. "
             "If omitted, processes all videos."
    )

    args = parser.parse_args()
    ag_unidepth = AgUniDepth(datapath=args.datapath)
    ag_unidepth.load_unidepth_model(args)
    ag_unidepth.run_ag_unidepth_estimation(args.split)


if __name__ == "__main__":
    main()