import argparse
import glob
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

from core.raft import RAFT
from core.utils.utils import InputPadder


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


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv2.remap(img, flow_new, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return res


def resize_flow(flow, img_h, img_w):
    flow_h, flow_w = flow.shape[0], flow.shape[1]
    flow[:, :, 0] *= float(img_w) / float(flow_w)
    flow[:, :, 1] *= float(img_h) / float(flow_h)
    flow = cv2.resize(flow, (img_w, img_h), cv2.INTER_LINEAR)
    return flow


class AgFlow:

    def __init__(self, datapath):
        self.datapath = datapath
        self.frames_path = os.path.join(self.datapath, "sampled_frames")
        self.video_list = sorted(os.listdir(self.frames_path))

        self._flow_root = os.path.join(self.datapath, 'ag4D', "flow", "raft")
        os.makedirs(self._flow_root, exist_ok=True)
        self._flow_estimation_model = None
        self._flow_model = None

    def load_flow_estimation_model(self, args):
        self._flow_estimation_model = torch.nn.DataParallel(RAFT(args))
        self._flow_estimation_model.load_state_dict(torch.load(args.model))
        print(f'Loaded checkpoint at {args.model}')
        self._flow_model = self._flow_estimation_model.module
        self._flow_model.cuda()
        self._flow_model.eval()

    def video_preprocess_flow(self, video_id, image_list):
        img_data = []
        for t, (image_file) in tqdm(enumerate(image_list)):
            image = cv2.imread(image_file)[..., ::-1]  # rgb
            h0, w0, _ = image.shape
            h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
            w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
            image = cv2.resize(image, (w1, h1))
            image = image[: h1 - h1 % 8, : w1 - w1 % 8].transpose(2, 0, 1)
            img_data.append(image)

        img_data = np.array(img_data)

        flows_low = []
        flows_high = []
        flow_masks_high = []
        flow_init = None
        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}

        ii = []
        jj = []
        flows_arr_up = []
        masks_arr_up = []

        for step in [1, 2, 4, 8, 15]:
            flows_arr_low = []
            for i in tqdm(range(max(0, -step), img_data.shape[0] - max(0, step))):
                image1 = (
                    torch.as_tensor(np.ascontiguousarray(img_data[i: i + 1]))
                    .float()
                    .cuda()
                )
                image2 = (
                    torch.as_tensor(
                        np.ascontiguousarray(img_data[i + step: i + step + 1])
                    )
                    .float()
                    .cuda()
                )

                ii.append(i)
                jj.append(i + step)

                with torch.no_grad():
                    padder = InputPadder(image1.shape)
                    image1, image2 = padder.pad(image1, image2)
                    if np.abs(step) > 1:
                        flow_init = np.stack(
                            [flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0
                        )
                        flow_init = (
                            torch.as_tensor(np.ascontiguousarray(flow_init))
                            .float()
                            .cuda()
                            .permute(0, 3, 1, 2)
                        )
                    else:
                        flow_init = None

                    flow_low, flow_up, _ = self._flow_model(
                        torch.cat([image1, image2], dim=0),
                        torch.cat([image2, image1], dim=0),
                        iters=22,
                        test_mode=True,
                        flow_init=flow_init,
                    )

                    flow_low_fwd = flow_low[0].cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].cpu().numpy().transpose(1, 2, 0)

                    flow_up_fwd = resize_flow(
                        flow_up[0].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2,
                    )
                    flow_up_bwd = resize_flow(
                        flow_up[1].cpu().numpy().transpose(1, 2, 0),
                        flow_up.shape[-2] // 2,
                        flow_up.shape[-1] // 2,
                    )

                    bwd2fwd_flow = warp_flow(flow_up_bwd, flow_up_fwd)
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = fwd_lr_error < 1.0

                    # flows_arr_low.append(flow_low_fwd)
                    flows_arr_low_bwd[i + step] = flow_low_bwd
                    flows_arr_low_fwd[i] = flow_low_fwd

                    # masks_arr_low.append(fwd_mask_low)
                    flows_arr_up.append(flow_up_fwd)
                    masks_arr_up.append(fwd_mask_up)

        iijj = np.stack((ii, jj), axis=0)
        flows_high = np.array(flows_arr_up).transpose(0, 3, 1, 2)
        flow_masks_high = np.array(masks_arr_up)[:, None, ...]

        video_flow_dir = os.path.join(self._flow_root, video_id)
        os.makedirs(video_flow_dir, exist_ok=True)

        np.save(os.path.join(video_flow_dir, "flows.npy"), np.float16(flows_high))
        np.save(os.path.join(video_flow_dir, "flow_masks.npy"), flow_masks_high)
        np.save(os.path.join(video_flow_dir, "ii-jj.npy"), iijj)

    def run_flow_estimation(self, args):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = sorted([int(Path(p).stem) for p in glob.glob(os.path.join(video_frames_path, "*.png"))])
            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            if get_video_belongs_to_split(video_id) == args.split:
                self.video_preprocess_flow(video_id, img_paths)
        print("Flow estimation completed for all videos.")


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
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}."
             "If omitted, processes all videos."
    )

    parser.add_argument(
        '--model', default='/home/rxp190007/CODE/Scene4Cast/datasets/preprocess/cvd_opt/raft-things.pth', help='restore checkpoint'
    )
    parser.add_argument('--small', action='store_true', help='use small model')

    parser.add_argument('--path', help='dataset for evaluation')
    parser.add_argument(
        '--num_heads',
        default=1,
        type=int,
        help='number of heads in attention and aggregation',
    )
    parser.add_argument(
        '--position_only',
        default=False,
        action='store_true',
        help='only use position-wise attention',
    )
    parser.add_argument(
        '--position_and_content',
        default=False,
        action='store_true',
        help='use position and content-wise attention',
    )
    parser.add_argument(
        '--mixed_precision', action='store_true', help='use mixed precision'
    )

    args = parser.parse_args()

    ag_flow = AgFlow(datapath=args.datapath)
    ag_flow.load_flow_estimation_model(args)
    ag_flow.run_flow_estimation(args)


if __name__ == "__main__":
    main()
