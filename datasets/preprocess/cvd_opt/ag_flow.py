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
        # 1) Load & normalize frames (C,H,W) with consistent H,W
        img_data = []
        for t, image_file in tqdm(enumerate(image_list), total=len(image_list)):
            image = cv2.imread(image_file)[..., ::-1]  # BGR->RGB
            h0, w0, _ = image.shape
            # keep ~constant area ~ (384*512)
            scale = np.sqrt((384 * 512) / float(h0 * w0))
            h1 = int(h0 * scale)
            w1 = int(w0 * scale)
            image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LINEAR)
            # make divisible by 8
            h1 = h1 - (h1 % 8)
            w1 = w1 - (w1 % 8)
            image = image[:h1, :w1].transpose(2, 0, 1)  # (C,H,W)
            img_data.append(image)

        img_data = np.array(img_data)  # (T, 3, H_ref, W_ref) if consistent
        if img_data.ndim != 4:
            raise RuntimeError(
                f"Inconsistent frame sizes; got array with ndim={img_data.ndim}. "
                f"Ensure all frames resize to a single (H,W)."
            )
        T, C, H_ref, W_ref = img_data.shape

        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}

        ii = []
        jj = []
        flows_arr_up = []
        masks_arr_up = []

        for step in [1, 2, 4, 8, 15]:
            # skip steps longer than available frames
            if T - step <= 0:
                continue

            for i in tqdm(range(0, T - step), leave=False):
                image1 = torch.as_tensor(np.ascontiguousarray(img_data[i:i + 1])).float().cuda()
                image2 = torch.as_tensor(np.ascontiguousarray(img_data[i + step:i + step + 1])).float().cuda()

                ii.append(i)
                jj.append(i + step)

                with torch.no_grad():
                    padder = InputPadder(image1.shape)  # pads to multiples of 8
                    image1p, image2p = padder.pad(image1, image2)

                    # Optional RAFT warm-start
                    if step > 1 and (i in flows_arr_low_fwd) and ((i + step) in flows_arr_low_bwd):
                        flow_init = np.stack([flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]],
                                             axis=0)  # (2, H, W, 2)
                        flow_init = (
                            torch.as_tensor(np.ascontiguousarray(flow_init))
                            .float().cuda().permute(0, 3, 1, 2)  # (2, 2, H, W)
                        )
                    else:
                        flow_init = None

                    flow_low, flow_up, _ = self._flow_model(
                        torch.cat([image1p, image2p], dim=0),
                        torch.cat([image2p, image1p], dim=0),
                        iters=22,
                        test_mode=True,
                        flow_init=flow_init,
                    )
                    # flow_*: (2, 2, Hp, Wp) where dim0 is fwd/bwd, dim1 is xy

                    # Convert to (H,W,2)
                    flow_low_fwd = flow_low[0].detach().cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].detach().cpu().numpy().transpose(1, 2, 0)

                    flow_up_fwd = flow_up[0].detach().cpu().numpy().transpose(1, 2, 0)
                    flow_up_bwd = flow_up[1].detach().cpu().numpy().transpose(1, 2, 0)

                    # --- Normalize all flows to a single reference size (H_ref, W_ref)
                    if (flow_up_fwd.shape[0] != H_ref) or (flow_up_fwd.shape[1] != W_ref):
                        flow_up_fwd = resize_flow(flow_up_fwd, H_ref, W_ref)
                    if (flow_up_bwd.shape[0] != H_ref) or (flow_up_bwd.shape[1] != W_ref):
                        flow_up_bwd = resize_flow(flow_up_bwd, H_ref, W_ref)

                    # Consistent low-resolution cache for warm-starting RAFT across steps
                    flows_arr_low_fwd[i] = flow_low_fwd
                    flows_arr_low_bwd[i + step] = flow_low_bwd

                    # --- Mask from forward-backward consistency at reference size
                    bwd2fwd_flow = warp_flow(flow_up_bwd.astype(np.float32), flow_up_fwd.astype(np.float32))
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = (fwd_lr_error < 1.0)  # (H_ref, W_ref), bool

                    flows_arr_up.append(flow_up_fwd.astype(np.float32))  # (H_ref, W_ref, 2)
                    masks_arr_up.append(fwd_mask_up.astype(np.bool_))  # (H_ref, W_ref)

        # Stack safely
        if len(flows_arr_up) == 0:
            raise RuntimeError("No flows were computed (check number of frames vs steps).")

        try:
            flows_high = np.stack(flows_arr_up, axis=0).transpose(0, 3, 1, 2)  # (N, 2, H, W)
        except Exception as e:
            # Helpful debug if something ever goes ragged again
            shapes = [a.shape for a in flows_arr_up]
            raise RuntimeError(f"Ragged flow list; shapes seen: {shapes}") from e

        flow_masks_high = np.stack(masks_arr_up, axis=0)[:, None, ...]  # (N, 1, H, W)
        iijj = np.stack((np.array(ii, dtype=np.int32), np.array(jj, dtype=np.int32)), axis=0)

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
