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

    def __init__(self, datapath,
                 viz_stride: int = 16,
                 viz_scale: float = 1.0,
                 viz_alpha: float = 0.55,
                 viz_use_mask: bool = True,
                 viz_backend: str = "raft",
                 viz_side_by_side: bool = True,
                 viz_max_pairs: int = None):
        self.datapath = datapath
        self.frames_path = os.path.join(self.datapath, "sampled_frames")
        self.video_list = sorted(os.listdir(self.frames_path))

        self._flow_root = os.path.join(self.datapath, 'ag4D', "flow", "raft")
        os.makedirs(self._flow_root, exist_ok=True)
        self._flow_vis_root = os.path.join(self.datapath, 'ag4D', "flow_vis", viz_backend)
        os.makedirs(self._flow_vis_root, exist_ok=True)

        self._flow_estimation_model = None
        self._flow_model = None

        # viz settings
        self.viz_stride = viz_stride
        self.viz_scale = viz_scale
        self.viz_alpha = viz_alpha
        self.viz_use_mask = viz_use_mask
        self.viz_backend = viz_backend
        self.viz_side_by_side = viz_side_by_side
        self.viz_max_pairs = viz_max_pairs

    def load_flow_estimation_model(self, args):
        self._flow_estimation_model = torch.nn.DataParallel(RAFT(args))
        self._flow_estimation_model.load_state_dict(torch.load(args.model))
        print(f'Loaded checkpoint at {args.model}')
        self._flow_model = self._flow_estimation_model.module
        self._flow_model.cuda()
        self._flow_model.eval()

    # ---------- Visualization helpers (embedded) ----------
    @staticmethod
    def _flow_to_color(flow: np.ndarray, clip_mag: float = None) -> np.ndarray:
        """flow (H,W,2)-> BGR color image using HSV wheel."""
        fx, fy = flow[..., 0], flow[..., 1]
        mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)
        if clip_mag is None:
            clip_mag = np.percentile(mag, 95) + 1e-6
        mag = np.clip(mag / (clip_mag + 1e-6), 0.0, 1.0)
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.float32)
        hsv[..., 0] = ang / 2.0
        hsv[..., 1] = 1.0
        hsv[..., 2] = mag
        hsv_8u = np.uint8(hsv * 255.0)
        return cv2.cvtColor(hsv_8u, cv2.COLOR_HSV2BGR)

    @staticmethod
    def _overlay_color_flow(image_bgr: np.ndarray, flow_bgr: np.ndarray, alpha: float = 0.55) -> np.ndarray:
        img = image_bgr if image_bgr.dtype == np.uint8 else np.clip(image_bgr, 0, 255).astype(np.uint8)
        flow_img = flow_bgr if flow_bgr.dtype == np.uint8 else np.clip(flow_bgr, 0, 255).astype(np.uint8)
        return cv2.addWeighted(img, 1.0 - alpha, flow_img, alpha, 0.0)

    @staticmethod
    def _draw_quiver_on_image(image_bgr: np.ndarray,
                              flow: np.ndarray,
                              mask: np.ndarray = None,
                              step: int = 16,
                              scale: float = 1.0,
                              thickness: int = 1,
                              tip_length: float = 0.3) -> np.ndarray:
        """Draw sparse arrowed flow vectors on image."""
        H, W = flow.shape[:2]
        out = image_bgr.copy()
        y_coords = range(step // 2, H, step)
        x_coords = range(step // 2, W, step)
        for y in y_coords:
            for x in x_coords:
                if mask is not None and not mask[y, x]:
                    continue
                dx, dy = float(flow[y, x, 0]), float(flow[y, x, 1])
                x2 = int(round(x + dx * scale))
                y2 = int(round(y + dy * scale))
                cv2.arrowedLine(out, (x, y), (x2, y2), (0, 0, 255), thickness=thickness, tipLength=tip_length)
        return out

    def _save_visualizations(self,
                             video_id: str,
                             image_list: list,
                             flows_high: np.ndarray,  # (N,2,H,W)
                             flow_masks_high: np.ndarray,  # (N,1,H,W) or None
                             iijj: np.ndarray):  # (2,N)
        out_dir = os.path.join(self._flow_vis_root, video_id)
        os.makedirs(out_dir, exist_ok=True)

        N, _, H, W = flows_high.shape
        total = N if self.viz_max_pairs is None else min(N, self.viz_max_pairs)

        # Helper to get nice stem from path (falls back to index if not numeric)
        def _stem6(path_str: str, idx: int) -> str:
            s = Path(path_str).stem
            if s.isdigit():
                return f"{int(s):06d}"
            return f"{idx:06d}"

        for k in tqdm(range(total), desc=f"[viz] {video_id}", leave=False):
            i, j = int(iijj[0, k]), int(iijj[1, k])
            if i < 0 or j < 0 or i >= len(image_list) or j >= len(image_list):
                continue

            img_i = cv2.imread(str(image_list[i]))
            img_j = cv2.imread(str(image_list[j]))
            if img_i is None or img_j is None:
                continue
            img_i = cv2.resize(img_i, (W, H), interpolation=cv2.INTER_LINEAR)
            img_j = cv2.resize(img_j, (W, H), interpolation=cv2.INTER_LINEAR)

            flow = flows_high[k].astype(np.float32).transpose(1, 2, 0)  # (H,W,2)
            mask = flow_masks_high[k, 0].astype(bool) if (flow_masks_high is not None and self.viz_use_mask) else None

            flow_bgr = self._flow_to_color(flow)
            color_overlay = self._overlay_color_flow(img_i, flow_bgr, alpha=self.viz_alpha)
            quiver_overlay = self._draw_quiver_on_image(
                img_i, flow, mask=mask, step=self.viz_stride, scale=self.viz_scale, thickness=1, tip_length=0.3
            )

            stem = f"{_stem6(image_list[i], i)}_{_stem6(image_list[j], j)}"
            cv2.imwrite(os.path.join(out_dir, f"{stem}_color.png"), color_overlay)
            cv2.imwrite(os.path.join(out_dir, f"{stem}_quiver.png"), quiver_overlay)

            if self.viz_side_by_side:
                side_by_side = cv2.hconcat([img_i, img_j, color_overlay, quiver_overlay])
                cv2.imwrite(os.path.join(out_dir, f"{stem}_all.png"), side_by_side)

    # ---------- Flow processing with embedded viz ----------
    def video_preprocess_flow(self, video_id, image_list):
        # 1) Load & normalize frames (C,H,W) with consistent H,W
        img_data = []
        for t, image_file in tqdm(enumerate(image_list), total=len(image_list), desc=f"[load] {video_id}", leave=False):
            image = cv2.imread(image_file)[..., ::-1]  # BGR->RGB
            h0, w0, _ = image.shape
            scale = np.sqrt((384 * 512) / float(h0 * w0))
            h1 = int(h0 * scale)
            w1 = int(w0 * scale)
            image = cv2.resize(image, (w1, h1), interpolation=cv2.INTER_LINEAR)
            h1 = h1 - (h1 % 8)
            w1 = w1 - (w1 % 8)
            image = image[:h1, :w1].transpose(2, 0, 1)  # (C,H,W)
            img_data.append(image)

        img_data = np.array(img_data)  # (T, 3, H_ref, W_ref)
        if img_data.ndim != 4:
            raise RuntimeError("Inconsistent frame sizes after resize.")
        T, C, H_ref, W_ref = img_data.shape

        flows_arr_low_bwd = {}
        flows_arr_low_fwd = {}
        ii, jj = [], []
        flows_arr_up, masks_arr_up = [], []

        for step in [1, 2, 4, 8, 15]:
            if T - step <= 0:
                continue
            for i in tqdm(range(0, T - step), desc=f"[flow step={step}] {video_id}", leave=False):
                image1 = torch.as_tensor(np.ascontiguousarray(img_data[i:i + 1])).float().cuda()
                image2 = torch.as_tensor(np.ascontiguousarray(img_data[i + step:i + step + 1])).float().cuda()

                ii.append(i);
                jj.append(i + step)

                with torch.no_grad():
                    padder = InputPadder(image1.shape)
                    image1p, image2p = padder.pad(image1, image2)

                    if step > 1 and (i in flows_arr_low_fwd) and ((i + step) in flows_arr_low_bwd):
                        flow_init = np.stack([flows_arr_low_fwd[i], flows_arr_low_bwd[i + step]], axis=0)  # (2,H,W,2)
                        flow_init = torch.as_tensor(np.ascontiguousarray(flow_init)).float().cuda().permute(0, 3, 1, 2)
                    else:
                        flow_init = None

                    flow_low, flow_up, _ = self._flow_model(
                        torch.cat([image1p, image2p], dim=0),
                        torch.cat([image2p, image1p], dim=0),
                        iters=22, test_mode=True, flow_init=flow_init,
                    )

                    flow_low_fwd = flow_low[0].detach().cpu().numpy().transpose(1, 2, 0)
                    flow_low_bwd = flow_low[1].detach().cpu().numpy().transpose(1, 2, 0)

                    flow_up_fwd = flow_up[0].detach().cpu().numpy().transpose(1, 2, 0)
                    flow_up_bwd = flow_up[1].detach().cpu().numpy().transpose(1, 2, 0)

                    # Ensure consistent size (H_ref, W_ref)
                    if (flow_up_fwd.shape[0] != H_ref) or (flow_up_fwd.shape[1] != W_ref):
                        flow_up_fwd = resize_flow(flow_up_fwd, H_ref, W_ref)
                    if (flow_up_bwd.shape[0] != H_ref) or (flow_up_bwd.shape[1] != W_ref):
                        flow_up_bwd = resize_flow(flow_up_bwd, H_ref, W_ref)

                    flows_arr_low_fwd[i] = flow_low_fwd
                    flows_arr_low_bwd[i + step] = flow_low_bwd

                    # Consistency mask at reference size
                    bwd2fwd_flow = warp_flow(flow_up_bwd.astype(np.float32), flow_up_fwd.astype(np.float32))
                    fwd_lr_error = np.linalg.norm(flow_up_fwd + bwd2fwd_flow, axis=-1)
                    fwd_mask_up = (fwd_lr_error < 1.0)

                    flows_arr_up.append(flow_up_fwd.astype(np.float32))  # (H_ref,W_ref,2)
                    masks_arr_up.append(fwd_mask_up.astype(np.bool_))  # (H_ref,W_ref)

        if len(flows_arr_up) == 0:
            print(f"[warn] No flows for {video_id}")
            return

        # Pack -> (N,2,H,W) and (N,1,H,W)
        try:
            flows_high = np.stack(flows_arr_up, axis=0).transpose(0, 3, 1, 2)
        except Exception as e:
            shapes = [a.shape for a in flows_arr_up]
            raise RuntimeError(f"Ragged flows for {video_id}; shapes: {shapes}") from e

        flow_masks_high = np.stack(masks_arr_up, axis=0)[:, None, ...]
        iijj = np.stack((np.array(ii, dtype=np.int32), np.array(jj, dtype=np.int32)), axis=0)

        # 2) Save arrays
        video_flow_dir = os.path.join(self._flow_root, video_id)
        os.makedirs(video_flow_dir, exist_ok=True)
        np.save(os.path.join(video_flow_dir, "flows.npy"), np.float16(flows_high))
        np.save(os.path.join(video_flow_dir, "flow_masks.npy"), flow_masks_high)
        np.save(os.path.join(video_flow_dir, "ii-jj.npy"), iijj)

        # 3) Save visualizations alongside (uses in-memory tensors to avoid re-loading)
        self._save_visualizations(video_id, image_list, flows_high, flow_masks_high, iijj)

    def run_flow_estimation(self, args):
        for video_id in tqdm(self.video_list):
            video_frames_path = os.path.join(self.frames_path, video_id)
            img_paths = []
            frame_id_list = sorted([int(Path(p).stem) for p in glob.glob(os.path.join(video_frames_path, "*.png"))])

            if len(frame_id_list) == 0:
                print(f"[warn] No frames found for video {video_id}, skipping.")
                continue

            for frame_id in frame_id_list:
                img_path = os.path.join(video_frames_path, f"{frame_id:06d}.png")
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                else:
                    assert False, f"Image {img_path} does not exist."

            if get_video_belongs_to_split(video_id) == args.split:
                # Check the flow output already exists
                flow_output_path = os.path.join(self._flow_root, video_id, "flows.npy")
                if os.path.exists(flow_output_path):
                    print(f"[skip] Flow already exists for video {video_id}, skipping.")
                    continue
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
        '--model', default='/home/rxp190007/CODE/Scene4Cast/datasets/preprocess/cvd_opt/raft-things.pth',
        help='restore checkpoint'
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
