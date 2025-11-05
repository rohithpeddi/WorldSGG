import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import joblib
import torch
from tqdm import tqdm

from datasets.preprocess.human.pipeline.ag_pipeline import AgPipeline

sys.path.insert(0, os.path.dirname(__file__) + '/..')
from datasets.preprocess.human.data_config import SMPLX_PATH
from datasets.preprocess.human.prompt_hmr.smpl_family import SMPLX as SMPLX_Layer
from datasets.preprocess.human.prompt_hmr.utils.rotation_conversions import axis_angle_to_matrix
from datasets.preprocess.human.prompt_hmr.vis.traj import get_floor_mesh
from datasets.preprocess.human.prompt_hmr.vis import rerun_vis as rrvis


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


class AgPromptHMR:

    def __init__(
            self,
            output_root,
            root_dir_path
    ):
        self.pipeline = AgPipeline(static_cam=False)
        self.smplx = SMPLX_Layer(SMPLX_PATH).cuda()
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        self.root_dir_path = root_dir_path

    def process_video_intermediate_steps(self, video_id):
        print(f"[{video_id}] Starting processing...")
        video_output_folder = os.path.join(self.output_root, video_id)
        os.makedirs(video_output_folder, exist_ok=True)
        results = self.pipeline.__call__(video_id,
                                         video_output_folder,
                                         save_only_essential=True)

    def process_video(self, video_id):
        smplx = SMPLX_Layer(SMPLX_PATH).cuda()
        results = self.pipeline.__call__(video_id, save_only_essential=False)

        # Downsample for viser visualization
        images = self.pipeline.images
        world4d = self.pipeline.create_world4d()
        world4d = {i: world4d[k] for i, k in enumerate(world4d)}

        # Get vertices
        all_verts = []
        for k in world4d:
            world3d = world4d[k]
            if len(world3d['track_id']) == 0:  # no people
                continue
            rotmat = axis_angle_to_matrix(world3d['pose'].reshape(-1, 55, 3))
            verts = smplx(global_orient=rotmat[:, :1].cuda(),
                          body_pose=rotmat[:, 1:22].cuda(),
                          betas=world3d['shape'].cuda(),
                          transl=world3d['trans'].cuda()).vertices.cpu().numpy()

            world3d['vertices'] = verts
            all_verts.append(torch.tensor(verts, dtype=torch.bfloat16))

        all_verts = torch.cat(all_verts)
        [gv, gf, gc] = get_floor_mesh(all_verts, scale=2)

        rrvis.rerun_vis_world4d(
            video_id=video_id,
            images=images,
            world4d=world4d,
            results=results,
            pipeline=self.pipeline,
            faces=smplx.faces,
            floor=(gv, gf),
            init_fps=10,
            img_maxsize=480,
        )

        print('Rerun visualization running. Please open the Rerun app to view the results.')
        print('Press Ctrl+C to terminate.')
        while True:
            time.sleep(1)

    def infer_all_videos(self, split):
        video_id_list = os.listdir(self.root_dir_path)
        # video_id_list = ["0DJ6R.mp4"]
        for video_id in tqdm(video_id_list, desc=f"Processing videos in split {split}", unit="video"):
            if get_video_belongs_to_split(video_id) != split:
                print(f"Skipping video {video_id} not in split {split}")
                continue
            # self.process_video(video_id)
            try:
                self.process_video_intermediate_steps(video_id)
            except Exception as e:
                print(f"[ERROR] Error processing video {video_id}: {e}")


def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(f"Invalid split '{s}'. Choose one of: {sorted(valid)}")
    return val


def parse_args():
    parser = argparse.ArgumentParser(description="Sample frames from videos based on homography-overlap filtering.")
    parser.add_argument(
        "--output_dir_path", type=str, default="/data/rohith/ag/ag4D/human/",
        help="Path to root dataset directory (must contain 'videos', 'frames', etc.)"
    )
    parser.add_argument(
        "--root_dir_path", type=str, default="/data/rohith/ag/videos",
        help="Path to directory containing input videos."
    )
    parser.add_argument(
        "--split", default="04",
        help="Optional shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}."
             "If omitted, processes all videos."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    processor = AgPromptHMR(output_root=args.output_dir_path, root_dir_path=args.root_dir_path)
    processor.infer_all_videos(split=args.split)


if __name__ == '__main__':
    main()