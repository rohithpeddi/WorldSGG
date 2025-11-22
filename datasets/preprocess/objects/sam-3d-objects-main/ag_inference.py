import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import cv2
import imageio
import pickle

import numpy as np
from IPython.display import Image as ImageDisplay
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.standard.action_genome.ag_dataset import StandardAG
from notebook.inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer

# =====================================================================
# COMMON HELPERS
# =====================================================================
def get_video_belongs_to_split(video_id: str) -> Optional[str]:
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


class AgSam3DInference:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
    ):
        # -------------------- Data Folders --------------------
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)
        self.ag_root_directory = Path(ag_root_directory)

        self.dataset_classnames = [
            '__background__', 'person', 'bag', 'bed', 'blanket', 'book', 'box', 'broom', 'chair',
            'closet/cabinet', 'clothes', 'cup/glass/bottle', 'dish', 'door', 'doorknob', 'doorway',
            'floor', 'food', 'groceries', 'laptop', 'light', 'medicine', 'mirror', 'paper/notebook',
            'phone/camera', 'picture', 'pillow', 'refrigerator', 'sandwich', 'shelf', 'shoe',
            'sofa/couch', 'table', 'television', 'towel', 'vacuum', 'window'
        ]
        self.name_to_catid = {name: idx for idx, name in enumerate(self.dataset_classnames) if idx > 0}
        self.catid_to_name_map = {v: k for k, v in self.name_to_catid.items()}

        self.categories_json: List[Dict[str, Any]] = [
            {"id": cid, "name": name} for name, cid in self.name_to_catid.items()
        ]

        self.dynamic_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.static_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'
        self.frame_annotated_dir_path = self.ag_root_directory / "frames_annotated"
        self.sampled_frames_idx_root_dir = self.ag_root_directory / "sampled_frames_idx"

        self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        os.makedirs(self.bbox_3d_root_dir, exist_ok=True)

        self.sam3d_annotations_dir = self.world_annotations_root_dir / "sam3d_annotations"
        os.makedirs(self.sam3d_annotations_dir, exist_ok=True)

        self.gt_annotations_root_dir = self.ag_root_directory / "gt_annotations"

        # segmentation dirs
        self.dynamic_masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.dynamic_masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.dynamic_masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.dynamic_masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        self.dynamic_masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.dynamic_masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.dynamic_masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.static_masks_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masks' / 'image_based'
        self.static_masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masks' / 'video_based'
        self.static_masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

        # -------------------- Inference Setup --------------------
        PATH = os.getcwd()
        TAG = "hf"
        self.config_path = f"{PATH}/checkpoints/{TAG}/pipeline.yaml"
        self.inference = Inference(self.config_path, compile=False)

    def get_video_gt_annotations(self, video_id):
        video_gt_annotations_json_path = self.gt_annotations_root_dir / video_id / "gt_annotations.json"
        if not video_gt_annotations_json_path.exists():
            raise FileNotFoundError(f"GT annotations file not found: {video_gt_annotations_json_path}")

        with open(video_gt_annotations_json_path, "r") as f:
            video_gt_annotations = json.load(f)

        video_gt_bboxes = {}
        for frame_idx, frame_items in enumerate(video_gt_annotations):
            frame_name = frame_items[0]["frame"].split("/")[-1]
            boxes = []
            labels = []
            for item in frame_items:
                if 'person_bbox' in item:
                    boxes.append(item['person_bbox'][0])
                    labels.append('person')
                    continue
                category_id = item['class']
                category_name = self.catid_to_name_map[category_id]
                if category_name:
                    if category_name == "closet/cabinet":
                        category_name = "closet"
                    elif category_name == "cup/glass/bottle":
                        category_name = "cup"
                    elif category_name == "paper/notebook":
                        category_name = "paper"
                    elif category_name == "sofa/couch":
                        category_name = "sofa"
                    elif category_name == "phone/camera":
                        category_name = "phone"
                    boxes.append(item['bbox'])
                    labels.append(category_name)
            if boxes:
                video_gt_bboxes[frame_name] = {
                    'boxes': boxes,
                    'labels': labels
                }

        return video_gt_bboxes, video_gt_annotations

    def labels_for_frame(self, video_id: str, stem: str, is_static: bool) -> List[str]:
        lbls = set()
        if is_static:
            image_root_dir_list = [self.static_masks_im_dir_path, self.static_masks_vid_dir_path]
        else:
            image_root_dir_list = [self.dynamic_masks_im_dir_path, self.dynamic_masks_vid_dir_path]
        for root in image_root_dir_list:
            vdir = root / video_id
            if not vdir.exists():
                continue
            for fn in os.listdir(vdir):
                if not fn.endswith(".png"):
                    continue
                if "__" in fn:
                    st, lbl = fn.split("__", 1)
                    lbl = lbl.rsplit(".png", 1)[0]
                    if st == stem:
                        lbls.add(lbl)
        return sorted(lbls)

    def get_union_mask(self, video_id: str, stem: str, label: str, is_static) -> Optional[np.ndarray]:
        if is_static:
            im_p = self.static_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.static_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
        else:
            im_p = self.dynamic_masks_im_dir_path / video_id / f"{stem}__{label}.png"
            vd_p = self.dynamic_masks_vid_dir_path / video_id / f"{stem}__{label}.png"
        m_im = cv2.imread(str(im_p), cv2.IMREAD_GRAYSCALE) if im_p.exists() else None
        m_vd = cv2.imread(str(vd_p), cv2.IMREAD_GRAYSCALE) if vd_p.exists() else None
        if m_im is None and m_vd is None:
            return None
        if m_im is None:
            m = (m_vd > 127)
        elif m_vd is None:
            m = (m_im > 127)
        else:
            m = (m_im > 127) | (m_vd > 127)
        return m.astype(bool)

    def update_frame_map(
            self,
            frame_stems,
            video_id,
            frame_map: Dict[str, Dict[str, np.ndarray]],
            is_static
    ):
        all_labels = set()
        for stem in frame_stems:
            lbls = self.labels_for_frame(video_id, stem, is_static)
            if not lbls:
                continue
            all_labels.update(lbls)
            if stem not in frame_map:
                frame_map[stem] = {}
            for lbl in lbls:
                m = self.get_union_mask(video_id, stem, lbl, is_static)
                if m is not None:
                    frame_map[stem][lbl] = m
        return frame_map, all_labels

    def create_label_wise_masks_map(
            self,
            video_id,
            gt_annotations
    ) -> Tuple[Dict[str, Dict[str, Dict[str, np.ndarray]]], set, set]:
        video_to_frame_to_label_mask: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        frame_stems = []
        for frame_items in gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
            stem = Path(frame_name).stem
            frame_stems.append(stem)

        frame_map: Dict[str, Dict[str, np.ndarray]] = {}
        frame_map, all_static_labels = self.update_frame_map(
            frame_stems=frame_stems,
            video_id=video_id,
            frame_map=frame_map,
            is_static=True
        )
        frame_map, all_dynamic_labels = self.update_frame_map(
            frame_stems=frame_stems,
            video_id=video_id,
            frame_map=frame_map,
            is_static=False
        )
        if frame_map:
            video_to_frame_to_label_mask[video_id] = frame_map

        return video_to_frame_to_label_mask, all_static_labels, all_dynamic_labels

    def process_frame(
            self,
            frame_name: str,
            image_path: str,
            masks: List[np.ndarray],
            video_output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Run SAM3D on a single frame, save all outputs for that frame, and return basic metadata.

        Saved artifacts (inside video_output_dir / frame_name):
        - {frame_name}.gif                : rendered SAM3D turntable video
        - {frame_name}_sam3d.pkl          : dict with image_path, masks, raw outputs, scene_gaussians, video_path
        """

        # ------------------------------------------------------------------
        # 0) Prepare per-frame directory
        # ------------------------------------------------------------------
        frame_output_dir = video_output_dir / frame_name
        gaussians_dir = frame_output_dir / "gaussians"

        os.makedirs(frame_output_dir, exist_ok=True)
        os.makedirs(gaussians_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Load image & run inference per mask
        # ------------------------------------------------------------------
        image = load_image(image_path)
        display_image(image, masks)

        # Run SAM3D for each mask and collect outputs
        outputs = []
        for mask in masks:
            out = self.inference(image, mask, seed=42)
            outputs.append(out)

        # ------------------------------------------------------------------
        # 2) Build scene gaussians and render video
        # ------------------------------------------------------------------
        scene_gs = make_scene(*outputs)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        video = render_video(scene_gs, r=1, fov=60, resolution=512)["color"]

        # ------------------------------------------------------------------
        # 3) Save rendered video (GIF)
        # ------------------------------------------------------------------
        video_frame_file_path = frame_output_dir / f"{frame_name}.gif"
        imageio.mimsave(
            video_frame_file_path,
            video,
            format="GIF",
            duration=1000 / 30,  # assume 30fps
            loop=0,  # 0 means loop indefinitely
        )

        # ------------------------------------------------------------------
        # 4) Save all outputs + gaussians in a single pickle
        # ------------------------------------------------------------------
        pkl_output_path = frame_output_dir / f"{frame_name}_sam3d.pkl"

        frame_dump: Dict[str, Any] = {
            "frame_name": frame_name,
            "image_path": str(image_path),
            "masks": masks,
            "outputs": outputs,  # list of per-mask SAM3D outputs
            "scene_gaussians": scene_gs,  # combined gaussian scene
        }

        with open(pkl_output_path, "wb") as f:
            pickle.dump(frame_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

        # (Optional) if you want to also save gaussians separately per frame, you
        # could add something like:
        # gaussians_pkl_path = gaussians_dir / f"{frame_name}_scene_gaussians.pkl"
        # with open(gaussians_pkl_path, "wb") as f:
        #     pickle.dump(scene_gs, f, protocol=pickle.HIGHEST_PROTOCOL)

        # ------------------------------------------------------------------
        # 5) Return basic metadata (optional)
        # ------------------------------------------------------------------
        return {
            "frame_name": frame_name,
            "frame_output_dir": frame_output_dir,
            "pickle_path": pkl_output_path,
            "gif_path": video_frame_file_path,
        }

    def get_all_labels_in_video(self, video_gt_annotations):
        all_labels = set()
        for frame_items in video_gt_annotations:
            for item in frame_items:
                category_id = item['class']
                category_name = self.catid_to_name_map[category_id]
                if category_name == "closet/cabinet":
                    category_name = "closet"
                elif category_name == "cup/glass/bottle":
                    category_name = "cup"
                elif category_name == "paper/notebook":
                    category_name = "paper"
                elif category_name == "sofa/couch":
                    category_name = "sofa"
                elif category_name == "phone/camera":
                    category_name = "phone"
                all_labels.add(category_name)
        return all_labels

    def select_frames_for_video(self, video_id, video_gt_annotations, video_to_frame_to_label_mask, all_labels):
        # TODO:
        # 1. We first construct a frame_id, label, bbox area_map
        # 2. We select frames ranked such that - we minimize the number of frames while maximizing label coverage
        # a. We pick frames with maximum labels present and in them rank them with maximum area coverage.
        # b. We repeat until all labels are covered.
        selected_frames = []

        return selected_frames

    def process_video(self, video_id):
        video_gt_annotations = self.get_video_gt_annotations(video_id)[1]
        video_to_frame_to_label_mask, _, _ = self.create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_gt_annotations
        )

        # 1. Frame selection logic
        selected_frames = []
        all_labels = self.get_all_labels_in_video(video_gt_annotations)
        selected_frames = self.select_frames_for_video(
            video_id,
            video_gt_annotations,
            video_to_frame_to_label_mask,
            all_labels
        )

        # 2. Mask compilation per frame
        frame_masks = {}
        for frame_items in video_gt_annotations:
            frame_name = frame_items[0]["frame"].split("/")[-1]
            if frame_name not in selected_frames:
                continue
            stem = Path(frame_name).stem
            if video_id not in video_to_frame_to_label_mask:
                continue
            if stem not in video_to_frame_to_label_mask[video_id]:
                continue
            label_to_mask_map = video_to_frame_to_label_mask[video_id][stem]
            masks = []
            for item in frame_items:
                category_id = item['class']
                category_name = self.catid_to_name_map[category_id]
                if category_name == "closet/cabinet":
                    category_name = "closet"
                elif category_name == "cup/glass/bottle":
                    category_name = "cup"
                elif category_name == "paper/notebook":
                    category_name = "paper"
                elif category_name == "sofa/couch":
                    category_name = "sofa"
                elif category_name == "phone/camera":
                    category_name = "phone"
                if category_name in label_to_mask_map:
                    masks.append(label_to_mask_map[category_name])
            if masks:
                frame_masks[frame_name] = masks

        # 3. Processing the selected frames and saving outputs
        video_output_dir = self.sam3d_annotations_dir / video_id / "sam3d_outputs"
        os.makedirs(video_output_dir, exist_ok=True)

    def generate_sam3d_annotations(self, dataloader, split) -> None:
        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                out_path = self.bbox_3d_root_dir / f"{video_id[:-4]}.pkl"
                if out_path.exists():
                    print(f"[bbox] floor-aligned 3D bboxes already exist for video {video_id}, skipping...")
                    continue
                try:
                    print(f"[bbox] processing video {video_id}...")
                    self.process_video(video_id)
                except Exception as e:
                    print(f"[bbox] failed to process video {video_id}: {e}")
            else:
                print(f"[bbox] video {video_id} does not belong to split {split}, skipping...")

    def generate_sample_sam3d_annotations(self, video_id: str) -> None:
        self.process_video(video_id)

def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False
    )

    dataloader_train = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0
    )

    dataloader_test = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False
    )

    return train_dataset, test_dataset, dataloader_train, dataloader_test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combined: (a) floor-aligned 3D bbox generator + (b) SMPL↔PI3 human mesh aligner (sampled frames only)."
    )
    parser.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    parser.add_argument("--dynamic_scene_dir_path", type=str,
                        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic")
    parser.add_argument("--output_human_dir_path", type=str, default="/data/rohith/ag/ag4D/human/")
    parser.add_argument("--split", type=str, default="04")
    parser.add_argument("--include_dense", action="store_true",
                        help="use dense correspondences for human aligner")
    return parser.parse_args()

def main():
    args = parse_args()
    bbox_3d_generator = AgSam3DInference(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    bbox_3d_generator.generate_sam3d_annotations(dataloader=dataloader_train, split=args.split)
    bbox_3d_generator.generate_sam3d_annotations(dataloader=dataloader_test, split=args.split)

def main_sample():
    args = parse_args()

    bbox_3d_generator = AgSam3DInference(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory
    )
    video_id = "0DJ6R.mp4"
    bbox_3d_generator.generate_sample_sam3d_annotations(video_id=video_id)


if __name__ == "__main__":
    main_sample()
    # main()
