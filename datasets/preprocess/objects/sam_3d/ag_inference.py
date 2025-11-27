import argparse
import gzip
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
from datasets.preprocess.objects.sam_3d.notebook.inference import Inference, ready_gaussian_for_video_rendering, load_image, load_masks, display_image, make_scene, render_video, interactive_visualizer

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
            output_directory: Optional[str] = None,
    ):
        # -------------------- Data Folders --------------------
        self.inference = None
        self.config_path = None
        self.dynamic_scene_dir_path = Path(dynamic_scene_dir_path)
        self.ag_root_directory = Path(ag_root_directory)
        self.output_directory = Path(output_directory)

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

        # self.world_annotations_root_dir = self.ag_root_directory / "world_annotations"
        # self.bbox_3d_root_dir = self.world_annotations_root_dir / "bbox_annotations_3d"
        # os.makedirs(self.bbox_3d_root_dir, exist_ok=True)
        self.world_annotations_root_dir = self.output_directory

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
        self.setup_inference()

    def setup_inference(self):
        self.config_path = "/home/rxp190007/CODE/Scene4Cast/datasets/preprocess/objects/sam_3d/checkpoints/hf/pipeline.yaml"
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
            save_gif: bool = False,
            compress_pickle: bool = True,
    ) -> Dict[str, Any]:
        """
        Run SAM3D on a single frame, save all outputs for that frame, and return basic metadata.

        Saved artifacts (inside video_output_dir / frame_name):

        - {frame_name}_sam3d.pkl.gz      : gzip-compressed pickle with:
            {
                "frame_name": frame_name,
                "outputs": outputs,          # list of per-mask SAM3D outputs
                "scene_gaussians": scene_gs, # combined gaussian scene
            }

        Optionally (if save_gif=True):
        - {frame_name}.gif               : rendered SAM3D turntable video
        """

        # ------------------------------------------------------------------
        # 0) Prepare per-frame directory
        # ------------------------------------------------------------------
        frame_output_dir = video_output_dir / frame_name
        gaussians_dir = frame_output_dir / "gaussians"  # kept for future use if needed

        os.makedirs(frame_output_dir, exist_ok=True)
        os.makedirs(gaussians_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1) Load image & run inference per mask
        # ------------------------------------------------------------------
        image = load_image(image_path)

        outputs = []
        for mask in masks:
            out = self.inference(image, mask, seed=42)
            outputs.append(out)

        # ------------------------------------------------------------------
        # 2) Build scene gaussians
        # ------------------------------------------------------------------
        scene_gs = make_scene(*outputs)
        scene_gs = ready_gaussian_for_video_rendering(scene_gs)

        # ------------------------------------------------------------------
        # 3) (Optional) render GIF video
        # ------------------------------------------------------------------
        video_frame_file_path = None
        if save_gif:
            video = render_video(scene_gs, r=1, fov=60, resolution=512)["color"]
            video_frame_file_path = frame_output_dir / f"{frame_name}.gif"
            imageio.mimsave(
                video_frame_file_path,
                video,
                format="GIF",
                duration=1000 / 30,  # assume 30fps
                loop=0,  # 0 means loop indefinitely
            )

        # ------------------------------------------------------------------
        # 4) Save compressed pickle
        # ------------------------------------------------------------------
        frame_dump: Dict[str, Any] = {
            "frame_name": frame_name,
            "outputs": outputs,
            "scene_gaussians": scene_gs,
        }

        if compress_pickle:
            pkl_output_path = frame_output_dir / f"{frame_name}_sam3d.pkl.gz"
            with gzip.open(pkl_output_path, "wb") as f:
                pickle.dump(frame_dump, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pkl_output_path = frame_output_dir / f"{frame_name}_sam3d.pkl"
            with open(pkl_output_path, "wb") as f:
                pickle.dump(frame_dump, f, protocol=pickle.HIGHEST_PROTOCOL)

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
                if "class" not in item:
                    category_name = "person"
                else:
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

    def select_all_frames_for_video(self, video_id, video_gt_annotations):
        selected_frames = []
        for frame_items in video_gt_annotations:
            if not frame_items:
                continue
            frame_name = frame_items[0]["frame"].split("/")[-1]
            selected_frames.append(frame_name)
        return selected_frames

    def select_frames_for_video(self, video_id, video_gt_annotations, all_labels):
        """
        Greedy frame subsampling using *bbox area only* (no masks).
        Strategy:
        1. For each frame, compute:
           - which labels are present (using GT annotations)
           - total bbox area per label (sum of all bboxes of that label in the frame)
        2. Greedy set-cover:
           - At each step, pick the frame that covers the largest number of *new* labels.
           - Break ties by choosing the frame with the largest total bbox area for those new labels.
        3. Stop when no frame can cover any new labels.
        Args:
            video_id: str, e.g. "0DJ6R.mp4"
            video_gt_annotations: list of per-frame annotation lists (raw JSON)
            all_labels: set of label strings we want to cover
        Returns:
            selected_frames: list[str] of frame file names (e.g. "0DJ6R_000123.jpg")
        """

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------
        def _canonical_label_from_category_id(category_id: int) -> str:
            """Match the remapping used everywhere else (closet/cabinet → closet, etc.)."""
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
            return category_name

        def _xywh_to_xyxy(b):  # [x,y,w,h] -> [x1,y1,x2,y2]
            x, y, w, h = [float(v) for v in b]
            return [x, y, x + w, y + h]

        def _bbox_area(box) -> float:
            """
            Compute area of a single bbox.
            Assumes [x1, y1, x2, y2] format (adjust if your GT uses something else).
            """
            if box is None or len(box) != 4:
                return 0.0
            x1, y1, x2, y2 = box
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            return w * h

        # ------------------------------------------------------------------
        # 1) Build per-frame label and bbox-area maps
        # ------------------------------------------------------------------
        frame_labels: Dict[str, set] = {}
        frame_label_area: Dict[str, Dict[str, float]] = {}
        frame_order: Dict[str, int] = {}  # to optionally sort frames chronologically

        for frame_idx, frame_items in enumerate(video_gt_annotations):
            if not frame_items:
                continue

            frame_name = frame_items[0]["frame"].split("/")[-1]
            frame_order[frame_name] = frame_idx

            labels_here = set()
            label_area_map: Dict[str, float] = {}

            for item in frame_items:
                # 1) Person bbox(es)
                if "person_bbox" in item:
                    label = "person"
                    if all_labels and label not in all_labels:
                        continue

                    pb = item["person_bbox"]
                    # pb can be a single bbox or list of bboxes
                    if isinstance(pb, (list, tuple)) and pb and isinstance(pb[0], (int, float)):
                        person_bboxes = [pb]
                    else:
                        person_bboxes = pb

                    total_area = 0.0
                    if person_bboxes is not None:
                        for b in person_bboxes:
                            xyxy_b = _xywh_to_xyxy(b)
                            total_area += _bbox_area(xyxy_b)

                    if total_area > 0.0:
                        labels_here.add(label)
                        label_area_map[label] = label_area_map.get(label, 0.0) + total_area
                    continue

                # 2) Object bbox
                if "class" not in item or "bbox" not in item:
                    continue

                category_id = item["class"]
                label = _canonical_label_from_category_id(category_id)
                if all_labels and label not in all_labels:
                    continue

                bbox = item["bbox"]
                # If bbox could be a list of multiple boxes, handle that too
                if isinstance(bbox, (list, tuple)) and bbox and isinstance(bbox[0], (int, float)):
                    bboxes = [bbox]
                else:
                    bboxes = bbox

                total_area = 0.0
                if bboxes is not None:
                    if isinstance(bboxes, (list, tuple)) and bboxes and isinstance(bboxes[0], (list, tuple)):
                        # list of boxes
                        for b in bboxes:
                            total_area += _bbox_area(b)
                    else:
                        # single box
                        total_area += _bbox_area(bboxes)

                if total_area > 0.0:
                    labels_here.add(label)
                    label_area_map[label] = label_area_map.get(label, 0.0) + total_area

            # If nothing useful in this frame, skip it
            if not labels_here:
                continue

            frame_labels[frame_name] = labels_here
            frame_label_area[frame_name] = label_area_map

        if not frame_labels:
            print(f"[sam3d] No frames with usable bboxes for video {video_id}.")
            return []

        # ------------------------------------------------------------------
        # 2) Universe of labels we can actually cover from bboxes
        # ------------------------------------------------------------------
        labels_universe = set()
        for lbls in frame_labels.values():
            labels_universe.update(lbls)

        if all_labels:
            labels_to_cover = labels_universe.intersection(all_labels)
        else:
            labels_to_cover = set(labels_universe)

        if not labels_to_cover:
            print(f"[sam3d] No overlap between GT labels and bbox labels for video {video_id}.")
            return []

        # ------------------------------------------------------------------
        # 3) Greedy set-cover using bbox area
        # ------------------------------------------------------------------
        selected_frames: List[str] = []
        remaining_labels = set(labels_to_cover)

        while remaining_labels:
            best_frame = None
            best_new_labels_count = 0
            best_new_labels_area = 0.0

            for frame_name, labels_here in frame_labels.items():
                if frame_name in selected_frames:
                    continue

                new_labels = labels_here & remaining_labels
                if not new_labels:
                    continue

                # Total bbox area for just the new labels
                new_area = sum(
                    frame_label_area[frame_name].get(lbl, 0.0) for lbl in new_labels
                )

                # Primary key: number of new labels
                # Secondary key: total area of those new labels
                if (
                    len(new_labels) > best_new_labels_count
                    or (
                        len(new_labels) == best_new_labels_count
                        and new_area > best_new_labels_area
                    )
                ):
                    best_frame = frame_name
                    best_new_labels_count = len(new_labels)
                    best_new_labels_area = new_area

            if best_frame is None:
                # No frame can cover any remaining labels
                break

            selected_frames.append(best_frame)
            remaining_labels -= frame_labels[best_frame]

        selected_frames_sorted = sorted(selected_frames, key=lambda fn: frame_order.get(fn, 0))
        return selected_frames_sorted

    def process_video(self, video_id: str, save_gif: bool = True, compress_pickle: bool = True):
        # ------------------------------------------------------------------
        # 1) Load GT annotations + segmentation masks
        # ------------------------------------------------------------------
        video_gt_annotations = self.get_video_gt_annotations(video_id)[1]
        video_to_frame_to_label_mask, _, _ = self.create_label_wise_masks_map(
            video_id=video_id,
            gt_annotations=video_gt_annotations,
        )

        # ------------------------------------------------------------------
        # 2) Frame selection
        #   - selected_frames: frames we run SAM3D on
        #   - gif_frames: subset of frames we render GIFs for
        # ------------------------------------------------------------------
        all_labels = self.get_all_labels_in_video(video_gt_annotations)
        gif_frames = self.select_frames_for_video(
            video_id,
            video_gt_annotations,
            all_labels,
        )
        selected_frames = self.select_all_frames_for_video(video_id, video_gt_annotations)

        # NEW: use stems for comparison, since process_frame uses stem names
        gif_frame_stems = {Path(f).stem for f in gif_frames}

        if not selected_frames:
            print(f"[sam3d] No selected frames for video {video_id}")
            return

        # ------------------------------------------------------------------
        # 3) Build masks for each selected frame
        # ------------------------------------------------------------------
        frame_masks: Dict[str, List[np.ndarray]] = {}

        for frame_items in video_gt_annotations:
            if not frame_items:
                continue

            frame_name = frame_items[0]["frame"].split("/")[-1]  # e.g. 0DJ6R_000123.jpg
            if frame_name not in selected_frames:
                continue

            stem = Path(frame_name).stem  # e.g. 0DJ6R_000123

            if (
                video_id not in video_to_frame_to_label_mask
                or stem not in video_to_frame_to_label_mask[video_id]
            ):
                continue

            # --- segmentation masks: raw_label -> bool mask ---
            raw_label_to_mask: Dict[str, np.ndarray] = video_to_frame_to_label_mask[video_id][stem]

            # Canonicalise segmentation labels to match GT label remapping
            # (closet/cabinet -> closet, cup/glass/bottle -> cup, etc.)
            label_to_mask: Dict[str, np.ndarray] = {}
            for lbl, m in raw_label_to_mask.items():
                if lbl == "closet/cabinet":
                    canon = "closet"
                elif lbl == "cup/glass/bottle":
                    canon = "cup"
                elif lbl == "paper/notebook":
                    canon = "paper"
                elif lbl == "sofa/couch":
                    canon = "sofa"
                elif lbl == "phone/camera":
                    canon = "phone"
                else:
                    canon = lbl

                # if multiple raw labels map to same canonical label, union them
                if canon in label_to_mask:
                    label_to_mask[canon] = np.logical_or(label_to_mask[canon], m)
                else:
                    label_to_mask[canon] = m

            # --- which labels do we actually need masks for in this frame? ---
            needed_labels = set()
            for item in frame_items:
                # person annotations can come via 'person_bbox' and may not have 'class'
                if "person_bbox" in item:
                    needed_labels.add("person")
                    continue

                if "class" not in item:
                    continue

                category_id = item["class"]
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

                needed_labels.add(category_name)

            # --- gather masks for those labels (one mask per label) ---
            masks: List[np.ndarray] = []
            for lbl in sorted(needed_labels):
                if lbl in label_to_mask:
                    masks.append(label_to_mask[lbl])

            if masks:
                frame_masks[frame_name] = masks

        if not frame_masks:
            print(f"[sam3d] No masks found for any selected frame in video {video_id}")
            return

        # ------------------------------------------------------------------
        # 4) Run SAM3D on each selected frame and save compressed outputs
        # ------------------------------------------------------------------
        video_output_dir = self.sam3d_annotations_dir / video_id / "sam3d_outputs"
        os.makedirs(video_output_dir, exist_ok=True)

        processed_count = 0

        for frame_name, masks in frame_masks.items():
            print("--------------------------------------------------------------------")
            print(f"Processing video [{video_id}] frame [{frame_name}]")
            print("--------------------------------------------------------------------")

            image_path = self.frame_annotated_dir_path / video_id / frame_name
            if not image_path.exists():
                print(f"[sam3d] Frame image not found: {image_path}, skipping")
                continue

            frame_stem = Path(frame_name).stem

            # NEW: only create GIFs for gif_frames
            save_gif_for_this_frame = save_gif and (frame_stem in gif_frame_stems)
            print(f"[sam3d] save_gif_for_this_frame={save_gif_for_this_frame}, frame_stem={frame_stem}")

            try:
                # Use stem as frame_name for cleaner directory/file names
                self.process_frame(
                    frame_name=frame_stem,
                    image_path=str(image_path),
                    masks=masks,
                    video_output_dir=video_output_dir,
                    save_gif=save_gif_for_this_frame,
                    compress_pickle=compress_pickle,
                )
                processed_count += 1
            except Exception as e:
                print(f"[sam3d] Failed to run SAM3D for {video_id}, frame {frame_name}: {e}")

            print("--------------------------------------------------------------------")

        print(
            f"[sam3d] Finished video {video_id}: "
            f"{processed_count}/{len(selected_frames)} selected frames processed "
            f"(save_gif={save_gif}, compress_pickle={compress_pickle}). "
            f"GIFs created for {len(gif_frame_stems & {Path(f).stem for f in frame_masks.keys()})} frames."
        )

    def generate_sam3d_annotations(self, dataloader, split) -> None:
        for data in tqdm(dataloader):
            video_id = data['video_id']
            if get_video_belongs_to_split(video_id) == split:
                out_path = self.sam3d_annotations_dir / video_id / "sam3d_outputs"
                if out_path.exists() and len(os.listdir(out_path)) > 0:
                    print(f"[sam3d] floor-aligned 3D bboxes already exist for video {video_id}, skipping...")
                    continue
                try:
                    print(f"[sam3d] processing video {video_id}...")
                    self.process_video(video_id)
                except Exception as e:
                    print(f"[sam3d] failed to process video {video_id}: {e}")
            else:
                print(f"[sam3d] video {video_id} does not belong to split {split}, skipping...")

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
    parser.add_argument("--output_directory", type=str, default="/data2/rohith/ag/ag4D/world_annotations/")
    parser.add_argument("--split", type=str, default="04")
    return parser.parse_args()

def main():
    args = parse_args()
    bbox_3d_generator = AgSam3DInference(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_directory=args.output_directory
    )
    train_dataset, test_dataset, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)
    bbox_3d_generator.generate_sam3d_annotations(dataloader=dataloader_train, split=args.split)
    bbox_3d_generator.generate_sam3d_annotations(dataloader=dataloader_test, split=args.split)

def main_sample():
    args = parse_args()

    bbox_3d_generator = AgSam3DInference(
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
        ag_root_directory=args.ag_root_directory,
        output_directory=args.output_directory
    )
    video_id = "0DJ6R.mp4"
    bbox_3d_generator.generate_sample_sam3d_annotations(video_id=video_id)


if __name__ == "__main__":
    main_sample()
    # main()
