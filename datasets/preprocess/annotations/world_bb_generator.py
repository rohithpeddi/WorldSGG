# This block of code takes in each annotations frame and the corresponding 3D points from the world.
# For each annotated 2D bounding box, estimates its corresponding 3D bounding box in the world coordinate system.
# It uses two types (a) Axis Aligned Bounding Boxes (AABB) and (b) Oriented Bounding Boxes (OBB).
# It also helps in visualization of the 3D bounding boxes.
import argparse
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Any


class WorldBBGenerator:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_directory: Optional[str] = None,
    ) -> None:
        self.ag_root_directory = Path(ag_root_directory)
        self.dynamic_scene_dir_path = dynamic_scene_dir_path

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

        # ------------------------------ Directory Paths ------------------------------ #
        # Detections paths
        self.dynamic_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes'
        self.static_detections_root_path = self.ag_root_directory / "detection" / 'gdino_bboxes_static'

        # Segmentation masks paths
        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation" / "masks" / "combined"

        self.masked_frames_im_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'image_based'
        self.masked_frames_vid_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'video_based'
        self.masked_frames_combined_dir_path = self.ag_root_directory / "segmentation_static" / 'masked_frames' / 'combined'
        self.masked_videos_dir_path = self.ag_root_directory / "segmentation_static" / "masked_videos"

        # Internal (per-object) mask stores
        self.masks_im_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "image_based"
        self.masks_vid_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "video_based"
        self.masks_combined_dir_path = self.ag_root_directory / "segmentation_static" / "masks" / "combined"

    def create_gt_annotations_map(self, dataloader):
        video_id_gt_annotations_map = {}
        video_id_gt_bboxes_map = {}
        for data in dataloader:
            video_id = data['video_id']
            gt_annotations = data['gt_annotations']
            video_id_gt_annotations_map[video_id] = gt_annotations

        # video_id, gt_bboxes for the gt detections
        for video_id, gt_annotations in video_id_gt_annotations_map.items():
            video_gt_bboxes = {}
            for frame_idx, frame_items in enumerate(gt_annotations):
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
            video_id_gt_bboxes_map[video_id] = video_gt_bboxes
        return video_id_gt_bboxes_map, video_id_gt_annotations_map

    def create_gdino_annotations_map(self, dataloader):
        video_id_gdino_annotations_map = {}
        for data in dataloader:
            video_id = data['video_id']

            # 1. Load dynamic gdino annotations
            video_dynamic_gdino_prediction_file_path = self.dynamic_detections_root_path / f"{video_id}.pkl"
            video_dynamic_predictions = None
            with open(video_dynamic_gdino_prediction_file_path, 'rb') as f:
                video_dynamic_predictions = pickle.load(f)

            # 2. Load static gdino annotations
            video_static_gdino_prediction_file_path = self.static_detections_root_path / f"{video_id}.pkl"
            video_static_predictions = None
            with open(video_static_gdino_prediction_file_path, 'rb') as f:
                video_static_predictions = pickle.load(f)

            # 3. Frame wise combined gdino annotations, use frame_id as the key for the map
            combined_gdino_predictions = {}
            for frame_name, dynamic_pred in video_dynamic_predictions.items():
                static_pred = video_static_predictions.get(frame_name, None)
                if static_pred:
                    combined_boxes = dynamic_pred['boxes'] + static_pred['boxes']
                    combined_labels = dynamic_pred['labels'] + static_pred['labels']
                    combined_scores = dynamic_pred['scores'] + static_pred['scores']
                else:
                    combined_boxes = dynamic_pred['boxes']
                    combined_labels = dynamic_pred['labels']
                    combined_scores = dynamic_pred['scores']
                combined_gdino_predictions[frame_name] = {
                    'boxes': combined_boxes,
                    'labels': combined_labels,
                    'scores': combined_scores
                }
            video_id_gdino_annotations_map[video_id] = combined_gdino_predictions

        return video_id_gdino_annotations_map

    def create_label_wise_masks_map(self, dataloader):
        pass

    def generate_gt_world_bb_annotations(self, dataloader) -> None:
        # For every frame in the video, (1) Person bbox is in xywh format and the object bbox is in xyxy format.
        # For the category of the object in the frame we have to get the 3D points corresponding to that object.

        # 1. Ground truth annotations for specific frames.
        # This primarily includes bounding boxes for persons and objects in the frame.
        video_id_gt_bboxes_map, video_id_gt_annotations_map = self.create_gt_annotations_map(dataloader)

        # 2. Grounding Dino bounding boxes for specific frames.
        # Combined detections of dynamic objects and static objects.
        video_id_gdino_annotations_map = self.create_gdino_annotations_map(dataloader)

        # 3. Label wise masks for each object in specific frames.

        # 4. For every ground truth bounding box detection, we need to make sure that we have corresponding gdino bounding box may be some union of boxes.

        # 5. Load 3D points for specific frames.
        # We need to match specific frames with the subsampled frames for the complete video

        # 6. We need to extract the 3D points corresponding to each object in the frame using the masks.

        # 7. Using the 3D points, we need to estimate the Axis Aligned Bounding Box (AABB) and Oriented Bounding Box (OBB) for each object in the frame.

        # 8. We need to run a rerun visualization for all the things frame by frame.
        # Gdino detections, Ground truth detection, Final label wise masks, 3D points, AABB and OBB boxes.

        # 9. Finally, we need to save the world bounding box annotations in a pkl file.

        pass




def _parse_split(s: str) -> str:
    valid = {"04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"}
    val = s.strip().upper()
    if val not in valid:
        raise argparse.ArgumentTypeError(
            f"Invalid split '{s}'. Choose one of: {sorted(valid)}"
        )
    return val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize static + per-frame 3D points with Rerun (AG-Pi3 unified)."
    )
    # Paths
    parser.add_argument(
        "--frames_annotated_dir_path",
        type=str,
        default="/data/rohith/ag/frames_annotated",
        help="Optional: directory containing annotated frames (unused here).",
    )
    parser.add_argument(
        "--mask_dir_path",
        type=str,
        default="/data/rohith/ag/segmentation/masks/rectangular_overlayed_masks",
        help="Path to directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--static_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/static_scenes/pi3_static",
        help="Path to output directory where predictions folders live (e.g., <video>_10/).",
    )
    parser.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )
    parser.add_argument(
        "--grounded_dynamic_scene_dir_path",
        type=str,
        default="/data2/rohith/ag/ag4D/dynamic_scenes/pi3_grounded_dynamic"
    )
    # Selection
    parser.add_argument(
        "--split",
        type=_parse_split,
        default="04",
        help="Shard to process: one of {04, 59, AD, EH, IL, MP, QT, UZ}.",
    )


    return parser.parse_args()


def main() -> None:
    args = parse_args()

    world_bb_annotations = WorldBBGenerator(
        phase="train",
        mode=args.split,
        datasize="full",
        data_path="/data/rohith/ag/action_genome",
        filter_nonperson_box_frame=True,
        filter_small_box=False,
        enable_coco_gt=False
    )


if __name__ == "__main__":
    main()