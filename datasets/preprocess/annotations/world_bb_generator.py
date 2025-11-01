# This block of code takes in each annotations frame and the corresponding 3D points from the world.
# For each annotated 2D bounding box, estimates its corresponding 3D bounding box in the world coordinate system.
# It uses two types (a) Axis Aligned Bounding Boxes (AABB) and (b) Oriented Bounding Boxes (OBB).
# It also helps in visualization of the 3D bounding boxes.
import argparse
from typing import Optional

from dataloader.base_ag_dataset import BaseAG


class WorldBBGenerator:

    def __init__(
            self,
            dynamic_scene_dir_path: Optional[str] = None,
            ag_root_dir_path: Optional[str] = None,
            frame_annotated_dir_path: Optional[str] = None
    ) -> None:
        self.dynamic_scene_dir_path = dynamic_scene_dir_path
        self.ag_root_dir_path = ag_root_dir_path
        self.frame_annotated_dir_path = frame_annotated_dir_path

        self.image_mask_dir_path = "/data/rohith/ag/segmentation/masks/image_based"
        self.video_mask_dir_path = "/data/rohith/ag/segmentation/masks/video_based"

    def generate_video_world_bb_annotations(self, video_id: str, dataset) -> None:
        gt_annotations = dataset.gt_annotations(video_id)

        # For every frame in the video, (1) Person bbox is in xywh format and the object bbox is in xyxy format.
        # For the category of the object in the frame we have to get the 3D points corresponding to that object. 


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