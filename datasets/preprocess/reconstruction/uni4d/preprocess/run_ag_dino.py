import argparse
import json
import os
import pickle

import cv2
import numpy as np
import supervision as sv
import torch
import torchvision
from PIL import Image
from supervision.draw.color import ColorPalette
from tqdm import tqdm

CUSTOM_COLOR_MAP = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#0082c8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#d2f53c",
    "#fabebe",
    "#008080",
    "#e6beff",
    "#aa6e28",
    "#fffac8",
    "#800000",
    "#aaffc3",
]

torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16).__enter__()


def id_to_colors(id):  # id to color
    rgb = np.zeros((3,), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb


class AgDinoV2Mask:

    def __init__(
            self,
            args,
            ag_root_dir
    ):
        self.ag_root_dir = ag_root_dir

        self.ag_frames_dir = os.path.join(ag_root_dir, "frames")
        self.ag_4D_dir = os.path.join(ag_root_dir, "ag4D")
        self.uni4D_dir = os.path.join(self.ag_4D_dir, "uni4D")
        self.unidepth_dir = os.path.join(self.uni4D_dir, "unidepth")
        os.makedirs(self.unidepth_dir, exist_ok=True)

        self.frames_path = os.path.join(self.ag_root_dir, "frames")
        self.annotations_path = os.path.join(self.ag_root_dir, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))
        print("Total number of ground truth annotations: ", len(self.gt_annotations))

        self.idx_to_id = [i for i in range(256 * 256 * 256)]
        np.random.shuffle(self.idx_to_id)  # mapping to randomize idx to id to get random color

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        video_id_frame_id_list_pkl_file_path = os.path.join(self.ag_root_dir, "4d_video_frame_id_list.pkl")
        if os.path.exists(video_id_frame_id_list_pkl_file_path):
            with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
                self.video_id_frame_id_list = pickle.load(f)
        else:
            assert False, f"Please generate {video_id_frame_id_list_pkl_file_path} first"

        self.object_labels = [
            "person", "bag", "blanket", "book", "box", "broom", "chair", "clothes", "cup", "dish", "food", "laptop",
            "paper", "phone", "picture", "pillow", "sandwich", "shoe", "towel", "vacuum", "glass", "bottle", "notebook",
            "camera"
        ]
        self.gdino_output_dir = os.path.join(self.uni4D_dir, "gdino")
        self.gdino_mask_output_dir = os.path.join(self.uni4D_dir, "gdino_mask")

    def get_frame_paths(self, video_name):
        frame_paths = []
        video_frames_path = os.path.join(self.frames_path, video_name)
        frame_id_list = self.video_id_frame_id_list[video_name]
        frame_id_list = sorted(np.unique(frame_id_list))
        for frame_id in frame_id_list:
            frame_paths.append(os.path.join(video_frames_path, f"{frame_id:06d}.png"))
        return frame_id_list, frame_paths

    def fetch_gdino_detections(self, video_name):
        gdino_output_path = os.path.join(self.gdino_output_dir, f"{video_name[:-4]}.pkl")
        if not os.path.exists(gdino_output_path):
            raise FileNotFoundError(f"Grounding DINO detections not found for {video_name}. Please run GDino first.")
        with open(gdino_output_path, 'rb') as f:
            gdino_detections = torch.load(f)

        # Map object detections to frame IDs
        object_to_frame_id_map = {}
        for frame_id, detections in gdino_detections.items():
            objects = detections['labels']
            for obj in objects:
                if obj not in object_to_frame_id_map:
                    object_to_frame_id_map[obj] = []
                object_to_frame_id_map[obj].append(frame_id)

        # Frame id to information maps
        frame_id_to_bbox_map = {}
        frame_id_to_scores_map = {}
        frame_id_to_labels_map = {}
        for frame_id, detections in gdino_detections.items():
            bboxes = detections['boxes']
            frame_id_to_bbox_map[frame_id] = bboxes
            scores = detections['scores']
            frame_id_to_scores_map[frame_id] = scores
            labels = detections['labels']
            frame_id_to_labels_map[frame_id] = labels

        return object_to_frame_id_map, frame_id_to_bbox_map, frame_id_to_scores_map, frame_id_to_labels_map

    def get_gdino_masks_framewise(
            self,
            video_name: str,
            frame_id_to_bbox_map: dict,
            frame_id_to_scores_map: dict,
            frame_id_to_labels_map: dict,
    ):
        """
        Generate per-frame Grounding-DINO masks, run NMS first,
        and store colour masks + visualisations on disk.
        """
        video_output_path_vis = os.path.join(self.gdino_mask_output_dir, video_name, "vis")
        video_output_path_mask = os.path.join(self.gdino_mask_output_dir, video_name, "mask")
        os.makedirs(video_output_path_vis, exist_ok=True)
        os.makedirs(video_output_path_mask, exist_ok=True)

        # Frames available for this video
        frame_id_list, frame_paths = self.get_frame_paths(video_name)

        for frame_id, frame_path in tqdm(
                zip(frame_id_list, frame_paths), total=len(frame_id_list), desc=f"{video_name}"
        ):
            # ------------------------------------------------------------------
            # 1) Load image & fetch GD-DINO detections for this frame
            # ------------------------------------------------------------------
            img_bgr = cv2.imread(frame_path)  # (H, W, 3) BGR for drawing
            if img_bgr is None:
                print(f"Could not read {frame_path}. Skipping.")
                continue
            H, W = img_bgr.shape[:2]

            boxes_tensor = frame_id_to_bbox_map.get(frame_id, None)  # (n, 4) torch.float32
            scores_tensor = frame_id_to_scores_map.get(frame_id, None)  # (n,)   torch.float32
            labels_list = frame_id_to_labels_map.get(frame_id, None)  # list[str] length n

            if (
                    boxes_tensor is None
                    or scores_tensor is None
                    or labels_list is None
                    or len(boxes_tensor) == 0
            ):
                # Nothing detected for this frame
                continue

            boxes_np = boxes_tensor.cpu().numpy()  # (n, 4)
            scores_np = scores_tensor.cpu().numpy()  # (n,)
            class_ids = np.arange(len(labels_list))  # (n,) temp id -> 0 .. n-1
            class_names = np.array(labels_list)  # (n,) ndarray[str]

            # ------------------------------------------------------------------
            # 2)  Non-max suppression *before* mask generation
            # ------------------------------------------------------------------
            keep_idx = torchvision.ops.nms(
                torch.from_numpy(boxes_np).float(),
                torch.from_numpy(scores_np).float(),
                0.50,  # IoU threshold
            ).cpu().numpy()

            boxes_np = boxes_np[keep_idx]
            scores_np = scores_np[keep_idx]
            class_ids = class_ids[keep_idx]
            class_names = class_names[keep_idx]

            if len(boxes_np) == 0:
                # All detections suppressed
                continue

            # ------------------------------------------------------------------
            # 3)  Build rectangular masks for kept boxes
            # ------------------------------------------------------------------
            masks_bool = np.zeros((len(boxes_np), H, W), dtype=bool)

            for i, (x1, y1, x2, y2) in enumerate(boxes_np):
                # clamp to image bounds & convert to int
                x1 = max(0, int(np.floor(x1)))
                y1 = max(0, int(np.floor(y1)))
                x2 = min(W - 1, int(np.ceil(x2)))
                y2 = min(H - 1, int(np.ceil(y2)))
                masks_bool[i, y1: y2 + 1, x1: x2 + 1] = True

            # ------------------------------------------------------------------
            # 4)  Pack into a supervision.Detections object
            # ------------------------------------------------------------------
            detections = sv.Detections(
                xyxy=boxes_np,  # (m, 4)
                mask=masks_bool,  # (m, H, W)
                class_id=class_ids,  # (m,)
                confidence=scores_np,  # (m,)
            )

            # ------------------------------------------------------------------
            # 5)  Draw VISUALISATIONS (boxes + labels + masks)
            # ------------------------------------------------------------------
            palette = ColorPalette.from_hex(CUSTOM_COLOR_MAP)
            box_annotator = sv.BoxAnnotator(color=palette)
            label_annotator = sv.LabelAnnotator(color=palette)
            mask_annotator = sv.MaskAnnotator(color=palette)

            labels_for_vis = [
                f"{labels_list[i]} {detections.confidence[j]:.2f}"
                for j, i in enumerate(detections.class_id)
            ]

            img_vis = box_annotator.annotate(img_bgr.copy(), detections=detections)
            img_vis = label_annotator.annotate(img_vis, detections=detections, labels=labels_for_vis)
            img_vis = mask_annotator.annotate(img_vis, detections=detections)

            # ------------------------------------------------------------------
            # 6)  Convert masks to colour map & write outputs
            # ------------------------------------------------------------------
            color_mask = np.zeros((H, W, 3), dtype=np.uint8)
            obj_info_json = []

            # sort masks by area so larger masks are painted first
            areas = [np.sum(m) for m in detections.mask]
            for idx in np.argsort(areas)[::-1]:
                mask = detections.mask[idx]
                unique_id = self.idx_to_id[idx]
                color_mask[mask] = id_to_colors(unique_id)
                obj_info_json.append(
                    {
                        "id": int(unique_id),
                        "label": str(class_names[idx]),
                        "score": float(detections.confidence[idx]),
                    }
                )

            frame_name = f"{frame_id:06d}.png"
            cv2.imwrite(os.path.join(video_output_path_vis, frame_name), img_vis)  # visualisation
            cv2.imwrite(os.path.join(video_output_path_mask, frame_name), cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR))
            with open(os.path.join(video_output_path_mask, frame_name.replace(".png", ".json")), "w") as f:
                json.dump(obj_info_json, f, indent=2)

    def process_ag_video(self, video_name):
        video_output_dir = os.path.join(self.gdino_mask_output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        (object_to_frame_id_map, frame_id_to_bbox_map,
         frame_id_to_scores_map, frame_id_to_labels_map) = self.fetch_gdino_detections(video_name)

        self.get_gdino_masks_framewise(
            video_name=video_name,
            frame_id_to_bbox_map=frame_id_to_bbox_map,
            frame_id_to_scores_map=frame_id_to_scores_map,
            frame_id_to_labels_map=frame_id_to_labels_map
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--workdir", default="/data/rohith/ag")
    parser.add_argument("--sam2-checkpoint", default="./preprocess/pretrained/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    parser.add_argument("--output-dir", default="gsam2")
    parser.add_argument("--input-dir", default="ram")
    parser.add_argument("--force-cpu", action="store_true")
    args = parser.parse_args()

    ag_gdino_sam2 = AgDinoV2Mask(
        ag_root_dir=args.workdir,
        args=args
    )

    # for video_name in tqdm(ag_unidepth.video_list):
    #     ag_unidepth.process_ag_video(video_name)

    video_name = "00T1E.mp4"  # Replace with an actual video name from your dataset
    ag_gdino_sam2.process_ag_video(video_name)


if __name__ == "__main__":
    main()
