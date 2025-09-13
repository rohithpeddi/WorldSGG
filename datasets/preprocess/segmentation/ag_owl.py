from typing import Dict, Any, Union, Optional
import argparse
import os
import pickle
from pathlib import Path
from typing import Dict, Any, Union, Optional
from typing import List

import torch
from PIL import Image
from PIL import ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import pipeline


def visualize_owl_predictions(
        image: Image.Image,
        predictions: List[Dict[str, Any]],
        score_threshold: float = 0.3,
        font: Optional[ImageFont.ImageFont] = None,
        frame_idx: int = 0,
        output_dir: Optional[Path] = None,
) -> Image.Image:
    draw = ImageDraw.Draw(image := image.copy())
    try:
        if font is None:
            font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        print("Font file not found. Using default font.")
        font = ImageFont.load_default()

    for pred in predictions:
        if pred["score"] < score_threshold:
            continue

        xmin, ymin, xmax, ymax = pred["box"].values()
        label = f"{pred['label']} {pred['score']:.2f}"

        draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=2)

        text_w = font.getbbox(label)[2] - font.getbbox(label)[0]
        text_h = font.getbbox(label)[3] - font.getbbox(label)[1]

        draw.rectangle((xmin, ymin - text_h, xmin + text_w, ymin), fill="red")
        draw.text((xmin, ymin - text_h), label, fill="white", font=font)

        if output_dir is not None:
            out_path = output_dir / f"frame_{frame_idx:04d}.jpg"
            image.save(out_path)
    return image


def visualize_owl_predictions_batch(
        images: List[Union[str, Image.Image]],
        preds_batch: List[List[Dict[str, Any]]],
        score_threshold: float = 0.3,
        output_dir: Optional[str] = None,
        overwrite: bool = False,
) -> List[Image.Image]:
    # Prepare output folder
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    annotated_imgs: List[Image.Image] = []

    for idx, (img, preds) in enumerate(zip(images, preds_batch)):
        pil_img = Image.open(img) if isinstance(img, (str, Path)) else img
        annotated = visualize_owl_predictions(
            pil_img, preds, score_threshold=score_threshold
        )
        annotated_imgs.append(annotated)

        if output_dir is not None:
            out_path = output_dir / f"frame_{idx:04d}.jpg"
            if overwrite or not out_path.exists():
                annotated.save(out_path)

    return annotated_imgs


class OwlVideoDataset(Dataset):
    """
    Iterates over every video_id and returns: video_id, [PIL.Image]  (all frames in that video)
    """

    def __init__(self, ag_root_dir: str):
        self.frames_root = Path(ag_root_dir) / "frames"
        self.video_ids = sorted(os.listdir(self.frames_root))

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        frame_dir = self.frames_root / video_id
        imgs = [
            Image.open(frame_dir / f).convert("RGB")
            for f in sorted(os.listdir(frame_dir))
        ]
        return video_id, imgs


def cuda_collate_fn(batch):
    """
    don't need to zip the tensor
    """
    return batch[0]


def run_detection(
        ag_root_dir: str,
        batch_size: int = 15
):
    # initialize detector
    detector = pipeline(
        task="zero-shot-object-detection",
        model="google/owlv2-base-patch16-ensemble",
    )
    candidate_labels = [
        "person", "bag", "blanket", "book", "box", "broom", "chair", "clothes", "cup", "dish", "food", "laptop",
        "paper", "phone", "picture", "pillow", "sandwich", "shoe",  "towel", "vacuum", "glass", "bottle", "notebook", "camera"
    ]
    
    out_dir = Path(ag_root_dir) / "detection" / "owl"
    out_dir.mkdir(parents=True, exist_ok=True)

    # prepare data loader
    dataset = OwlVideoDataset(ag_root_dir)
    video_frames_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=cuda_collate_fn,
        pin_memory=False,
    )

    # will accumulate per-video predictions
    results = {}

    for video_id, images in tqdm(video_frames_dataloader, desc="Detecting"):
        vis_out_dir = Path(ag_root_dir) / "detection" / "visualization" / "owl" / video_id
        os.makedirs(vis_out_dir, exist_ok=True)
        # run pipeline on all images at once (it will chunk internally by pipeline batch_size)
        predictions = detector(
            images[0],
            candidate_labels=candidate_labels,
            batch_size=batch_size
        )

        # save results
        with open(out_dir / f"{video_id}.pkl", "wb") as f:
            pickle.dump(predictions, f)
            
    print(f"✅ Done! Processed {len(results)} videos.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ag_root_directory",
        type=str,
        default="/data/rohith/ag/",
        help="Root dir with 'frames/' subfolder"
    )
    parser.add_argument("--batch_size", type=int, default=15)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    run_detection(
        ag_root_dir=args.ag_root_directory,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
