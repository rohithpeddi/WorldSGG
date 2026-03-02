#!/usr/bin/env python3
"""
Precompute CLIP Text Embeddings
================================

One-time script to extract CLIP text embeddings for all Action Genome
object class labels. Stores the result as a small .npy file so the CLIP
model never needs to be loaded during training.

Usage:
    python scripts/precompute_clip_embeddings.py \
        --data_path /data/rohith/ag

Output:
    {data_path}/features/clip_features/clip_text_embeddings.npy   (37, 512)
"""

import argparse
import os

import numpy as np
import torch

# Action Genome object classes (37 total, including __background__)
OBJECT_CLASSES = [
    "__background__", "person", "bag", "bed", "blanket", "book", "box",
    "broom", "chair", "closet/cabinet", "clothes", "cup/glass/bottle",
    "dish", "door", "doorknob", "doorway", "floor", "food", "groceries",
    "laptop", "light", "medicine", "mirror", "paper/notebook",
    "phone/camera", "picture", "pillow", "refrigerator", "sandwich",
    "shelf", "shoe", "sofa/couch", "table", "television", "towel",
    "vacuum", "window",
]


def main():
    parser = argparse.ArgumentParser(description="Precompute CLIP text embeddings for AG classes")
    parser.add_argument("--data_path", type=str, default="/data/rohith/ag", help="AG dataset root")
    parser.add_argument("--clip_model", type=str, default="ViT-B/32", help="CLIP model variant")
    args = parser.parse_args()

    # Import CLIP (only needed for this one-time script)
    try:
        import clip
    except ImportError:
        raise ImportError(
            "CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git"
        )

    # Output directory
    out_dir = os.path.join(args.data_path, "features", "clip_features")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "clip_text_embeddings.npy")

    print(f"Loading CLIP model: {args.clip_model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(args.clip_model, device=device)

    # Create text prompts — use human-readable forms
    prompts = [f"a photo of a {cls.replace('/', ' or ')}" for cls in OBJECT_CLASSES]
    print(f"Encoding {len(prompts)} class labels:")
    for i, p in enumerate(prompts):
        print(f"  [{i:2d}] {p}")

    # Encode
    with torch.no_grad():
        tokens = clip.tokenize(prompts).to(device)
        embeddings = model.encode_text(tokens).float().cpu().numpy()  # (37, 512)

    print(f"\nEmbeddings shape: {embeddings.shape}")
    print(f"Saving to: {out_path}")
    np.save(out_path, embeddings)
    print("Done!")


if __name__ == "__main__":
    main()
