"""Bundle the real intermediates of the 3D bounding-box annotation pipeline for one video
(supplementary Fig. sup:fig:bbox_pipeline), run on CS93371.

    python scripts/paper_figures/extract_bbox_pipeline.py --video 00T1E --frames 000087 000163 \
        --out /data3/rohith/ag/runs/bbox_pipeline/00T1E

Writes into <out>:
  frames/<stem>.png              raw annotated frame (270x480)
  sam2/<stem>__<label>__{image,video}.png   SAM2 image-mode / video-mode masks
  sam2/<stem>__combined.png, <stem>__rect.png  combined dynamic mask, rectangular inpainting mask
  detections.json                Grounding DINO (+fused GT) boxes/scores/labels for every sampled frame
  ag_gt.json                     Action Genome GT 2D boxes for the requested frames
  labels.json                    AG active labels, LLM moving-object list
  pi3_<stem>.npz                 full-resolution pi3 dynamic points / conf / image for that frame's clip
  boxes_3d.json                  per frame, per object: aabb run (bbox_annotations_3d) and obb run
                                 (bbox_annotations_3d_obb) incl. all multiscale candidates
  final_4d.json                  bbox_annotations_4d objects per frame (final boxes)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np

AG = Path("/data/rohith/ag")
AG2 = Path("/data2/rohith/ag")


def jsonable(x):
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy().tolist()
    except ImportError:
        pass
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="00T1E")
    ap.add_argument("--frames", nargs="+", default=["000087", "000163"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    v, out = a.video, Path(a.out)
    for sub in ("frames", "sam2"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # ---- labels
    lab = {}
    for key, rel in (("ag_active", "active_objects/annotations"), ("ag_active_sampled", "active_objects/sampled_videos"),
                     ("llm_moving", "moving_objects/llama3.1")):
        p = AG / rel / f"{v}.txt"
        lab[key] = open(p).read().split() if p.exists() else None
    json.dump(lab, open(out / "labels.json", "w"), indent=1)

    # ---- detections (all frames) and AG GT for requested frames
    g = pickle.load(open(AG / "detection" / "gdino_bboxes" / f"{v}.mp4.pkl", "rb"))
    json.dump(jsonable(g), open(out / "detections.json", "w"))
    gt = {}
    obj = pickle.load(open(AG / "annotations" / "object_bbox_and_relationship.pkl", "rb"))
    per = pickle.load(open(AG / "annotations" / "person_bbox.pkl", "rb"))
    for s in a.frames:
        k = f"{v}.mp4/{s}.png"
        gt[s] = {"objects": jsonable(obj.get(k)), "person": jsonable(per.get(k))}
    json.dump(gt, open(out / "ag_gt.json", "w"), indent=1)

    # ---- images and masks
    for s in a.frames:
        shutil.copy(AG / "frames" / f"{v}.mp4" / f"{s}.png", out / "frames" / f"{s}.png")
        for mode in ("image_based", "video_based"):
            d = AG / "segmentation" / "masks" / mode / f"{v}.mp4"
            for n in os.listdir(d):
                if n.startswith(f"{s}__"):
                    tag = "image" if mode == "image_based" else "video"
                    label = n[len(s) + 2:-4] or "unlabeled"
                    shutil.copy(d / n, out / "sam2" / f"{s}__{label}__{tag}.png")
        for mode, tag in (("combined_1", "combined"), ("rectangular_overlayed_masks", "rect")):
            p = AG / "segmentation" / "masks" / mode / f"{v}.mp4" / f"{s}.png"
            if p.exists():
                shutil.copy(p, out / "sam2" / f"{s}__{tag}.png")

    # ---- pi3 clip of each requested frame (clip k <-> sampled_idx[k]; the clip shows file sampled_idx[k]+1)
    sampled = np.load(AG / "sampled_frames_idx" / f"{v}.npy", allow_pickle=True).astype(int)
    z = np.load(AG2 / "ag4D" / "dynamic_scenes" / "pi3_dynamic" / f"{v}_10" / "predictions.npz", allow_pickle=True)
    Z = {k: z[k] for k in ("points", "local_points", "conf", "camera_poses", "images")}
    for s in a.frames:
        idx = int(s)
        ks = np.nonzero(sampled[: len(Z["points"])] == idx)[0]
        if not len(ks):
            print("no pi3 clip for", s); continue
        k = int(ks[0])
        np.savez_compressed(out / f"pi3_{s}.npz", k=k, points=Z["points"][k].astype(np.float32),
                            local=Z["local_points"][k].astype(np.float32), conf=Z["conf"][k][..., 0].astype(np.float32),
                            image=(Z["images"][k] * 255).clip(0, 255).astype(np.uint8), pose=Z["camera_poses"][k])
        print("pi3 clip", k, "for frame", s)

    # ---- 3D boxes: both runs + final
    B = {}
    for run in ("bbox_annotations_3d", "bbox_annotations_3d_obb"):
        d = pickle.load(open(AG / "world_annotations" / run / f"{v}.pkl", "rb"))
        B[run] = {"frames": jsonable(d["frames"]), "global_floor_sim": jsonable(d["global_floor_sim"])}
    json.dump(B, open(out / "boxes_3d.json", "w"))
    d4 = pickle.load(open(AG / "world_annotations" / "bbox_annotations_4d" / f"{v}.pkl", "rb"))
    json.dump(jsonable({k: d4[k] for k in ("frames", "frame_names", "all_labels", "world_to_final", "global_floor_sim")}),
              open(out / "final_4d.json", "w"))
    print("wrote", out)


if __name__ == "__main__":
    main()
