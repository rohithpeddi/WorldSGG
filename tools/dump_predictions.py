"""
Dump per-frame WSGG predictions + GT + visibility for offline analysis.
=======================================================================

Runs a trained checkpoint once over the test set (ALL frames, single forward
pass per video -- same as tools/reeval_test.py) and pickles, for every frame,
the arrays needed to reproduce the recall metric offline:

    attention/spatial/contacting_distribution, gt_attention/spatial/contacting,
    pair_valid, person_idx, object_idx, object_classes, bboxes_2d, valid_mask,
    visibility_mask (+ sgdet: pred_labels, pred_scores, gt_bboxes_2d)

This is the "dump" half of the dump-once-then-analyze workflow: the model runs
one time per checkpoint, and tools/bucketed_breakdown.py then computes the
per-visibility-bucket breakdown offline without re-running inference (so the
bucket / trivial-set / IoU / frame-scheme definitions can be re-sliced freely).

Usage:
    python tools/dump_predictions.py --config <cfg> --ckpt checkpoint_19 \
        --frames all --out results/bucket_dumps/<exp>__all.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from wsgg_base import load_wsgg_config                     # noqa: E402
from tools.reeval_test import (                            # noqa: E402
    build_model, forward_all_frames, load_checkpoint, build_pred_pkl, make_test_dataset,
)


def _slim(pkl):
    """Downcast float arrays to float32 to keep dumps small; keep ints."""
    out = {}
    for k, v in pkl.items():
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            out[k] = v.astype(np.float32)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="checkpoint_19")
    ap.add_argument("--frames", default="all", choices=["all", "last"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--videos", nargs="*", default=[],
                    help="restrict the dump to these video ids (with or without the "
                         "'.mp4' suffix); default is the whole test split")
    ap.add_argument("--keep-3d", action="store_true",
                    help="keep the 3-D arrays: the input corners (bboxes_3d, gt_corners) "
                         "and, for a model with a 3-D head, its predicted corners "
                         "(pred_corners_slot / pred_corners_free).  Off by default "
                         "because the full-split dumps only need 2-D bucket recall")
    args = ap.parse_args()

    conf = load_wsgg_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from dataloader.world_ag_dataset import world_collate_fn
    from torch.utils.data import DataLoader
    ds = make_test_dataset(conf)
    if args.videos:
        wanted = {v for x in args.videos for v in (x, f"{x}.mp4", x.replace(".mp4", ""))}
        keep = [v for v in ds.video_list if v in wanted or v.replace(".mp4", "") in wanted]
        missing = {x.replace(".mp4", "") for x in args.videos} - {v.replace(".mp4", "") for v in keep}
        if missing:
            raise SystemExit(f"[dump] videos not in the test split: {sorted(missing)}")
        ds.video_list = keep
        print(f"[dump] restricted to {len(keep)} video(s): {keep}")
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=world_collate_fn)
    print(f"[dump] {conf.experiment_name} | mode={conf.mode} | "
          f"method={conf.method_name} | videos={len(ds)} | frames={args.frames}")

    model = build_model(conf, ds, device)
    load_checkpoint(model, conf, args.ckpt, device)
    model.eval()

    records = []
    n = len(dl) if args.limit <= 0 else min(args.limit, len(dl))
    it = iter(dl)
    with torch.no_grad():
        for _ in tqdm(range(n), desc="Dumping"):
            batch = next(it)
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
            pred = forward_all_frames(model, conf, b)
            T = int(batch["T"])
            if T <= 0:
                continue
            frame_idxs = range(T) if args.frames == "all" else [T - 1]
            for t in frame_idxs:
                pkl = build_pred_pkl(batch, pred, t, conf.mode)
                # the one field the standard dump omits -- needed for bucketing
                pkl["visibility_mask"] = batch["visibility_mask"][t].numpy()
                pkl["frame_t"] = int(t)
                pkl["is_last"] = bool(t == T - 1)
                if not args.keep_3d:
                    # drop heavy 3D arrays we don't need for 2D bucket recall
                    for key in ("bboxes_3d", "gt_corners", "pred_corners_slot",
                                "pred_corners_free", "pred_free_logits",
                                "pred_boxes_2d_free", "pred_boxes_2d_slot"):
                        pkl.pop(key, None)
                records.append(_slim(pkl))

    out = args.out or os.path.join(
        REPO, "results", "bucket_dumps", f"{conf.experiment_name}__{args.frames}.pkl")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    meta = {
        "experiment": conf.experiment_name, "mode": conf.mode,
        "method": conf.method_name, "ckpt": args.ckpt, "frames": args.frames,
        "feature_model": getattr(conf, "feature_model", "?"),
        "n_videos": n, "n_frames": len(records),
    }
    with open(out, "wb") as f:
        pickle.dump({"meta": meta, "records": records}, f, protocol=4)
    size_mb = os.path.getsize(out) / 1e6
    print(f"[dump] wrote {len(records)} frames -> {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
