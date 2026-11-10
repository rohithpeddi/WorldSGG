"""TRAIN: dump several checkpoints of one rebuttal run on the 1,511-video test split in one data pass.

Same records as ``tools/dump_predictions.py`` (``--frames all``, 3-D arrays dropped): it reuses that tool's
helpers from ``tools/reeval_test.py`` (``build_model``, ``load_checkpoint``, ``forward_all_frames``,
``build_pred_pkl``) unchanged. The one difference is the dataset: it is built by the trainer's own
``_make_dataset("test")`` hook, so every dataset option in the run's config (``slot_alignment``,
``visibility_from_annotation``, ``rescale_boxes_to_grid``) applies exactly as in training.

  PY tools/rebuttal/train_dump.py --config <cfg> --ckpts best checkpoint_19 --out_dir <dir>
  -> <dir>/<experiment>__<ckpt>__all.pkl  (+ <...>.meta.json)
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

# DataLoader workers hand tensors to the main process as file descriptors. At the default soft limit of 1,024 open
# files this runs out on cs94085 ("RuntimeError: received 0 items of ancdata"), so raise the soft limit.
try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _want = 65536 if _hard == resource.RLIM_INFINITY else min(_hard, 65536)
    if _soft != resource.RLIM_INFINITY and _soft < _want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_want, _hard))
except (ImportError, ValueError, OSError):
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from wsgg_base import load_wsgg_config                     # noqa: E402
from tools.reeval_test import (                            # noqa: E402
    build_model, forward_all_frames, load_checkpoint, build_pred_pkl,
)

DROP_3D = ("bboxes_3d", "gt_corners", "pred_corners_slot", "pred_corners_free", "pred_free_logits",
           "pred_boxes_2d_free", "pred_boxes_2d_slot")
# --keep_3d keeps what tools/rebuttal/rescore_3d.py reads: the input corners (bboxes_3d, canonical frame), the GT
# corners and WorldWise++'s refined per-slot corners; the free-query (DETR) outputs are still dropped.
KEEP_3D = ("bboxes_3d", "gt_corners", "pred_corners_slot")


def _slim(pkl):
    return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float64 else v)
            for k, v in pkl.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", default=["best", "checkpoint_19"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--video_list", default=None, help="restrict to these test videos (smoke); skips count check")
    ap.add_argument("--expect_videos", type=int, default=1511)
    ap.add_argument("--expect_frames", type=int, default=48834)
    ap.add_argument("--keep_3d", action="store_true",
                    help="keep the 3-D corners for rescore_3d.py; writes <exp>__<ckpt>__all3d.pkl")
    args = ap.parse_args()
    drop = tuple(k for k in DROP_3D if k not in KEEP_3D) if args.keep_3d else DROP_3D
    tag = "all3d" if args.keep_3d else "all"

    sys.argv = [sys.argv[0], "--config", args.config]      # load_wsgg_config also parses argv
    conf = load_wsgg_config(args.config)
    import train_wsgg_methods as tm
    trainer = tm.METHOD_MAP[conf.method_name](conf)
    from wsgg_base import read_video_list
    vids = read_video_list(args.video_list) if args.video_list else None
    ds = trainer._make_dataset("test") if vids is None else trainer._make_dataset("test", video_ids=vids)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from dataloader.world_ag_dataset import world_collate_fn
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=int(getattr(conf, "num_workers", 0)),
                    collate_fn=world_collate_fn)
    n = len(dl) if args.limit <= 0 else min(args.limit, len(dl))
    print(f"[train_dump] {conf.experiment_name} mode={conf.mode} method={conf.method_name} videos={len(ds)} "
          f"ckpts={args.ckpts} slot_alignment={getattr(ds, '_slot_alignment', '?')} "
          f"visibility_from_annotation={getattr(ds, '_visibility_from_annotation', '?')}", flush=True)

    models, paths, epochs = [], [], []
    for c in args.ckpts:
        m = build_model(conf, ds, device)
        paths.append(load_checkpoint(m, conf, c, device))
        epochs.append(torch.load(paths[-1], map_location="cpu").get("epoch", None))
        m.eval()
        models.append(m)

    records = [[] for _ in models]
    t0 = time.time()
    it = iter(dl)
    with torch.no_grad():
        for _ in tqdm(range(n), desc="Dumping", mininterval=30):
            batch = next(it)
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            T = int(batch["T"])
            if T <= 0:
                continue
            for mi, m in enumerate(models):
                pred = forward_all_frames(m, conf, b)
                for t in range(T):
                    pkl = build_pred_pkl(batch, pred, t, conf.mode)
                    pkl["visibility_mask"] = batch["visibility_mask"][t].numpy()
                    pkl["frame_t"] = int(t)
                    pkl["is_last"] = bool(t == T - 1)
                    if args.keep_3d:   # without a pose, bboxes_3d / pred_corners_slot stay in the camera frame
                        pkl["has_cam_pose"] = batch.get("camera_poses") is not None
                    for key in drop:
                        pkl.pop(key, None)
                    records[mi].append(_slim(pkl))

    os.makedirs(args.out_dir, exist_ok=True)
    n_vid = len({r["video_id"] for r in records[0]}) if records and records[0] else 0
    for c, path, ep, recs in zip(args.ckpts, paths, epochs, records):
        out = os.path.join(args.out_dir, f"{conf.experiment_name}__{c}__{tag}.pkl")
        meta = {"experiment": conf.experiment_name, "mode": conf.mode, "method": conf.method_name,
                "ckpt": c, "ckpt_path": path, "ckpt_epoch": ep, "frames": "all", "keep_3d": bool(args.keep_3d),
                "feature_model": getattr(conf, "feature_model", "?"), "n_videos": n_vid, "n_frames": len(recs),
                "slot_alignment": getattr(ds, "_slot_alignment", None),
                "visibility_from_annotation": getattr(ds, "_visibility_from_annotation", None),
                "seconds": round(time.time() - t0, 1)}
        with open(out, "wb") as f:
            pickle.dump({"meta": meta, "records": recs}, f, protocol=4)
        with open(out.replace(".pkl", ".meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[train_dump] wrote {len(recs)} frames ({n_vid} videos) -> {out}", flush=True)
    if args.limit <= 0 and vids is None and (n_vid != args.expect_videos or len(records[0]) != args.expect_frames):
        raise SystemExit(f"[train_dump] count check FAILED: {n_vid} videos / {len(records[0])} frames, "
                         f"expected {args.expect_videos} / {args.expect_frames}")


if __name__ == "__main__":
    main()
