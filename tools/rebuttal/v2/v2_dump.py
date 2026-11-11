"""V2: test-split dumps of a V2 run (val-selected + final checkpoint), with 3-D fields for scoring. GPU.

Like tools/rebuttal/train_dump.py (same dataset, same build_pred_pkl records), plus, per frame and slot:
  bboxes_3d          the input corners in the canonical (world) frame (PredCls: annotated boxes; SGDet: detector boxes,
                     held boxes for held slots). For hidden slots these are what the copy reference would use; the V2
                     model never reads them (gate: tools/rebuttal/v2/v2_gate.py).
  pred_corners_slot  V2 box per slot, world frame: the decoded box for hidden slots, the input box otherwise
                     (rescore_3d.py reads it as the "refined" box source).
  gt_corners         annotated box (world frame) of the slot's object (for SGDet held slots, the object of that class).
  v2_hidden          hidden slots (valid, not visible) of this frame.
  geo_cos            cosine between g_hat and the geometry encoding of gt_corners (latent check; computed after the
                     forward with the model's own geometry encoder; diagnostics only).
Output: <out_dir>/<exp>__<ckpt>__all3d.pkl, then the caller writes <exp>.dumped.

    python tools/rebuttal/v2/v2_dump.py --config configs/methods/<mode>/rebuttal3/<exp>.yaml --out_dir <dir>
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

try:
    import resource
    _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    _want = 65536 if _hard == resource.RLIM_INFINITY else min(_hard, 65536)
    if _soft != resource.RLIM_INFINITY and _soft < _want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (_want, _hard))
except (ImportError, ValueError, OSError):
    pass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for p in (REPO, os.path.join(REPO, "tools", "rebuttal")):
    if p not in sys.path:
        sys.path.insert(0, p)
from wsgg_base import load_wsgg_config                     # noqa: E402
from tools.reeval_test import load_checkpoint, build_pred_pkl  # noqa: E402
from lib.supervised.v2.trainers import build_v2_model, forward_v2  # noqa: E402
from lib.supervised.v2.geometry import to_camera, to_world  # noqa: E402

DROP = ("pred_corners_free", "pred_free_logits", "pred_boxes_2d_free", "pred_boxes_2d_slot")


def _slim(pkl):
    out = {}
    for k, v in pkl.items():
        if isinstance(v, np.ndarray) and v.dtype == np.float64:
            v = v.astype(np.float32)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", default=["best", "checkpoint_19"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--video_list", default=None)
    ap.add_argument("--expect_videos", type=int, default=1511)
    ap.add_argument("--expect_frames", type=int, default=48834)
    args = ap.parse_args()
    sys.argv = [sys.argv[0], "--config", args.config]
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
    print(f"[v2_dump] {conf.experiment_name} mode={conf.mode} method={conf.method_name} videos={len(ds)}", flush=True)
    models, paths, epochs = [], [], []
    for c in args.ckpts:
        m = build_v2_model(conf, ds, device)
        paths.append(load_checkpoint(m, conf, c, device))
        epochs.append(torch.load(paths[-1], map_location="cpu").get("epoch", None))
        m.eval()
        models.append(m)
    records = [[] for _ in models]
    t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(dl, desc="Dumping", mininterval=30):
            T = int(batch["T"])
            if T <= 0:
                continue
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            pose = b.get("camera_poses")
            cam_frame = conf.mode == "sgdet" and pose is not None
            corners_w = to_world(b["corners"].float(), pose.float()) if cam_frame else b["corners"].float()
            for mi, m in enumerate(models):
                pred = forward_v2(m, conf, b)
                v2 = pred["v2"]
                H = v2["H"]
                merged = torch.where(H[..., None, None], v2["corners_pred_world"].float(), corners_w)
                gt_w = b["gt_corners"].float()
                gt_in = to_camera(gt_w, pose.float()) if cam_frame else gt_w
                g_gt, _ = m.global_structural_encoder(gt_in, b["valid_mask"].bool())
                cos = torch.nn.functional.cosine_similarity(v2["g_hat"].float(), g_gt.float(), dim=-1)
                for t in range(T):
                    pkl = build_pred_pkl(batch, pred, t, conf.mode)
                    for key in DROP:
                        pkl.pop(key, None)
                    pkl["visibility_mask"] = batch["visibility_mask"][t].numpy()
                    pkl["frame_t"] = int(t)
                    pkl["is_last"] = bool(t == T - 1)
                    pkl["has_cam_pose"] = pose is not None
                    pkl["bboxes_3d"] = corners_w[t].cpu().numpy()
                    pkl["pred_corners_slot"] = merged[t].cpu().numpy()
                    pkl["gt_corners"] = gt_w[t].cpu().numpy()
                    pkl["v2_hidden"] = H[t].cpu().numpy()
                    pkl["geo_cos"] = cos[t].cpu().numpy()
                    pkl["v2_scale"] = float(v2["scale"])
                    records[mi].append(_slim(pkl))
    os.makedirs(args.out_dir, exist_ok=True)
    n_vid = len({r["video_id"] for r in records[0]}) if records and records[0] else 0
    for c, path, ep, recs in zip(args.ckpts, paths, epochs, records):
        out = os.path.join(args.out_dir, f"{conf.experiment_name}__{c}__all3d.pkl")
        meta = {"experiment": conf.experiment_name, "mode": conf.mode, "method": conf.method_name, "version": "V2",
                "ckpt": c, "ckpt_path": path, "ckpt_epoch": ep, "frames": "all", "keep_3d": True,
                "feature_model": getattr(conf, "feature_model", "?"), "n_videos": n_vid, "n_frames": len(recs),
                "slot_alignment": getattr(ds, "_slot_alignment", None), "inventory": getattr(ds, "_inventory", None),
                "visibility_from_annotation": getattr(ds, "_visibility_from_annotation", None),
                "seconds": round(time.time() - t0, 1)}
        with open(out + ".tmp", "wb") as f:
            pickle.dump({"meta": meta, "records": recs}, f, protocol=4)
        os.replace(out + ".tmp", out)
        with open(out.replace(".pkl", ".meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print(f"[v2_dump] wrote {len(recs)} frames ({n_vid} videos) -> {out}", flush=True)
    if vids is None and (n_vid != args.expect_videos or len(records[0]) != args.expect_frames):
        raise SystemExit(f"[v2_dump] count check FAILED: {n_vid} videos / {len(records[0])} frames")


if __name__ == "__main__":
    main()
