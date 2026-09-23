"""
Re-evaluate a trained WSGG checkpoint on the test set — all frames, with a
per-predicate mean-recall diagnostic.

Why this exists: the training-time evaluation (train_wsgg_base) and the
standalone tester (test_wsgg_base) both score only the LAST frame of each
video. For mean-recall over 26 predicates that is a tiny, biased sample —
especially in sgdet (detector boxes, higher variance), where rare contacting
predicates may barely appear in last frames, making mR unstable and, at K,
misleading. This script scores EVERY frame (the model already produces
all-frame predictions in one forward pass) and reports:

  * wc / nc  R@{10,20,50,100} and mR@{...}
  * a per-predicate table: GT sample count + with-constraint recall @ K
    (so you can see which predicates drag mR, and whether last-frame-only
     was starving some classes of samples)

--frames both runs the all-frame and last-frame schemes side by side in one
pass, so the discrepancy (if any) is explicit.

Usage (on the training server):
    python tools/reeval_test.py --config configs/methods/sgdet/worldwise_v2e_sgdet_dinov3l.yaml
    python tools/reeval_test.py --config <cfg> --ckpt best --frames both
    python tools/reeval_test.py --config <cfg> --ckpt checkpoint_19 --k 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from wsgg_base import load_wsgg_config              # noqa: E402
from lib.supervised.evaluation_recall import (       # noqa: E402
    BasicSceneGraphEvaluator, evaluate_wsgg_video,
)


# --------------------------------------------------------------------------
# Model construction (mirrors train/test entry points, all frames kept)
# --------------------------------------------------------------------------
def build_model(conf, test_dataset, device):
    nobj = len(test_dataset.object_classes)
    natt = len(test_dataset.attention_relationships)
    nspa = len(test_dataset.spatial_relationships)
    ncon = len(test_dataset.contacting_relationships)
    m = conf.method_name
    if m == "w_sttran":
        from lib.supervised.baselines.w_sttran.w_sttran import WSTTran as C
    elif m == "w_sttran_pp":
        from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP as C
    elif m == "w_dsgdetr":
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr as C
    elif m == "w_dsgdetr_pp":
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP as C
    elif m == "w_usg":
        from lib.supervised.baselines.w_usg.w_usg import WUSG as C
    elif m == "worldwise":
        from lib.supervised.worldwise.worldwise import WorldWise as C
    elif m in ("worldwise_plus", "worldformer_c1"):
        from lib.supervised.worldwise_plus import WorldWisePlus as C
    elif m == "worldwise_pp":
        from lib.supervised.worldwise_pp import WorldWisePP as C
    else:
        raise ValueError(f"Unknown method_name: {m}")
    return C(conf, nobj, natt, nspa, ncon).to(device)


def make_test_dataset(conf):
    """Test-split dataset for the config's method (WorldAGGrid for WorldWise++,
    which also needs the token-grid cache; plain WorldAG otherwise)."""
    from wsgg_base import annot_dir_for
    kw = dict(phase="test", data_path=conf.data_path, mode=conf.mode,
              feature_model=getattr(conf, "feature_model", "dinov2b"),
              include_invisible=getattr(conf, "include_invisible", True),
              max_objects=getattr(conf, "max_objects", 64),
              annot_dir_name=annot_dir_for(conf, "test"))
    if conf.method_name == "worldwise_pp":
        from lib.supervised.worldwise_pp.dataset import WorldAGGrid
        return WorldAGGrid(grid_cache_root=conf.grid_cache_root,
                           allow_missing_grids=getattr(conf, "grid_cache_allow_missing", False), **kw)
    from dataloader.world_ag_dataset import WorldAG
    return WorldAG(**kw)


def forward_all_frames(model, conf, b):
    """Full-sequence forward — returns dict of (T, K, C) distributions."""
    kw = dict(
        visual_features_seq=b["visual_features"], corners_seq=b["corners"],
        valid_mask_seq=b["valid_mask"], visibility_mask_seq=b["visibility_mask"],
        person_idx_seq=b["person_idx"], object_idx_seq=b["object_idx"],
        pair_valid=b["pair_valid"], camera_pose_seq=b.get("camera_poses"),
        union_features_seq=b.get("union_features"),
    )
    # GT node labels feed the text pathway in predcls only (task input)
    kw["node_labels_seq"] = b.get("object_classes") if conf.mode == "predcls" else None
    if conf.method_name in ("worldwise", "worldwise_plus", "worldformer_c1", "worldwise_pp"):
        kw["p_mask_visible"] = 0.0
    if conf.method_name == "worldwise_pp":
        kw.update(grid_dino_seq=b["grid_dino"], grid_pi3_seq=b["grid_pi3"],
                  image_hw=b["image_hw"], bboxes_2d_seq=b.get("bboxes_2d"))
    return model(**kw)


# --------------------------------------------------------------------------
# Checkpoint loading (best_model.pth or checkpoint_<N>/checkpoint_state.pth)
# --------------------------------------------------------------------------
def load_checkpoint(model, conf, ckpt, device):
    exp_dir = os.path.join(conf.save_path, conf.experiment_name)
    if os.path.isfile(ckpt):
        path = ckpt
    elif ckpt in ("best", "best_model"):
        path = os.path.join(exp_dir, "best_model.pth")
    else:
        path = os.path.join(exp_dir, ckpt, "checkpoint_state.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=device)
    sd = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load] {len(missing)} missing, {len(unexpected)} unexpected keys "
              f"(strict=False)")
    print(f"  [load] {path} (epoch {state.get('epoch', '?')})")
    return path


# --------------------------------------------------------------------------
# Evaluator factory (mirrors wsgg_base._init_evaluators)
# --------------------------------------------------------------------------
def make_evaluator(conf, test_dataset, constraint):
    from dataloader.world_ag_dataset import (
        ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS, CONTACTING_RELATIONSHIPS,
    )
    all_pred = (list(ATTENTION_RELATIONSHIPS) + list(SPATIAL_RELATIONSHIPS)
                + list(CONTACTING_RELATIONSHIPS))
    return BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=test_dataset.object_classes,
        AG_all_predicates=all_pred,
        AG_attention_predicates=list(ATTENTION_RELATIONSHIPS),
        AG_spatial_predicates=list(SPATIAL_RELATIONSHIPS),
        AG_contacting_predicates=list(CONTACTING_RELATIONSHIPS),
        iou_threshold=0.5,
        save_file=os.devnull,
        constraint=constraint,
    )


def _cam_to_world(corners, cam_pose_t):
    """(M,8,3) camera-frame corners -> the canonical ("final") frame.

    ``camera_poses[t]`` is the 4x4 camera->final pose, so this is R @ c + t,
    the same transform ``build_pred_pkl`` applies to the detector input corners
    when it writes ``bboxes_3d``.  ``None`` pose leaves the corners untouched.
    """
    if cam_pose_t is None:
        return corners
    Tm = cam_pose_t.numpy() if hasattr(cam_pose_t, "numpy") else np.asarray(cam_pose_t)
    return np.einsum('ij,nkj->nki', Tm[:3, :3], corners) + Tm[:3, 3]


def build_pred_pkl(batch, pred, t, mode):
    """Per-frame prediction dict for evaluate_wsgg_video (frame index t).

    Two 3-D fields are written and they must never be confused:

    ``bboxes_3d``          the *input* corners of frame t (in sgdet the detector's
                           ``boxes_3d``, in predcls the annotation ``corners_final``),
                           mapped to the canonical frame.  This is what the model was
                           handed, not what it said.
    ``pred_corners_slot``  WorldWise++'s per-slot 3-D refinement, i.e. those same
                           input corners plus the learned ``slot_corner_head`` delta,
                           in the same canonical frame.
    ``pred_corners_free``  WorldWise++'s free-query (DETR) 3-D detections, which are
                           produced in the camera frame by pinhole back-projection and
                           are therefore always mapped to canonical here.

    Only WorldWise++ returns a ``det`` block; WorldWise and WorldWise+ have no 3-D
    head at all, so the two ``pred_corners_*`` keys are simply absent for them and no
    caller may fall back to ``bboxes_3d`` in their place.
    """
    pkl = {
        "video_id": batch["video_id"],
        "attention_distribution": pred["attention_distribution"][t].cpu().numpy(),
        "spatial_distribution": pred["spatial_distribution"][t].cpu().numpy(),
        "contacting_distribution": pred["contacting_distribution"][t].cpu().numpy(),
        "gt_attention": batch["gt_attention"][t].numpy(),
        "gt_spatial": batch["gt_spatial"][t].numpy(),
        "gt_contacting": batch["gt_contacting"][t].numpy(),
        "pair_valid": batch["pair_valid"][t].numpy(),
        "person_idx": batch["person_idx"][t].numpy(),
        "object_idx": batch["object_idx"][t].numpy(),
        "object_classes": batch["object_classes"][t].numpy(),
        "bboxes_2d": batch["bboxes_2d"][t].numpy(),
        "valid_mask": batch["valid_mask"][t].numpy(),
    }
    cam_pose = batch.get("camera_poses")
    cam_pose_t = cam_pose[t] if cam_pose is not None else None
    if mode == "sgdet":
        pkl["pred_labels"] = batch["object_classes"][t].numpy()
        pkl["pred_scores"] = np.ones(batch["object_classes"][t].shape[0], dtype=np.float32)
        pkl["gt_bboxes_2d"] = batch["gt_bboxes_2d"][t].numpy()
        pkl["gt_corners"] = batch["gt_corners"][t].numpy()
        corners_raw = batch.get("corners")
        if corners_raw is not None:
            # input corners (detector boxes_3d) live in the camera frame in sgdet
            pkl["bboxes_3d"] = _cam_to_world(corners_raw[t].numpy(), cam_pose_t)

    # --- model-predicted 3-D, WorldWise++ only -------------------------------
    det = pred.get("det") if isinstance(pred, dict) else None
    if det is not None:
        # slot_corners = the input corners_seq + slot_corner_head delta, so they
        # sit in whatever frame corners_seq is in: camera in sgdet (map it, exactly
        # as bboxes_3d above), already canonical in predcls (leave it).
        slot = det["slot_corners"][t].detach().cpu().numpy().astype(np.float32)
        pkl["pred_corners_slot"] = _cam_to_world(slot, cam_pose_t) if mode == "sgdet" else slot
        # free-query corners come out of _compute_3d_corners in the camera frame
        # in both modes, so they are always mapped.
        free = det["corners"][t].detach().cpu().numpy().astype(np.float32)
        pkl["pred_corners_free"] = _cam_to_world(free, cam_pose_t)
        pkl["pred_free_logits"] = det["logits"][t].detach().cpu().float().numpy()
        pkl["pred_boxes_2d_free"] = det["boxes_xyxy"][t].detach().cpu().float().numpy()
        pkl["pred_boxes_2d_slot"] = det["slot_boxes"][t].detach().cpu().float().numpy()
        pkl["pred_3d_frame"] = "canonical" if cam_pose_t is not None else "camera"
        pkl["pred_3d_source"] = ("slot_corners = input corners + slot_corner_head delta; "
                                 "free = free-query detections")
    return pkl


def stats_block(ev, ks):
    s = ev.fetch_stats_json()
    return {
        "R": {k: round(s["recall"].get(k, 0.0), 6) for k in ks},
        "mR": {k: round(s["mean_recall"].get(k, 0.0), 6) for k in ks},
        "hR": {k: round(s["harmonic_mean_recall"].get(k, 0.0), 6) for k in ks},
    }


def per_predicate_diag(ev, k):
    """(predicate, n_samples, mean_recall) — n_samples = #frames where the
    predicate appeared in GT (count>0). Zero-sample predicates still divide
    the mR denominator (num_rel), so they reveal deflation."""
    collect = ev.result_dict[ev.mode + "_mean_recall_collect"][k]
    out = []
    for i, vi in enumerate(collect):
        name = ev.AG_all_predicates[i]
        out.append((name, len(vi), float(np.mean(vi)) if vi else 0.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="best",
                    help="'best' (best_model.pth), a checkpoint dir name "
                         "(checkpoint_19), or an explicit .pth path")
    ap.add_argument("--frames", default="all", choices=["all", "last", "both"],
                    help="which frames to score (default all — the fix)")
    ap.add_argument("--k", type=int, default=20, help="K for the per-predicate table")
    ap.add_argument("--out", default=None, help="output JSON (default results/reeval_<exp>.json)")
    ap.add_argument("--limit", type=int, default=0, help="score only N videos (debug)")
    args = ap.parse_args()

    conf = load_wsgg_config(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ks = [10, 20, 50, 100]

    from dataloader.world_ag_dataset import world_collate_fn
    from torch.utils.data import DataLoader
    ds = make_test_dataset(conf)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=world_collate_fn)
    print(f"Test videos: {len(ds)} | mode={conf.mode} | method={conf.method_name} "
          f"| backbone={getattr(conf, 'feature_model', '?')}")

    model = build_model(conf, ds, device)
    load_checkpoint(model, conf, args.ckpt, device)
    model.eval()

    schemes = ["all", "last"] if args.frames == "both" else [args.frames]
    evals = {s: {"wc": make_evaluator(conf, ds, "with"),
                 "nc": make_evaluator(conf, ds, "no")} for s in schemes}

    n = len(dl) if args.limit <= 0 else min(args.limit, len(dl))
    it = iter(dl)
    with torch.no_grad():
        for _ in tqdm(range(n), desc="Re-evaluating"):
            batch = next(it)
            b = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
            pred = forward_all_frames(model, conf, b)
            T = int(batch["T"])
            if T <= 0:
                continue
            frame_sets = {"all": range(T), "last": [T - 1]}
            for s in schemes:
                for t in frame_sets[s]:
                    pkl = build_pred_pkl(batch, pred, t, conf.mode)
                    evaluate_wsgg_video(pkl, evals[s]["wc"], mode=conf.mode, verbose=False)
                    evaluate_wsgg_video(pkl, evals[s]["nc"], mode=conf.mode, verbose=False)

    # ---- report ----
    result = {"experiment": conf.experiment_name, "mode": conf.mode,
              "method": conf.method_name, "ckpt": args.ckpt, "schemes": {}}
    for s in schemes:
        result["schemes"][s] = {
            "wc": stats_block(evals[s]["wc"], ks),
            "nc": stats_block(evals[s]["nc"], ks),
        }
        wc, nc = result["schemes"][s]["wc"], result["schemes"][s]["nc"]
        print(f"\n=== scheme: {s} frames ===")
        print(f"  wc  R@20={wc['R'][20]:.4f}  mR@20={wc['mR'][20]:.4f}  hR@20={wc['hR'][20]:.4f}"
              f"   |  mR@10={wc['mR'][10]:.4f} mR@50={wc['mR'][50]:.4f}")
        print(f"  nc  R@20={nc['R'][20]:.4f}  mR@20={nc['mR'][20]:.4f}  hR@20={nc['hR'][20]:.4f}")

    # per-predicate diagnostic (with-constraint, first scheme)
    s0 = schemes[0]
    diag = per_predicate_diag(evals[s0]["wc"], args.k)
    result["per_predicate_wc"] = {"k": args.k, "scheme": s0,
                                  "rows": [{"predicate": nm, "n_samples": ns,
                                            "recall": round(r, 6)} for nm, ns, r in diag]}
    n_absent = sum(1 for _, ns, _ in diag if ns == 0)
    print(f"\n=== per-predicate (wc, scheme={s0}, K={args.k}) ===")
    print(f"  {'predicate':<22} {'n_samples':>9} {'recall':>8}")
    for nm, ns, r in diag:
        flag = "  <- ZERO SAMPLES (deflates mR)" if ns == 0 else ""
        print(f"  {nm:<22} {ns:>9} {r:>8.4f}{flag}")
    print(f"\n  {n_absent}/{len(diag)} predicates never appeared in GT for scheme "
          f"'{s0}' — each still divides the mR denominator (num_rel={len(diag)}).")

    out = args.out or os.path.join(REPO, "results", f"reeval_{conf.experiment_name}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
