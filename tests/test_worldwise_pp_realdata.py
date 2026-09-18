"""
Real-data forward/backward check for WorldWise++ (server, GPU):

  * loads N test videos through WorldAGGrid (worldbbox annotations + the pp grid cache)
    for predcls and sgdet, checks the cache geometry against the feature PKLs
    (image_hw == feature target_size, boxes inside the image, grid rows in frame order);
  * runs WorldWisePP -> WorldWisePPLoss -> backward under bf16 autocast (as the trainer
    does), reporting seconds / video and peak GPU memory;
  * saves an untrained checkpoint per mode as <save_dir>/<experiment>/best_model.pth so
    tools/reeval_test.py / tools/dump_predictions.py can be exercised end to end:

    CUDA_VISIBLE_DEVICES=2 python tests/test_worldwise_pp_realdata.py --data_path /data/rohith/ag \
        --save_dir /data3/rohith/ag/runs/worldwise_pp_smoke
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def load_conf(stem):
    mode = stem.rsplit("_", 1)[-1]
    with open(os.path.join(REPO, "configs", "methods", mode, f"{stem}.yaml"), encoding="utf-8") as f:
        return SimpleNamespace(**yaml.safe_load(f))


def make_ds(conf, allow_missing):
    from lib.supervised.worldwise_pp.dataset import WorldAGGrid
    return WorldAGGrid(phase="test", data_path=conf.data_path, mode=conf.mode, feature_model=conf.feature_model,
                       include_invisible=getattr(conf, "include_invisible", True),
                       max_objects=getattr(conf, "max_objects", 64), annot_dir_name=conf.test_annot_dir,
                       grid_cache_root=conf.grid_cache_root, allow_missing_grids=allow_missing)


def check_geometry(ds, item):
    """Cache target_size must equal the feature PKL target size (W, H) of every frame,
    and the feature boxes must lie inside that image."""
    H, W = item["image_hw"]
    feat = ds._load_feature_pkl(item["video_id"])["frames"]
    for f in item["frame_names"]:
        ts = feat[f].get("target_size")
        if ts is not None:
            assert (int(ts[0]), int(ts[1])) == (W, H), (item["video_id"], f, ts, (W, H))
    bb = item["bboxes_2d"][item["valid_mask"]]
    assert bb.numel() == 0 or (bb[:, 2].max() <= W + 1 and bb[:, 3].max() <= H + 1), (bb.max(0).values, (W, H))
    hp, wp = item["grid_hw"]
    assert item["grid_dino"].shape == (item["T"], hp, wp, 256), item["grid_dino"].shape
    return H, W, hp, wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--videos", nargs="+", default=["015XE", "02DPI", "03PRW"])
    ap.add_argument("--modes", nargs="+", default=["predcls", "sgdet"])
    ap.add_argument("--save_dir", default="/data3/rohith/ag/runs/worldwise_pp_smoke")
    ap.add_argument("--allow_missing", action="store_true", help="drop videos without a grid cache")
    ap.add_argument("--steps", type=int, default=2, help="optimizer steps per video (loss must not blow up)")
    args = ap.parse_args()

    from lib.supervised.worldwise_pp import WorldWisePP, WorldWisePPLoss
    from dataloader.world_ag_dataset import (ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS,
                                             CONTACTING_RELATIONSHIPS)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = dev.type == "cuda"
    print(f"device={dev} amp={use_amp}")

    for mode in args.modes:
        stem = f"worldwise_pp_dinov3_{mode}"
        conf = load_conf(stem)
        conf.data_path = args.data_path
        ds = make_ds(conf, args.allow_missing)
        idx = {v: i for i, v in enumerate(ds.video_list)}
        vids = [v for v in args.videos if v in idx]
        assert vids, f"none of {args.videos} in the {mode} test set / cache"
        model = WorldWisePP(conf, num_object_classes=len(ds.object_classes),
                            attention_class_num=len(ATTENTION_RELATIONSHIPS),
                            spatial_class_num=len(SPATIAL_RELATIONSHIPS),
                            contact_class_num=len(CONTACTING_RELATIONSHIPS)).to(dev)
        crit = WorldWisePPLoss(num_object_classes=len(ds.object_classes), lambda_det=conf.lambda_det,
                               lambda_slot_box=conf.lambda_slot_box, det_one_to_many_iou=conf.det_one_to_many_iou,
                               lambda_vlm=conf.lambda_vlm, lambda_recon=conf.lambda_reconstruction,
                               lambda_recon_dominance=conf.lambda_recon_dominance,
                               p_simulate_unseen=conf.p_simulate_unseen, label_smoothing=conf.label_smoothing_vlm,
                               mode=mode, use_logit_adjustment=conf.use_logit_adjustment,
                               logit_adjustment_tau=conf.logit_adjustment_tau,
                               predicate_priors_path=conf.predicate_priors_path, data_path=conf.data_path).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)
        model.train()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[{stem}] {len(ds)} test videos, model {n_par / 1e6:.2f}M params")
        for v in vids:
            t_load = time.time()
            item = ds[idx[v]]
            H, W, hp, wp = check_geometry(ds, item)
            t_load = time.time() - t_load
            b = {k: (x.to(dev) if torch.is_tensor(x) else x) for k, x in item.items()}
            labels = b["object_classes"] if mode == "predcls" else None
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            t0 = time.time()
            hist = []
            for step in range(args.steps):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    out = model(visual_features_seq=b["visual_features"], corners_seq=b["corners"],
                                valid_mask_seq=b["valid_mask"], visibility_mask_seq=b["visibility_mask"],
                                person_idx_seq=b["person_idx"], object_idx_seq=b["object_idx"],
                                pair_valid=b["pair_valid"], p_mask_visible=conf.p_mask_visible,
                                camera_pose_seq=b.get("camera_poses"), node_labels_seq=labels,
                                gt_contacting_seq=b["gt_contacting"], grid_dino_seq=b["grid_dino"],
                                grid_pi3_seq=b["grid_pi3"], image_hw=b["image_hw"], bboxes_2d_seq=b["bboxes_2d"])
                    losses = crit(predictions=out, gt_attention=b["gt_attention"], gt_spatial=b["gt_spatial"],
                                  gt_contacting=b["gt_contacting"], pair_valid=b["pair_valid"],
                                  visibility_mask=b["visibility_mask"], person_idx=b["person_idx"],
                                  object_idx=b["object_idx"], valid_mask=b["valid_mask"], corners=b["corners"],
                                  gt_node_labels=b["object_classes"], gt_bboxes_2d=b["gt_bboxes_2d"],
                                  gt_corners=b["gt_corners"], camera_poses=b.get("camera_poses"))
                loss = losses["total"]
                assert torch.isfinite(loss), {k: float(x) for k, x in losses.items()}
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf.grad_clip)
                opt.step()
                hist.append(float(loss))
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) / args.steps
            mem = torch.cuda.max_memory_allocated() / 1e9 if dev.type == "cuda" else 0.0
            print(f"  {v}: T={item['T']} N={item['N_max']} K={item['K_max']} img={H}x{W} grid={hp}x{wp} "
                  f"load={t_load:.2f}s fwd+bwd={dt:.2f}s/step peak={mem:.2f}GB total={hist} "
                  f"det_n_matched={float(losses.get('det_n_matched', -1)):.0f} "
                  f"det_cls={float(losses.get('det_cls', -1)):.3f} slot_box={float(losses.get('slot_box', -1)):.4f} "
                  f"gate={model.gate_summary()}")
        exp_dir = os.path.join(args.save_dir, stem)
        os.makedirs(exp_dir, exist_ok=True)
        path = os.path.join(exp_dir, "best_model.pth")
        torch.save({"epoch": 0, "score": 0.0, "model_state_dict": model.state_dict()}, path)
        print(f"  saved untrained checkpoint -> {path}")
        del model, crit, opt
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    print("WorldWise++ real-data check OK")


if __name__ == "__main__":
    main()
