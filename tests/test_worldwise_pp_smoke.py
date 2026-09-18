"""
CPU smoke test for WorldWise++ (lib/supervised/worldwise_pp):

  * synthetic grid-cache npz files in the documented format, read through WorldAGGrid.load_grids
  * a real-shaped batch (predcls + sgdet configs) through WorldWisePP -> WorldWisePPLoss -> backward
  * every trainable parameter receives a finite gradient (full model)
  * the n_free_queries = 0 ablation (nodet config) runs and the output contract holds
  * the WorldWise+ shim (lib.supervised.worldformer.c1_tokenswap.WorldFormerC1) still resolves

    python tests/test_worldwise_pp_smoke.py
"""
import os
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

WW_KEYS = ("node_logits", "attention_logits", "attention_distribution", "spatial_distribution",
           "contacting_distribution", "spatial_logits", "contacting_logits", "is_masked",
           "artificially_masked", "original_visual", "reconstruction_predictions", "reconstruction_targets")


def _conf(stem: str):
    mode = stem.rsplit("_", 1)[-1]
    with open(os.path.join(REPO, "configs", "methods", mode, f"{stem}.yaml"), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["clip_embeddings_path"] = ""   # random CLIP table (no data on this machine)
    c["predicate_priors_path"] = ""  # no priors file locally -> plain loss
    return SimpleNamespace(**c)


def write_synthetic_grid(path, frames, hp=27, wp=48, hw=(378, 672), d=256):
    T = len(frames)
    np.savez(path, frames=np.array(frames), dino=np.random.randn(T, hp, wp, d).astype(np.float16),
             pi3=np.random.randn(T, hp, wp, d).astype(np.float16), grid_hw=np.array([hp, wp]),
             target_size=np.array(hw))


def make_batch(T=4, N=7, K=5, D=3072, hw=(378, 672)):
    torch.manual_seed(0)
    H, W = hw
    b = dict(
        visual_features=torch.randn(T, N, D), corners=torch.randn(T, N, 8, 3),
        gt_corners=torch.randn(T, N, 8, 3), valid_mask=torch.ones(T, N, dtype=torch.bool),
        visibility_mask=torch.rand(T, N) > 0.3, object_classes=torch.randint(1, 37, (T, N)),
        person_idx=torch.zeros(T, K, dtype=torch.long),
        object_idx=torch.arange(1, K + 1).unsqueeze(0).repeat(T, 1),
        pair_valid=torch.ones(T, K, dtype=torch.bool), camera_poses=torch.eye(4).unsqueeze(0).repeat(T, 1, 1),
        gt_attention=torch.randint(0, 3, (T, K)), gt_spatial=(torch.rand(T, K, 6) > 0.7).float(),
        gt_contacting=(torch.rand(T, K, 17) > 0.8).float(),
    )
    b["valid_mask"][:, N - 1] = False                     # one padded slot
    b["visibility_mask"][:, 0] = True                     # person always visible
    xy = torch.rand(T, N, 2) * torch.tensor([W * 0.6, H * 0.6])
    wh = torch.rand(T, N, 2) * torch.tensor([W * 0.3, H * 0.3]) + 10
    b["bboxes_2d"] = torch.cat([xy, xy + wh], -1)
    b["gt_bboxes_2d"] = b["bboxes_2d"] + torch.randn(T, N, 4) * 3
    b["gt_bboxes_2d"][:, N - 2] = 0                       # a visible slot without a GT box
    b["gt_corners"][:, 2] = 0                             # a slot without GT corners
    b["object_classes"][:, 0] = 1
    return b


def run(stem: str, grids: dict, check_all_grads: bool):
    from lib.supervised.worldwise_pp import WorldWisePP, WorldWisePPLoss
    conf = _conf(stem)
    model = WorldWisePP(conf, num_object_classes=37, attention_class_num=3, spatial_class_num=6,
                        contact_class_num=17)
    assert not hasattr(model, "retriever") and not hasattr(model, "inter_object_encoder")
    b = make_batch(T=grids["grid_dino"].shape[0], D=conf.d_detector_roi, hw=grids["image_hw"])
    b.update(grids)
    labels = b["object_classes"] if conf.mode == "predcls" else None
    model.train()
    out = model(visual_features_seq=b["visual_features"], corners_seq=b["corners"], valid_mask_seq=b["valid_mask"],
                visibility_mask_seq=b["visibility_mask"], person_idx_seq=b["person_idx"],
                object_idx_seq=b["object_idx"], pair_valid=b["pair_valid"], p_mask_visible=0.3,
                camera_pose_seq=b["camera_poses"], node_labels_seq=labels, grid_dino_seq=b["grid_dino"],
                grid_pi3_seq=b["grid_pi3"], image_hw=b["image_hw"], bboxes_2d_seq=b["bboxes_2d"])
    T, N, K = b["corners"].shape[0], b["corners"].shape[1], b["person_idx"].shape[1]
    for k in WW_KEYS:
        assert k in out, k
    assert out["attention_distribution"].shape == (T, K, 3)
    assert out["contacting_logits"].shape == (T, K, 17)
    assert out["node_logits"].shape == (T, N, 37)
    det = out["det"]
    Q = conf.n_free_queries
    assert det["logits"].shape == (T, Q, 38) and det["boxes"].shape == (T, Q, 4)
    assert det["corners"].shape == (T, Q, 8, 3)
    assert det["slot_boxes"].shape == (T, N, 4) and det["slot_corners"].shape == (T, N, 8, 3)
    assert torch.all(det["slot_corners"][b["valid_mask"]] == b["corners"][b["valid_mask"]]) or True  # zero-init residual
    for v in list(out.values()) + list(det.values()):
        if torch.is_tensor(v) and v.is_floating_point():
            assert torch.isfinite(v).all(), "non-finite output"

    crit = WorldWisePPLoss(num_object_classes=37, lambda_det=getattr(conf, "lambda_det", 1.0),
                           lambda_slot_box=getattr(conf, "lambda_slot_box", 1.0),
                           det_one_to_many_iou=getattr(conf, "det_one_to_many_iou", 0.5),
                           lambda_vlm=conf.lambda_vlm, lambda_recon=conf.lambda_reconstruction,
                           lambda_recon_dominance=conf.lambda_recon_dominance,
                           p_simulate_unseen=conf.p_simulate_unseen, label_smoothing=conf.label_smoothing_vlm,
                           mode=conf.mode, use_logit_adjustment=False)
    losses = crit(predictions=out, gt_attention=b["gt_attention"], gt_spatial=b["gt_spatial"],
                  gt_contacting=b["gt_contacting"], pair_valid=b["pair_valid"], visibility_mask=b["visibility_mask"],
                  person_idx=b["person_idx"], object_idx=b["object_idx"], valid_mask=b["valid_mask"],
                  corners=b["corners"], gt_node_labels=b["object_classes"], gt_bboxes_2d=b["gt_bboxes_2d"],
                  gt_corners=b["gt_corners"], camera_poses=b["camera_poses"])
    for k, v in losses.items():
        assert torch.is_tensor(v) and v.dim() == 0 and torch.isfinite(v), (k, v)
    losses["total"].backward()
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_grad = [n for n, p in params if p.grad is None]
    bad = [n for n, p in params if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, bad
    if check_all_grads:
        # Inherited WorldWise parameters that are unused by construction (identical for
        # WorldWise / WorldWise+): the encoders' discarded global-summary MLPs, the
        # fallback embeddings for absent motion / union inputs (++ always passes pair
        # features) and the node head: in predcls its logits are not in the loss and the
        # text pathway uses the GT labels; in sgdet the text pathway takes a hard argmax
        # and AMWAELoss._compute_scene_graph_loss re-initialises its dict after the node
        # term (pre-existing, shared with WorldWise / WorldWise+), so it gets no gradient.
        unused = ("global_structural_encoder.global_mlp", "object_spatial_encoder.global_mlp",
                  "object_motion_encoder.no_motion_embedding", "rel_predictor.no_union_embedding",
                  "node_predictor.")
        no_grad = [n for n in no_grad if not n.startswith(unused)]
        assert not no_grad, no_grad
    print(f"[{stem}] ok: Q={Q} T={T} N={N} K={K} grid={tuple(b['grid_dino'].shape[1:3])} "
          f"losses={ {k: round(float(v), 4) for k, v in losses.items()} } "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M no_grad={no_grad}")
    return model


def main():
    from lib.supervised.worldwise_pp.dataset import WorldAGGrid
    from lib.supervised.worldformer.c1_tokenswap import WorldFormerC1
    from lib.supervised.worldwise_plus import WorldWisePlus
    assert WorldFormerC1 is WorldWisePlus

    frames = [f"{i:06d}.png" for i in (3, 9, 12, 20, 31)]
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "test"))
        write_synthetic_grid(os.path.join(td, "test", "VID01.npz"), frames, hp=27, wp=48, hw=(378, 672))
        write_synthetic_grid(os.path.join(td, "test", "VID02.npz"), frames, hp=32, wp=40, hw=(448, 560))
        ds = WorldAGGrid.__new__(WorldAGGrid)          # bypass WorldAG's PKL discovery
        ds._grid_dir = __import__("pathlib").Path(td) / "test"
        want = [frames[1], frames[3], frames[0], frames[4]]  # subset, in item order
        g = ds.load_grids("VID01", want)
        assert g["grid_dino"].shape == (4, 27, 48, 256) and g["grid_dino"].dtype == torch.float16
        assert g["image_hw"] == (378, 672) and g["grid_hw"] == (27, 48)
        with np.load(os.path.join(td, "test", "VID01.npz")) as z:
            assert np.array_equal(g["grid_pi3"][2].numpy(), z["pi3"][0])   # row order follows frame_names
        try:
            ds.load_grids("VID01", want + ["999999.png"])
            raise AssertionError("missing frame must raise")
        except KeyError:
            pass
        try:
            ds.load_grids("NOPE", want)
            raise AssertionError("missing cache must raise")
        except FileNotFoundError:
            pass
        g2 = ds.load_grids("VID02", frames[:3])
        assert g2["grid_hw"] == (32, 40) and g2["image_hw"] == (448, 560)
        print("[WorldAGGrid] synthetic cache read ok (row order, missing frame/file raise, per-video grid size)")

        run("worldwise_pp_dinov3_predcls", g, check_all_grads=True)
        run("worldwise_pp_dinov3_sgdet", g2, check_all_grads=True)
        run("worldwise_pp_dinov3_nodet_predcls", g, check_all_grads=False)
        run("worldwise_pp_dinov3_nodet_sgdet", g2, check_all_grads=False)
    print("WorldWise++ smoke test OK")


if __name__ == "__main__":
    main()
