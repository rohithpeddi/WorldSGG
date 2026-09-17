"""
CPU forward/backward smoke test for the WorldFormer C1 token-swap model.

    python tests/test_worldformer_c1_smoke.py
"""
import os
import sys
from types import SimpleNamespace

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def _conf(stem: str):
    mode = stem.rsplit("_", 1)[-1]
    with open(os.path.join(REPO, "configs", "methods", mode, f"{stem}.yaml"), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    c["clip_embeddings_path"] = ""  # random CLIP table (no data on this machine)
    return SimpleNamespace(**c)


def run(stem: str, T: int = 3, N: int = 6, K: int = 5) -> None:
    from lib.supervised.worldformer.c1_tokenswap import WorldFormerC1, GatedFusionProjector

    conf = _conf(stem)
    model = WorldFormerC1(conf, num_object_classes=37, attention_class_num=3,
                          spatial_class_num=6, contact_class_num=17)
    D = conf.d_detector_roi
    assert isinstance(model.scaffold_tokenizer.visual_projector, GatedFusionProjector)
    assert isinstance(model.rel_predictor.union_proj, GatedFusionProjector)
    torch.manual_seed(0)
    vis = torch.randn(T, N, D)
    corners = torch.randn(T, N, 8, 3)
    valid = torch.ones(T, N, dtype=torch.bool)
    visible = torch.rand(T, N) > 0.3
    person_idx = torch.zeros(T, K, dtype=torch.long)
    object_idx = torch.arange(1, K + 1).unsqueeze(0).repeat(T, 1)
    pair_valid = torch.ones(T, K, dtype=torch.bool)
    cam = torch.eye(4).unsqueeze(0).repeat(T, 1, 1)
    union = torch.randn(T, K, D)
    labels = torch.randint(0, 37, (T, N))
    model.train()
    out = model(visual_features_seq=vis, corners_seq=corners, valid_mask_seq=valid,
                visibility_mask_seq=visible, person_idx_seq=person_idx, object_idx_seq=object_idx,
                pair_valid=pair_valid, p_mask_visible=0.3, camera_pose_seq=cam,
                union_features_seq=union, node_labels_seq=labels if conf.mode == "predcls" else None)
    loss = sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.is_floating_point())
    loss.backward()
    g = model.scaffold_tokenizer.visual_projector
    grads = [p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters()]
    assert all(grads), grads
    print(f"[{stem}] ok: D={D} fusion={g.fusion} outputs={sorted(k for k in out)} "
          f"gate={model.gate_summary()}")


if __name__ == "__main__":
    for stem in ("worldformer_c1_dinov3tok_predcls", "worldformer_c1_fused_predcls",
                 "worldformer_c1_fused_sgdet"):
        run(stem)
