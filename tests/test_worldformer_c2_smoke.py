"""
Forward/backward smoke test for the WorldFormer C2 scaffold (CPU, random grids).

    python tests/test_worldformer_c2_smoke.py
"""
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    from lib.supervised.worldformer.c2_worldformer import WorldFormerC2, WorldFormerC2Loss

    torch.manual_seed(0)
    T, N, C = 2, 8, 37
    image_hw, dino_padded_hw = (378, 672), (384, 672)
    dino = torch.randn(T, 24, 42, 1024).half()
    pi3 = torch.randn(T, 27, 48, 1024).half()
    slot_corners = torch.randn(T, N, 8, 3)
    slot_valid = torch.ones(T, N, dtype=torch.bool)
    slot_valid[:, 6:] = False
    slot_visible = torch.rand(T, N) > 0.4
    slot_tokens = torch.randn(T, N, 2048).half()

    model = WorldFormerC2(num_classes=C, n_layers=2, n_queries=20, max_slots=N)
    model.train()
    out = model(dino, pi3, dino_padded_hw, image_hw, slot_corners=slot_corners, slot_valid=slot_valid,
                slot_visible=slot_visible, slot_tokens=slot_tokens, p_mask_visible=0.3)
    assert out["logits"].shape == (T, 20, C + 1)
    assert out["corners"].shape == (T, 20, 8, 3)
    assert out["rel_logits"].shape == (T, 20 + N, 20 + N, 26)
    assert out["reconstruction_predictions"].shape == (T, N, model.d_model)

    targets = []
    for t in range(T):
        G = 3
        cxcy = torch.rand(G, 2) * 0.6 + 0.2
        wh = torch.rand(G, 2) * 0.3 + 0.05
        targets.append(dict(labels=torch.randint(0, C, (G,)), boxes=torch.cat([cxcy, wh], 1),
                            corners=torch.randn(G, 8, 3), rel_pairs=torch.tensor([[0, 1], [0, 2]]),
                            rel_labels=(torch.rand(2, 26) > 0.8).float()))
    crit = WorldFormerC2Loss(C, one_to_many_iou=0.5)
    losses = crit(out, targets, image_hw)
    losses["total"].backward()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None
                 and torch.isfinite(p.grad).all())
    n_param = sum(1 for p in model.parameters() if p.requires_grad)
    print({k: round(float(v), 4) for k, v in losses.items()})
    print(f"params with finite grads: {n_grad}/{n_param}; "
          f"total params {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    assert torch.isfinite(losses["total"])
    assert n_grad >= n_param - 2  # target_proj is frozen; rot/off heads always used
    print("C2 scaffold smoke test OK")


if __name__ == "__main__":
    main()
