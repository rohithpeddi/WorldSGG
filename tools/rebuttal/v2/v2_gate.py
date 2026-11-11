"""V2 gate: hidden and masked slots carry no box-derived evidence. CPU, randomly initialised model, read-only.

For a few test videos of a V2 config:
  1. eval mode: perturbing the input corners, appearance and union features of hidden slots leaves every relation
     distribution and every decoded box unchanged;
  2. train mode (fixed seed): the same holds for hidden slots and the masked items the model draws;
  3. train mode: the masked-item 3-D losses are finite and backpropagate; masked items exist.
Exit 1 on any failure.

    python tools/rebuttal/v2/v2_gate.py --config configs/methods/predcls/rebuttal3/<exp>.yaml --n 2
"""
import argparse
import copy
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
from wsgg_base import load_wsgg_config  # noqa: E402
from lib.supervised.v2.trainers import build_v2_model, forward_v2  # noqa: E402

KEYS = ("attention_distribution", "spatial_distribution", "contacting_distribution")


def perturb(b, mask):
    b2 = {k: (v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)) for k, v in b.items()}
    m = mask.bool()
    g = torch.Generator().manual_seed(7)
    b2["corners"][m] = b2["corners"][m] + 5.0 * torch.randn(b2["corners"][m].shape, generator=g)
    b2["visual_features"][m] = torch.randn(b2["visual_features"][m].shape, generator=g)
    if b2.get("union_features") is not None:
        N = m.shape[1]
        og = torch.gather(m, 1, b2["object_idx"].clamp(0, N - 1))
        b2["union_features"][og] = torch.randn(b2["union_features"][og].shape, generator=g)
    return b2


def diff(p1, p2):
    d = max(float((p1[k] - p2[k]).abs().max()) for k in KEYS)
    return max(d, float((p1["v2"]["corners_pred_world"] - p2["v2"]["corners_pred_world"]).abs().max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    sys.argv = [sys.argv[0], "--config", a.config]
    conf = load_wsgg_config(a.config)
    import train_wsgg_methods as tm
    trainer = tm.METHOD_MAP[conf.method_name](conf)
    vids = sorted(f.split(".")[0] for f in os.listdir(f"{conf.data_path}/world4d_rel_annotations_worldbbox/test"))[:a.n * 3]
    ds = trainer._make_dataset("test", video_ids=vids)
    torch.manual_seed(0)
    model = build_v2_model(conf, ds, torch.device("cpu"))
    bad, n_hidden, n_masked = [], 0, 0
    for i in range(min(a.n, len(ds))):
        b = ds[i]
        H = b["valid_mask"].bool() & ~b["visibility_mask"].bool()
        n_hidden += int(H.sum())
        model.eval()
        with torch.no_grad():
            d = diff(forward_v2(model, conf, b), forward_v2(model, conf, perturb(b, H)))
        if d > 1e-5:
            bad.append(f"{b['video_id']}: eval hidden leak {d:.2e}")
        model.train()
        kw = dict(slot_classes_seq=b["object_classes"], gt_corners_seq=b["gt_corners"],
                  gt_box_valid_seq=b["gt_box_valid"])

        def fwd(bb):
            args = dict(visual_features_seq=bb["visual_features"], corners_seq=bb["corners"],
                        valid_mask_seq=bb["valid_mask"], visibility_mask_seq=bb["visibility_mask"],
                        person_idx_seq=bb["person_idx"], object_idx_seq=bb["object_idx"], pair_valid=bb["pair_valid"],
                        p_mask_visible=0.3, camera_pose_seq=bb.get("camera_poses"),
                        node_labels_seq=bb["object_classes"] if conf.mode == "predcls" else None, **kw)
            if conf.method_name == "worldwise_pp_3d":
                args.update(grid_dino_seq=bb["grid_dino"], grid_pi3_seq=bb["grid_pi3"], image_hw=bb["image_hw"],
                            bboxes_2d_seq=bb.get("bboxes_2d"))
            else:
                args["union_features_seq"] = bb.get("union_features")
            return model(**args)
        torch.manual_seed(1)
        p1 = fwd(b)
        G = p1["v2"]["G"]
        n_masked += int(p1["v2"]["M"].sum())
        torch.manual_seed(1)
        with torch.no_grad():
            p2 = fwd(perturb(b, G))
        d = diff({k: v.detach() for k, v in p1.items() if k in KEYS} | {"v2": {"corners_pred_world": p1["v2"]["corners_pred_world"].detach()}}, p2)
        if d > 1e-5:
            bad.append(f"{b['video_id']}: train hidden/masked leak {d:.2e}")
        L = p1.get("v2_losses", {})
        tot = L.get("geo_latent", torch.tensor(float("nan"))) + L.get("geo_box", torch.tensor(float("nan")))
        if not torch.isfinite(tot):
            bad.append(f"{b['video_id']}: non-finite geometry loss")
        else:
            (tot + sum(p1[k].sum() * 0 for k in KEYS)).backward()
            if model.geo.decoder[1].weight.grad is None:
                bad.append(f"{b['video_id']}: no gradient into the box decoder")
            model.zero_grad(set_to_none=True)
        print(f"[v2-gate] {b['video_id']}: T={int(b['T'])} hidden={int(H.sum())} masked={int(p1['v2']['M'].sum())} "
              f"items_with_target={float(L.get('geo_n_items', 0)):.0f} latent={float(L.get('geo_latent', 0)):.3f} "
              f"box={float(L.get('geo_box', 0)):.3f}")
    if n_masked == 0:
        bad.append("no masked items drawn")
    print(f"[v2-gate] {conf.method_name} {conf.mode}: hidden slot-frames {n_hidden}, masked {n_masked}; "
          f"failures {bad or 'none'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
