"""
End-to-end check of the C1c derived feature PKLs (server, CPU):

  * for each stream in {dinov3_tok, pi3_tok, fused} and mode in {predcls, sgdet}, load the
    derived PKLs through WorldAG (worldbbox annotations) and compare every tensor except
    the appearance features against the reference dinov3l loader output (same frames,
    objects, 2D boxes, corners, pair indices, visibility, union pair count);
  * run WorldFormerC1 (fused config) forward + backward on the real batch.

    CUDA_VISIBLE_DEVICES="" python tests/test_worldformer_c1_realdata.py --data_path /data/rohith/ag
"""
import argparse
import os
import sys
from types import SimpleNamespace

import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

STREAM_DIM = {"dinov3_tok": 3072, "pi3_tok": 3072, "fused": 6144}
COMPARE_KEYS = ("corners", "valid_mask", "visibility_mask", "person_idx", "object_idx", "pair_valid",
                "gt_bboxes_2d", "object_classes", "camera_poses")


def load(data_path, mode, fm, videos):
    from dataloader.world_ag_dataset import WorldAG
    ds = WorldAG(phase="test", data_path=data_path, mode=mode, feature_model=fm,
                 annot_dir_name="world4d_rel_annotations_worldbbox")
    idx = {v: i for i, v in enumerate(ds.video_list)}
    return ds, {v: ds[idx[v]] for v in videos if v in idx}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--videos", nargs="+", default=["015XE", "02DPI", "03PRW"])
    args = ap.parse_args()

    for mode in ("predcls", "sgdet"):
        _, ref = load(args.data_path, mode, "dinov3l", args.videos)
        assert len(ref) == len(args.videos), (mode, sorted(ref))
        for fm, D in STREAM_DIM.items():
            _, got = load(args.data_path, mode, fm, args.videos)
            assert sorted(got) == sorted(ref), (mode, fm, sorted(got))
            for v in args.videos:
                r, g = ref[v], got[v]
                assert int(r["T"]) == int(g["T"]), (v, r["T"], g["T"])
                assert r["visual_features"].shape[:2] == g["visual_features"].shape[:2]
                assert g["visual_features"].shape[-1] == D, g["visual_features"].shape
                assert torch.isfinite(g["visual_features"]).all()
                for k in COMPARE_KEYS:
                    a, b = r.get(k), g.get(k)
                    if a is None and b is None:
                        continue
                    if torch.is_tensor(a):
                        assert a.shape == b.shape, (v, k, a.shape, b.shape)
                        assert torch.equal(a, b) if not a.is_floating_point() else torch.allclose(a, b), (v, k)
                    else:
                        assert a == b, (v, k)
                ru, gu = r.get("union_features"), g.get("union_features")
                if ru is not None:
                    assert gu is not None and gu.shape[:2] == ru.shape[:2] and gu.shape[-1] == D, (v, "union")
            print(f"[{mode}/{fm}] {len(got)} videos match the dinov3l reference "
                  f"(T={[int(got[v]['T']) for v in args.videos]}, D={D})")

    # forward/backward of the fused C1 model on a real predcls batch
    from lib.supervised.worldformer.c1_tokenswap import WorldFormerC1
    from dataloader.world_ag_dataset import (ATTENTION_RELATIONSHIPS, SPATIAL_RELATIONSHIPS,
                                             CONTACTING_RELATIONSHIPS)
    with open(os.path.join(REPO, "configs/methods/predcls/worldformer_c1_fused_predcls.yaml"),
              encoding="utf-8") as f:
        conf = SimpleNamespace(**yaml.safe_load(f))
    conf.data_path = args.data_path
    ds, got = load(args.data_path, "predcls", "fused", args.videos)
    model = WorldFormerC1(conf, num_object_classes=len(ds.object_classes),
                          attention_class_num=len(ATTENTION_RELATIONSHIPS),
                          spatial_class_num=len(SPATIAL_RELATIONSHIPS),
                          contact_class_num=len(CONTACTING_RELATIONSHIPS))
    model.train()
    b = got[args.videos[0]]
    out = model(visual_features_seq=b["visual_features"], corners_seq=b["corners"],
                valid_mask_seq=b["valid_mask"], visibility_mask_seq=b["visibility_mask"],
                person_idx_seq=b["person_idx"], object_idx_seq=b["object_idx"], pair_valid=b["pair_valid"],
                p_mask_visible=0.3, camera_pose_seq=b.get("camera_poses"),
                union_features_seq=b.get("union_features"), node_labels_seq=b.get("object_classes"))
    loss = sum(v.float().mean() for v in out.values() if torch.is_tensor(v) and v.is_floating_point())
    loss.backward()
    assert torch.isfinite(loss)
    print(f"[worldformer_c1_fused_predcls] real-batch forward/backward OK: T={int(b['T'])} "
          f"loss={float(loss):.4f} gate={model.gate_summary()}")


if __name__ == "__main__":
    main()
