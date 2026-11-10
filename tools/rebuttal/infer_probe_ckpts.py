"""INFER probe (CPU, read-only): checkpoint inventory for the rebuttal inference runs, plus 3-D annotation fields.

For every run: number of checkpoint_<e> dirs, best_model.pth epoch, and which checkpoint_<e> holds the same weights
(max |diff| over all tensors). Also prints, for 3 worldbbox test PKLs and 3 held-out PKLs, whether person_info and
objects (visible / unobserved) carry an (8,3) ``corners_final``.

    CUDA_VISIBLE_DEVICES="" $PY tools/rebuttal/infer_probe_ckpts.py
"""
import glob
import json
import os
import pickle
import sys

import numpy as np
import torch

RUNS = {
    "w_sttran": "/data/rohith/ag/checkpoints/w_sttran_{m}_resnet50_v2",
    "w_sttran_pp": "/data/rohith/ag/checkpoints/w_sttran_pp_{m}_resnet50_v2",
    "w_dsgdetr": "/data/rohith/ag/checkpoints/w_dsgdetr_{m}_resnet50_v2",
    "w_dsgdetr_pp": "/data/rohith/ag/checkpoints/w_dsgdetr_pp_{m}_resnet50_v2",
    "w_usg": "/data/rohith/ag/checkpoints/w_usg_{m}_resnet50_v2",
    "worldwise_dinov3l": "/data/rohith/ag/checkpoints/worldwise_v2e_{m}_dinov3l_v2",
    "worldwise_plus_dinov3tok": "/data3/rohith/ag/runs/worldformer/worldformer_c1_dinov3tok_{m}",
    "worldwise_pp_dinov3": "/data3/rohith/ag/runs/worldwise_pp/worldwise_pp_dinov3_{m}",
    **{f"abl_{x}": f"/data/rohith/ag/checkpoints/worldwise_abl_{x}_{{m}}_dinov3l_v2"
       for x in ("notau", "nomask", "noema", "noego", "nomotion", "notempedge", "nospatial")},
}


def sd_of(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    return s.get("model_state_dict", s), s.get("epoch", None)


def same(a, b):
    if a.keys() != b.keys():
        return float("inf")
    d = 0.0
    for k in a:
        if a[k].shape != b[k].shape:
            return float("inf")
        if a[k].dtype.is_floating_point:
            d = max(d, float((a[k].float() - b[k].float()).abs().max()) if a[k].numel() else 0.0)
        elif not torch.equal(a[k], b[k]):
            return float("inf")
    return d


def main():
    out = {}
    for name, tmpl in RUNS.items():
        for m in ("predcls", "sgdet"):
            d = tmpl.format(m=m)
            eps = sorted(int(p.rsplit("_", 1)[1]) for p in glob.glob(f"{d}/checkpoint_*")
                         if os.path.isfile(f"{p}/checkpoint_state.pth"))
            rec = {"dir": d, "n_epochs": len(eps), "epochs": [min(eps), max(eps)] if eps else None,
                   "best_exists": os.path.isfile(f"{d}/best_model.pth")}
            if rec["best_exists"]:
                bsd, bep = sd_of(f"{d}/best_model.pth")
                rec["best_epoch_field"] = bep
                n_params = sum(v.numel() for v in bsd.values() if v.dtype.is_floating_point)
                rec["n_params_M"] = round(n_params / 1e6, 2)
                cands = ([bep] if isinstance(bep, int) and bep in eps else []) + [e for e in eps if e != bep]
                for e in cands:
                    csd, cep = sd_of(f"{d}/checkpoint_{e}/checkpoint_state.pth")
                    diff = same(bsd, csd)
                    if diff == 0.0:
                        rec["best_equals_checkpoint"] = e
                        rec["checkpoint_epoch_field"] = cep
                        break
                else:
                    rec["best_equals_checkpoint"] = None
            out[f"{name}|{m}"] = rec
            print(f"{name:26s} {m:7s} {json.dumps(rec)}", flush=True)
    # 3-D annotation fields
    def corners_ok(o):
        c = o.get("corners_final", None)
        return c is not None and np.asarray(c).shape == (8, 3)
    for label, files in (("worldbbox", sorted(glob.glob("/data/rohith/ag/world4d_rel_annotations_worldbbox/test/*.pkl"))[:3]),
                         ("heldout", sorted(glob.glob("/data3/rohith/ag/runs/rebuttal/annot_heldout/test/*.pkl"))[:3])):
        for f in files:
            ann = pickle.load(open(f, "rb"))
            st = {"person": [0, 0], "vis": [0, 0], "unobs": [0, 0]}
            for fr in ann["frames"].values():
                st["person"][0] += 1
                st["person"][1] += int(corners_ok(fr.get("person_info", {})))
                for o in fr.get("object_info_list", []):
                    k = "vis" if o.get("visible", True) else "unobs"
                    st[k][0] += 1
                    st[k][1] += int(corners_ok(o))
            print(label, os.path.basename(f), "[n, with corners_final]", st,
                  "person keys", sorted(next(iter(ann["frames"].values())).get("person_info", {}).keys()))
    os.makedirs("/data3/rohith/ag/runs/rebuttal/infer", exist_ok=True)
    with open("/data3/rohith/ag/runs/rebuttal/infer/ckpt_inventory.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())
