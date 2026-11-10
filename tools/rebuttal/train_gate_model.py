"""TRAIN gate: with every new option off, each model family computes exactly what it computed before this round.

Run once per code tree (the pre-round snapshot and the TRAIN worktree), then compare:

  PY tools/rebuttal/train_gate_model.py run --repo <tree> --out <tree_tag>.pkl
  PY tools/rebuttal/train_gate_model.py compare a.pkl b.pkl

``run`` builds every Table 2 family from its published config (CPU, fp32, fixed seeds), loads one real video
through the trainer's own dataset hook, then records (a) the training losses of one forward pass, (b) the gradient
of every parameter after backward, (c) the eval-mode distributions. ``compare`` requires bit equality.
"""
import argparse
import os
import pickle
import sys

FAMILIES = [  # (method config stem, mode) -- the published Table 2 configs
    ("w_sttran_{m}_resnet50", "w_sttran"), ("w_sttran_pp_{m}_resnet50", "w_sttran_pp"),
    ("w_dsgdetr_{m}_resnet50", "w_dsgdetr"), ("w_dsgdetr_pp_{m}_resnet50", "w_dsgdetr_pp"),
    ("w_usg_{m}_resnet50", "w_usg"), ("worldwise_{m}_dinov3l", "worldwise"),
    ("worldwise_plus_dinov3tok_{m}", "worldwise_plus"), ("worldwise_pp_dinov3_{m}", "worldwise_pp"),
]
VIDEO = {"train": "001YG"}   # first val750 video (train split)


def run(repo, out, modes):
    sys.path.insert(0, repo)
    os.chdir(repo)
    import torch
    import numpy as np
    from wsgg_base import load_wsgg_config
    import train_wsgg_methods as tm

    torch.use_deterministic_algorithms(True, warn_only=True)
    res = {}
    for mode in modes:
        for stem, method in FAMILIES:
            cfg = f"{repo}/configs/methods/{mode}/{stem.format(m=mode)}.yaml"
            sys.argv = [sys.argv[0], "--config", cfg]
            conf = load_wsgg_config(cfg)
            conf.use_amp = False
            tr = tm.METHOD_MAP[conf.method_name](conf)
            tr._device = torch.device("cpu")

            # dataset restricted to one training video without scanning the split
            from dataloader.world_ag_dataset import WorldAG
            orig_build = WorldAG._build_video_list
            WorldAG._build_video_list = lambda self: setattr(self, "video_list", [VIDEO["train"]])
            try:
                dset = tr._make_dataset("train")
            finally:
                WorldAG._build_video_list = orig_build
            tr._train_dataset = dset
            tr._object_classes = dset.object_classes
            torch.manual_seed(0)
            tr.init_model()
            tr.init_loss_fn()
            batch = dset[0]
            tr._model.train()
            torch.manual_seed(1)
            losses = tr.process_train_video(batch)
            total = losses.get("total", sum(losses.values()))
            total.backward()
            grads = {n: p.grad.detach().clone() for n, p in tr._model.named_parameters() if p.grad is not None}
            tr._model.eval()
            with torch.no_grad():
                pred = tr.process_test_video(batch)
            key = f"{mode}/{method}"
            res[key] = {
                "losses": {k: float(v.detach()) for k, v in losses.items()},
                "grads": {n: g.numpy() for n, g in grads.items()},
                "pred": {k: v.detach().numpy() for k, v in pred.items()},
                "n_params": sum(p.numel() for p in tr._model.parameters()),
            }
            print(key, "total", float(total), "params", res[key]["n_params"], "grads", len(grads), flush=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)


def compare(a_path, b_path):
    import numpy as np
    a, b = pickle.load(open(a_path, "rb")), pickle.load(open(b_path, "rb"))
    ok = True
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print(key, "missing in one tree"); ok = False; continue
        x, y = a[key], b[key]
        bad = []
        if x["losses"] != y["losses"]:
            bad.append(f"losses {x['losses']} vs {y['losses']}")
        if x["n_params"] != y["n_params"]:
            bad.append(f"n_params {x['n_params']} vs {y['n_params']}")
        if set(x["grads"]) != set(y["grads"]):
            bad.append("grad keys differ")
        else:
            nd = [n for n in x["grads"] if not np.array_equal(x["grads"][n], y["grads"][n])]
            if nd:
                bad.append(f"{len(nd)} grads differ, e.g. {nd[:3]}")
        for k in set(x["pred"]) | set(y["pred"]):
            if k not in x["pred"] or k not in y["pred"] or not np.array_equal(x["pred"][k], y["pred"][k]):
                bad.append(f"pred {k} differs")
        print(key, "OK" if not bad else bad)
        ok &= not bad
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--repo", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--modes", nargs="+", default=["predcls", "sgdet"])
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.repo, a.out, a.modes)
    else:
        sys.exit(0 if compare(a.a, a.b) else 1)
