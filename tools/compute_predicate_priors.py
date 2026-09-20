"""
Compute predicate class priors for WorldWise's tail-aware logit adjustment.

Counts how often each attention / spatial / contacting predicate class occurs
over all valid (person, object) pairs in the training split, and writes the
frequencies to a JSON consumed by lib/supervised/worldwise/loss.py.

  attention  : single-label  -> prior_c = count_c / sum_c count   (sums to 1)
  spatial    : multi-label   -> prior_c = positives_c / n_pairs    (per-class)
  contacting : multi-label   -> prior_c = positives_c / n_pairs    (per-class)

Label priors are backbone-independent, so any available --feature_model works.
Run on the training server (needs torch + the dataset):

    python tools/compute_predicate_priors.py --data_path /data/rohith/ag \
        --feature_model dinov2b --mode predcls
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader.world_ag_dataset import WorldAG, world_collate_fn

N_ATT, N_SPA, N_CON = 3, 6, 17


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="/data/rohith/ag")
    ap.add_argument("--feature_model", default="dinov2b")
    ap.add_argument("--mode", default="predcls")
    ap.add_argument("--phase", default="train")
    ap.add_argument("--max_objects", type=int, default=64)
    ap.add_argument("--annot_dir", default="world4d_rel_annotations",
                    help="annotation folder under data_path (e.g. world4d_rel_annotations_worldbbox)")
    ap.add_argument(
        "--out", default=None,
        help="output JSON (default: <data_path>/features/predicate_priors_<mode>.json; "
             "mode-specific so predcls and sgdet priors never overwrite each other)",
    )
    args = ap.parse_args()

    ds = WorldAG(
        phase=args.phase, data_path=args.data_path, mode=args.mode,
        feature_model=args.feature_model, include_invisible=True,
        max_objects=args.max_objects, annot_dir_name=args.annot_dir,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=world_collate_fn)

    att_counts = torch.zeros(N_ATT, dtype=torch.float64)
    spa_counts = torch.zeros(N_SPA, dtype=torch.float64)
    con_counts = torch.zeros(N_CON, dtype=torch.float64)
    total_pairs = 0

    for batch in tqdm(dl, desc="counting predicates"):
        valid = batch["pair_valid"].bool()
        if not valid.any():
            continue
        att = batch["gt_attention"][valid]       # (P,)
        spa = batch["gt_spatial"][valid].double()   # (P, 6)
        con = batch["gt_contacting"][valid].double()  # (P, 17)

        total_pairs += int(att.shape[0])
        att_valid = att[att >= 0].long()
        if att_valid.numel() > 0:
            att_counts += torch.bincount(att_valid, minlength=N_ATT).double()
        spa_counts += spa.sum(0)
        con_counts += con.sum(0)

    denom = max(total_pairs, 1)
    att_sum = max(att_counts.sum().item(), 1.0)

    priors = {
        "attention": (att_counts / att_sum).tolist(),
        "spatial": (spa_counts / denom).tolist(),
        "contacting": (con_counts / denom).tolist(),
        "n_pairs": total_pairs,
        "mode": args.mode,
        "feature_model": args.feature_model,
    }

    out = args.out or os.path.join(
        args.data_path, "features", f"predicate_priors_{args.mode}.json"
    )
    Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(priors, f, indent=2)

    print(f"\n[predicate priors]  n_pairs={total_pairs}")
    print(f"  attention : {[round(x, 4) for x in priors['attention']]}")
    print(f"  spatial   : {[round(x, 4) for x in priors['spatial']]}")
    print(f"  contacting: {[round(x, 4) for x in priors['contacting']]}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
