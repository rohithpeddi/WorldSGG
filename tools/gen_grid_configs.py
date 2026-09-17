"""
Generate the WorldSGG experiment config grid (final campaign structure).

WorldWise's MAIN configuration is the round-2 winner **v2e**:
  I-0 architecture + pair geometry (I-1) + tau=0.5 logit adjustment +
  lambda_vlm=0 (no noisy VLM supervision) + p_mask=0.3.
Its experiment names keep the `worldwise_v2e` stem so every already-trained
v2e cell is reused as-is.

Campaign structure:
  1. baselines x resnet50            — method-comparison table (unchanged)
  2. worldwise (=v2e) x 4 backbones  — main method + backbone scaling
  3. ablations x dinov3l             — component-wise ablation of the main
                                       config (one change per tier)
  x 2 tasks (predcls, sgdet)

Writes to configs/methods/<mode>/<method>_<mode>_<backbone>.yaml, ablations
as configs/methods/<mode>/worldwise_<tier>_<mode>_dinov3l.yaml. Stale configs
from retired campaign rounds are deleted on regeneration.

Run (stdlib only, no torch needed):
    python tools/gen_grid_configs.py           # base grid only
    python tools/gen_grid_configs.py --tiers   # + ablation configs
"""

import argparse
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METHODS = ["w_sttran", "w_sttran_pp", "w_dsgdetr", "w_dsgdetr_pp", "w_usg", "worldwise"]
BACKBONES = ["resnet50", "dinov2b", "dinov2l", "dinov3l"]
MODES = ["predcls", "sgdet"]
HERO_BACKBONE = "dinov3l"        # WorldWise⁺ tiers run here
METHODS_BACKBONE = "resnet50"    # baselines run here only (method comparison)
VERSION = "v2"

SEED = 0


def backbones_for(method):
    """Baselines run only at the common backbone; WorldWise scales across all."""
    return BACKBONES if method == "worldwise" else [METHODS_BACKBONE]

# Keys shared by every method (ladder is realised in code, not via flags).
COMMON = [
    ("data_path", "/data/rohith/ag"),
    ("save_path", "/data/rohith/ag/checkpoints"),
    # Annotation folders per split (2026-09-17: test = WorldBBox release set)
    ("train_annot_dir", "world4d_rel_annotations"),
    ("test_annot_dir", "world4d_rel_annotations_worldbbox"),
    ("results_path", "results"),
    ("ckpt", None),
    ("world_sg_dir", ""),
    ("seed", SEED),
    # ---- Shared architecture ----
    ("d_model", 256),
    ("d_struct", 256),
    ("d_visual", 256),
    ("d_detector_roi", 1024),
    ("d_camera", 128),
    ("d_motion", 64),
    ("n_heads", 4),
    ("d_feedforward", 512),
    ("max_objects", 64),
    ("dropout", 0.1),
    # ---- Relationship predictor ----
    ("d_rel", 256),
    ("d_text", 128),
    ("d_union_roi", 1024),
    ("n_rel_layers", 2),
    ("n_rel_heads", 4),
    ("n_temporal_edge_layers", 1),
    ("n_temporal_obj_layers", 2),
    ("clip_embeddings_path", "features/clip_features/clip_text_embeddings.npy"),
    # ---- Training (identical budget across the whole grid) ----
    ("nepoch", 20),
    ("lr", 0.0001),
    ("weight_decay", 0.0001),
    ("warmup_fraction", 0.1),
    ("grad_clip", 5.0),
    ("optimizer", "adamw"),
    ("bce_loss", True),
    ("use_wandb", True),
    ("wandb_project", "worldsgg-v2"),
    ("use_amp", True),
    ("log_every", 100),
    ("include_invisible", True),
    ("datasize", "large"),
    ("skip_test", False),
    ("n_gnn_layers", 3),
    # ---- Loss ----
    ("lambda_vlm", 0.2),
    ("label_smoothing_vlm", 0.2),
]

# WorldWise-only keys — the MAIN configuration is v2e (round-2 winner):
# MWAE core + pair geometry + tau=0.5 + lambda_vlm=0 + mask 0.3.
# Keys here OVERRIDE same-named COMMON keys (render() dedupes) — that is how
# WorldWise gets lambda_vlm=0 while baselines keep 0.2.
WORLDWISE_EXTRA = [
    ("n_cross_attn_layers", 2),
    ("n_self_attn_layers", 3),
    ("d_memory", 256),
    ("lambda_reconstruction", 0.5),
    ("lambda_recon_dominance", 0.1),
    ("p_simulate_unseen", 0.3),
    ("p_mask_visible", 0.3),
    ("use_object_spatial_encoder", True),
    ("use_camera_temporal", True),
    ("use_object_motion_encoder", True),
    ("use_temporal_edge_attn", True),
    # v2e loss recipe: no noisy VLM supervision on unseen pairs — the
    # simulated-unseen objective (clean labels on artificially masked pairs)
    # is the only edge supervision for the masked pathway. Round-2 result:
    # +4 R / +9 mR / +9.5 occpair vs lambda_vlm=0.2.
    ("lambda_vlm", 0.0),
    # Tail-aware logit adjustment at the round-2 knee (tau=1.0 was unstable
    # and traded 13 R points; 0.75 is the published mR-max point).
    ("use_logit_adjustment", True),
    ("logit_adjustment_tau", 0.5),
    # predicate_priors_path is mode-specific — filled in by render()
    ("predicate_priors_path", "features/predicate_priors_{mode}.json"),
    ("use_ema_recon_target", True),
    # v2e architecture plugin: explicit pair geometry in relation tokens
    ("use_pair_geometry", True),
    # Retired plugins (round-2 gate: all failed) — kept as explicit flags so
    # the config fully documents the architecture.
    ("use_soft_text_embedding", False),
    ("use_geometric_attn_bias", False),
    ("attn_bias_keep_pe", False),
    ("use_confidence_weighted_vlm", False),
    ("use_predicate_prototypes", False),
    ("use_cross_object_retrieval", False),
    ("use_energy_refinement", False),
    ("lambda_stability", 0.0),
]

# W-USG-only keys — external baseline (USG-Par relation machinery on the
# WSGG substrate). Same relation supervision as the ladder baselines
# (lambda_vlm stays at the COMMON 0.2); adds only the USG-specific
# architecture depth and the text-centric alignment weight.
W_USG_EXTRA = [
    ("n_usg_decoder_layers", 2),
    ("lambda_align", 0.1),
    ("align_temperature", 0.07),
]

# Component-wise ablations of the MAIN (v2e) configuration — one change per
# tier, all @ the hero backbone. Overrides apply on top of WORLDWISE_EXTRA.
# Three tiers keep their historical names so already-trained cells are reused
# verbatim (the override reproduces the identical historical config):
#   v2a = main with the noisy VLM supervision added back (loss ablation)
#   v2f = main at tau=0.75 (the published mR-max operating point)
#   v2g = main without pair geometry (I-1 attribution)
ABLATION_TIERS = {
    # ---- loss ablations ----
    "v2a":            {"lambda_vlm": 0.2},                # + noisy VLM supervision back
    "v2f":            {"logit_adjustment_tau": 0.75},     # tau operating point
    "abl_notau":      {"use_logit_adjustment": False},    # − logit adjustment
    "abl_nomask":     {"p_mask_visible": 0.0,             # − artificial masking
                       "p_simulate_unseen": 0.0},         #   (recon/sim vanish)
    "abl_noema":      {"use_ema_recon_target": False},    # − EMA recon target
    # ---- architecture ablations ----
    "v2g":            {"use_pair_geometry": False},          # − pair geometry (I-1)
    "abl_nospatial":  {"use_object_spatial_encoder": False}, # − camera-frame features
    "abl_noego":      {"use_camera_temporal": False},        # − ego-motion encoder
    "abl_nomotion":   {"use_object_motion_encoder": False},  # − object motion encoder
    "abl_notempedge": {"use_temporal_edge_attn": False},     # − temporal edge attention
}

# Config files from retired campaign rounds — deleted on regeneration.
RETIRED_TIER_FILES = [
    "plus1", "plus2", "plus3", "plus3pe", "noema",
    "tau025", "tau05", "tau075",
    "notau", "lowmask", "nomask", "novlm", "nomotion", "noego",
    "conf", "proto", "xobj", "energy",
    "v2b", "v2c", "v2d", "v2e",  # v2e is now the MAIN config, not a tier
]


def fmt(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return "''" if v == "" else v
    return repr(v) if isinstance(v, float) else str(v)


def render(method, mode, backbone, tier=None):
    stem = f"{method}_{tier}" if tier else method
    # The main WorldWise config keeps the historical `worldwise_v2e` experiment
    # stem so every already-trained v2e cell is reused verbatim.
    exp_stem = "worldwise_v2e" if (method == "worldwise" and tier is None) else stem
    exp = f"{exp_stem}_{mode}_{backbone}_{VERSION}"
    title = f"{method}  |  mode={mode}  |  backbone={backbone}"
    if tier:
        title += f"  |  ablation={tier} ({', '.join(sorted(ABLATION_TIERS[tier]))})"
    elif method == "worldwise":
        title += "  |  MAIN config = v2e (pair geometry, tau=0.5, lambda_vlm=0)"
    lines = [
        "# ============================================================================",
        f"# {title}",
        "# Auto-generated by tools/gen_grid_configs.py - final campaign structure.",
        "# Usage:",
        f"#   python train_wsgg_methods.py --config configs/methods/{mode}/{stem}_{mode}_{backbone}.yaml",
        "# ============================================================================",
        "",
    ]
    # Dict-merge so WORLDWISE_EXTRA overrides same-named COMMON keys
    # (e.g. lambda_vlm: baselines 0.2, WorldWise 0.0)
    merged = dict(COMMON)
    if method == "worldwise":
        merged.update(dict(WORLDWISE_EXTRA))
        merged["predicate_priors_path"] = merged["predicate_priors_path"].format(mode=mode)
        if tier:
            merged.update(ABLATION_TIERS[tier])
    elif method == "w_usg":
        merged.update(dict(W_USG_EXTRA))
    merged.update({
        "method_name": method,
        "mode": mode,
        "feature_model": backbone,
        "task_name": "worldsgg",
        "experiment_name": exp,
    })
    for k, v in merged.items():
        lines.append(f"{k}: {fmt(v)}")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", action="store_true",
                    help="also generate the component-ablation configs @ hero backbone")
    args = ap.parse_args()

    n = 0
    removed = 0
    for mode in MODES:
        out_dir = os.path.join(REPO, "configs", "methods", mode)
        os.makedirs(out_dir, exist_ok=True)
        for method in METHODS:
            for backbone in backbones_for(method):
                fname = f"{method}_{mode}_{backbone}.yaml"
                with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                    f.write(render(method, mode, backbone))
                n += 1
                print(f"  wrote configs/methods/{mode}/{fname}")
            # Drop stale configs for combos no longer in the design
            # (baselines at non-resnet50 backbones)
            for backbone in BACKBONES:
                if backbone in backbones_for(method):
                    continue
                stale = os.path.join(out_dir, f"{method}_{mode}_{backbone}.yaml")
                if os.path.exists(stale):
                    os.remove(stale)
                    removed += 1
                    print(f"  removed stale configs/methods/{mode}/{fname}")

        if args.tiers:
            for tier in ABLATION_TIERS:
                fname = f"worldwise_{tier}_{mode}_{HERO_BACKBONE}.yaml"
                with open(os.path.join(out_dir, fname), "w", encoding="utf-8") as f:
                    f.write(render("worldwise", mode, HERO_BACKBONE, tier=tier))
                n += 1
                print(f"  wrote configs/methods/{mode}/{fname}")

        # Delete configs from retired campaign rounds (and ablation configs at
        # non-hero backbones left over from the v2-candidate round)
        stale_names = [
            (t, b) for t in RETIRED_TIER_FILES for b in BACKBONES
        ] + [
            (t, b) for t in ABLATION_TIERS for b in BACKBONES if b != HERO_BACKBONE
        ]
        for tier, backbone in stale_names:
            stale = os.path.join(out_dir, f"worldwise_{tier}_{mode}_{backbone}.yaml")
            if os.path.exists(stale):
                os.remove(stale)
                removed += 1
                print(f"  removed retired configs/methods/{mode}/worldwise_{tier}_{mode}_{backbone}.yaml")
    print(f"\nGenerated {n} configs ({removed} stale removed).")


if __name__ == "__main__":
    main()
