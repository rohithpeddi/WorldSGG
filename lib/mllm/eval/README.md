# lib/mllm/eval — how MLLM runs are scored on the worldbbox test set

All numbers are produced by `python -m lib.mllm.eval.score_run --method <m> --model <k> --mode <predcls|sgdet>`
over the locked split `/data3/rohith/ag/splits/test_worldbbox_1511.txt` (`annotation_version`
`test_worldbbox` in `/data3/rohith/ag/cache/manifest.json`). Videos without an output are
reported as *missing* (`n_missing`), never dropped silently.

## Pipeline

1. `dump_adapter.py` converts a run (vendored baseline PKLs or Track A/B outputs) into
   per-frame records in the exact `tools/dump_predictions.py` format, one record per annotated
   frame (48,834 frames), slot 0 = person, one slot per GT object (unique by short label, the
   WorldAG rule), plus — sgdet only — one slot per predicted object that is not in the GT.
   * 2D boxes (`bboxes_2d`, `gt_bboxes_2d`) are in **Pi-3 feature space** (WorldAG rule after
     commit 64c7114); objects without a `bbox_2d` (unobserved) get the full-frame placeholder
     for both GT and pred, so PredCls IoU is 1 and they are scored on relations only.
   * `visibility_mask[i] = visible and source not in {rag, gdino, correction}` (same as WorldAG).
   * Scores: a predicted label gets its verification `yes_prob` (1.0 with `--skip-verification`),
     every other predicate 0; a head the model left `unknown` is all-zero.
   * Extra keys: `pred_pair_valid` (pairs the model emitted), `gt_corners` / `pred_corners`
     (8×3 OBB corners in the canonical floor frame), `has_pred_3d`.
2. Regimes:
   * **`wsgg`** (predcls): the stock `evaluate_wsgg_video` + `evaluate_wsgg_video_bucketed`
     (R/mR/hR@{10,20,50,100}, with/no constraint, OO/OU buckets, OU non-trivial) — identical
     protocol to the WorldSGG tables, so MLLM rows are directly comparable.
   * **`loc3d`** (`recall3d.py`): same GT construction and ranking as `evaluate_from_dict`, but
     (a) only `pred_pair_valid` pairs emit triplets, and (b) the triplet matcher requires class
     equality **and oriented 3D IoU ≥ τ on both endpoints** (`iou3d.py` = verbatim copy of
     `lib/detector/monocular3d/evaluation/evaluate_3d.py::compute_iou_3d_obb`, bottom-face
     polygon ∩ × z-overlap). τ ∈ {0 (“unloc”, class match only), 0.15, 0.25}. In predcls the
     predicted corners are the GT corners, so `loc3d/unloc` = `wsgg` restricted to emitted pairs.
     **MLLM SGDet is never matched by 2D box** — MLLMs emit OBBs (Track A/B) or nothing;
     baselines without 3D output get `has_pred_3d = False` and score 0 at τ > 0 by design
     (they are only comparable under `unloc`).
   * **`legacy`** (`legacy_f1.py`): the box-free P/R/F1 of the MLLM baseline paper section
     (vendored `legacy/evaluate_relationships.py`) on `wsg_2d_augmentations` restricted to the
     split (symlinked GT dir `/data3/rohith/ag/cache/mllm/legacy_gt_test_worldbbox_1511/`).
3. The records are written as `<outputs.dumps>/<method>__<model>__<mode>.pkl` (same container as
   `results/bucket_dumps/*.pkl`) so `tools/bucketed_breakdown.py` re-slices them offline.

## Caveats that matter for the paper text

* “With constraint” takes one predicate per (pair, head) in pair order and R@K over the first K
  (stock behaviour, no re-ranking); MLLM labels are mostly ties at score 1.0, so “no constraint”
  ordering among pairs is arbitrary-but-deterministic (`argsort_desc`). Report both, prefer
  R@50/100 for MLLMs.
* mR divides by all 26 predicates (stock), so predicates an MLLM never emits count as 0.
* Frames: all annotated frames of every video (not last-frame-only).
