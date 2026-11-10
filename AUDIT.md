# Paper audit — ICLR 2027 submission 46061 (WSGG / ActionGenome4D)

Date: 2026-11-06. Scope: the reviewed LaTeX source (`WSGG_Paper/updated_submission` @9dcbb11; its `.aux` numbering
matches the tables the reviewers cite), checked against the code (`WorldSGG` @8e3ad0c, `WorldSceneGraphAnnotationTool`,
`3DBBoxAnnotationTool`), the saved results (`assets/paper/evidence/*`, CS93371 `/data3/rohith/ag/runs/*`) and the
saved annotations and feature files. Seven read-only auditors, one per area. No training, no inference, no edits to
the paper or code. Small CPU probes read saved files only.

Per-area reports with full evidence (paper line, code line, probe output): [docs/paper_audit/](docs/paper_audit/)
(`MAIN`, `DATA`, `SETUP`, `WORLDWISE`, `EVAL`, `ZEROSHOT`, `FRAME`). IDs below point there. The reports cite their
probes as `SCRATCH\audit\...`; copies are in `docs/paper_audit/probes/`, and server-side copies and outputs are in
`/data3/rohith/ag/runs/rebuttal/audit/`.
"Checked" = I re-read the code or text myself, beyond the auditor's report.

---

## 0. Bottom line

1. **The printed numbers are right.** Every cell of Table 2 (192 cells, 50 marks, 24 Δ), App G Tables 15–19 (960
   cells, 114 marks), the support tables 20–21, Table 3 and Tables 26–30 reproduces from the saved results, and the 28
   checkpoint hashes match the manifest. The problems are in **what the numbers measure** and **what the text claims**.
2. **Five implementation facts change what the paper may claim** (all verified in code):
   - Object "persistence" does not exist in the model input: slot *n* is the *n*-th entry of each frame's list, with no
     cross-frame association (C1).
   - In SGDet, the visibility flag of every detection is set from the ground truth at test time (C2).
   - With-Constraint R@K is not ranked by confidence (C3).
   - Table 3's SGDet scorer never counts relations of hallucinated objects, contrary to App H.3.1 (C7).
   - Table 3 runs on undisclosed 184- and 14-video subsets (C8).
3. **Several headline sentences are contradicted by the paper's own appendix**: "persists even under a matched
   detector", "significantly improve ... unobserved", "WorldRAG enhances MLLMs' reasoning", "metrically accurate",
   "3D-localised detection and tracking", and the abstract's retriever mechanism, which WorldWise++ does not have.
4. **The dataset section claims validation that has no record** (§1b): the 0.385 / >0.9 box-agreement numbers, the
   92/90/89% relationship agreement, and the hand correction of test boxes. The dataset counts themselves check out.
   Resolve this with the coauthors and annotators before the rebuttal: these are the claims the reviewers praised.
5. **All reviewer points are valid**; none is a misreading. Several are worse than the reviewer states (y9Am W1, SRrf W1,
   SRrf Q4). Section 2 lists what existing material answers each one.
6. Most fixes are text: narrow the claims, disclose the protocol in the main text, restore the dropped appendix file.
   The identity and visibility facts (C1, C2) cannot be fixed in text alone. They need disclosure now, and any method
   claim built on "an object's visible history" must be withdrawn or re-established with new runs.

Corrections to things I said earlier in this session:
- The protocol table `evaluation_protocols.tex` ("Predicted 2D/3D boxes") is **not** in the paper.
- `sup_evaluation_details.tex` is **not** `\input`, so the coverage paragraph (85,639 objects, 81.3%), the compute
  table and the run manifest are not in the reviewed paper either. The test-selection disclosure the reviewers saw
  comes from `sup_methods_shared.tex`.
- G.2's wording "an unobserved object ... is matched by class, the identity its object representation carries" is
  itself inaccurate (EVAL-3), so G.2 is not a fully correct description of the scorer.

---

## 1. Critical findings (deduplicated)

| # | Finding | Where (printed) | Evidence | Auditor IDs | Fix without new experiments |
|---|---|---|---|---|---|
| C1 | **No object identity across frames.** "Object representation *n* refers to the same physical object in every frame ... identity by construction" is false. Slot *n* is position *n* in that frame's feature list. In PredCls a slot changes class between consecutive frames in 17.1% of slot pairs (88% of test videos); in SGDet 68.4%. Only 36.6% of PredCls unobserved slot-frames have any visible frame of the same class in their own slot (300 videos), although the class is visible somewhere in the video in 100%. The retriever, WW++ temporal attention, motion encoder, temporal edge attention and the baselines' LKS buffer all index by this position. Fig. 3's "tracked blanket" (TM0BV) occupies two slots. | App F.1.1 (`sup_methods_shared.tex:25`), F.3, abstract, intro L42, Sec. 5.1 | `world_ag_dataset.py:524-606` (per-frame list), `:805-863` (stacked by index); `associative_retriever.py:129-131` (per-slot sequence). **Checked:** UG4M2 slot 2 = `chair` in frame 000092, `closet` in 000435. | SETUP-1, WORLDWISE-1 | State plainly that slots are per-frame list positions without association; give the drift rates; drop "visible history", "identity by construction", "tracked". Withdraw claims that gains come from completion over an object's own history. |
| C2 | **SGDet visibility flags come from the ground truth at test time, for every method.** An unmatched detection is looked up in the annotation by class and gets that object's `visible` flag; 77% of the slots the scorer credits as unobserved carry this GT-derived flag. In PredCls the flag is also GT (GroundingDINO fill = unobserved), which the main text does not list among the given inputs. | Sec. 5 (`evaluation_wsgg.tex:3`), F.1.1, G.2 | `world_ag_dataset.py:551-558`, `:604-606`. **Checked.** Probes `setup_probe4/9.py` (200/300 videos). | SETUP-2 | Disclose in Sec. 5 and F.1.1; label SGDet UO columns "with GT visibility flags". Measuring the effect needs inference. |
| C3 | **With-Constraint R@K / mR@K are not top-K by confidence.** The arg-max rows are kept in construction order (all attention rows, then spatial, then contacting) and the first K are scored; the sort is commented out. Little effect on PredCls @20 (few pairs per frame), large in SGDet @10 (mostly attention only). Sorting the saved dumps changes WW++ SGDet WC mR@10 from 6.6 to 17.4 (W-DSGDetr++ 4.8 → 10.5) and flips two pairwise orders. | Sec. 5 metric definition, Table 2, App G | `rescore_common_gt.py:123` (no sort), `:184` (`rows[:k]`); `evaluation_recall.py:487-490` (sort commented out, apparently inherited from the STTran-style evaluator). **Checked.** | EVAL-1, MAIN-16 | Disclose the convention, or re-score the saved dumps with sorting (CPU only, no inference; the user decides whether that counts as an experiment). |
| C4 | **"Advantage ... driven by superior detection that persists even under a matched detector" is contradicted by App G.1.** No WW++ run uses the baselines' ResNet-50 detector. WorldWise (base) on that detector has WC R@20 22.9 vs 23.4–24.3 for the baselines ("gains No-Constraint mean recall but not raw recall"). SGDet scoring uses the frozen detector's boxes and labels, not WW++'s refined ones. The main text never says the two families use different detectors. | Sec. 5.2 Insights (`experiments_wsgg.tex:98`) | App G.1 (`sup_experiments_wsgg.tex:9`). **Checked.** | MAIN-2, EVAL-2, WORLDWISE-2, SETUP-4 | Replace with what G.1 says; state the detector and feature difference in Sec. 5. |
| C5 | **Conclusion: WW++ "significantly improve[s] relationship recovery for both observed and unobserved entities."** Table 2 shows SGDet UO raw recall far below the baselines (NC R@20 14.1 vs 22.1). One seed, no test, no interval. | Conclusion L1 | Table 2; App G.1, App J. **Checked.** | MAIN-1, FRAME-6 | Restrict to class-balanced recall; drop "significantly". |
| C6 | **"WorldRAG ... enhances MLLMs' reasoning"** (intro contribution) is contradicted by Table 30: GraphRAG 47.4 / 28.8 vs no context 47.8 / 30.6; a random training graph gets 50.9. App H.7 concedes it. The main text never cites Table 30. | Intro L48, abstract | Table 30 (`ext_mllm_context150.tex`). **Checked.** | MAIN-3, ZEROSHOT-4 | Concede in the main text; reposition GraphRAG as an analysed baseline. |
| C7 | **Table 3 SGDet scorer ignores relations of hallucinated objects.** App H.3.1 says predicted objects absent from the GT count as false positives; the code loops over GT objects only. Under the stated rule (CPU re-score of saved predictions): Qwen2.5-VL RAG µF1 43.7 → 16.6; RAG beats Caption on 2/5 backbones instead of 4/5; best backbone becomes Ovis2.5. Caption and RAG also use different object discovery, though Table 24 says they share it. | Table 3 (SGDet half), App H.3.1 (`sup_mllm_protocols.tex:65-70`) | `WorldSceneGraphAnnotationTool/backend/evaluation/evaluate_relationships.py:434` (loop over `gt_objects`). **Checked** (code); re-score numbers from ZEROSHOT `zs_sgdet_fp.py`. | ZEROSHOT-1 | Describe the coded rule and report the saved object-detection P/R/F1, or drop/flag Table 3's SGDet half. |
| C8 | **Table 3 sample sizes undisclosed:** PredCls 184 videos (38 outside the 1,511 test split), SGDet 14 videos / 271 frames / 852 pairs (2 outside). The set is whatever overlapped across all runs, including unreported ones. App H.3.5 wrongly says no manifest exists; the table's own `.tex` comment lists the counts. | Table 3 (main, p.9), App H.3.5 | `uwsgg_gt_plus_corrections.tex` comment; E: folders. Found independently by two auditors. | MAIN-4, FRAME-3, ZEROSHOT-2 | State the counts and selection rule in the caption; fix H.3.5. |
| C9 | **Most Table 3 Caption→RAG deltas are not retrieval effects.** Ovis2.5 had no captions on 172/184 videos, MiniCPM on 96/184; KeyeVL's Caption run returned no valid answer for 58–64% of queries. Only InternVL2.5 and Qwen2.5-VL are clean comparisons. | Table 3 | ZEROSHOT probes (`zs_zero_caps.py`, `zs_parse_rate*.py`) | ZEROSHOT-3 | Footnote the failures, or restrict the comparison to the two clean backbones. |
| C10 | **"Metrically accurate ... inter-annotator 3D IoU > 0.9" in the main text.** No script or result behind 0.385 or >0.9 has been found; App D.2.3 and App J qualify it. Commit 1e1394d restored the unhedged main-text wording about 20 minutes after an internal pass had hedged it. | §4 (`scene_construction.tex:108-120`) | `git log` of the paper repo; App D (`sup_manual_correction_3dbbox.tex:145`). See also §1b (DATA). | FRAME-1, DATA (§1b) | Remove "metrically accurate"; state what the numbers are and where they come from, or drop them. |
| C11 | **"3D-localised detection and tracking" (Sec. 5.2) and other 3D claims.** No tracking (C1); SGDet uses camera-frame boxes; the frozen detector has mean IoU3D ≈ 0.13 and Hit@0.50 ≈ 1% (Table 14); no supervised output is evaluated in 3D; WW++ 3D outputs were saved for 8 videos only. | Sec. 5.2, abstract | Table 14; `worldwise_pp/model.py:461` (`slot_corners`, never scored) | MAIN-5, WORLDWISE-11, WORLDWISE-14 | Remove the tracking/3D-localisation claims, or scope them to inputs. |

### 1b. Dataset and annotation (DATA)

The headline counts check out against the annotation files: 9,027 videos, the 7,516 / 1,511 split, 48,834 test
frames, the triplet and OBB counts. The problems are in what the paper says was done to produce and validate them.

These four are integrity-level. For DATA-1 and DATA-2, only the authors or annotators can say whether a study was run
outside the repos and tools. Until a record turns up, the rebuttal must not defend these numbers.

| # | Finding | Where (printed) | Evidence | Fix without new experiments |
|---|---|---|---|---|
| DATA-1 (= C10) | **The 3D-box agreement numbers have no source, and there are no corrected boxes to measure.** The sentence first read "Mean IoU3D of 0.1285", the monocular detector's own IoU3D (Table 14, DINOv3-L Joint). Two hours later it became 0.385, with no data. The June Firebase snapshots contain zero edited boxes, and no exported object carries a `manual_correction` tag. | §4 (`scene_construction.tex:108-120`), App D.2.3, App J | Paper repo `git show 3e46ac6` (2026-05-07 04:22: `-0.1285 / +0.385`); `eecf6a9` (02:07: 0.1285 added). **Checked.** | Remove both numbers and "metrically accurate", unless a record is found. |
| DATA-2 | **The semantic agreement study (92 / 90 / 89% on 200 videos) has no record.** It first read "300 videos" (`eecf6a9`) and became "200" in `3e46ac6`, the same commit as DATA-1, with the percentages unchanged. The correction tool stores one edit stack per video and no annotator ID, so independent labels were never stored. `docs/ICLR_PLAN.md` (Aug 2026) still lists IAA as future work. The NeurIPS rebuttal promised a from-scratch control annotation that App J lists as future work. | §4 (`scene_construction.tex:118-121`), App E.2.2, App J | `wsg_api.py:341-362`; ICLR_PLAN L105-121 | Remove the sentences unless a documented study exists; keep the edit rate (LFR), described as editing activity. |
| DATA-3 | **The paper says test boxes and trajectories were corrected by hand; only the floor transform was.** All 1,511 test videos have a human floor correction. Every exported test box is an automatic refit or a nearest-frame copy: `detected` 124,926; `hold_prev` 29,039; `hold_next` 26,450; `static_copy` 51,144. | Fig. 2 caption, §4 (d, f), intro L45, App D.2.2 | `videos_floor_corrections.txt`; per-object `source` fields in the test PKLs (`data_audit.json`) | Rewrite: annotators correct the floor transform, then boxes are refit automatically; unobserved relationships are corrected by hand. |
| DATA-4 | **The described reconstruction does not produce the released geometry.** The trimmed-ICP call is commented out, the box generator reads the raw dynamic π3 pass, and a static π3 run exists only for 0DJ6R. | App C, §4 | Code locations in DATA.md | Describe the pipeline that was run. |

Major dataset findings:
- **DATA-5:** training geometry ignores the floor corrections; all 165 corrected training videos use the automatic floor.
- **DATA-6:** training pseudo-labels come from Qwen3-VL with verification skipped (every score is 1.0), not from the
  models and verification step the appendix describes; the test seeds are very likely the same model. This links to
  ZEROSHOT-6.
- **DATA-7:** there is a large undisclosed train/test shift in unobserved-object labels (`in` is 44.6% in train vs 9.3%
  in test).
- **DATA-8:** the inventory, identity and visibility are never defined. In practice identity = the AG class label,
  "unobserved" = no AG box in that frame, and frames with no visible object are dropped. This answers y9Am W1 and
  explains C1.
- **DATA-9:** the rule for selecting the 1,511 test videos is undisclosed: they are the videos whose floor correction
  was completed.
- **DATA-10:** the release does not contain the interpolation and slerp the paper describes; boxes of hidden objects
  are frozen copies of the nearest seen box.
- **DATA-11:** the metric scale is not stored in the test PKLs.
- **DATA-12:** the LFR comes from a partial log (745 of 1,525 videos), and the main text calls it a quality metric.
- **DATA-13:** annotators saw the VLM proposals, and the rule for relations that cannot be determined from the video
  exists only in an internal document.

Minor: DATA-14..16 (the "beneath and in are rare" statement, the claim that the MLLM labels every pair, and appendix
running examples taken from outside the released splits).

---

## 2. Major findings, grouped

**Supervised protocol not stated in the main text** (MAIN-6, EVAL-4, FRAME-7, MAIN-12, SETUP-3, EVAL-9, EVAL-11)
- An unobserved object is scored only if GroundingDINO fires on its class in that frame (score ≥ 0.3): 18.8% of
  unobserved object-frames. The paper calls the rest "objects without pre-extracted features". Covered pairs also get a
  union feature from the current-frame GroundingDINO box.
- SGDet matching is 2D IoU for visible objects and class only for unobserved objects; the baselines use ResNet-50
  features and detector, WorldWise+/++ use DINOv3; only person–object relations exist and are scored. None of this is in
  Sec. 5.
- Table 23 / App G.6 present counts of feature-slotted objects as if they were the full annotation.

**Class-only fallback is more lenient than described** (EVAL-3; checked in `rescore_common_gt.py:137-150`)
- A detection that fails 2D IoU against every visible object of its class is credited to an unobserved object of that
  class. Such mislabelled or mislocalised visible detections carry 15–27% of SGDet UO hits (about +3 points of UO NC
  R@20). G.2 describes the rule as the slot's own identity.

**Test-set checkpoint selection** (MAIN-7, FRAME-2, EVAL-6, SETUP-5)
- Not in the main text (the sentence is commented out, although `HIGH_PRIORITY_REVIEW_FIXES.md` marks it fixed) and
  missing from App J.
- Asymmetric: the July runs (baselines, WorldWise) were selected on the older 1,525/1,526-video annotation, the
  September runs (WW+, WW++) on the reported 1,511.
- It picks epochs 2–4 for the PredCls baselines. In the saved per-epoch logs (last frame, older annotation), the
  WorldWise vs best-baseline WC mR@20 gap is +11.3 at the selected epochs and +3.4 at epoch 20; the raw-recall gap
  widens. Two auditors computed this independently.

**Dropped appendix file** (EVAL-5, WORLDWISE-12, FRAME-9)
- `sup_evaluation_details.tex` is never `\input`: coverage, selection provenance, compute/inference cost (t10) and the
  manifest (t22) are absent. Re-including it answers y9Am's inference-cost request from saved numbers.

**Abstract/intro describe a mechanism WW++ does not have** (MAIN-10, FRAME-5, WORLDWISE-7)
- "Camera-pose-conditioned cross-attention over an object's visible history" is the AssociativeRetriever, which
  WorldWise++ removed.

**Attribution of gains** (MAIN-9, WORLDWISE-4/5/6/8)
- Baselines have no logit adjustment and are trained with VLM pseudo-labels (λ_vlm 0.2), which the paper's own
  ablation finds harmful.
- Ablation Table 22: removing logit adjustment drops WC mR@20 49.5 → 41.7; removing artificial masking (which also
  removes L_recon and L_sim, contrary to the text) only to 49.0. The mR advantage comes mostly from the loss recipe.
- Table 22 was scored on the legacy 1,526-video annotation with the legacy evaluator, All bucket only, selected
  epochs 2–12; this is not disclosed, and its SGDet counterpart (where removing masking raises mR) was dropped.
- Artificial masking does not hide the masked object's appearance (unmasked union features, full-frame cross-attention).
- The PUF / last-known-state comparison does not support "learned completion": it is confounded by logit adjustment,
  and the bidirectional hold beats WorldWise (ResNet-50) on the same PredCls inputs in UO NC R@20 (82.8 vs 81.2), though not in mR (36.0 vs 44.6).

**Other method facts** (WORLDWISE-3, SETUP-6)
- L_node is never trained in any WorldWise SGDet model; the node-head weights are at initialisation.
- SGDet training: 39% of training pairs are unmatched detections given a default `looking_at` label; one pair per class
  per frame; 33% of test pairs are duplicates.

**Explanations that contradict App G.1** (MAIN-8/11, WORLDWISE-9/10, EVAL-7/8)
- "Raw recall on unobserved objects is bounded by proposal misses": G.1 shows the gap persists with the same detector.
- "Ablations confirm that detection queries and supervision are critical": holds in PredCls only; in SGDet no cell
  moves by more than 0.5.

**Unanswered research questions** (MAIN-13, FRAME-13, SETUP-7, EVAL-14)
- RQ1 (direct transfer) is answered only by a footnote with no number; Sec. 5.2 has no result prose; RQ4 and the
  conclusion's "gap" rest on appendix-only results.

**Zero-shot side** (ZEROSHOT-5..9, MAIN-14, FRAME-10)
- Caption-key rename bug: all three Table 30 GraphRAG rows (150/150 videos) and 442 videos of the full-split GraphRAG
  runs had no captions; those runs feed the App H.7 SGDet hallucination rates.
- The unobserved-object test labels were likely seeded by Qwen2.5-VL RAG proposals (then corrected); Qwen2.5-VL agrees
  with the seeds 2.5–10× more than the other backbones and is Table 3's top row. Undisclosed. (Status: likely.)
- Zero-shot SGDet R@K ranking places hallucinated objects after GT objects (GT-informed order), biasing K=10/20 in
  Tables 26–29.
- Table 3 has no no-context control, although saved no-context predictions exist for all five backbones.
- The abstract credits WorldRAG with geometric inputs and outputs it does not have; Methods A/B appear nowhere in the
  main text.
- The 150-video subset (Table 30) is a seed-0 sample of the 442 videos with pre-existing graphs, the same 442 hit by
  the caption bug; the rule is undisclosed.

**Framing and statements** (FRAME-4, FRAME-8, FRAME-11, FRAME-12, MAIN-15, MAIN-20)
- Novelty: the abstract and Table 1 call persistence unique, while App I.4 and the PUF baseline show persistent 3D/4D
  graph methods exist; Table 1's 4D-SGG row is wrong.
- Limitations: Sec. 6 has almost none and no pointer to App J; App J omits test selection, seeds, the 150-video rule
  and the 3D-IoU provenance, and calls the 81% coverage loss "some".
- The AI-use statement omits that AI agents built and ran method and evaluation code, and differs from the OpenReview
  form.
- No reproducibility, release, license or annotator statement for a dataset paper (the NeurIPS `checklist.tex` is not
  included).
- The unobserved-mR lead rests on 12 predicates with under 100 instances, one seed, no interval.

Minor findings (wording, notation, citations, figure captions, rounding, the 1,525 vs 1,526 count, the DSG-DETR
miscitation, frames-only remnants in figures, a figure video outside the test split, etc.) are listed in each area
report: MAIN-16..21, FRAME-14..18, WORLDWISE-13..21, ZEROSHOT-10..18, EVAL-10/12..15, SETUP-8..14.

---

## 3. Reviewer points

| Point | Verdict | What the code / paper shows | Answerable from existing material? |
|---|---|---|---|
| **SRrf W1** — no 3D IoU; occluded objects matched by class | **Valid, and worse** | No 3D test anywhere in supervised scoring (`bbox_overlaps_3d` is never called). The class fallback also credits mislabelled visible detections (15–27% of SGDet UO hits). Visibility flags are GT (C2). | Text now. Table 14 already bounds a 3D-matched result: the frozen detector has Hit@0.50 ≈ 1%. A 3D-matched re-score of WW++ needs new inference (dumps have no 3D fields). |
| **SRrf W2 / Q1** — 67,347 of 358,793 UO triplets; filtering criteria; rescore with misses | **Valid** | Criterion: a same-class GroundingDINO box (score ≥ 0.3) in that frame (SETUP-3). | **Yes (pooled).** All methods miss the same 291,593 triplets, so pooled full-inventory recall = pooled hits / 358,793 from `common_gt/results/*.json`. PredCls UO NC R@20 becomes 14.5–15.5 for all eight Table 2 rows (WW++ 15.5, best baseline W-USG 15.4); SGDet UO NC R@20 2.3–4.0 (baselines 3.7–4.0 ahead of WW++ 2.7). Pooled rankings unchanged. Per-frame values need a CPU re-score of saved dumps. |
| **SRrf Q2** — 3D IoU of WW++ boxes, visible vs occluded | **Valid** | WW++ predicts `slot_corners` but the saved full-split dumps contain no 3D fields; 3D outputs exist for 8 SGDet videos only. | **No** — needs an inference dump (PLAN.md A3). |
| **SRrf Q3** — geometric-consistency sample, reference boxes, independent and blind re-annotation, undeterminable relations | **Valid** | (a) No geometric agreement study exists and no hand-edited boxes exist (DATA-1, DATA-3). (b) There is no blind re-annotation; annotators saw the VLM proposals (DATA-13). (c) A rule for relations that cannot be determined exists only internally. | (a) Concede and remove the claims. (b) Needs new annotation. (c) Add the internal rule as text. |
| Dataset praise (all three) — "human correction ... annotation agreement" | **Partly warranted** | The correction of unobserved relationships and of the test floors is real. The agreement figures, the box correction and the ICP step are not supported. | Reword the dataset claims so the praise survives the rebuttal (§1b). |
| **SRrf Q4** — GraphRAG backbone-dependent; Table 30 ≤ no context | **Valid, and stronger** | Plus C7 (SGDet scorer), C8 (subsets), C9 (caption failures), caption-key bug, label seeding. | Concede in text. A full-split caption / GraphRAG / no-context comparison is already scored on the shared scorer (ZEROSHOT report). |
| **wqgS W1** — coverage and class-only matching | **Valid** | = SRrf W1 + W2. | As above. |
| **wqgS W2 / Q2** — test-set selection; fixed final checkpoint | **Valid, and it matters for mR** | Selection picks baseline epochs 2–4; asymmetric annotation. | **Partly.** Saved per-epoch logs give a last-frame proxy: WC mR@20 gap +11.3 at selected epochs vs +3.4 at epoch 20; raw-recall gap widens. checkpoint_19 all-frame dumps exist for 5 July runs (old annotation). All-frame scores at a fixed epoch for WW+/WW++ need inference; a non-test selection needs a validation split. |
| **wqgS Q1** — rescore against all relations; 3D localisation matching | **Valid** | | Part 1 yes (pooled, as SRrf Q1). Part 2 needs inference. |
| **y9Am W1** — preconstructed representations, supplied identities, complete videos, person–object | **Valid, and worse** | Identities are *not* supplied: slots drift (C1). SGDet visibility is GT (C2). Offline and bidirectional over keyframes. Person–object only. | Text disclosure only. End-to-end evaluation needs new runs. |
| **y9Am W2** — fair controls for the completion mechanism | **Valid** | Families differ in loss (logit adjustment, λ_vlm), features, detector, selected epochs. | **Partly.** WorldWise (ResNet-50) shares the baselines' features and detector (PredCls WC R@20 68.0 vs 68.5; WC mR@20 49.1 vs 38.9). Table 22 shows the loss drives mR. PUF bidirectional LKS: UO NC R@20 82.8 vs 81.2, mR 36.0 vs 44.6. A direct retriever/decoder ablation under aligned recipes needs runs, and is moot until slots are aligned (C1). |
| **y9Am W3** — independent validation, unified protocol, repeats, inference cost | **Valid** | Test selection, detector and coverage differences, WC ordering (C3), seed 0 only. | Inference cost: yes, re-include t10 (state that preprocessing is excluded). Seeds and validation split: new runs. |
| Presentation 2 (all three) | **Valid** | Too many names, protocols and object inventories; key tables need cross-reading; stale appendix guide; caveats under "Related Work". | Text (FRAME-16, FRAME-17). |

---

## 4. Text fixes, in priority order (no new experiments)

0. Dataset validation claims (§1b DATA-1..4): find the records or remove the numbers; rewrite Fig. 2 and §4 so they
   describe what was corrected (floors and unobserved relationships) and what was automatic (boxes).
1. Withdraw or rewrite the contradicted headline claims: C4, C5, C6, C10, C11, the abstract's retriever mechanism,
   the novelty "unique" claim, and the "bounded by proposal misses" and "ablations confirm" sentences.
2. Disclose the identity and visibility facts (C1, C2) wherever the method and the setting are described.
3. Add a main-text protocol paragraph to Sec. 5: coverage rule and 18.8%, matching rule (2D IoU / class only, with the
   fallback), GT visibility flags, detectors and features per family, person–object only, test-set selection and its
   asymmetry, single seed, WC ordering convention (C3).
4. Table 3: state the 184 / 14-video sets and their selection, describe the coded SGDet rule (C7), footnote the caption
   and parse failures (C9), disclose label seeding (ZEROSHOT-6); or move Table 3 to the appendix.
5. Re-include `sup_evaluation_details.tex` (coverage, compute, manifest) after correcting it, and fix H.3.5.
6. Disclose Table 22's protocol (legacy annotation, legacy evaluator, All bucket) and that "−masking" removes the
   reconstruction objective.
7. Expand Sec. 6 and App J: every item above, plus the 3D-IoU provenance.
8. Statements: AI use, reproducibility, release, license, annotators.
9. Minor items from the area reports.

## 5. Implications for PLAN.md (not edited; yours to decide)

- **C1 undercuts the planned completion ablations (B3).** Comparing `mwae` / `hold_last_visible` / `mask_token`
  measures little while slots are not the same object across frames. Either align slots by identity first (code change
  plus retraining) or drop the "completion over an object's history" framing.
- **C2:** any new SGDet run or re-score should also report a variant with visibility flags that are not taken from the
  ground truth.
- **A3 (3D re-score):** Table 14 suggests detector-box 3D matching will leave about 1% of detections matched at IoU 0.5;
  only WW++'s refined `slot_corners` could do better, and they need an inference dump.
- **A1 (full-inventory re-score)** is answerable now in pooled form from the saved results (Section 3, SRrf Q1).
- **C3:** decide whether the rebuttal re-scores WC with sorting; it changes SGDet WC @10 substantially.
- **A4 (fixed final epoch):** the saved per-epoch logs already give the last-frame proxy (+11.3 → +3.4).
