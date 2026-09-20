# MLLM tracks — status and metrics (2026-09-19)

Split: `world4d_rel_annotations_worldbbox/test`, 1,511 videos / 48,834 frames,
`annotation_version test_worldbbox = 40decd1785af3c3470290be4b67443e1`.
All frames, stock WorldSGG evaluator (`lib/mllm/eval/`), values in %.

## Track-level metrics (predcls, complete runs)

| method | model | context | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU nc mR@50 | OU-nt nc mR@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot (unlocalized, Graph-RAG baseline) | qwen25vl_7b | per-object prompt | 1511 | 46.9 | 25.1 | 60.9 | 43.1 | 40.3 | 45.7 | 42.8 |
| **Track A** (localized: marked frames + BEV) | qwen3vl_8b | per-frame, 6 context frames + BEV | 1511 | **51.7** | **30.5** | **62.8** | **47.2** | **46.5** | 44.4 | 40.0 |

442-video subset whose Stage-1 graphs pre-existed (sanity check that the 1,069
filled graphs behave like the old ones): zero_shot 46.2 / 24.7, Track A 51.0 / 28.8
— within 0.7 / 1.7 pt of the full split, so the filled graphs are not an artefact.

Reading: the localized track beats the unlocalized baseline by **+4.8 wc R@20 /
+5.4 wc mR@20**, and the gain is concentrated on *observed* objects
(OO mR@50 46.5 vs 40.3). On the unobserved buckets it is slightly *behind*
(OU-nt 40.0 vs 42.8): marking the frame and the BEV helps what is visible,
whereas the retrieved graph context is what carries permanence reasoning.
That gap is exactly what Track B (tool-loop agent + geometric critic) targets.

## Truncation bug found and fixed (2026-09-19, commit 92e1441)

The tracks answer with a single JSON array over the frame's object list and keep
inventing ids past the end of it, so they hit `max_new_tokens=1024` on most
frames. `extract_json` required a balanced object, so a cut-off response made the
**whole frame** count as an empty prediction:

| | frames parsed | object predictions | wc R@20 |
|---|---:|---:|---:|
| Track A predcls, strict parser | 18,295 / 48,834 (37.5 %) | 64,543 (27.9 % of the 231,639 GT pairs) | 24.9 |
| Track A predcls, after salvage  | 48,834 / 48,834 (100 %) | 231,632 (99.997 %) | **51.7** |

Fix: `extract_json` now falls back to collecting the complete `{...}` entries
before the cut, and `lib/mllm/tools/repair_truncated.py` re-parses runs that were
already generated (predcls payloads rebuilt from the annotations, sgdet from the
detection / lift caches) — no generation is repeated, the cached responses are
reused. `max_new_tokens` is deliberately left at 1024: the tail that gets cut is
the model's over-generation past the real object list (predcls recovers 99.997 %
of GT pairs), so raising it would only buy slower decoding.

**Runs produced before commit 92e1441 must be passed through
`repair_truncated` before scoring** — that is Track A sgdet (in flight, started
under the old parser) and nothing else; Track B and every later job start with
the fixed parser.

    python -m lib.mllm.tools.repair_truncated --method track_a --mode sgdet --model qwen3vl_8b

## Queue status (3 workers, one per GPU, `b3.jobs`)

| job | state |
|---|---|
| zero_shot predcls | done (7.1 h) |
| Stage-1 graph fill, 1,069 videos + register | done (4.9 h) |
| Track A predcls | done (12.6 h), repaired + scored |
| Track A sgdet | running GPU 0, ~12.5 h left |
| caption_all predcls | running GPU 1 |
| rag_all predcls | running GPU 2 |
| wsg_agent predcls, Track B predcls+sgdet, 4 sgdet baselines, 4 Thinking-150 rows, final scoring | queued |

Remaining ≈ 146 GPU-hours over 3 GPUs ≈ 2–2.5 days: all predcls rows by
~Sep 20 midday, Track B by ~Sep 21, Thinking-150 + `score_all` by ~Sep 22.
