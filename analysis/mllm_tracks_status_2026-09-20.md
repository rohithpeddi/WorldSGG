# MLLM tracks — status and metrics (2026-09-20)

Split: `world4d_rel_annotations_worldbbox/test`, 1,511 videos / 48,834 frames,
`annotation_version test_worldbbox = 40decd1785af3c3470290be4b67443e1`.
All frames, values in %. predcls uses the stock WorldSGG evaluator; sgdet uses
`lib/mllm/eval/recall3d.py` (class match + oriented 3D-IoU), **not** the 2D-IoU
protocol of the supervised table — the two sgdet columns are not comparable.

## predcls

| method | context | model | pair coverage | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU nc mR@50 | OU-nt nc R@50 | OU-nt nc mR@50 | legacy uF1 |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| zero_shot | none (per-object prompt) | qwen25vl_7b | 231,639/231,639 | 46.9 | 25.1 | 60.9 | 43.1 | 40.3 | 45.7 | 41.4 | 42.8 | 49.8 |
| caption_all | object captions | qwen25vl_7b | 231,639/231,639 | 45.3 | 26.1 | 59.9 | 43.6 | 40.6 | 46.7 | 40.6 | 44.1 | 48.2 |
| rag_all | Graph-RAG retrieval | qwen25vl_7b | 231,639/231,639 | 46.9 | 26.0 | 61.1 | 43.8 | 40.8 | 47.2 | 41.9 | 44.6 | 49.9 |
| **Track A** | localized: marked frames + BEV + OBB table | qwen3vl_8b | 231,632/231,639 | **51.7** | **30.5** | **62.8** | **47.2** | **46.5** | 44.4 | 30.8 | 40.0 | — |

Track A wins overall and on observed objects (OO mR@50 +5.7 over the best
baseline) but is clearly behind on the non-trivial unobserved pairs
(OU-nt R@50 30.8 vs 41.9): localized rendering helps what is visible, the
retrieved graph is what carries permanence. Part of the gap is the backbone
(Qwen3-VL-8B vs Qwen2.5-VL-7B) — the Thinking-150 rows separate prompt from
model. The three unlocalized baselines are within ~1.5 pt of each other.

## sgdet (3D localization)

| method | model | videos | unloc nc R@50 | unloc nc mR@50 | IoU.15 nc R@50 | IoU.15 nc mR@50 | IoU.25 nc R@50 | IoU.25 nc mR@50 | slots with a predicted 3D box |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Track A | qwen3vl_8b | 1511/1511 | 33.3 | 20.2 | 11.0 | 6.6 | 7.9 | 4.7 | 80.8 % |

The headline MLLM-localization result: relations survive class-only matching
(33.3 R@50) but collapse once the emitted OBB has to overlap the GT box
(11.0 at IoU 0.15, 7.9 at 0.25). Track B's geometric critic targets exactly this.

## Run / queue state

| run | mode | state |
|---|---|---|
| zero_shot, caption_all, rag_all | predcls | done 1,511/1,511, scored |
| Stage-1 graph fill (1,069 videos) + register | — | done |
| Track A | predcls | done, repaired, scored |
| Track A | sgdet | done, repaired, scored |
| Track B | predcls / sgdet | running, ETA Sep 21 ~02:00 / ~15:00 |
| zero_shot, caption_all, rag_all | sgdet | queued |
| Track A + B, Thinking-150 | both | queued last |

## Two defects found while scoring

1. **Response truncation** (fixed, `92e1441`). The tracks answer with one JSON
   array and keep inventing ids past the real object list, so they hit
   `max_new_tokens` on most frames; the strict parser then dropped the whole
   frame. Track A predcls parsed 18,295/48,834 frames and scored 24.9 wc R@20.
   `extract_json` now salvages the complete entries, and
   `lib/mllm/tools/repair_truncated.py` re-parses already-generated runs from
   their stored `raw_response`. predcls: 48,834 frames, 231,632 pairs, 51.7
   wc R@20. sgdet: 8,389 -> 48,785 frames with predictions, 36,295 -> 344,216
   object predictions. Any run generated before `92e1441` must go through the
   repair tool before scoring.
2. **Silent engine death** (fixed, `c1195d8`). One unlocalized predcls run OOMed at
   `gpu_memory_utilization: 0.90`; the vendored runner swallowed one
   `EngineDeadError` per remaining video and exited 0, so the queue recorded
   "done" with 77/1,511 videos written. The budget is now 0.85 and the queue
   marks a job failed when its log shows the engine died.
