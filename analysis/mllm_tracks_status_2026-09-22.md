# MLLM tracks on the worldbbox test set (2026-09-22)
annotation_version test_worldbbox = `40decd1785af3c3470290be4b67443e1`; split = 1,511 videos / 48,834 frames;
protocol: lib/mllm/eval/README.md (stock WorldSGG evaluator for predcls, 3D-IoU matching for sgdet).

### full split


**predcls** (all frames; R/mR in %, K=50 unless noted)

| method | model | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU nc mR@50 | OU-nt nc mR@50 | legacy uF1 | legacy corr-only uF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag_all | qwen3vl_8b_thinking_think150 | 150/1511 |  50.4 |  30.3 |  63.8 |  49.9 |  47.7 |  55.5 |  53.8 |   -   |   -   |
| track_b | qwen3vl_8b | 1511/1511 |  52.4 |  31.7 |  63.4 |  48.4 |  47.6 |  45.2 |  40.8 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b | 1511/1511 |  52.1 |  31.1 |  63.2 |  47.9 |  47.3 |  44.5 |  40.0 |   -   |   -   |
| track_b | qwen3vl_8b_thinking_think150 | 150/1511 |  27.2 |  16.4 |  43.2 |  38.2 |  37.8 |  39.6 |  38.2 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b_thinking_think150 | 150/1511 |  27.3 |  16.2 |  43.3 |  38.1 |  37.8 |  39.1 |  37.6 |   -   |   -   |

**sgdet** (all frames; R/mR in %, K=50 unless noted)

| method | model | videos | unloc nc R@50 | unloc nc mR@50 | IoU.15 nc R@50 | IoU.15 nc mR@50 | IoU.25 nc R@50 | IoU.25 nc mR@50 | slots w/ 3D | legacy uF1 | legacy obj F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag_all | qwen3vl_8b_thinking_think150 | 150/1511 |  22.3 |  20.6 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |   -   |   -   |
| track_b | qwen3vl_8b | 1511/1511 |  33.5 |  23.8 |  10.7 |   7.7 |   7.7 |   5.5 |  79.5 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b | 1511/1511 |  33.4 |  23.2 |  11.2 |   7.8 |   8.0 |   5.6 |  79.4 |   -   |   -   |
| track_b | qwen3vl_8b_thinking_think150 | 150/1511 |   0.3 |   0.2 |   0.1 |   0.1 |   0.1 |   0.1 |   0.2 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b_thinking_think150 | 150/1511 |   0.3 |   0.2 |   0.1 |   0.1 |   0.1 |   0.1 |   0.2 |   -   |   -   |

### test_worldbbox_graphs442 split


**predcls** (all frames; R/mR in %, K=50 unless noted)

| method | model | videos | wc R@20 | wc mR@20 | nc R@50 | nc mR@50 | OO nc mR@50 | OU nc mR@50 | OU-nt nc mR@50 | legacy uF1 | legacy corr-only uF1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag_all | qwen3vl_8b_thinking_think150 | 150/442 |  50.4 |  30.3 |  63.8 |  49.9 |  47.7 |  55.5 |  53.8 |   -   |   -   |
| track_b | qwen3vl_8b | 442/442 |  51.9 |  30.4 |  62.1 |  47.8 |  45.8 |  47.1 |  42.8 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b | 442/442 |  51.6 |  29.6 |  61.9 |  47.1 |  45.2 |  46.3 |  41.8 |   -   |   -   |
| track_b | qwen3vl_8b_thinking_think150 | 150/442 |  27.2 |  16.4 |  43.2 |  38.2 |  37.8 |  39.6 |  38.2 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b_thinking_think150 | 150/442 |  27.3 |  16.2 |  43.3 |  38.1 |  37.8 |  39.1 |  37.6 |   -   |   -   |

**sgdet** (all frames; R/mR in %, K=50 unless noted)

| method | model | videos | unloc nc R@50 | unloc nc mR@50 | IoU.15 nc R@50 | IoU.15 nc mR@50 | IoU.25 nc R@50 | IoU.25 nc mR@50 | slots w/ 3D | legacy uF1 | legacy obj F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rag_all | qwen3vl_8b_thinking_think150 | 150/442 |  22.3 |  20.6 |   0.0 |   0.0 |   0.0 |   0.0 |   0.0 |   -   |   -   |
| track_b | qwen3vl_8b | 442/442 |  33.3 |  23.2 |  10.1 |   6.8 |   7.2 |   4.8 |  79.0 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b | 442/442 |  33.2 |  22.8 |  10.6 |   7.0 |   7.6 |   4.9 |  78.9 |   -   |   -   |
| track_b | qwen3vl_8b_thinking_think150 | 150/442 |   0.3 |   0.2 |   0.1 |   0.1 |   0.1 |   0.1 |   0.2 |   -   |   -   |
| track_b(-critic) | qwen3vl_8b_thinking_think150 | 150/442 |   0.3 |   0.2 |   0.1 |   0.1 |   0.1 |   0.1 |   0.2 |   -   |   -   |
