"""U-WSGG-Zero (``zero_shot``): the context-free per-object prompt, hero style, three
stages (source: setup/MLLM_ZERO_SHOT.md and lib/mllm/methods/zero_shot/runner.py):

    Stage 1  Object Set And Visual Input Construction (frames only)
    Stage 2  Per-Object Relationship Query (one batched call)
    Stage 3  Parsing And Scene Graph

Every intermediate is an image slot; without ``--images`` each slot draws the
schematic of what it will hold.  Slot keys (PNG names) are listed in ``IMAGES``.

    python scripts/paper_figures/dark/fig_zero_shot.py [--images <dir>] [--video <id>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIM, MUTED, ORANGE, RULE, image_map  # noqa: E402
from mllm_common import (MCanvas, object_set_block, parse_block, ph_answer_p4, ph_prompt_p4,  # noqa: E402
                         visual_input_block)

IMAGES = ["frame_0", "frame_1", "frame_2", "context_frames", "query_tensor", "video_v", "objects",
          "discovery_card", "prompt_card", "answer_card", "scene_graph"]
MODEL = "Qwen2.5-VL-7B"


def build(images, video: str) -> MCanvas:
    c = MCanvas(1720, 1264, images)
    c.band_title(46, "U-WSGG-Zero · The Context-Free Per-Object Prompt")

    # ================= Stage 1: object set, then the visual input =================
    c.stage(88, "Stage 1 · Object Set And Visual Input Construction (Frames Only)")
    ob = object_set_block(c, 136, MODEL, captions=False)
    ox, oy, ow, oh = ob["objects"]
    c.flow(ox + ow / 2, oy + oh, ox + ow / 2, oy + oh + 40, "O → Stage 2, One Question Per o", lsize=8.3,
           loff=(-8, 24), anchor="end")
    c.note(30, 334, "SGDet discovers the object set from V alone: P₆ is told “(No captions available for this video.)”, "
                    "its JSON list is intersected with the vocabulary by exact string match (the GT list is never "
                    "consulted), and each survivor",
           size=9, fill=DIM)
    c.note(30, 347, "is P₅-scored; P(Yes) = p_yes / (p_yes + p_no) is the object confidence multiplied into every "
                    "triplet score, never a filter. PredCls asks about every GT object of the video at every frame, "
                    "unseen ones included.",
           size=9, fill=DIM)
    vi = visual_input_block(c, 366)
    qx, qy, qw, qh = vi["q"]
    c.flow(qx + qw / 2, qy + qh, qx + qw / 2, 628, "Q(f) → Stage 2, Every Frame f", col=ORANGE, lsize=8.5,
           loff=(10, 0), anchor="start")

    # ================= Stage 2: per-object relationship query =================
    c.stage(640, "Stage 2 · Per-Object Relationship Query (One Batched Call)")
    c.a(f'<rect x="24" y="676" width="1272" height="200" rx="12" fill="none" stroke="{RULE}" stroke-dasharray="6 4"/>')
    c.note(38, 866, "F Frames × |O| Objects Prompts · One Batched vLLM Call In Chunks Of 64 · No Deduplication, No "
                    "Text Prefix", size=9.5, fill=MUTED)
    c.tensor(36, 700, 130, 40, "Object o ∈ O", "From Stage 1", col=MUTED)
    c.flow(166, 720, 190, 720, "Name", lsize=8.3, loff=(0, -7))
    c.slot_title(192, 692, "Question P₄(o) · Text Depends Only On o")
    c.image_slot(192, 692, 262, 146, "prompt_card", placeholder=ph_prompt_p4())
    c.tensor(36, 782, 130, 44, "Visual Input Q(f)", "From Stage 1", col=ORANGE)
    c.flow(454, 740, 480, 740, "Text", lsize=8.3, loff=(0, -7))
    c.arrow(166, 804, 178, 804, col=ORANGE, head=False)
    c.arrow(178, 804, 178, 852, col=ORANGE, head=False)
    c.arrow(178, 852, 560, 852, col=ORANGE, head=False)
    c.arrow(560, 852, 560, 812, col=ORANGE)
    c.note(370, 848, "Q(f) · Video", size=8.3, fill=ORANGE, anchor="middle")
    c.vlm(482, 714, 160, 96, MODEL, prompts=(4,), sub2="One Call Per (f, o) · ≤ 128 Tokens", tsize=12)
    c.flow(642, 762, 668, 762, "JSON", lsize=8.3, loff=(0, -7))
    c.slot_title(670, 692, "Raw Answer For (f, o)")
    c.image_slot(670, 692, 220, 116, "answer_card", placeholder=ph_answer_p4())
    c.note(670, 826, "Stored verbatim as raw_response", size=8.5, fill=DIM)
    c.note(912, 700, "Decoding", size=10, fill=MUTED)
    c.wrap(912, 717, ["Qwen2.5-VL-7B-Instruct through vLLM, bfloat16, one GPU;",
                      "temperature 0.2 (fixed in the wrapper), max 128 new tokens,",
                      "one video per prompt, chunks of 64 prompts decoded with",
                      "the largest budget in the chunk. A 32-frame video with",
                      "4 objects makes 128 answer calls, about 30 s on one A40."], size=9, fill=DIM, lh=14)
    c.flow(780, 808, 780, 892, "Raw Response → Stage 3", lsize=8.5, loff=(8, 14), anchor="start")

    # ================= Stage 3: parsing and scene graph =================
    c.stage(904, "Stage 3 · Parsing And Scene Graph")
    parse_block(c, 984, verify_clip="[f ; ±15 Frames, Every 2nd]", verify_prefix="No Text Prefix")
    c.note(30, 1128, "The frame-local clips are decoded only when verification runs; with --skip-verification the "
                     "runner never loads them. Every reported zero_shot row uses --skip-verification.", size=9, fill=DIM)

    # ================= cards, legend, caption =================
    CX, CW = 1320, 380
    c.card(CX, 88, CW, 530, 1, "Frames And Nothing Else",
           ["In SGDet the frozen VLM names the objects from",
            "the whole video alone (P₆) and scores each",
            "survivor with one Yes/No token (P₅); PredCls",
            "takes the GT list. Every query then receives one",
            "video: the target frame first, then up to 15",
            "annotated key frames shared by every frame."])
    c.card(CX, 640, CW, 240, 2, "Ask Per Object",
           ["One three-field question per (frame, object): an",
            "attention label, contacting labels and spatial",
            "labels. The text depends only on the object name;",
            "there is no caption, no graph and no retrieval."])
    c.card(CX, 904, CW, 236, 3, "Parse And Emit",
           ["Labels outside the 3 / 17 / 6 sets become",
            "unknown; every predicted label scores 1.0 because",
            "the Yes/No verification is off. The output is a",
            "class-only world scene graph per frame."])
    c.legend(1166, [("frozen", "Frozen Model"), ("tool", "Program"),
                    ("ghost", "In Code, Unused In Reported Runs")],
             extra_swatches=[(ORANGE, "Target Frame · Visual Input Q(f)")])
    c.badge_legend(30, 1194, [4, 5, 6])
    c.caption(30, 1226, "U-WSGG-Zero.",
              ["Stage 1 fixes the object set (ground truth in PredCls, discovered from the frames in SGDet) and turns the "
               "annotated frames into one video per query, the target frame first.",
               "Stage 2 asks the frozen VLM one three-field question per (frame, object) with no other context. Stage 3 "
               "validates the labels and emits a class-only scene graph. Nothing is trained or placed in 3-D."],
              size=11)
    c.note(1690, 1256, f"panels: {video}", size=8.5, fill=DIM, anchor="end")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="directory of intermediate PNGs named after the slot keys")
    ap.add_argument("--video", default="schematic")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/zero_shot.svg"))
    ap.add_argument("--theme", default="light", choices=["light", "dark"],
                    help="white background (light) or the dark hero palette; read by common.py at import")
    ap.add_argument("--caps", default="title", choices=["title", "upper", "none"],
                    help="capitalise every text: Title Case, UPPER CASE or as written; read by common.py at import")
    args = ap.parse_args()
    c = build(image_map(args.images, IMAGES), args.video)
    p = c.write(Path(args.out))
    print(f"wrote {p}" + (f"  (schematic slots: {len(set(c.missing))})" if c.missing else ""))


if __name__ == "__main__":
    main()
