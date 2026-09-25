"""U-WSGG-Sub (``caption_all``): per-object prompting with a caption transcript, hero
style, three stages (source: setup/MLLM_CAPTION_ALL.md, lib/mllm/methods/caption_all/runner.py
and the Stage-1 build in lib/mllm/methods/graphs/runner.py):

    Stage 1  Caption Transcript Generation (the Stage-1 build, once per video)
    Stage 2  Caption-Prefixed Per-Object Query
    Stage 3  Parsing And Scene Graph

Every intermediate is an image slot; without ``--images`` each slot draws the
schematic of what it will hold.  Slot keys (PNG names) are listed in ``IMAGES``.

    python scripts/paper_figures/dark/fig_caption_all.py [--images <dir>] [--video <id>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIM, MUTED, ORANGE, RULE, image_map  # noqa: E402
from mllm_common import (MCanvas, object_set_block, parse_block, ph_answer_p4, ph_caption_prefix, ph_film,  # noqa: E402
                         ph_text, ph_transcript, visual_input_block)

IMAGES = ["frame_0", "frame_1", "frame_2", "segments", "caption_card", "transcript", "context_frames",
          "query_tensor", "video_v", "objects", "discovery_card", "prompt_card", "answer_card", "scene_graph"]
MODEL = "Qwen2.5-VL-7B"


def build(images, video: str) -> MCanvas:
    c = MCanvas(1720, 1548, images)
    c.band_title(46, "U-WSGG-Sub · Per-Object Prompting With A Caption Transcript")

    # ================= Stage 1: caption transcript (the Stage-1 build) =================
    c.stage(88, "Stage 1 · Caption Transcript Generation (Stage-1 Build, Once Per Video)")
    c.frames(30, 130, ["frame_0", "frame_1", "frame_2"], w=84, h=56, labels=("k₁", "k₂", "k₃"), max_total_h=196)
    c.note(72, 344, "Annotated Key Frames", size=10, fill=MUTED, anchor="middle")
    c.flow(114, 218, 140, 218, "Video", lsize=8.5)
    c.tool(142, 180, 150, 76, "Key-Frame Segmentation", "Cut At Midpoints Between", "Key Frames · Every 2nd Frame")
    c.flow(292, 218, 318, 218, "")
    c.slot_title(320, 150, "Segments Sᵢ, One Per Key Frame · ≤ 19 Frames")
    c.image_slot(320, 150, 236, 128, "segments", placeholder=ph_film())
    c.elbow(556, 214, 612, 163, xm=582)
    c.elbow(556, 214, 612, 293, xm=582, col=DIM, dashed=True)
    c.note(584, 250, "Sᵢ", size=8.5, fill=MUTED, anchor="middle")
    c.vlm(614, 132, 136, 62, MODEL, prompts=(1,), sub2="≤ 100 Tokens")
    c.vlm(614, 262, 136, 62, MODEL, prompts=(0,), ghost=True, sub2="Graph Node, ≤ 512 Tokens")
    c.flow(750, 163, 788, 163, "")
    c.arrow(750, 293, 788, 293, col=DIM, dashed=True)
    c.image_slot(790, 128, 196, 70, "caption_card",
                 placeholder=ph_text(["“<the main action or event of Sᵢ>”", "concise, visual content only"],
                                     head="Caption cᵢ  (≤ 100 Tokens)"))
    c.box(790, 250, 196, 86, "Entities · Actions · Scenes", "Built In The Same Pass (P₀)", kind="ghost",
          sub2="Read By rag_all And Track B, Not Here", tsize=10, ssize=8.3)
    c.arrow(840, 198, 840, 228, dashed=True, head=False, col=DIM)
    c.arrow(840, 228, 682, 228, dashed=True, head=False, col=DIM)
    c.arrow(682, 228, 682, 260, dashed=True, col=DIM)
    c.note(761, 222, "Caption As Prompt Prefix", size=8.5, fill=DIM, anchor="middle")
    c.flow(986, 163, 1036, 163, "cᵢ", lsize=8.5)
    c.slot_title(1038, 130, "Caption Transcript · All F Captions")
    c.image_slot(1038, 130, 252, 150, "transcript", placeholder=ph_transcript())
    c.note(1038, 296, "[Frame kᵢ] cᵢ, one line per segment", size=8.5, fill=DIM)
    c.note(30, 374, "Stored once as graphs/qwen25vl_7b/<video>.mp4.pkl and shared with rag_all and Track B; "
                    "caption_all reads only its caption field. The fast path makes two batched calls per video "
                    "(all captions, then all graph prompts).",
           size=9, fill=DIM)
    c.note(30, 387, "A video with no pickle runs with an empty prefix, i.e. exactly as zero_shot.", size=9, fill=DIM)

    # ================= Stage 2: caption-prefixed per-object query =================
    c.stage(418, "Stage 2 · Caption-Prefixed Per-Object Query")
    ob = object_set_block(c, 466, MODEL, captions=True)
    ox, oy, ow, oh = ob["objects"]
    c.flow(ox + ow / 2, oy + oh, ox + ow / 2, oy + oh + 34, "O → One Question Per o", lsize=8.3, loff=(-8, 20),
           anchor="end")
    c.tensor(308, 548, 160, 30, "All F Captions · Stage 1", None, col=MUTED)
    c.elbow(468, 563, 512, 531, xm=512)
    c.note(30, 664, "SGDet differs from zero_shot here: P₆ and P₅ read the real transcript (every caption as a "
                    "[Frame kᵢ] line), so the discovered set is a different set. PredCls takes the GT objects.",
           size=9, fill=DIM)
    vi = visual_input_block(c, 684)
    qx, qy, qw, qh = vi["q"]
    c.flow(qx + qw / 2, qy + qh, qx + qw / 2, 926, "Q(f) → Every Prompt Of Frame f", col=ORANGE, lsize=8.5,
           loff=(10, 0), anchor="start")
    # the per-(frame, object) loop
    c.a(f'<rect x="24" y="936" width="1272" height="232" rx="12" fill="none" stroke="{RULE}" stroke-dasharray="6 4"/>')
    c.note(38, 1158, "F Frames × |O| Objects Prompts · One Batched vLLM Call In Chunks Of 64 · The Prefix Is "
                     "Memoised Per Frame", size=9.5, fill=MUTED)
    c.tensor(36, 968, 140, 40, "All F Captions", "From Stage 1", col=MUTED)
    c.flow(176, 988, 200, 988, "")
    c.box(202, 954, 140, 68, "Caption Context c(f)", "Sort By |kᵢ − f|, Nearest First", kind="new",
          sub2="[Frame kᵢ] cᵢ Lines", tsize=11, ssize=8.8)
    c.flow(342, 988, 366, 988, "c(f)", lsize=8.3, loff=(0, -7))
    c.slot_title(368, 950, "Prompt Text · c(f) + P₄(o) · One Per (f, o)")
    c.image_slot(368, 950, 290, 150, "prompt_card", placeholder=ph_caption_prefix())
    c.tensor(36, 1108, 140, 36, "Object o ∈ O", "From The Object Set", col=MUTED)
    c.elbow(176, 1126, 400, 1102, xm=400)
    c.note(290, 1122, "Name In P₄(o)", size=8.3, fill=MUTED, anchor="middle")
    c.flow(658, 1010, 698, 1010, "Text", lsize=8.3, loff=(0, -7))
    c.vlm(700, 962, 160, 96, MODEL, prompts=(4,), sub2="One Call Per (f, o) · ≤ 128 Tokens", tsize=12)
    c.tensor(700, 1090, 160, 40, "Visual Input Q(f)", "From Above", col=ORANGE)
    c.flow(780, 1090, 780, 1060, "Video", col=ORANGE, lsize=8.3, loff=(10, 4), anchor="start")
    c.flow(860, 1010, 886, 1010, "JSON", lsize=8.3, loff=(0, -7))
    c.slot_title(888, 950, "Raw Answer For (f, o)")
    c.image_slot(888, 950, 220, 116, "answer_card", placeholder=ph_answer_p4())
    c.wrap(888, 1084, ["Decoding: Qwen2.5-VL-7B-Instruct, vLLM, T = 0.2 (fixed),",
                       "≤ 128 new tokens, one video per prompt, chunks of 64;",
                       "the response is stored verbatim as raw_response."], size=8.6, fill=DIM, lh=13)
    c.flow(1128, 1010, 1154, 1010, "")
    c.tensor(1156, 990, 134, 40, "→ Stage 3", "Raw Response", col=MUTED)

    # ================= Stage 3: parsing and scene graph =================
    c.stage(1196, "Stage 3 · Parsing And Scene Graph")
    parse_block(c, 1276, verify_clip="[f ; The Stage-1 Segment Of f]", verify_prefix="Caption Prefix c(f) Reused")
    c.note(30, 1420, "The Stage-1 segments are decoded only when verification runs; with --skip-verification the "
                     "runner never loads them. Every reported caption_all row uses --skip-verification.", size=9,
           fill=DIM)

    # ================= cards, legend, caption =================
    CX, CW = 1320, 380
    c.card(CX, 88, CW, 308, 1, "Caption Every Segment",
           ["The Stage-1 build cuts the video at its annotated",
            "key frames and captions every segment with the",
            "frozen VLM (P₁). caption_all keeps the caption",
            "transcript and discards the graph nodes."])
    c.card(CX, 418, CW, 756, 2, "Prefix, Then Ask Per Object",
           ["The visual input is zero_shot's: the target frame",
            "first, then up to 15 shared key frames. The text is",
            "the whole transcript, nearest caption first, followed",
            "by the same three-field question per (frame,",
            "object). In SGDet the transcript also feeds the",
            "object discovery (P₆) and the Yes/No scores (P₅)."])
    c.card(CX, 1196, CW, 240, 3, "Parse And Emit",
           ["Labels outside the 3 / 17 / 6 sets become",
            "unknown; every predicted label scores 1.0 because",
            "the Yes/No verification is off. The output is a",
            "class-only world scene graph per frame."])
    c.legend(1450, [("frozen", "Frozen Model"), ("tool", "Program"), ("new", "Introduced Component"),
                    ("ghost", "In Code, Unused In Reported Runs")],
             extra_swatches=[(ORANGE, "Key Frame / Target Frame · Visual Input Q(f)")])
    c.badge_legend(30, 1478, [0, 1, 4, 5, 6])
    c.caption(30, 1510, "U-WSGG-Sub.",
              ["Stage 1 cuts each video at its annotated key frames and captions every segment with one frozen VLM. "
               "Stage 2 fixes the objects, builds zero_shot's visual input and prepends the whole",
               "transcript, nearest caption first, to the same per-object question. Stage 3 validates the labels and "
               "emits a class-only scene graph. Nothing is retrieved, trained or placed in 3-D."],
              size=11)
    c.note(1690, 1540, f"panels: {video}", size=8.5, fill=DIM, anchor="end")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="directory of intermediate PNGs named after the slot keys")
    ap.add_argument("--video", default="schematic")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] /
                                         "outputs/paper_figures/dark/caption_all.svg"))
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
