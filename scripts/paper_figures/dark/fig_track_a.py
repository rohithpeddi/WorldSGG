"""Localized Track A (``track_a``): the localized prompt with shared ids, hero style,
three stages (source: setup/LOCALIZED_MLLM.md, setup/MLLM_TRACK_A.md and
lib/mllm/methods/track_a_prompt/{runner,prompts}.py):

    Stage 1  Perception Layer In The Canonical Frame (offline, no VLM)
    Stage 2  Localized Prompt With Shared Ids (the payload builder)
    Stage 3  One Call Per Frame, Parsing And Localized Scene Graph

Every intermediate is an image slot; without ``--images`` each slot draws the
schematic of what it will hold.  Slot keys (PNG names) are listed in ``IMAGES``.

    python scripts/paper_figures/dark/fig_track_a.py [--images <dir>] [--video <id>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIM, MUTED, ORANGE, image_map  # noqa: E402
from mllm_common import MCanvas, payload_images, perception_block, ph_bev, ph_text  # noqa: E402

IMAGES = ["frame_0", "frame_1", "frame_2", "cloud", "bev_base", "proposals_2d", "lifted_obbs",
          "marked_frame", "context_0", "context_1", "marked_bev", "prompt_card", "answer_card", "localized_graph"]
MODEL = "Qwen3-VL-8B"


def build(images, video: str) -> MCanvas:
    c = MCanvas(1720, 1236, images)
    c.band_title(46, "Localized Track A · Marked Frames, A Marked Map And A Metric Table In One Id Space")

    # ================= Stage 1: perception layer =================
    c.stage(88, "Stage 1 · Perception Layer In The Canonical Frame (Offline, No VLM)")
    perception_block(c, 130)

    # ================= Stage 2: the localized prompt =================
    c.stage(412, "Stage 2 · Localized Prompt With Shared Ids (The Payload Builder)")
    c.tensor(30, 500, 150, 46, "SGDet Object Table", "Proposals By Score", col=MUTED)
    c.tensor(30, 556, 150, 46, "PredCls Object Table", "GT Objects Of f", col=MUTED)
    c.tensor(30, 612, 150, 46, "Map + Window", "Stage 1", col=MUTED)
    c.flow(180, 523, 206, 560, "")
    c.flow(180, 579, 206, 579, "Ids 1…N, P", lsize=8.3, loff=(0, -7))
    c.flow(180, 635, 206, 598, "")
    c.box(208, 542, 132, 74, "Payload Builder", "Same Id In Frame,", kind="new", sub2="Map And Text", tsize=11.5,
          ssize=8.8)
    c.note(274, 632, "mark_frame · mark_bev", size=8.3, fill=DIM, anchor="middle")
    c.note(274, 644, "predcls_prompt / sgdet_prompt", size=8.3, fill=DIM, anchor="middle")
    c.flow(340, 579, 364, 579, "")
    payload_images(c, 366, 446)
    c.wrap(830, 470, ["Three encodings, one id space:", "the number on the target frame,",
                      "the footprint on the map and the", "row of the metric table are the",
                      "same integer, and the answer", "must use it."], size=10, fill=MUTED, lh=15)
    c.wrap(830, 576, ["PredCls: ids = the frame's GT objects (one per label), GT OBBs,",
                      "'visible' / 'NOT visible in the target frame'; the person's GT box.",
                      "SGDet: ids = GDino proposals by score with their lifted OBBs or",
                      "'3D box unknown', the detector score, the map extent and the",
                      "35-name vocabulary; the person = the largest person detection."],
           size=9, fill=DIM, lh=14)
    c.wrap(830, 668, ["Sizes: target ≤ 640 px, context ≤ 320 px, map ≤ 640 px.",
                      "Context = linspace over the other annotated frames (first and last).",
                      "The text: system rules (frame, OBB, 3 / 17 / 6 vocabulary), the image",
                      "legend, the scene table, the task and a one-line JSON schema."],
           size=9, fill=DIM, lh=14)
    c.note(30, 760, "The payload builder is the piece Track A adds: without the marks, the map and the metric",
           size=9, fill=DIM)
    c.note(30, 773, "table the VLM has no frame of reference for a metric box or an out-of-view object.", size=9,
           fill=DIM)

    # ================= Stage 3: the call, parsing, the localized graph =================
    c.stage(796, "Stage 3 · One Call Per Frame, Parsing And Localized Scene Graph")
    c.tensor(30, 862, 150, 46, "Payload Of Frame f", "4 Images + Text", col=ORANGE)
    c.flow(586, 742, 586, 790, "Payload → Stage 3", lsize=8.5, loff=(8, 6), anchor="start")
    c.flow(180, 885, 206, 885, "")
    c.vlm(208, 838, 160, 94, MODEL, sub2="4 Images + Text · ≤ 1,024 Tokens", tsize=12)
    c.tool(208, 948, 160, 40, "Response Cache", "sha256(Model, Text, PNGs, T, Tokens)", tsize=10, ssize=7.8)
    c.arrow(288, 948, 288, 934, col=DIM, dashed=True, head=False)
    c.flow(368, 885, 394, 885, "JSON", lsize=8.3, loff=(0, -7))
    c.slot_title(396, 826, "Answer · One JSON Array Over All Ids")
    c.image_slot(396, 826, 272, 116, "answer_card",
                 placeholder=ph_text(['{"objects": [{"id": 1,', '   "label": "<vocab name>",',
                                      '   "center": [x, y, z], "size": [l, w, h],', '   "yaw_deg": θ,',
                                      '   "attention": "<label>",', '   "contacting": ["<label>", …],',
                                      '   "spatial": ["<label>", …]}, …]}'],
                                     cols=[MUTED, ORANGE, ORANGE, ORANGE, MUTED, MUTED, MUTED], size=8.2))
    if "answer_card" in c.missing:        # explains the schematic's orange lines; a real answer has none
        c.note(664, 953, "orange lines: SGDet only", size=8.3, fill=ORANGE, anchor="end")
    c.flow(668, 885, 694, 885, "")
    c.tool(696, 838, 128, 94, "Extract JSON", "Drop </think> · Fenced Or Raw", "First Balanced {…} With objects",
           tsize=11, ssize=8.2)
    c.flow(824, 885, 850, 885, "")
    c.tool(852, 838, 128, 94, "Salvage", "Cut At 1,024 Tokens →", "Complete Entries Only", tsize=11, ssize=8.2)
    c.flow(980, 885, 1006, 885, "")
    c.tool(1008, 838, 140, 94, "Parse Objects", "Id → Payload Object", "First Entry Per Label Wins", tsize=11,
           ssize=8.2)
    c.flow(1148, 885, 1174, 885, "")
    c.tensor(1176, 862, 114, 46, "Graph G(f)", "Localized", col=ORANGE)
    c.flow(1233, 908, 1233, 968, "", col=ORANGE)
    c.slot_title(1018, 984, "Localized Scene Graph At Frame f", col=ORANGE)
    c.image_slot(1018, 984, 272, 66, "localized_graph", placeholder=ph_bev(marks=True, predicted=True))
    c.wrap(396, 974, ["PredCls: an entry binds by id; the label and the box are the ground truth's, a box the",
                      "model writes is ignored. SGDet: the label is normalised to the vocabulary (person",
                      "rejected); the box is the model's center / size / yaw when given (src = model,",
                      "proposal+model), else the proposal's lift (src = proposal); no box → dropped.",
                      "Predicates are filtered to their vocabularies; the score is the detector score",
                      "for a proposal and 1.0 otherwise. Output: {label: predicates, corners, obb, score, src}."],
           size=9, fill=DIM, lh=13.5)
    c.note(30, 1002, "Decoding", size=10, fill=MUTED)
    c.wrap(30, 1019, ["Qwen3-VL-8B-Instruct through vLLM, native multi-image",
                      "prompts (max_images = 4), max_model_len 24,576;",
                      "temperature 0.2, top_p 0.95, seed 0, ≤ 1,024 new tokens;",
                      "the prompts of 4 videos pooled into one generate() call;",
                      "every call keyed by content hash and served from disk on",
                      "a hit. About 30 s per video on one A40."], size=9, fill=DIM, lh=14)

    # ================= cards, legend, caption =================
    CX, CW = 1320, 380
    c.card(CX, 88, CW, 300, 1, "Ground In Metric 3-D",
           ["Frozen Pi-3 and GDino express each video in one",
            "canonical floor frame: a top-down map, 2-D",
            "proposals and their lifts to floor-parallel boxes.",
            "No model is trained; the caches are built once."])
    c.card(CX, 412, CW, 360, 2, "Prompt With Shared Ids",
           ["One multi-image prompt per annotated frame: the",
            "marked target frame, two context frames, the",
            "marked map and a metric table carry the same ids.",
            "PredCls gives the GT boxes and asks for predicates;",
            "SGDet gives lifted proposals and asks for a box",
            "per object as well."])
    c.card(CX, 796, CW, 316, 3, "One Call, One Array",
           ["The frozen VLM answers every id in one JSON",
            "array. A truncated array is salvaged entry by",
            "entry, ids are bound back to the payload, and the",
            "result is a localized scene graph: predicates on",
            "GT boxes in PredCls, predicates plus oriented",
            "boxes in SGDet."])
    c.legend(1144, [("frozen", "Frozen Model"), ("tool", "Program"), ("new", "Introduced Component")],
             extra_swatches=[(ORANGE, "Target Frame / Localized Output")])
    c.caption(30, 1178, "Localized Track A.",
              ["Stage 1 expresses each video in a canonical floor frame with frozen tools. Stage 2 builds, per annotated "
               "frame, a marked target frame, two context frames, a marked top-down map and a metric",
               "table that share one id space. Stage 3 makes one frozen-VLM call per frame, salvages and parses the "
               "JSON array, and emits a localized scene graph. Nothing is trained."],
              size=11)
    c.note(1690, 1226, f"panels: {video}", size=8.5, fill=DIM, anchor="end")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="directory of intermediate PNGs named after the slot keys")
    ap.add_argument("--video", default="schematic")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/track_a.svg"))
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
