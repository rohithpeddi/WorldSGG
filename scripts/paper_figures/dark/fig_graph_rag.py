"""Unlocalized Graph-RAG (``rag_all``, the README's WorldRAG) architecture figure,
dark hero style, three stages (source: setup/UNLOCALIZED_GRAPH_RAG.md):

    Stage 1  Coarse Event-Graph Construction (offline, once per video)
    Stage 2  Object Discovery And Per-Object Graph RAG
    Stage 3  Per-Frame Relationship Prediction

Every intermediate is an image slot; without ``--images`` each slot draws the
schematic of what it will hold.  Slot keys (PNG names) are listed in ``IMAGES``.

    python scripts/paper_figures/dark/fig_graph_rag.py [--images <dir>] [--video <id>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIM, MUTED, ORANGE, RULE, TXT, image_map  # noqa: E402
from mllm_common import (MCanvas, ph_event_graph, ph_film, ph_frames_strip, ph_objects, ph_ranked,  # noqa: E402
                         ph_scene_graph, ph_text)

IMAGES = ["frame_0", "frame_1", "frame_2", "segments", "caption_card", "node_card", "event_graph",
          "video_v", "objects", "keywords_card", "ranked_nodes", "context_card",
          "query_tensor", "answer_card", "scene_graph"]
MODEL = "Qwen2.5-VL-7B"


def build(images, video: str) -> MCanvas:
    c = MCanvas(1720, 1180, images)
    c.band_title(46, "Unlocalized Graph-RAG · Retrieval Over A VLM-Built Event Graph")

    # ================= Stage 1: coarse event-graph construction =================
    c.stage(88, "Stage 1 · Coarse Event-Graph Construction (Offline, Once Per Video)")
    c.frames(30, 130, ["frame_0", "frame_1", "frame_2"], w=84, h=56, labels=("k₁", "k₂", "k₃"), max_total_h=196)
    c.note(72, 344, "Annotated Key Frames", size=10, fill=MUTED, anchor="middle")
    c.flow(114, 218, 140, 218, "Video", lsize=8.5)
    c.tool(142, 180, 150, 76, "Key-Frame Segmentation", "Cut At Midpoints Between", "Key Frames · Every 2nd Frame")
    c.flow(292, 218, 318, 218, "")
    c.slot_title(320, 150, "Segments Sᵢ, One Per Key Frame · ≤ 19 Frames")
    c.image_slot(320, 150, 236, 128, "segments", placeholder=ph_film())
    c.elbow(556, 214, 612, 163, xm=582)
    c.elbow(556, 214, 612, 293, xm=582)
    c.note(584, 250, "Sᵢ", size=8.5, fill=MUTED, anchor="middle")
    c.vlm(614, 132, 136, 62, MODEL, prompts=(1,), sub2="≤ 100 Tokens")
    c.vlm(614, 262, 136, 62, MODEL, prompts=(0,), sub2="≤ 512 Tokens · JSON")
    c.flow(750, 163, 788, 163, "cᵢ", lsize=8.5)
    c.flow(750, 293, 788, 293, "nᵢ", lsize=8.5)
    c.image_slot(790, 128, 196, 70, "caption_card",
                 placeholder=ph_text(["“<the main action or event of Sᵢ>”", "concise, visual content only"],
                                     head="Caption cᵢ  (≤ 100 Tokens)"))
    c.image_slot(790, 244, 196, 100, "node_card",
                 placeholder=ph_text(['{"entities": [{"entity name",', '     "description"}],',
                                      ' "actions": [{"entity name",', '     "action description"}],',
                                      ' "scenes": [{"location"}]}'], head="Node nᵢ  (JSON, ≤ 512 Tokens)"))
    c.arrow(840, 198, 840, 228, dashed=True, head=False)
    c.arrow(840, 228, 682, 228, dashed=True, head=False)
    c.arrow(682, 228, 682, 260, dashed=True)
    c.note(761, 222, "Caption As Prompt Prefix", size=8.5, fill=MUTED, anchor="middle")
    c.flow(986, 294, 1036, 294, "F Nodes", lsize=8.5)
    c.slot_title(1038, 130, "Event Graph G")
    c.image_slot(1038, 130, 252, 214, "event_graph", placeholder=ph_event_graph())
    c.note(1038, 360, "Merge Per-Segment Nodes · Entity Index: Name → {Nodes}", size=8.8, fill=MUTED)
    c.note(30, 374, "Stored once as graphs/qwen25vl_7b/<video>.mp4.pkl and read by Stage 2 and by the localized "
                    "Track B. Every VLM call in this figure is the same frozen model; P-badges name its prompts.",
           size=9, fill=DIM)

    # ================= Stage 2: object discovery and per-object graph RAG =================
    c.stage(404, "Stage 2 · Object Discovery And Per-Object Graph RAG")
    c.note(30, 440, "SGDet", size=10, fill=ORANGE)
    c.image_slot(30, 448, 160, 58, "video_v",
                 placeholder=ph_frames_strip(n=12, lowres=True, label="V · ≤ 19 Frames Of The Video"))
    c.tensor(30, 514, 160, 34, "All Captions + Class List", "[Frame kᵢ] cᵢ · 36 AG Names", col=MUTED)
    c.flow(190, 477, 236, 486, "V", lsize=8.5, loff=(0, -5))
    c.flow(190, 531, 236, 500, "Text", lsize=8.5, loff=(4, 10))
    c.vlm(238, 464, 124, 58, MODEL, prompts=(6,), sub2="≤ 256 Tokens")
    c.flow(362, 493, 392, 493, "")
    c.tensor(394, 474, 110, 38, "Candidates", "JSON, Free Names", col=MUTED)
    c.flow(504, 493, 530, 493, "")
    c.tool(532, 470, 124, 46, "∩ AG Vocabulary", "Exact String Match")
    c.flow(656, 493, 684, 493, "")
    c.vlm(686, 464, 124, 58, MODEL, prompts=(5,), sub2="“Is There A <o>?”")
    c.note(748, 536, "1 Token At T = 0 · V + Captions", size=8, fill=DIM, anchor="middle")
    c.flow(810, 493, 842, 486, "")
    c.slot_title(844, 446, "Objects O + Object Score")
    c.image_slot(844, 446, 186, 96, "objects", placeholder=ph_objects())
    c.tensor(686, 544, 124, 30, "GT Video Objects", None, col=MUTED)
    c.note(686, 587, "PredCls", size=10, fill=ORANGE)
    c.elbow(810, 559, 842, 530, xm=826)
    c.wrap(1050, 462, ["PredCls queries every GT object of the video", "at every frame. SGDet discovers the set:",
                       "P₆ reads V, all captions and the class list;", "the P₅ probability is kept as the object",
                       "confidence for scoring, never as a filter."], size=9.5, fill=DIM, lh=15)
    # per-object loop
    c.a(f'<rect x="24" y="592" width="1272" height="180" rx="12" fill="none" stroke="{RULE}" '
        'stroke-dasharray="6 4"/>')
    c.note(38, 610, "Once Per Unique Object o (Prompts Deduplicated Over Frames)", size=9.5, fill=MUTED)
    c.flow(937, 542, 937, 590, "O → One Question Per o", lsize=8.3, loff=(8, 4), anchor="start")
    c.tensor(36, 648, 128, 44, "Question P₄(o)", "Depends Only On o", col=MUTED)
    c.flow(164, 670, 196, 670, "Text", lsize=8.3, loff=(0, -6))
    c.vlm(198, 642, 124, 58, MODEL, prompts=(2,), sub2="Text Only · ≤ 256 Tokens")
    c.flow(322, 671, 350, 671, "JSON", lsize=8.3, loff=(0, -6))
    c.image_slot(352, 640, 132, 62, "keywords_card",
                 placeholder=ph_text(['["<keyword>",', ' "<keyword>", …]'], head="Keywords"))
    c.flow(484, 671, 514, 671, "Queries", lsize=8.3, loff=(0, -6))
    c.box(516, 640, 132, 62, "BGE-Large", "Frozen Text Encoder · CLS", kind="frozen", frozen=True,
          sub2="Keywords ∪ {P₄(o)}", tsize=11.5, ssize=9)
    c.tensor(516, 602, 132, 26, "Event Graph G · Stage 1", None, col=MUTED)
    c.flow(582, 628, 582, 638, "Entity Names · Node Texts", lsize=8.3, loff=(70, -12), anchor="start")
    c.flow(648, 671, 678, 671, "cos", lsize=8.5)
    c.box(680, 628, 170, 86, "Mean-Cosine Node Retrieval", "Entity Name Or Node Text > 0.5", kind="new",
          sub2="Rank By Node Text · Keep 20", tsize=11, ssize=8.8)
    c.flow(850, 671, 878, 671, "Top 20", lsize=8.3, loff=(0, -6))
    c.slot_title(880, 626, "Ranked Nodes")
    c.image_slot(880, 626, 176, 96, "ranked_nodes", placeholder=ph_ranked())
    c.flow(1056, 671, 1084, 671, "Top-1", lsize=8.5)
    c.slot_title(1086, 626, "Context Block c(o)", col=ORANGE)
    c.image_slot(1086, 626, 204, 96, "context_card",
                 placeholder=ph_text(["Relevant scene context from", "video analysis:",
                                      "<entities>; <actions>; <scenes>", "(the top-1 node, no caption)"], mono=True))
    c.arrow(968, 722, 968, 734, head=False, col=DIM)
    c.box(880, 734, 176, 32, "Node Relevance Check", "P₃ · Picks A Verification Clip", kind="ghost", tsize=10,
          ssize=8.5)
    c.a('<g opacity="0.45">')
    c.badge(886, 725, 3, s=16)
    c.a('</g>')
    c.note(870, 746, "In code; changes no reported prediction", size=8.8, fill=DIM, anchor="end")
    c.note(870, 759, "(verification is off in every reported run)", size=8.8, fill=DIM, anchor="end")
    c.flow(1188, 722, 1188, 790, "c(o) → Stage 3", col=ORANGE, loff=(8, 4), anchor="start")

    # ================= Stage 3: per-frame relationship prediction =================
    c.stage(800, "Stage 3 · Per-Frame Relationship Prediction")
    c.slot_title(30, 846, "Visual Input Q(f) · One Video")
    c.image_slot(30, 846, 300, 104, "query_tensor",
                 placeholder=ph_frames_strip(n=15, target=True, label="[ target frame f ; ≤ 15 key frames ] · 168×336"))
    c.tensor(30, 962, 146, 38, "Context Block c(o)", "From Stage 2", col=ORANGE)
    c.tensor(184, 962, 146, 38, "Question P₄(o)", "Same Prompt As zero_shot", col=MUTED)
    c.flow(330, 898, 386, 900, "Video", lsize=8.3, loff=(0, -6))
    c.elbow(330, 981, 386, 940, xm=358)
    c.note(352, 993, "Text", size=8.3, fill=MUTED, anchor="middle")
    c.arrow(103, 1000, 103, 1014, col=ORANGE, head=False)
    c.arrow(103, 1014, 430, 1014, col=ORANGE, head=False)
    c.arrow(430, 1014, 430, 966, col=ORANGE)
    c.note(266, 1024, "Prefix: c(o) + P₄(o)", size=8.3, fill=ORANGE, anchor="middle")
    c.vlm(388, 868, 150, 96, MODEL, prompts=(4,), sub2="One Call Per (f, o) · ≤ 128 Tokens", tsize=12)
    c.note(463, 1040, "T = 0.2 · Chunks Of 64 · One Video Per Prompt", size=8.3, fill=DIM, anchor="middle")
    c.flow(538, 916, 570, 916, "JSON", lsize=8.3, loff=(0, -6))
    c.image_slot(572, 858, 210, 116, "answer_card",
                 placeholder=ph_text(['{"attention": "<label>",', ' "contacting": ["<label>", …],',
                                      ' "spatial": ["<label>", …]}', "", "3 / 17 / 6 label sets"],
                                     head="Answer For (f, o)"))
    c.flow(782, 916, 812, 916, "")
    c.tool(814, 882, 140, 68, "Parse And Validate", "Labels In 3 / 17 / 6 Sets", "Invalid Head → Unknown")
    c.flow(954, 916, 984, 916, "")
    c.slot_title(986, 846, "World Scene Graph At Frame f")
    c.image_slot(986, 846, 304, 140, "scene_graph", placeholder=ph_scene_graph())
    c.arrow(884, 950, 884, 992, col=DIM, dashed=True)
    c.box(760, 992, 194, 34, "Yes / No Verification", "P₅ Per Label · Off In Reported Runs", kind="ghost",
          tsize=10, ssize=8.5)
    c.note(750, 1004, "Skipped: every predicted label scores 1.0", size=8.8, fill=DIM, anchor="end")
    c.note(750, 1017, "and every other predicate 0", size=8.8, fill=DIM, anchor="end")
    c.note(986, 1004, "SGDet has no boxes, so it is scored by class only", size=8.8, fill=DIM)
    c.note(986, 1017, "(loc3d at τ = 0)", size=8.8, fill=DIM)

    # ================= cards, legend, caption =================
    CX, CW, CH = 1320, 380, 304
    c.card(CX, 88, CW, CH, 1, "Build An Event Graph",
           ["Every annotated key frame opens a segment. The",
            "frozen VLM captions it (P₁) and, prompted with",
            "that caption, lists its entities, actions and",
            "scenes (P₀). Nodes naming one entity are linked."])
    c.card(CX, 412, CW, CH, 2, "Retrieve Per Object",
           ["Each object's question becomes keywords (P₂). A",
            "frozen BGE encoder scores every node by mean",
            "cosine; nodes above 0.5 are ranked, and the top",
            "node's entities, actions and scenes form c(o)."])
    c.card(CX, 736, CW, CH, 3, "Answer Per Frame",
           ["For every frame f and object o the VLM reads the",
            "target frame, 15 key frames and c(o), and returns",
            "one attention label plus contacting and spatial",
            "lists (P₄): a class-only world scene graph."])
    c.legend(1064, [("frozen", "Frozen Model"), ("tool", "Program"), ("new", "Introduced Component"),
                    ("ghost", "In Code, Unused In Reported Runs")],
             extra_swatches=[(ORANGE, "Key Or Target Frame · Context c(o)")])
    c.badge_legend(30, 1092, [0, 1, 2, 3, 4, 5, 6])
    c.caption(30, 1126, "Unlocalized Graph-RAG.",
              ["Stage 1 cuts each video at its annotated key frames and turns every segment into a caption and an entity / "
               "action / scene node with one frozen VLM. Stage 2 fixes the objects and retrieves,",
               "per object, the node closest to its question under a frozen text encoder. Stage 3 asks the same VLM for "
               "attention, contacting and spatial predicates at every (frame, object). Nothing is trained or placed in 3-D."],
              size=11)
    c.note(1690, 1170, f"panels: {video}", size=8.5, fill=DIM, anchor="end")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="directory of intermediate PNGs named after the slot keys")
    ap.add_argument("--video", default="schematic")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/graph_rag.svg"))
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
