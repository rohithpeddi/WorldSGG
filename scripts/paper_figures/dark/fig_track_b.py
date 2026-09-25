"""Localized Track B (``track_b``): a fixed tool loop with a geometric critic and one
repair, hero style, three stages (source: setup/LOCALIZED_MLLM.md, setup/MLLM_TRACK_B.md,
lib/mllm/methods/track_b_agent/{runner,critic}.py):

    Stage 1  Perception Layer In The Canonical Frame (offline, no VLM)
    Stage 2  Perceive And Retrieve: Track A's Payload Plus A Temporal Window Of The Event Graph
    Stage 3  Relate, Verify, Repair, Emit (the critic is a program)

Every intermediate is an image slot; without ``--images`` each slot draws the
schematic of what it will hold.  Slot keys (PNG names) are listed in ``IMAGES``.

    python scripts/paper_figures/dark/fig_track_b.py [--images <dir>] [--video <id>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIM, MUTED, ORANGE, RED, image_map  # noqa: E402
from mllm_common import (MONO, MCanvas, payload_images, perception_block, ph_before_after, ph_bev,  # noqa: E402
                         ph_text, ph_violations)

IMAGES = ["frame_0", "frame_1", "frame_2", "cloud", "bev_base", "proposals_2d", "lifted_obbs",
          "marked_frame", "context_0", "context_1", "marked_bev", "prompt_card", "retrieved_card",
          "proposal_card", "violations_card", "repair_card", "repaired_graph", "critic_before_after"]
MODEL = "Qwen3-VL-8B"


def build(images, video: str) -> MCanvas:
    c = MCanvas(1720, 1420, images)
    c.band_title(46, "Localized Track B · A Fixed Tool Loop With A Geometric Critic And One Repair")

    # ================= Stage 1: perception layer =================
    c.stage(88, "Stage 1 · Perception Layer In The Canonical Frame (Offline, No VLM)")
    perception_block(c, 130)

    # ================= Stage 2: perceive and retrieve =================
    c.stage(412, "Stage 2 · Perceive And Retrieve: Track A's Payload Plus A Temporal Window Of The Event Graph")
    c.tensor(30, 500, 150, 46, "Object Table", "Stage 1 · Ids 1…N, P", col=MUTED)
    c.tensor(30, 556, 150, 46, "Map + Window", "Stage 1", col=MUTED)
    c.flow(180, 523, 206, 560, "")
    c.flow(180, 579, 206, 598, "")
    c.tool(208, 542, 132, 74, "Payload Builder", "Track A's build_payload,", "Imported Unchanged")
    c.flow(340, 579, 364, 579, "")
    payload_images(c, 366, 446, title_col=MUTED)
    c.tensor(830, 470, 150, 42, "Event Graph G", "Graph-RAG Stage 1", col=MUTED)
    c.flow(980, 491, 1006, 491, "")
    c.tool(1008, 460, 160, 62, "query_graph", "Nearest Segment To f, ± 2", "Captions + ≤ 8 Entity Names")
    c.flow(1088, 522, 1088, 546, "≤ 900 Characters", lsize=8.3, loff=(8, 4), anchor="start")
    c.image_slot(830, 548, 460, 72, "retrieved_card",
                 placeholder=ph_text(["- [earlier, frame k] <caption> (entities: …)",
                                      "- [at the target frame, frame k] <caption> (…)",
                                      "- [later, frame k] <caption> (entities: …)"],
                                     head="Retrieved Context · The Segments Around f", size=8.2))
    c.flow(1060, 620, 1060, 646, "")
    c.tool(960, 648, 200, 44, "with_context", "Inserted Before “Task:” · Text Only", tsize=10.5)
    c.arrow(960, 670, 812, 670, head=False)
    c.arrow(812, 670, 812, 640, head=False)
    c.arrow(812, 640, 806, 640, head=True)
    c.note(886, 664, "Retrieved Block → Text", size=8.3, fill=MUTED, anchor="middle")
    c.wrap(830, 712, ["Retrieval by time, not by similarity: the Stage-1 segments nearest f (two on either side),",
                      "each as '- [earlier | at the target frame | later, frame k] <caption> (entities: …)'.",
                      "It always reads the Qwen2.5-VL-7B graphs, whatever model generates. The images are",
                      "Track A's; what the model sees differs from Track A only by this block."],
           size=9, fill=DIM, lh=13.5)
    c.note(30, 760, "The tools are called by the driver in a fixed order for every frame (perceive → retrieve →",
           size=9, fill=DIM)
    c.note(30, 773, "relate → verify → repair → emit); the model never selects a tool.", size=9, fill=DIM)

    # ================= Stage 3: relate, verify, repair, emit =================
    c.stage(796, "Stage 3 · Relate, Verify, Repair, Emit (The Critic Is A Program)")
    c.tensor(30, 852, 140, 46, "Payload Of Frame f", "4 Images + Text", col=MUTED)
    c.flow(586, 742, 586, 790, "Payload → Stage 3", lsize=8.5, loff=(8, 6), anchor="start")
    c.flow(170, 875, 196, 875, "")
    c.vlm(198, 838, 140, 74, MODEL, title="Frozen VLM · Call 1", sub2="≤ 1,024 Tokens · T = 0.2", tsize=11)
    c.flow(338, 875, 364, 875, "JSON", lsize=8.3, loff=(0, -7))
    pw, ph_ = c.fit("proposal_card", w=200, default=(200, 96))
    c.image_slot(366, 828, pw, ph_, "proposal_card",
                 placeholder=ph_text(['{"objects": [{"id": 1, "label": …,', '  "center": […], "size": […],',
                                      '  "attention": …, "contacting": […],', '  "spatial": […]}, …]}'],
                                     head="Call 1 · Raw Proposal", size=8))
    c.flow(566, 875, 592, 875, "")
    c.tool(594, 846, 110, 58, "Extract · Salvage", "Then Parse (Track A's)", tsize=10, ssize=8.5)
    c.flow(704, 875, 730, 875, "")
    c.tensor(732, 852, 110, 46, "First Proposal", "G_pre · objects_pre", col=MUTED)
    c.flow(842, 875, 868, 875, "")
    c.box(870, 830, 170, 90, "Geometric Critic", "A Program · check_geometry", kind="new",
          sub2="Schema · Floor · Size · Extent · Contact · Vertical", tsize=12, ssize=8.1)
    c.tensor(732, 936, 110, 30, "Person Box · Window", None, col=MUTED)
    c.elbow(842, 951, 900, 922, xm=900)
    c.flow(1040, 875, 1066, 875, "")
    vw, vh = c.fit("violations_card", w=222, default=(222, 100))
    c.image_slot(1068, 826, vw, vh, "violations_card",
                 placeholder=ph_violations(["<o>: ['holding'] claimed but the", "  box is d m away from the person",
                                            "<o>: box bottom is d m below the floor",
                                            "<o>: 'above' but its centre is lower"]))
    c.note(787, 912, "−Critic Arm, Same Run", size=8.3, fill=DIM, anchor="middle")
    # row 2: the repair
    c.arrow(1179, 826 + vh, 1179, 990, head=False, col=RED)
    c.arrow(1179, 990, 1082, 990, head=False, col=RED)
    c.arrow(1082, 990, 1082, 1006, col=RED)
    c.note(1172, 950, "Only Flagged Frames", size=8.3, fill=RED, anchor="end")
    c.note(1172, 962, "Whose Proposal Parsed", size=8.3, fill=RED, anchor="end")
    c.tool(952, 1008, 260, 58, "Repair Prompt", "Text + “Your previous answer was:” + G_pre As JSON",
           "+ ≤ 12 Violation Lines + “Fix them …”", tsize=11, ssize=8.2)
    c.flow(952, 1037, 926, 1037, "")
    c.vlm(776, 1000, 150, 74, MODEL, title="Frozen VLM · Call 2", sub2="Same 4 Images · ≤ 1,024 Tokens", tsize=11)
    c.flow(776, 1037, 750, 1037, "JSON", lsize=8.3, loff=(0, -7))
    c.image_slot(538, 990, 210, 96, "repair_card",
                 placeholder=ph_text(['{"objects": [{"id": 1, …,', '  "center": [x, y, z\'], …', '  "contacting": […],',
                                      '  "spatial": […]}, …]}'], head="Call 2 · Corrected Answer", size=8))
    c.flow(538, 1037, 512, 1037, "")
    c.tool(382, 1008, 130, 58, "Extract · Salvage · Parse", "Unparseable → Keep G_pre", tsize=10, ssize=8.5)
    c.flow(382, 1037, 356, 1037, "")
    c.tensor(206, 1014, 150, 46, "Repaired Graph", "G_post · objects", col=ORANGE)
    c.arrow(281, 1014, 281, 978, head=False, col=ORANGE, dashed=True)
    c.arrow(281, 978, 1020, 978, head=False, col=ORANGE, dashed=True)
    c.arrow(1020, 978, 1020, 922, col=ORANGE, dashed=True)
    c.note(300, 972, "Re-Checked Once → violations_post · max_repairs = 1, So No Second Repair", size=8.5,
           fill=ORANGE)
    c.flow(206, 1037, 192, 1037, "")
    c.tensor(30, 1014, 160, 46, "Emit", "objects · objects_pre · violations", col=MUTED)
    # row 3: the checks, one frame before / after, what a repair may change
    c.slot_title(30, 1150, "The Critic's Checks (Canonical Frame, Metres)")
    for i, ln in enumerate(["schema     attention is not exactly one valid label",
                            "floor      z_min < -0.15 m; contact claimed but z_min > person z_max + 0.3 m",
                            "size       any side > 3 m or < 1 cm",
                            "extent     centre outside the map window + 1 m margin",
                            "contact    holding/touching/carrying/... but gap > 0.5 m; sitting_on/lying_on/...",
                            "           but gap > 0.25 m or no vertical overlap; not_contacting inside the person",
                            "vertical   'above' with centre not higher than the person's; 'beneath' not lower; both",
                            "unchecked  in_front_of, behind, on_the_side_of, in (the person's facing is unknown)"]):
        c.text(30, 1164 + i * 13, ln.replace(" ", " "), size=8.2, fill=DIM, font=MONO)
    c.slot_title(560, 1150, "One Frame, Before And After Repair")
    c.image_slot(560, 1150, 360, 110, "critic_before_after", placeholder=ph_before_after())
    c.slot_title(950, 1150, "Emitted Graph At That Frame", col=ORANGE)
    c.image_slot(950, 1150, 272, 66, "repaired_graph", placeholder=ph_bev(marks=True, predicted=True))
    c.wrap(950, 1236, ["A repair may change: SGDet boxes, predicates, adding or dropping",
                       "objects; PredCls predicates or dropping an object (the GT boxes stay,",
                       "so floor / size / extent flags on GT boxes cannot be repaired)."],
           size=8.6, fill=DIM, lh=12.5)

    # ================= cards, legend, caption =================
    CX, CW = 1320, 380
    c.card(CX, 88, CW, 300, 1, "Ground In Metric 3-D",
           ["Frozen Pi-3 and GDino express each video in one",
            "canonical floor frame: a top-down map, 2-D",
            "proposals and their lifts to floor-parallel boxes.",
            "No model is trained; the caches are built once."])
    c.card(CX, 412, CW, 360, 2, "Perceive, Then Retrieve By Time",
           ["Track A's payload builder makes the marked frame,",
            "the context frames, the marked map and the metric",
            "table. query_graph adds the Stage-1 captions of",
            "the segments around the target frame, inserted",
            "before the task: a temporal window, not a",
            "similarity search."])
    c.card(CX, 796, CW, 480, 3, "Check, Then Repair Once",
           ["Call 1 proposes a localized graph. A program",
            "checks it against the person box and the map:",
            "boxes under the floor, implausible sizes, contact",
            "at a distance, wrong heights. Flagged frames get",
            "one repair call carrying the violations; the graph",
            "is checked again and emitted with its first",
            "proposal, the −critic arm."])
    c.legend(1300, [("frozen", "Frozen Model"), ("tool", "Program"), ("new", "Introduced Component")],
             extra_swatches=[(ORANGE, "Final Output / Re-Check"), (RED, "Violation / Repair Route")])
    c.caption(30, 1334, "Localized Track B.",
              ["Stage 1 expresses each video in a canonical floor frame with frozen tools. Stage 2 builds Track A's "
               "four-image payload and inserts the Stage-1 captions of the segments around the",
               "target frame. Stage 3 runs a fixed loop: one proposal call, a geometric critic that is a program, "
               "one repair call for flagged frames, one re-check. Both ablation arms come out of the same run."],
              size=11)
    c.note(1690, 1410, f"panels: {video}", size=8.5, fill=DIM, anchor="end")
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="directory of intermediate PNGs named after the slot keys")
    ap.add_argument("--video", default="schematic")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "outputs/paper_figures/dark/track_b.svg"))
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
