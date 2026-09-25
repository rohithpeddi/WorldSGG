#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turn AG intermediate JSON into chat jsonl (Swift-style, PVSG-like ``train_with_tags``).

Reads ``train_annotations_toon_sft.json`` / ``test_annotations_toon_sft.json`` from
``prepare_original_ag_sft.py``. Each ``image_path`` is **relative to the SceneGraphVLM repo root**;
this script joins it with ``--repo_root`` to fill ``images`` in the jsonl.

**Commands** (from repository root; path flags are repo-relative unless absolute):

  cd /path/to/SceneGraphVLM
  python datasets/annotations/AG_annot/tools/sft_to_jsonl_ag.py

Writes ``datasets/data_playground/AG_json/train.jsonl`` and ``test.jsonl`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

_TOOLS_DIR = Path(__file__).resolve().parent
_REPO_ROOT_DEFAULT = _TOOLS_DIR.parents[3]
EXPORT_ROOT_REL = "datasets/annotations/AG_annot/data_sft_original"
OUT_DIR_REL = "datasets/data_playground/AG_json"


def resolve_under_repo(repo_root: Path, path_arg: str) -> Path:
    p = Path(path_arg)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / path_arg).resolve()

OBJ_CLS: List[str] = [
    "person",
    "bag",
    "bed",
    "blanket",
    "book",
    "box",
    "broom",
    "chair",
    "closet/cabinet",
    "clothes",
    "cup/glass/bottle",
    "dish",
    "door",
    "doorknob",
    "doorway",
    "floor",
    "food",
    "groceries",
    "laptop",
    "light",
    "medicine",
    "mirror",
    "paper/notebook",
    "phone/camera",
    "picture",
    "pillow",
    "refrigerator",
    "sandwich",
    "shelf",
    "shoe",
    "sofa/couch",
    "table",
    "television",
    "towel",
    "vacuum",
    "window",
]

ATTENTION_CLS: List[str] = ["looking_at", "not_looking_at", "unsure"]

SPATIAL_CLS: List[str] = [
    "above",
    "behind",
    "beneath",
    "in",
    "in_front_of",
    "on_the_side_of",
]

CONTACTING_CLS: List[str] = [
    "carrying",
    "covered_by",
    "drinking_from",
    "eating",
    "have_it_on_the_back",
    "holding",
    "leaning_on",
    "lying_on",
    "not_contacting",
    "other_relationship",
    "sitting_on",
    "standing_on",
    "touching",
    "twisting",
    "wearing",
    "wiping",
    "writing_on",
]

MAX_ATTENTION = 1
MAX_SPATIAL = 5
MAX_CONTACTING = 4

IMG_PREFIX = "<image>\n"

# PVSG train_with_tags-style tail (first frame of clip)
TAIL_GENERATE = (
    "\nNow, generate the complete scene graph for the provided image. "
    "Wrap your scene graph in <answer>...</answer> tags.\n"
)

# Temporal block + tail for t>0 (three newlines before Now, PVSG-style)
TEMPORAL_BLOCK = (
    "You are also given the previous frame's ground-truth scene graph in TOON format.\n"
    "Use it as temporal context, but rely primarily on the current image.\n"
    "Important:\n"
    "- Include all objects visible in the CURRENT image, even if they did not exist in the previous graph.\n"
    "- Do NOT include objects that are NOT visible in the current image, even if they exist in the previous graph.\n"
    "- Output ONLY the complete scene graph for the CURRENT image: use the same TOON structure as in the "
    "format above, wrapped in <answer>...</answer> tags.\n\n"
    "Previous frame ground-truth scene graph (TOON), for reference:\n"
)

TAIL_GENERATE_FOLLOW = (
    "\n\n\nNow, generate the complete scene graph for the provided image. "
    "Wrap your scene graph in <answer>...</answer> tags.\n"
)


def _format_template_in_answer() -> str:
    return (
        "<answer>\n"
        "obj[N]{id,name,x1,y1,x2,y2}:\n"
        "  id,name,x1,y1,x2,y2\n"
        "  ...\n"
        "rel_pairs[M]{subj,attention,spatial,contacting,obj}:\n"
        "  subj,[attention_labels],[spatial_labels],[contacting_labels],obj\n"
        "  ...\n"
        "</answer>\n"
    )


def _guidelines_block_fixed() -> str:
    obj_s = json.dumps(OBJ_CLS, ensure_ascii=False)
    att_s = json.dumps(ATTENTION_CLS, ensure_ascii=False)
    spat_s = json.dumps(SPATIAL_CLS, ensure_ascii=False)
    cont_s = json.dumps(CONTACTING_CLS, ensure_ascii=False)
    return (
        "\nGuidelines (closed vocabulary):\n"
        "- Objects:\n"
        "  - Use integer IDs starting from 1 in the id field (e.g., 1, 2, 3).\n"
        "  - The name must belong to the predefined object set (person + interacted objects).\n"
        "  - Provide the bounding box [x1, y1, x2, y2] in integer pixel format.\n"
        "  - Include all visible objects that appear in the graph, even if some have no relationship row.\n"
        "- Relationship pairs:\n"
        "  - Each line is one (person, object) pair: subj is the person id, obj is the object id.\n"
        "  - attention, spatial, contacting are comma-separated lists inside square brackets, using exact "
        "labels from ATTENTION_CLS, SPATIAL_CLS, CONTACTING_CLS respectively.\n"
        "  - Use underscores as in the label names (e.g. in_front_of, not_looking_at).\n"
        "  - If a type has no label, use an empty list: [].\n"
        f"  - At most {MAX_ATTENTION} attention label, {MAX_SPATIAL} spatial labels, "
        f"and {MAX_CONTACTING} contacting labels per pair (limits match the training annotations).\n\n"
        "You are in the closed vocabulary setting. The object name in the name field must be chosen from "
        "OBJ_CLS below. Each bracket list must only contain values from its corresponding class list. "
        "If something does not match exactly, choose the closest category from the list.\n\n"
        f"OBJ_CLS (valid object categories): {obj_s}\n\n"
        f"ATTENTION_CLS: {att_s}\n\n"
        f"SPATIAL_CLS: {spat_s}\n\n"
        f"CONTACTING_CLS: {cont_s}\n\n"
    )


EXAMPLE_AG_IN_ANSWER = (
    "<answer>\n"
    "obj[3]{id,name,x1,y1,x2,y2}:\n"
    "  1,person,24,71,259,268\n"
    "  2,table,222,143,479,244\n"
    "  3,chair,56,179,249,269\n"
    "rel_pairs[2]{subj,attention,spatial,contacting,obj}:\n"
    "  1,[unsure],[in_front_of],[not_contacting],2\n"
    "  1,[not_looking_at],[beneath,behind],[sitting_on,leaning_on],3\n"
    "</answer>"
)


def build_user_prompt_first() -> str:
    return (
        "Generate a structured scene graph for an image of size (640 x 480) using the following format:\n"
        + _format_template_in_answer()
        + _guidelines_block_fixed()
        + "Example output:\n"
        + EXAMPLE_AG_IN_ANSWER
        + TAIL_GENERATE
    )


def build_prefix_through_example() -> str:
    """Shared user prefix through end of example (including closing </answer> of example)."""
    return (
        "Generate a structured scene graph for an image of size (640 x 480) using the following format:\n"
        + _format_template_in_answer()
        + _guidelines_block_fixed()
        + "Example output:\n"
        + EXAMPLE_AG_IN_ANSWER
    )


def build_user_prompt_follow(prev_assistant_body: str) -> str:
    """
    prev_assistant_body: formatted TOON of previous frame (as in assistant);
    wrapped in <answer>...</answer> inside user.
    """
    prev_block = f"<answer>\n{prev_assistant_body}\n</answer>"
    return build_prefix_through_example() + "\n" + TEMPORAL_BLOCK + prev_block + TAIL_GENERATE_FOLLOW


USER_PROMPT_FIRST = build_user_prompt_first()


def format_assistant_like_psg(answer_toon: str) -> str:
    """Section headers 6 spaces, data lines 8 spaces (PSG / PVSG style)."""
    out: List[str] = []
    for line in answer_toon.split("\n"):
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("obj[") or stripped.startswith("rel_pairs["):
            out.append("      " + stripped)
        else:
            out.append("        " + stripped)
    return "\n".join(out)


def wrap_assistant(answer_toon: str) -> str:
    body = format_assistant_like_psg(answer_toon)
    return "<answer>\n" + body + "\n</answer>\n"


def frame_sort_key(sample: Dict[str, Any]) -> Tuple[int, Any]:
    rel_img = sample.get("image_path") or ""
    stem = Path(rel_img).stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def video_key(sample: Dict[str, Any], repo_root: str) -> str:
    rel_img = sample.get("image_path") or ""
    if os.path.isabs(rel_img):
        p = Path(rel_img)
    else:
        p = Path(repo_root) / rel_img
    return str(p.parent.resolve())


def group_and_sort_samples(
    data: List[Dict[str, Any]], repo_root: str
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for s in data:
        k = video_key(s, repo_root)
        if k not in buckets:
            buckets[k] = []
            order.append(k)
        buckets[k].append(s)
    out: List[Dict[str, Any]] = []
    for k in order:
        rows = buckets[k]
        rows.sort(key=frame_sort_key)
        out.extend(rows)
    return out


def make_jsonl(in_path: str, out_path: str, repo_root: str) -> int:
    repo_root = os.path.abspath(repo_root)
    with open(in_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    data = group_and_sort_samples(data, repo_root)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    prev_formatted: str | None = None
    prev_clip: str | None = None

    with open(out_path, "w", encoding="utf-8") as fout:
        for sample in tqdm(data, desc=os.path.basename(out_path)):
            rel_img = sample["image_path"]
            if os.path.isabs(rel_img):
                abs_img = rel_img
            else:
                abs_img = os.path.normpath(os.path.join(repo_root, rel_img))

            clip = video_key(sample, repo_root)
            if clip != prev_clip:
                prev_formatted = None
                prev_clip = clip

            assistant_wrapped = wrap_assistant(sample["answer_toon"])

            if prev_formatted is None:
                user_body = USER_PROMPT_FIRST
            else:
                user_body = build_user_prompt_follow(prev_formatted)

            item = {
                "messages": [
                    {"role": "user", "content": IMG_PREFIX + user_body},
                    {
                        "role": "assistant",
                        "content": assistant_wrapped,
                    },
                ],
                "images": [abs_img],
            }
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

            inner = assistant_wrapped
            if inner.startswith("<answer>\n") and inner.endswith("</answer>\n"):
                prev_formatted = inner[len("<answer>\n") : -len("</answer>\n")].rstrip("\n")
            else:
                prev_formatted = format_assistant_like_psg(sample["answer_toon"])

    return len(data)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AG TOON JSON ==> jsonl with chat prompts (image_path relative to repo root)"
    )
    ap.add_argument(
        "--repo_root",
        default="",
        help="SceneGraphVLM repo root (empty ==> infer from this script location).",
    )
    ap.add_argument(
        "--export_root",
        default=EXPORT_ROOT_REL,
        help="Directory with train/test *_annotations_toon_sft.json, relative to repo unless absolute.",
    )
    ap.add_argument(
        "--out_dir",
        default=OUT_DIR_REL,
        help="Output directory for train.jsonl / test.jsonl, relative to repo unless absolute.",
    )
    args = ap.parse_args()

    repo_root = Path(args.repo_root).resolve() if args.repo_root else _REPO_ROOT_DEFAULT
    export_root = resolve_under_repo(repo_root, args.export_root)
    out_dir = resolve_under_repo(repo_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    train_in = str(export_root / "train_annotations_toon_sft.json")
    test_in = str(export_root / "test_annotations_toon_sft.json")
    train_out = str(out_dir / "train.jsonl")
    test_out = str(out_dir / "test.jsonl")

    n_train = make_jsonl(train_in, train_out, str(repo_root))
    n_test = make_jsonl(test_in, test_out, str(repo_root))
    print("Wrote:", train_out, f"({n_train} lines)")
    print("Wrote:", test_out, f"({n_test} lines)")
    if n_train < 10_000:
        print(
            "\n[warn] Few train lines - likely not a full export.\n"
            "       Rebuild JSON: python .../AG_annot/tools/prepare_original_ag_sft.py --num_workers 32\n",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
