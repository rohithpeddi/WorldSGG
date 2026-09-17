"""
label_constants.py
==================
Shared object / relationship label vocabularies and normalization utilities.

The AG dataset stores compound names (e.g. ``"phone/camera"``), while the
UI pipeline and MLLM predictions use short forms (e.g. ``"phone"``).  This
module provides a single source of truth for mapping between them.
"""

from typing import Dict

# ---- Object classes (Action Genome vocabulary, 1-indexed) ------------------

OBJECT_CLASSES = [
    "__background__", "person", "bag", "bed", "blanket", "book", "box",
    "broom", "chair", "closet/cabinet", "clothes", "cup/glass/bottle",
    "dish", "door", "doorknob", "doorway", "floor", "food", "groceries",
    "laptop", "light", "medicine", "mirror", "paper/notebook",
    "phone/camera", "picture", "pillow", "refrigerator", "sandwich",
    "shelf", "shoe", "sofa/couch", "table", "television", "towel",
    "vacuum", "window",
]

# Short-form ↔ full AG name mappings
LABEL_NORMALIZE_MAP: Dict[str, str] = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP: Dict[str, str] = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}

# Object name → class index  (both full and short forms)
NAME_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(OBJECT_CLASSES) if idx > 0
}
for _short, _full in LABEL_DENORMALIZE_MAP.items():
    NAME_TO_IDX[_short] = NAME_TO_IDX[_full]


# ---- Relationship classes --------------------------------------------------

ATTENTION_RELATIONSHIPS = ["looking_at", "not_looking_at", "unsure"]

CONTACTING_RELATIONSHIPS = [
    "carrying", "covered_by", "drinking_from", "eating",
    "have_it_on_the_back", "holding", "leaning_on", "lying_on",
    "not_contacting", "other_relationship", "sitting_on", "standing_on",
    "touching", "twisting", "wearing", "wiping", "writing_on",
]

SPATIAL_RELATIONSHIPS = [
    "above", "beneath", "in_front_of", "behind", "on_the_side_of", "in",
]

# Space-separated → underscore (MLLM outputs sometimes use spaces)
_LABEL_SPACE_TO_UNDERSCORE: Dict[str, str] = {
    "looking at": "looking_at",
    "not looking at": "not_looking_at",
    "covered by": "covered_by",
    "drinking from": "drinking_from",
    "have it on the back": "have_it_on_the_back",
    "leaning on": "leaning_on",
    "lying on": "lying_on",
    "not contacting": "not_contacting",
    "other relationship": "other_relationship",
    "sitting on": "sitting_on",
    "standing on": "standing_on",
    "writing on": "writing_on",
    "in front of": "in_front_of",
    "on the side of": "on_the_side_of",
}


# ---- Normalisation helpers -------------------------------------------------

def normalise_object_label(label: str) -> str:
    """Normalise an object class name.

    1. Strip whitespace, lowercase, replace spaces with underscores.
    2. Map compound AG names to their short form (e.g. "phone/camera" → "phone").
    """
    s = label.strip().lower().replace(" ", "_")
    return LABEL_NORMALIZE_MAP.get(s, s)


def normalise_relationship_label(label: str) -> str:
    """Normalise a relationship label.

    1. Strip whitespace, lowercase.
    2. Map space-separated LLM output to underscore form.
    3. Replace remaining spaces with underscores.
    """
    s = label.strip().lower()
    s = _LABEL_SPACE_TO_UNDERSCORE.get(s, s)
    return s.replace(" ", "_")
