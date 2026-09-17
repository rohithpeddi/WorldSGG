"""
Track A prompt templates (B5): marked frames + BEV -> JSON with predicates and
floor-parallel OBBs in the canonical frame.

Design (docs/ICLR_PLAN.md WS2 Track A): the model sees the target frame with
numbered object boxes (set-of-mark), a few unmarked context frames, and the
top-down BEV of the Pi3 reconstruction with the same ids stamped on the object
footprints, the person (P) and the camera (red arrow).  The text lists the same
objects with their metric boxes so the answer can be grounded numerically.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

from lib.mllm.data.worldbbox import (
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, OBJECT_CLASSES, SPATIAL_RELATIONSHIPS,
    to_short,
)

VOCAB = sorted({to_short(c) for c in OBJECT_CLASSES[2:]})

SYSTEM_RULES = """\
You are a careful 3D scene understanding model. Coordinates are metric, in a fixed \
world frame: z is up, the floor is z = 0, x/y follow the grid drawn on the top-down map \
(labels "x=..", "y=.." are metres). A box is {"center": [x, y, z], "size": [length, width, \
height], "yaw_deg": rotation of the length axis about z}. The person is the only subject; \
every relationship is between the person and one object.

Relationship vocabulary (use these exact strings):
- attention (exactly one): %s
- contacting (one or more): %s
- spatial (one or more, position of the OBJECT relative to the PERSON): %s
""" % (", ".join(ATTENTION_RELATIONSHIPS), ", ".join(CONTACTING_RELATIONSHIPS),
       ", ".join(SPATIAL_RELATIONSHIPS))


def _fmt_box(obb: Dict[str, Any]) -> str:
    c, s = obb["center"], obb["size"]
    return (f'center=({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m, size=({s[0]:.2f}, {s[1]:.2f}, {s[2]:.2f}) m, '
            f'yaw={obb.get("yaw_deg", 0.0):.0f} deg')


def image_legend(n_context: int, has_bev: bool) -> str:
    parts = ["Image 1: the TARGET frame with numbered boxes (P = person)."]
    if n_context:
        parts.append(f"Images 2-{1 + n_context}: unmarked frames from the same video, in time order, for context.")
    if has_bev:
        parts.append(f"Image {2 + n_context}: top-down map of the whole room reconstructed from the video "
                     "(grey dots = camera path, red arrow = camera position and viewing direction at the "
                     "target frame, numbered polygons = object footprints, P = person footprint).")
    return "\n".join(parts)


def predcls_prompt(frame_file: str, objects: Sequence[Dict[str, Any]], person_obb: Dict[str, Any] | None,
                   camera: Dict[str, Any] | None, n_context: int, has_bev: bool) -> str:
    lines = [SYSTEM_RULES, image_legend(n_context, has_bev), ""]
    lines.append("Scene at the target frame (all objects below are known to exist in this room, "
                 "some are outside the camera view or hidden):")
    if person_obb:
        lines.append(f"- P person: {_fmt_box(person_obb)}")
    if camera:
        lines.append(f"- camera: position=({camera['x']:.2f}, {camera['y']:.2f}, {camera['z']:.2f}) m, "
                     f"looking towards heading {camera['heading_deg']:.0f} deg")
    for o in objects:
        vis = "visible in the target frame" if o.get("visible") else "NOT visible in the target frame"
        box = _fmt_box(o["obb"]) if o.get("obb") else "3D box unknown"
        lines.append(f"- {o['id']} {o['label']}: {vis}; {box}")
    lines.append("")
    lines.append("Task: for EVERY listed object id, decide the person's attention, contacting and spatial "
                 "relationships with it at the target frame. Use the map and the boxes for objects that are "
                 "not visible; use the images for what the person is doing. Think briefly, then answer with "
                 "ONLY this JSON (no prose after it):")
    lines.append('{"objects": [{"id": 1, "attention": "<label>", "contacting": ["<label>", ...], '
                 '"spatial": ["<label>", ...]}, ...]}')
    return "\n".join(lines)


def sgdet_prompt(frame_file: str, detections: Sequence[Dict[str, Any]], person_obb: Dict[str, Any] | None,
                 camera: Dict[str, Any] | None, n_context: int, has_bev: bool, extent: Dict[str, float]) -> str:
    lines = [SYSTEM_RULES, image_legend(n_context, has_bev), ""]
    lines.append(f"The map covers x in [{extent['x0']:.1f}, {extent['x1']:.1f}] m and "
                 f"y in [{extent['y0']:.1f}, {extent['y1']:.1f}] m.")
    lines.append("Detector proposals at the target frame (may contain errors, duplicates and misses):")
    if person_obb:
        lines.append(f"- P person: {_fmt_box(person_obb)}")
    if camera:
        lines.append(f"- camera: position=({camera['x']:.2f}, {camera['y']:.2f}, {camera['z']:.2f}) m, "
                     f"looking towards heading {camera['heading_deg']:.0f} deg")
    for o in detections:
        box = _fmt_box(o["obb"]) if o.get("obb") else "3D box unknown"
        lines.append(f"- {o['id']} {o['label']} (detector score {o.get('score', 0):.2f}): {box}")
    lines.append("")
    lines.append("Object vocabulary (use these exact names): " + ", ".join(VOCAB))
    lines.append("")
    lines.append("Task: produce the scene graph at the target frame. List every object the person is "
                 "interacting with or that is relevant to the activity, including objects that are out of "
                 "view or hidden but whose location you can infer from the video and the map (e.g. the chair "
                 "the person sat on earlier). For each object give its label, a 3D box in the world frame "
                 "(copy or correct a proposal's box; estimate a plausible box for objects without a proposal), "
                 "and the person's attention / contacting / spatial relationships with it. Keep proposal ids "
                 "when you keep a proposal; use \"new\" for objects you add. Think briefly, then answer with "
                 "ONLY this JSON (no prose after it):")
    lines.append('{"objects": [{"id": 1, "label": "<vocab name>", "center": [x, y, z], "size": [l, w, h], '
                 '"yaw_deg": 0, "attention": "<label>", "contacting": ["<label>", ...], '
                 '"spatial": ["<label>", ...]}, ...]}')
    return "\n".join(lines)
