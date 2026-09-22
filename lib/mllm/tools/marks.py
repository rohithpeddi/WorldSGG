"""Set-of-mark frame rendering: numbered boxes on the original frame (B4/B5)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

from lib.mllm.tools.bev import _font

PALETTE = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 190), (0, 128, 128),
           (170, 110, 40), (128, 0, 0), (0, 0, 128), (128, 128, 0), (255, 215, 180)]


def mark_frame(img: Image.Image, objects: Sequence[Dict[str, Any]],
               person_bbox: Optional[Sequence[float]] = None, max_side: int = 640) -> Image.Image:
    """Draw ``{"id", "label", "bbox"}`` (ORIGINAL pixel xyxy) boxes with id tags; person in black.
    Objects without a box (unobserved) are listed in the top-left legend instead."""
    scale = min(1.0, max_side / max(img.size))
    im = img.convert("RGB")
    if scale < 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.BILINEAR)
    d = ImageDraw.Draw(im, "RGBA")
    f = _font(14)

    def box(bb, color, tag):
        x1, y1, x2, y2 = [float(v) * scale for v in bb]
        d.rectangle([x1, y1, x2, y2], outline=color + (255,), width=2)
        tw = d.textlength(tag, font=f) if hasattr(d, "textlength") else 8 * len(tag)
        d.rectangle([x1, max(0, y1 - 16), x1 + tw + 6, max(0, y1 - 16) + 16], fill=color + (230,))
        d.text((x1 + 3, max(0, y1 - 16)), tag, fill=(255, 255, 255, 255), font=f)

    if person_bbox is not None:
        box(person_bbox, (0, 0, 0), "P person")
    legend = []
    for i, o in enumerate(objects):
        c = PALETTE[i % len(PALETTE)]
        if o.get("bbox") is not None:
            box(o["bbox"], c, f"{o['id']} {o['label']}")
        else:
            legend.append((f"{o['id']} {o['label']} (not visible here)", c))
    y = 4
    for txt, c in legend:
        d.rectangle([2, y, 6 + (d.textlength(txt, font=f) if hasattr(d, "textlength") else 8 * len(txt)), y + 16],
                    fill=(255, 255, 255, 200))
        d.text((4, y), txt, fill=c + (255,), font=f)
        y += 17
    return im
