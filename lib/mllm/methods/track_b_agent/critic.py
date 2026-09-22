"""
Geometric / schema critic for Track B (B6): the ablatable, novel piece.

``check_geometry(graph, person_corners, room_extent)`` inspects a proposed
localised scene graph (objects with OBB corners in the canonical frame and
person-object predicates) and returns a list of violations with human-readable
explanations that the repair prompt feeds back to the model.  Only localised
predictions can be checked, which is the point of the track.

Checks (all in the canonical z-up floor frame, metres):
  schema      label in vocabulary, exactly one attention label, predicate strings valid
  floor       box bottom below the floor (z_min < -tol) or floating high above it
              while touching-type contact is claimed
  extent      box centre outside the reconstructed room window (+ margin)
  size        any side > 3 m or < 1 cm
  contact     holding / touching / carrying / eating / drinking_from / wiping /
              writing_on / twisting / wearing / have_it_on_the_back need the object
              within ``contact_dist`` of the person box; sitting_on / lying_on /
              standing_on / leaning_on / covered_by need it within ``support_dist``
              and overlapping the person in z; not_contacting with the object box
              inside the person box is contradictory
  vertical    above  -> object centre higher than the person centre
              beneath -> object centre lower than the person centre
              (front / behind / side are not checked: the person's facing is unknown)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from lib.mllm.data.worldbbox import (
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, NAME_TO_IDX, SPATIAL_RELATIONSHIPS,
)

CONTACT_PREDS = {"holding", "touching", "carrying", "eating", "drinking_from", "wiping",
                 "writing_on", "twisting", "wearing", "have_it_on_the_back"}
SUPPORT_PREDS = {"sitting_on", "lying_on", "standing_on", "leaning_on", "covered_by"}


def _aabb(c: np.ndarray):
    c = np.asarray(c, np.float64).reshape(8, 3)
    return c.min(0), c.max(0)


def _box_gap(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean gap between two axis-aligned envelopes (0 when they overlap)."""
    amin, amax = _aabb(a)
    bmin, bmax = _aabb(b)
    d = np.maximum(0.0, np.maximum(bmin - amax, amin - bmax))
    return float(np.linalg.norm(d))


def check_geometry(objects: Dict[str, Dict[str, Any]], person_corners: Optional[np.ndarray],
                   extent: Optional[Dict[str, float]] = None, floor_tol: float = 0.15,
                   contact_dist: float = 0.5, support_dist: float = 0.25, margin: float = 1.0
                   ) -> List[Dict[str, Any]]:
    """``objects``: {label: {"attention": [..], "spatial": [..], "contacting": [..],
    "corners": (8,3) | None}} -> list of {"label", "kind", "msg"}."""
    out: List[Dict[str, Any]] = []
    pmin = pmax = pc = None
    if person_corners is not None and np.any(person_corners):
        pmin, pmax = _aabb(person_corners)
        pc = 0.5 * (pmin + pmax)
    for label, o in objects.items():
        def flag(kind, msg):
            out.append({"label": label, "kind": kind, "msg": msg})
        # ---- schema ----
        if label not in NAME_TO_IDX:
            flag("schema", f"'{label}' is not in the object vocabulary")
        att = o.get("attention") or []
        if len(att) != 1 or att[0] not in ATTENTION_RELATIONSHIPS:
            flag("schema", f"{label}: attention must be exactly one of {ATTENTION_RELATIONSHIPS}")
        bad = [p for p in (o.get("spatial") or []) if p not in SPATIAL_RELATIONSHIPS]
        bad += [p for p in (o.get("contacting") or []) if p not in CONTACTING_RELATIONSHIPS]
        if bad:
            flag("schema", f"{label}: unknown predicate(s) {bad}")
        con = set(o.get("contacting") or [])
        spa = set(o.get("spatial") or [])
        c = o.get("corners")
        if c is None or not np.any(c):
            continue
        cmin, cmax = _aabb(c)
        cc = 0.5 * (cmin + cmax)
        size = cmax - cmin
        # ---- floor / size / extent ----
        if cmin[2] < -floor_tol:
            flag("floor", f"{label}: box bottom is {-cmin[2]:.2f} m below the floor (z=0)")
        if size.max() > 3.0 or size.min() < 0.01:
            flag("size", f"{label}: implausible box size {np.round(size, 2).tolist()} m")
        if extent is not None:
            if not (extent["x0"] - margin <= cc[0] <= extent["x1"] + margin and
                    extent["y0"] - margin <= cc[1] <= extent["y1"] + margin):
                flag("extent", f"{label}: box centre ({cc[0]:.2f}, {cc[1]:.2f}) is outside the reconstructed room")
        if pc is None:
            continue
        gap = _box_gap(c, person_corners)
        # ---- contact consistency ----
        if con & CONTACT_PREDS and gap > contact_dist:
            flag("contact", f"{label}: {sorted(con & CONTACT_PREDS)} claimed but the box is {gap:.2f} m away from the person")
        if con & SUPPORT_PREDS:
            z_overlap = min(cmax[2], pmax[2]) - max(cmin[2], pmin[2])
            if gap > support_dist or z_overlap <= 0:
                flag("contact", f"{label}: {sorted(con & SUPPORT_PREDS)} claimed but the box does not touch the person "
                                f"(gap {gap:.2f} m, vertical overlap {z_overlap:.2f} m)")
        if "not_contacting" in con and gap == 0.0 and np.all(cmin >= pmin - 0.05) and np.all(cmax <= pmax + 0.05):
            flag("contact", f"{label}: not_contacting but the box lies inside the person box")
        if con & CONTACT_PREDS and cmin[2] > pmax[2] + 0.3:
            flag("floor", f"{label}: contact claimed but the box floats {cmin[2] - pmax[2]:.2f} m above the person")
        # ---- vertical relations ----
        if "above" in spa and cc[2] <= pc[2]:
            flag("vertical", f"{label}: 'above' but its centre (z={cc[2]:.2f}) is not higher than the person's (z={pc[2]:.2f})")
        if "beneath" in spa and cc[2] >= pc[2]:
            flag("vertical", f"{label}: 'beneath' but its centre (z={cc[2]:.2f}) is not lower than the person's (z={pc[2]:.2f})")
        if "above" in spa and "beneath" in spa:
            flag("vertical", f"{label}: both 'above' and 'beneath'")
    return out


def summarize(violations: List[Dict[str, Any]]) -> Dict[str, int]:
    s: Dict[str, int] = {}
    for v in violations:
        s[v["kind"]] = s.get(v["kind"], 0) + 1
    return s
