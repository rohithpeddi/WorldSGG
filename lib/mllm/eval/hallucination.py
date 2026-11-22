"""
Hallucination metrics for any MLLM run (optional block, ``score_run --halluc``).
================================================================================

Computed from the run's per-frame predictions (``dump_adapter.prediction_lookup``) and the
worldbbox annotation, independently of the recall regimes.

* **UOR (unsupported-object rate)** = predicted objects whose class is absent from the
  video's WORLD GT inventory (every object annotated in any frame of the video, observed
  or not) / predicted objects.  A predicted object = one (frame, class) the model named,
  person excluded, relation rows or not (``objects_norel``); labels outside the 36-class
  vocabulary count as unsupported.  Because the inventory includes unobserved objects,
  naming an out-of-view object that really exists is not a hallucination.
  ``uor_frame`` is the stricter variant against the frame's own GT slots.
* **URR (unsupported-relation rate)** = per frame, predicted triplets (person, predicate,
  object) whose (subject, object) pair exists in the frame's GT but whose predicate is not
  among that pair's GT predicates / predicted triplets.  Reported overall (denominator =
  all predicted triplets) and split by the GT object's visibility: ``urr_observed`` /
  ``urr_unobserved`` = unsupported / predicted triplets on observed / unobserved GT
  objects.  Triplets on pairs absent from the frame's GT are counted in
  ``frac_triplets_pair_not_in_gt`` (they are UOR's business, not URR's).

AG labels are incomplete, so both rates are upper bounds (see docs/EXTERNAL_BASELINES_PLAN.md).
Counts are micro-averaged over all (frame, object) / triplets of the scored videos.
"""
from __future__ import annotations

from typing import Any, Dict

from lib.mllm.data.worldbbox import (
    ATTENTION_RELATIONSHIPS, CONTACTING_RELATIONSHIPS, NAME_TO_IDX, SPATIAL_RELATIONSHIPS,
)

_HEADS = (("attention", set(ATTENTION_RELATIONSHIPS)), ("spatial", set(SPATIAL_RELATIONSHIPS)),
          ("contacting", set(CONTACTING_RELATIONSHIPS)))


class HallucAccumulator:
    def __init__(self):
        self.c: Dict[str, int] = {k: 0 for k in (
            "frames", "frames_with_pred", "pred_objects", "unsupported_objects", "pred_objects_not_in_frame_gt",
            "pred_objects_outside_vocab", "triplets", "triplets_nonvocab_predicate", "triplets_pair_not_in_gt",
            "triplets_obs", "unsupported_obs", "triplets_unobs", "unsupported_unobs")}

    def add_video(self, video, preds: Dict[str, Dict[str, Dict[str, Any]]]) -> None:
        world = set(video.video_objects())
        c = self.c
        for fr in video.frames:
            c["frames"] += 1
            fp = preds.get(fr.file) or {}
            gt: Dict[str, Any] = {}
            for o in fr.objects:
                gt.setdefault(o.label, o)                 # unique by label, as build_records
            labs = [l for l in fp if not l.startswith("__") and l != "person"]
            if labs:
                c["frames_with_pred"] += 1
            for lab in labs:
                c["pred_objects"] += 1
                if lab not in NAME_TO_IDX:
                    c["pred_objects_outside_vocab"] += 1
                if lab not in world or lab not in NAME_TO_IDX:
                    c["unsupported_objects"] += 1
                if lab not in gt:
                    c["pred_objects_not_in_frame_gt"] += 1
                p = fp[lab]
                trip = []
                for head, vocab in _HEADS:
                    for rl, sc in p.get(head) or []:
                        if sc is not None and float(sc) <= 0:
                            continue
                        if rl not in vocab:
                            c["triplets_nonvocab_predicate"] += 1
                            continue
                        trip.append(rl)
                c["triplets"] += len(trip)
                o = gt.get(lab)
                if o is None:
                    c["triplets_pair_not_in_gt"] += len(trip)
                    continue
                gset = set(o.attention) | set(o.spatial) | set(o.contacting)
                n_uns = sum(1 for rl in trip if rl not in gset)
                key = "obs" if o.observed else "unobs"
                c["triplets_" + key] += len(trip)
                c["unsupported_" + key] += n_uns

    def summary(self) -> Dict[str, Any]:
        c = self.c
        d = lambda a, b: (a / b) if b else None  # noqa: E731
        return {
            "uor": d(c["unsupported_objects"], c["pred_objects"]),
            "uor_frame": d(c["pred_objects_not_in_frame_gt"], c["pred_objects"]),
            "urr": d(c["unsupported_obs"] + c["unsupported_unobs"], c["triplets"]),
            "urr_observed": d(c["unsupported_obs"], c["triplets_obs"]),
            "urr_unobserved": d(c["unsupported_unobs"], c["triplets_unobs"]),
            "urr_pairs_in_gt": d(c["unsupported_obs"] + c["unsupported_unobs"],
                                 c["triplets_obs"] + c["triplets_unobs"]),
            "frac_triplets_pair_not_in_gt": d(c["triplets_pair_not_in_gt"], c["triplets"]),
            "pred_objects_per_frame": d(c["pred_objects"], c["frames"]),
            "triplets_per_frame": d(c["triplets"], c["frames"]),
            "counts": dict(c),
        }
