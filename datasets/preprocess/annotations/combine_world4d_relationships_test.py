#!/usr/bin/env python3
"""
combine_world4d_relationships_test.py
======================================
Merge **human-corrected** relationship annotations (GT + corrections) with
world4D 3D bounding box annotations into a unified per-video PKL file for
the **test** split.

Unlike the train version (which reads RAG-augmented PKLs with ``object_bbox``
/ ``person_bbox`` dicts, source ``"gt"|"rag"``, and ``*_scores`` fields),
this script reads the test augmentation PKLs produced by
``augment_relationships_test.py``, which store per-frame data under a
``frames`` dict with simpler relationship keys (``attention``, ``contacting``,
``spatial``) and ``source`` values ``"gt"`` or ``"correction"``.

Inputs:
  - ``<ag_root>/wsg_2d_augmentations/<video>.pkl``
    (from ``augment_relationships_test.py``)
  - ``<ag_root>/world_annotations/bbox_annotations_4d/<video>.pkl``
    (from ``frame_to_world4D_annotations.py``)

Output:
  - ``<ag_root>/world4d_rel_annotations/test/<video>.pkl``

Pipeline:
  1) Load augmented_rel PKL (GT + correction relationships, 2D bboxes).
  2) Load world4D PKL (3D OBBs, camera poses, floor mesh, object permanence).
  3) Per-frame: verify object label sets match, extract person 3D data,
     merge each object's relationship labels with its 3D bbox data.
  4) Filter camera_poses to only frames present in augmented_rel.
  5) Save combined PKL; log label mismatches to ``verify.txt``.

OUTPUT PKL STRUCTURE
--------------------
{
    "video_id":        str,
    "num_frames":      int,
    "all_labels":      List[str],              # sorted unique short-form labels

    "frames": {
        "video_id/000042.png": {
            "person_info": {
                # 2D (from augmented_rel)
                "bbox_2d":             np.ndarray (4,),    # xyxy person bbox
                "bbox_size":           (w, h),             # frame dimensions
                # 3D (from world4D person object)
                "corners_world":       np.ndarray (8,3),   # WORLD-space OBB
                "corners_final":       np.ndarray (8,3),   # FINAL-space OBB
                "obb_final":           dict,               # {"corners_final": ...}
                "world4d_filled":      bool,
                "world4d_fill_method": str,
            },
            "object_info_list": [
                {
                    # From augmented_rel (test format)
                    "class":                    str,        # normalised object name
                    "label":                    str,        # short-form
                    "bbox_2d":                  np.ndarray | None,  # xyxy
                    "visible":                  bool,
                    "attention_relationship":   list[str],
                    "contacting_relationship":  list[str],
                    "spatial_relationship":     list[str],
                    "source":                   "gt"|"correction",
                    # From world4D (matched by short-form label)
                    "corners_world":            np.ndarray (8,3),
                    "corners_final":            np.ndarray (8,3),
                    "obb_final":                dict,
                    "world4d_filled":           bool,
                    "world4d_fill_method":      str,
                    "world4d_source":           "gt"|"gdino",
                },
                ...
            ],
        },
    },

    # Filtered camera poses (only valid frames, ordered)
    "camera_poses":      np.ndarray (N, 4, 4),   # FINAL coords
    "camera_frame_keys": List[str],              # corresponding frame keys

    # Pass-through from world4D
    "floor_mesh": {                           # raw floor mesh (WORLD space)
        "gv":          ndarray (V, 3),        # vertices
        "gf":          ndarray (F, 3),        # faces
        "gc":          ndarray (V, 3)|None,   # vertex colors
    } | None,
    "global_floor_sim": {                     # similarity transform params
        "s":           float,                 # scale
        "R":           ndarray (3, 3),        # rotation
        "t":           ndarray (3,),          # translation
    } | None,
    "world_to_final": {                       # pre-computed WORLD→FINAL transform
        "origin_world":       ndarray (3,),   # floor origin in WORLD
        "A_world_to_final":   ndarray (3, 3), # rotation+mirror matrix
    } | None,
    "floor_final":     dict | None,           # floor mesh already in FINAL coords
}

COORDINATE SYSTEMS
------------------
- corners_world:  Raw Pi3 reconstruction space.
- corners_final:  Floor-aligned space (Y=up, XZ=floor plane).
- camera_poses:   Already in FINAL coords.
- floor_final:    Already in FINAL coords.

Usage:
    python datasets/preprocess/annotations/combine_world4d_relationships_test.py \\
        --ag_root_directory /data/rohith/ag
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Label normalization (short-form ↔ full AG name)
# ---------------------------------------------------------------------------

LABEL_NORMALIZE_MAP = {
    "closet/cabinet": "closet",
    "cup/glass/bottle": "cup",
    "paper/notebook": "paper",
    "sofa/couch": "sofa",
    "phone/camera": "phone",
}
LABEL_DENORMALIZE_MAP = {v: k for k, v in LABEL_NORMALIZE_MAP.items()}


def _to_short(label: str) -> str:
    """Normalize a full AG label to short form.

    ``"closet/cabinet"`` → ``"closet"``, ``"bed"`` → ``"bed"``.
    """
    return LABEL_NORMALIZE_MAP.get(label, label)


def _to_full(label: str) -> str:
    """Expand a short-form label back to the full AG name.

    ``"closet"`` → ``"closet/cabinet"``, ``"bed"`` → ``"bed"``.
    """
    return LABEL_DENORMALIZE_MAP.get(label, label)


# ---------------------------------------------------------------------------
# Logger setup
# ---------------------------------------------------------------------------

_CODE_ROOT = Path(__file__).resolve().parents[3]  # WorldSGG/
_LOG_DIR = _CODE_ROOT / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("combine_w4d_rel_test")
logger.setLevel(logging.DEBUG)

_ch = logging.StreamHandler()
_ch.setLevel(logging.INFO)
_ch.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_ch)

_fh = logging.FileHandler(
    _LOG_DIR / "combine_world4d_relationships_test.log", mode="a",
)
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
))
logger.addHandler(_fh)


# ---------------------------------------------------------------------------
# 3D field extraction helpers
# ---------------------------------------------------------------------------

def _extract_3d_fields(w4d_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the relevant 3D fields from a world4D object dict."""
    obb_fp = w4d_obj.get("obb_floor_parallel", {})
    return {
        "corners_world": np.asarray(obb_fp.get("corners_world", []), dtype=np.float32)
                         if obb_fp.get("corners_world") is not None else None,
        "corners_final": np.asarray(w4d_obj["corners_final"], dtype=np.float32)
                         if w4d_obj.get("corners_final") is not None else None,
        "obb_final": w4d_obj.get("obb_final", None),
        "world4d_filled": w4d_obj.get("world4d_filled", False),
        "world4d_fill_method": w4d_obj.get("world4d_fill_method", "detected"),
        "world4d_source": w4d_obj.get("source", "gt"),
    }


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def process_video(
    video_id: str,
    rel_pkl_path: Path,
    w4d_pkl_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    """Merge test augmented_rel + world4D for a single video.

    The test augmented_rel PKL (from ``augment_relationships_test.py``) has
    a ``frames`` dict keyed by ``"<video_id>/<frame>.jpg"`` (or ``.png``),
    where each frame contains ``person_bbox`` and ``objects`` — a list of
    dicts with keys ``class``, ``source``, ``attention``, ``contacting``,
    ``spatial``, and ``bbox``.

    Returns
    -------
    success : bool
    mismatch_lines : list[str]
        Verification lines for label mismatches (empty if clean).
    output_record : dict or None
    """
    if output_path.exists() and not overwrite:
        logger.debug(f"[{video_id}] Skipping: output exists at {output_path}")
        return True, [], None

    # ------------------------------------------------------------------
    # 1) Load PKLs
    # ------------------------------------------------------------------
    if not rel_pkl_path.exists():
        logger.warning(f"[{video_id}] Augmented rel PKL not found: {rel_pkl_path}")
        return False, [], None

    if not w4d_pkl_path.exists():
        logger.warning(f"[{video_id}] World4D PKL not found: {w4d_pkl_path}")
        return False, [], None

    with open(rel_pkl_path, "rb") as f:
        rel_data = pickle.load(f)

    with open(w4d_pkl_path, "rb") as f:
        w4d_data = pickle.load(f)

    logger.info(f"[{video_id}] Loaded augmented_rel PKL: {rel_pkl_path.name}")
    logger.info(f"[{video_id}] Loaded world4D PKL: {w4d_pkl_path.name}")

    # ---- Detailed PKL structure logging ----
    logger.debug(
        f"[{video_id}] augmented_rel top-level keys: {sorted(rel_data.keys())}"
    )
    logger.debug(
        f"[{video_id}] augmented_rel video_id field: {rel_data.get('video_id', 'N/A')}"
    )
    logger.debug(
        f"[{video_id}] world4D top-level keys: {sorted(w4d_data.keys())}"
    )
    logger.debug(
        f"[{video_id}] world4D video_id field: {w4d_data.get('video_id', 'N/A')}"
    )
    logger.debug(
        f"[{video_id}] world4D all_labels field: {w4d_data.get('all_labels', 'N/A')}"
    )

    # ------------------------------------------------------------------
    # 2) Build stem→camera index from world4D
    # ------------------------------------------------------------------
    w4d_frame_stems = w4d_data.get("frame_stems", [])
    w4d_camera_poses = w4d_data.get("camera_poses", None)
    stem_to_cam_idx: Dict[str, int] = {
        s: i for i, s in enumerate(w4d_frame_stems)
    }

    w4d_frames = w4d_data.get("frames", {})

    logger.debug(
        f"[{video_id}] World4D: {len(w4d_frames)} frames, "
        f"{len(w4d_frame_stems)} camera stems, "
        f"camera_poses={'None' if w4d_camera_poses is None else w4d_camera_poses.shape}"
    )
    logger.debug(
        f"[{video_id}] World4D frame names (first 5): "
        f"{sorted(w4d_frames.keys())[:5]}"
    )

    # ------------------------------------------------------------------
    # 3) Per-frame merge
    # ------------------------------------------------------------------
    # Test augmented PKL stores data under "frames" dict keyed by
    # "<video_id>/<frame>.jpg" with each value having "person_bbox"
    # and "objects" list.
    rel_frames = rel_data.get("frames", {})

    logger.debug(
        f"[{video_id}] Augmented rel (test): {len(rel_frames)} frames"
    )
    logger.debug(
        f"[{video_id}] Augmented rel frame keys (first 5): "
        f"{sorted(rel_frames.keys())[:5]}"
    )

    combined_frames: Dict[str, Dict[str, Any]] = {}
    all_labels: Set[str] = set()
    mismatch_lines: List[str] = []

    valid_cam_indices: List[int] = []
    camera_frame_keys: List[str] = []

    n_person_matched = 0
    n_person_missing = 0
    n_obj_matched = 0
    n_obj_missing_3d = 0
    n_cam_matched = 0
    n_cam_missing = 0

    for frame_key in sorted(rel_frames.keys()):
        frame_data = rel_frames[frame_key]

        # Extract frame filename — handle both .jpg and .png suffixes
        frame_file = frame_key.split("/")[-1]  # e.g. "000042.jpg"
        frame_stem = frame_file.replace(".png", "").replace(".jpg", "")
        # World4D uses .png keys
        frame_name_png = f"{frame_stem}.png"

        # ---- World4D lookup ----
        w4d_frame_rec = w4d_frames.get(frame_name_png, None)

        if w4d_frame_rec is None:
            logger.warning(
                f"[{video_id}] Frame {frame_file}: NOT FOUND in world4D frames — "
                f"skipping 3D data for this frame"
            )
            w4d_objects = []
        else:
            w4d_objects = w4d_frame_rec.get("objects", [])

        # Build label→obj map from world4D (short-form keys)
        w4d_label_map: Dict[str, Dict[str, Any]] = {}
        w4d_person_obj: Optional[Dict[str, Any]] = None

        for obj in w4d_objects:
            lbl = obj.get("label", "")
            if lbl == "person":
                w4d_person_obj = obj
            else:
                # world4D already uses short-form labels
                if lbl not in w4d_label_map:
                    w4d_label_map[lbl] = obj

        logger.debug(
            f"[{video_id}] Frame {frame_file}: "
            f"w4d_objects={len(w4d_objects)} (person={'YES' if w4d_person_obj else 'NO'}, "
            f"objects={sorted(w4d_label_map.keys())})"
        )

        # ---- Verify object label sets ----
        rel_objs = frame_data.get("objects", [])
        # augment_relationships_test.py now stores class names in normalized
        # short form (e.g. "phone" not "phone/camera"), same as world4D.
        rel_labels = {o["class"] for o in rel_objs}
        w4d_labels = set(w4d_label_map.keys())

        logger.debug(
            f"[{video_id}] Frame {frame_file}: "
            f"rel_labels={sorted(rel_labels)}, w4d_labels={sorted(w4d_labels)}"
        )

        if rel_labels != w4d_labels:
            rel_only = rel_labels - w4d_labels
            w4d_only = w4d_labels - rel_labels
            line = (
                f"[{video_id}] Frame {frame_key}: label mismatch\n"
                f"  rel_only:  {sorted(rel_only) if rel_only else '{}'}\n"
                f"  w4d_only:  {sorted(w4d_only) if w4d_only else '{}'}"
            )
            mismatch_lines.append(line)
            logger.warning(line)

        # ---- Person info ----
        person_bbox_raw = frame_data.get("person_bbox", None)
        person_info: Dict[str, Any] = {
            # 2D from augmented_rel
            "bbox_2d": person_bbox_raw,
            "bbox_size": (0, 0),  # not available in test augmented PKL
        }
        if w4d_person_obj is not None:
            person_3d = _extract_3d_fields(w4d_person_obj)
            person_info.update(person_3d)
            n_person_matched += 1
            logger.debug(
                f"[{video_id}] Frame {frame_file}: person 3D MATCHED "
                f"(filled={person_3d['world4d_filled']}, "
                f"method={person_3d['world4d_fill_method']})"
            )
        else:
            # No person in world4D — fill with None
            person_info.update({
                "corners_world": None,
                "corners_final": None,
                "obb_final": None,
                "world4d_filled": None,
                "world4d_fill_method": None,
                "world4d_source": None,
            })
            n_person_missing += 1
            logger.warning(
                f"[{video_id}] Frame {frame_file}: person NOT FOUND in world4D — "
                f"3D fields set to None"
            )

        # ---- Object info list ----
        object_info_list: List[Dict[str, Any]] = []

        logger.debug(
            f"[{video_id}] Frame {frame_file}: processing {len(rel_objs)} "
            f"augmented_rel objects"
        )

        for obj_idx, rel_obj in enumerate(rel_objs):
            # augment_relationships_test.py now stores class in normalized
            # short form (e.g. "phone", "closet") — same space as world4D.
            cls_name = rel_obj["class"]           # already normalized short-form
            rel_source = rel_obj.get("source", "gt")
            visible = rel_source == "gt"          # GT objects are visible, corrections are not
            all_labels.add(cls_name)

            # ---- Log initial values from augmented_rel ----
            init_att = rel_obj.get("attention", [])
            init_cont = rel_obj.get("contacting", [])
            init_spa = rel_obj.get("spatial", [])
            init_bbox = rel_obj.get("bbox", None)
            logger.debug(
                f"[{video_id}] Frame {frame_file}: OBJ[{obj_idx}] INITIAL | "
                f"class={cls_name!r}, source={rel_source!r}, "
                f"bbox={'present' if init_bbox is not None else 'None'}, "
                f"attention={init_att}, contacting={init_cont}, spatial={init_spa}"
            )

            # Test augmented PKL uses "attention", "contacting", "spatial"
            # keys (list of str). Map them to the unified output keys.
            merged: Dict[str, Any] = {
                "class": _to_full(cls_name),   # expand to full AG name for consistency
                "label": cls_name,              # normalized short-form
                "bbox_2d": init_bbox,
                "visible": visible,
                "attention_relationship": init_att,
                "contacting_relationship": init_cont,
                "spatial_relationship": init_spa,
                "source": rel_source,
            }

            # Merge 3D from world4D (both use normalized short-form labels)
            w4d_match = w4d_label_map.get(cls_name)
            if w4d_match is not None:
                fields_3d = _extract_3d_fields(w4d_match)
                logger.debug(
                    f"[{video_id}] Frame {frame_file}: OBJ[{obj_idx}] 3D AUGMENT | "
                    f"label={cls_name!r}, w4d_source={fields_3d['world4d_source']!r}, "
                    f"filled={fields_3d['world4d_filled']}, "
                    f"fill_method={fields_3d['world4d_fill_method']!r}, "
                    f"corners_world={'present' if fields_3d['corners_world'] is not None else 'None'}, "
                    f"corners_final={'present' if fields_3d['corners_final'] is not None else 'None'}"
                )
                merged.update(fields_3d)
                n_obj_matched += 1
            else:
                # No world4D match — logged in verify
                merged.update({
                    "corners_world": None,
                    "corners_final": None,
                    "obb_final": None,
                    "world4d_filled": None,
                    "world4d_fill_method": None,
                    "world4d_source": None,
                })
                n_obj_missing_3d += 1
                logger.warning(
                    f"[{video_id}] Frame {frame_file}: OBJ[{obj_idx}] "
                    f"label={cls_name!r} (src={rel_source!r}) "
                    f"NO world4D match — 3D fields set to None"
                )

            # ---- Log final merged values ----
            logger.debug(
                f"[{video_id}] Frame {frame_file}: OBJ[{obj_idx}] FINAL | "
                f"class={merged['class']!r}, label={merged['label']!r}, "
                f"source={merged['source']!r}, visible={merged['visible']}, "
                f"bbox_2d={'present' if merged['bbox_2d'] is not None else 'None'}, "
                f"att={merged['attention_relationship']}, "
                f"cont={merged['contacting_relationship']}, "
                f"spa={merged['spatial_relationship']}, "
                f"w4d_filled={merged.get('world4d_filled')}, "
                f"w4d_method={merged.get('world4d_fill_method')!r}, "
                f"w4d_source={merged.get('world4d_source')!r}"
            )

            object_info_list.append(merged)

        # ---- Per-frame summary ----
        frame_labels_here = sorted({obj["label"] for obj in object_info_list})
        logger.debug(
            f"[{video_id}] Frame {frame_file}: FRAME_DONE | "
            f"n_objects={len(object_info_list)}, "
            f"labels={frame_labels_here}"
        )

        combined_frames[frame_key] = {
            "person_info": person_info,
            "object_info_list": object_info_list,
        }

        # ---- Camera pose index ----
        if frame_stem in stem_to_cam_idx:
            valid_cam_indices.append(stem_to_cam_idx[frame_stem])
            camera_frame_keys.append(frame_key)
            n_cam_matched += 1
        else:
            n_cam_missing += 1
            logger.warning(
                f"[{video_id}] Frame {frame_file}: stem '{frame_stem}' "
                f"NOT FOUND in world4D frame_stems — no camera pose for this frame"
            )

    # ------------------------------------------------------------------
    # 4) Validate: every frame must have len(object_info_list) == len(all_labels)
    # ------------------------------------------------------------------
    sorted_all_labels = sorted(all_labels)
    n_all_labels = len(sorted_all_labels)
    logger.info(
        f"[{video_id}] Validation: all_labels={sorted_all_labels} "
        f"(n={n_all_labels})"
    )

    for frame_key, frame_rec in combined_frames.items():
        n_objs_frame = len(frame_rec["object_info_list"])
        frame_obj_labels = sorted({obj["label"] for obj in frame_rec["object_info_list"]})
        if n_objs_frame != n_all_labels:
            logger.error(
                f"[{video_id}] Frame {frame_key}: VALIDATION FAILED | "
                f"len(object_info_list)={n_objs_frame} != len(all_labels)={n_all_labels} | "
                f"frame_labels={frame_obj_labels}, all_labels={sorted_all_labels} | "
                f"missing_from_frame={sorted(set(sorted_all_labels) - set(frame_obj_labels))} | "
                f"extra_in_frame={sorted(set(frame_obj_labels) - set(sorted_all_labels))}"
            )
            raise ValueError(
                f"[{video_id}] Frame {frame_key}: object_info_list length "
                f"({n_objs_frame}) does not match all_labels length "
                f"({n_all_labels}). "
                f"Missing: {sorted(set(sorted_all_labels) - set(frame_obj_labels))}, "
                f"Extra: {sorted(set(frame_obj_labels) - set(sorted_all_labels))}"
            )

    logger.info(
        f"[{video_id}] Validation PASSED: all {len(combined_frames)} frames "
        f"have {n_all_labels} objects each"
    )

    # ------------------------------------------------------------------
    # 5) Filter camera poses
    # ------------------------------------------------------------------
    camera_poses_filtered = None
    if w4d_camera_poses is not None and valid_cam_indices:
        camera_poses_filtered = np.asarray(w4d_camera_poses, dtype=np.float32)[
            valid_cam_indices
        ]
        logger.debug(
            f"[{video_id}] Camera poses filtered: "
            f"{w4d_camera_poses.shape} → {camera_poses_filtered.shape} "
            f"({n_cam_matched} matched, {n_cam_missing} missing)"
        )
    elif w4d_camera_poses is None:
        logger.warning(f"[{video_id}] No camera_poses in world4D PKL")
    else:
        logger.warning(
            f"[{video_id}] No valid camera pose indices found "
            f"(0/{len(rel_frames)} frames matched)"
        )

    # ------------------------------------------------------------------
    # 6) Build and save output
    # ------------------------------------------------------------------
    output_record = {
        "video_id": video_id,
        "num_frames": len(combined_frames),
        "all_labels": sorted(all_labels),

        "frames": combined_frames,

        # Filtered camera poses
        "camera_poses": camera_poses_filtered,
        "camera_frame_keys": camera_frame_keys,

        # Pass-through from world4D
        "floor_mesh": w4d_data.get("floor_mesh", None),
        "global_floor_sim": w4d_data.get("global_floor_sim", None),
        "world_to_final": w4d_data.get("world_to_final", None),
        "floor_final": w4d_data.get("floor_final", None),
    }

    with open(output_path, "wb") as f:
        pickle.dump(output_record, f)

    n_objs = sum(
        len(fr["object_info_list"]) for fr in combined_frames.values()
    )
    cam_shape = (
        camera_poses_filtered.shape if camera_poses_filtered is not None
        else None
    )
    logger.info(
        f"[{video_id}] Saved {len(combined_frames)} frames, "
        f"{n_objs} total objects, camera={cam_shape} → {output_path.name}"
    )
    logger.info(
        f"[{video_id}] Match stats: "
        f"person={n_person_matched}/{n_person_matched + n_person_missing}, "
        f"objects={n_obj_matched}/{n_obj_matched + n_obj_missing_3d}, "
        f"camera={n_cam_matched}/{n_cam_matched + n_cam_missing}, "
        f"labels={sorted(all_labels)}, "
        f"mismatches={len(mismatch_lines)}"
    )

    return True, mismatch_lines, output_record


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine human-corrected test relationship annotations with "
            "world4D 3D bounding box annotations into a unified per-video PKL."
        ),
    )
    parser.add_argument(
        "--ag_root_directory", type=str, default="/data/rohith/ag",
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Process a single video (e.g. '00607.mp4')",
    )
    return parser.parse_args()


def main():
    import random

    args = parse_args()
    ag_root = Path(args.ag_root_directory)

    # Test augmented PKLs from augment_relationships_test.py
    rel_dir = ag_root / "wsg_2d_augmentations"
    w4d_dir = ag_root / "world_annotations" / "bbox_annotations_4d"
    output_dir = ag_root / "world4d_rel_annotations" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Augmented rel dir:  {rel_dir}")
    logger.info(f"World4D dir:        {w4d_dir}")
    logger.info(f"Output dir:         {output_dir}")

    # ------------------------------------------------------------------
    # Discover videos
    # ------------------------------------------------------------------
    if args.video:
        vid = args.video
        video_ids = [vid]
    else:
        if not rel_dir.exists():
            logger.error(f"Augmented rel directory not found: {rel_dir}")
            sys.exit(1)
        # PKL files are named <video_id>.pkl (e.g. "00607.mp4.pkl")
        video_ids = sorted([
            p.name.replace(".pkl", "") for p in rel_dir.glob("*.pkl")
        ])

    if not video_ids:
        logger.error("No videos found to process.")
        sys.exit(1)

    random.shuffle(video_ids)
    logger.info(f"Processing {len(video_ids)} videos (overwrite={args.overwrite})")

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------
    success_count = 0
    error_count = 0
    all_mismatches: List[str] = []

    for video_id in tqdm(video_ids, desc="Combining (test)"):
        # Derive video stem for world4D PKL (world4D uses stem without .mp4)
        video_stem = video_id.replace(".mp4", "")

        rel_pkl = rel_dir / f"{video_id}.pkl"
        w4d_pkl = w4d_dir / f"{video_stem}.pkl"
        out_pkl = output_dir / f"{video_id}.pkl"

        try:
            ok, mismatches, output_record = process_video(
                video_id=video_id,
                rel_pkl_path=rel_pkl,
                w4d_pkl_path=w4d_pkl,
                output_path=out_pkl,
                overwrite=args.overwrite,
            )
            if ok:
                success_count += 1
            all_mismatches.extend(mismatches)

        except Exception as e:
            logger.error(f"[{video_id}] Error: {e}", exc_info=True)
            error_count += 1

    # ------------------------------------------------------------------
    # Write verify.txt
    # ------------------------------------------------------------------
    verify_path = output_dir / "verify.txt"
    with open(verify_path, "w", encoding="utf-8") as vf:
        if all_mismatches:
            vf.write(f"Label mismatches: {len(all_mismatches)} entries\n")
            vf.write("=" * 60 + "\n\n")
            for line in all_mismatches:
                vf.write(line + "\n\n")
        else:
            vf.write("No label mismatches found.\n")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info(
        f"Done: {success_count}/{len(video_ids)} succeeded, "
        f"{error_count} errors."
    )
    logger.info(f"Output:     {output_dir}")
    logger.info(f"Verify:     {verify_path} ({len(all_mismatches)} mismatches)")
    logger.info(f"Log:        {_LOG_DIR / 'combine_world4d_relationships_test.log'}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
