#!/usr/bin/env python3
"""
augment_annotations.py
======================
Load original GT annotations (``VideoAGLoader``) and couple them with
human-corrected annotations for missing objects (downloaded PKLs).

For each video and frame the output contains **all** objects, each tagged
with ``"source": "gt"`` or ``"source": "correction"`` so downstream
evaluation can slice metrics by source.

Output pkl format
-----------------
One pkl file per video, saved at ``<output_dir>/<video_id>.pkl``.

Each file contains a single pickled ``dict``::

    {
        "video_id":  str,
        "frames": {
            "<video_id>/<frame>.jpg": {       # same key format as VideoAGLoader
                "person_bbox": np.ndarray|None,
                "objects": [
                    {
                        "class":      str,       # normalised object name
                        "source":     str,       # "gt" or "correction"
                        "attention":  [str],     # list of relationship labels
                        "contacting": [str],
                        "spatial":    [str],
                        "bbox":       np.ndarray|None,
                    },
                    ...
                ]
            },
            ...
        }
    }

Usage
-----
::

    python -m backend.evaluation.augment_annotations \\
        --ag_root_directory /data/rohith/ag \\
        --corrections_dir ./corrections \\
        --output_dir ./augmented

    # Single video
    python -m backend.evaluation.augment_annotations \\
        --ag_root_directory /data/rohith/ag \\
        --corrections_dir ./corrections \\
        --output_dir ./augmented --video 00607.mp4
"""

import os
import sys
import glob
import pickle
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np

# Ensure project root is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))


from backend.gold.video_ag_loader import VideoAGLoader


# ---------------------------------------------------------------------------
# Logging setup — writes to <project_root>/augment_annotations.log
# ---------------------------------------------------------------------------

def _setup_logger() -> logging.Logger:
    """Create a file logger that writes detailed step-by-step output."""
    logger = logging.getLogger("augment_annotations")
    logger.setLevel(logging.DEBUG)
    # Avoid duplicate handlers on re-import
    if logger.handlers:
        return logger

    log_path = os.path.join(_PROJECT_ROOT, "augment_annotations.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Also add a concise console handler (INFO level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[augment] %(message)s"))
    logger.addHandler(ch)

    return logger


log = _setup_logger()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_label(label: str) -> str:
    """Strip and lowercase a relationship / object label."""
    return label.strip().lower().replace(" ", "_")


def _index_to_label(index: int, label_list: List[str]) -> str:
    """Convert a relationship index back to its string label."""
    if 0 <= index < len(label_list):
        return label_list[index]
    return f"unknown_{index}"


def _normalise_video_id(raw_id: str) -> str:
    """Extract a bare numeric video ID from various formats.

    Handles:
      - ``"00607.mp4"``  → ``"00607"``
      - ``"00607_mp4"``  → ``"00607"``
      - ``"00607"``      → ``"00607"``
    """
    vid = raw_id.replace(".mp4", "").replace("_mp4", "")
    vid = vid.strip("_").strip(".")
    return vid


def _extract_rel_labels(value) -> List[str]:
    """Robustly extract relationship labels from various stored formats.

    Handles:
      - ``str``               → ``["label"]``
      - ``list[str]``         → ``["label", ...]``
      - ``dict{str: float}``  → ``["label", ...]``  (from scored pipeline)
      - ``None``              → ``[]``
    """
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [_normalise_label(s)] if s else []
    if isinstance(value, dict):
        return [_normalise_label(k) for k in value.keys() if k]
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str) and v.strip():
                out.append(_normalise_label(v))
            elif isinstance(v, dict):
                lbl = v.get("label", "")
                if lbl and lbl != "unknown":
                    out.append(_normalise_label(lbl))
        return out
    return []


# ---------------------------------------------------------------------------
# Core augmentation
# ---------------------------------------------------------------------------

def augment_video(
    video_id: str,
    ag_loader: VideoAGLoader,
    corrections: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Merge GT annotations with correction annotations for one video.

    Returns ``None`` if the video is not found in the AG dataset.
    """
    gt_entry = ag_loader.video_id_annotations_map.get(video_id)
    if gt_entry is None:
        log.warning("VIDEO_NOT_FOUND | video_id=%s | not in AG dataset", video_id)
        return None

    frame_names: List[str] = gt_entry["frame_names"]
    gt_annotations: List[List[dict]] = gt_entry["gt_annotations"]

    has_corrections = corrections is not None and "frames" in corrections
    corr_video_id = corrections.get("video_id", "N/A") if corrections else "N/A"
    log.info(
        "VIDEO_START | video_id=%s | gt_frames=%d | corrections_available=%s | corrections_video_id=%s",
        video_id, len(frame_names), has_corrections, corr_video_id,
    )

    augmented_frames: Dict[str, Dict[str, Any]] = {}
    n_gt = 0
    n_corr = 0
    n_frames_with_corrections = 0

    for frame_idx, frame_key in enumerate(frame_names):
        frame_annots = gt_annotations[frame_idx]
        objects_out: List[Dict[str, Any]] = []

        # Person bbox
        person_entry = frame_annots[0] if frame_annots else {}
        person_bbox = person_entry.get("person_bbox", None)

        frame_stem = frame_key.split("/")[-1].replace(".png", "").replace(".jpg", "")

        log.debug(
            "  FRAME_START | frame_key=%s | frame_stem=%s | gt_objects=%d | person_bbox=%s",
            frame_key, frame_stem, max(0, len(frame_annots) - 1),
            "present" if person_bbox is not None else "absent",
        )

        # ---- GT objects (elements 1+) ------------------------------------
        for obj_idx, obj in enumerate(frame_annots[1:]):
            obj_class_idx = obj.get("class", -1)
            obj_class_name = (
                ag_loader.object_classes[obj_class_idx]
                if 0 <= obj_class_idx < len(ag_loader.object_classes)
                else f"unknown_{obj_class_idx}"
            )

            raw_att = obj.get("attention_relationship", [])
            raw_cont = obj.get("contacting_relationship", [])
            raw_spat = obj.get("spatial_relationship", [])

            attention = [_index_to_label(r, ag_loader.attention_relationships) for r in raw_att]
            contacting = [_index_to_label(r, ag_loader.contacting_relationships) for r in raw_cont]
            spatial = [_index_to_label(r, ag_loader.spatial_relationships) for r in raw_spat]

            normalised_class = _normalise_label(obj_class_name)

            log.debug(
                "    GT_OBJECT | idx=%d | Current: class_idx=%d, raw_att=%s, raw_cont=%s, raw_spat=%s | "
                "Changed: class_name=%s→%s, att_idx→labels=%s, cont_idx→labels=%s, spat_idx→labels=%s | "
                "Final: class=%s, source=gt, attention=%s, contacting=%s, spatial=%s, bbox=%s",
                obj_idx,
                obj_class_idx, raw_att, raw_cont, raw_spat,
                obj_class_name, normalised_class, attention, contacting, spatial,
                normalised_class, attention, contacting, spatial,
                "present" if obj.get("bbox") is not None else "None",
            )

            objects_out.append({
                "class": normalised_class,
                "source": "gt",
                "attention": attention,
                "contacting": contacting,
                "spatial": spatial,
                "bbox": obj.get("bbox", None),
            })
            n_gt += 1

        # ---- Correction objects ------------------------------------------
        n_corr_this_frame = 0
        if has_corrections:
            corr_frame = corrections["frames"].get(frame_stem, {})
            corr_preds = corr_frame.get("predictions", [])

            if corr_preds:
                log.debug(
                    "    CORRECTIONS_FOUND | frame_stem=%s | n_predictions=%d",
                    frame_stem, len(corr_preds),
                )

            for pred_idx, pred in enumerate(corr_preds):
                raw_obj = pred.get("missing_object", "")
                raw_att = pred.get("attention")
                raw_cont = pred.get("contacting")
                raw_spat = pred.get("spatial")

                missing_obj = _normalise_label(raw_obj)
                if not missing_obj:
                    log.debug(
                        "    CORRECTION_SKIPPED | pred_idx=%d | reason=empty_missing_object | raw=%r",
                        pred_idx, raw_obj,
                    )
                    continue

                att_list = _extract_rel_labels(raw_att)
                cont_list = _extract_rel_labels(raw_cont)
                spat_list = _extract_rel_labels(raw_spat)

                log.debug(
                    "    CORRECTION_OBJECT | pred_idx=%d | "
                    "Current: raw_obj=%r, raw_att=%r, raw_cont=%r, raw_spat=%r | "
                    "Changed: obj=%s, att→%s, cont→%s, spat→%s | "
                    "Final: class=%s, source=correction, attention=%s, contacting=%s, spatial=%s, bbox=None",
                    pred_idx,
                    raw_obj, raw_att, raw_cont, raw_spat,
                    missing_obj, att_list, cont_list, spat_list,
                    missing_obj, att_list, cont_list, spat_list,
                )

                objects_out.append({
                    "class": missing_obj,
                    "source": "correction",
                    "attention": att_list,
                    "contacting": cont_list,
                    "spatial": spat_list,
                    "bbox": None,
                })
                n_corr += 1
                n_corr_this_frame += 1

        if n_corr_this_frame > 0:
            n_frames_with_corrections += 1

        augmented_frames[frame_key] = {
            "person_bbox": person_bbox,
            "objects": objects_out,
        }

        n_gt_this = len(frame_annots) - 1 if frame_annots else 0
        log.debug(
            "  FRAME_DONE | frame_key=%s | Final: gt_objects=%d, correction_objects=%d, total_objects=%d",
            frame_key, n_gt_this, n_corr_this_frame, len(objects_out),
        )

    log.info(
        "VIDEO_DONE | video_id=%s | Final: total_frames=%d, gt_objects=%d, correction_objects=%d, "
        "frames_with_corrections=%d",
        video_id, len(frame_names), n_gt, n_corr, n_frames_with_corrections,
    )

    return {
        "video_id": video_id,
        "frames": augmented_frames,
        "_stats": {"n_gt": n_gt, "n_corr": n_corr},
    }


def augment_all(
    ag_root_directory: str,
    corrections_dir: str,
    output_dir: str,
    video_id: Optional[str] = None,
) -> Dict[str, bool]:
    """Augment all (or one) video(s) and save results.

    Returns ``{video_id: success_bool}``.
    """
    os.makedirs(output_dir, exist_ok=True)

    log.info("=" * 70)
    log.info("AUGMENT_START | timestamp=%s", datetime.now().isoformat())
    log.info(
        "CONFIG | ag_root_directory=%s | corrections_dir=%s | output_dir=%s | single_video=%s",
        ag_root_directory, corrections_dir, output_dir, video_id or "ALL",
    )

    # ---- Load AG dataset -------------------------------------------------
    log.info("LOADING_AG | path=%s", ag_root_directory)
    ag_loader = VideoAGLoader(data_path=ag_root_directory)
    log.info(
        "AG_LOADED | valid_videos=%d | object_classes=%d | attention_rels=%d | contacting_rels=%d | spatial_rels=%d",
        len(ag_loader.valid_video_names),
        len(ag_loader.object_classes),
        len(ag_loader.attention_relationships),
        len(ag_loader.contacting_relationships),
        len(ag_loader.spatial_relationships),
    )

    # ---- Load correction PKLs -------------------------------------------
    correction_map: Dict[str, Dict[str, Any]] = {}
    n_unique_corrections = 0
    if os.path.isdir(corrections_dir):
        pkl_paths = sorted(glob.glob(os.path.join(corrections_dir, "*.pkl")))
        log.info("LOADING_CORRECTIONS | dir=%s | pkl_files=%d", corrections_dir, len(pkl_paths))

        for pkl_path in pkl_paths:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            raw_vid = data.get("video_id", os.path.basename(pkl_path).replace(".pkl", ""))
            norm_vid = _normalise_video_id(raw_vid)
            file_stem = os.path.basename(pkl_path).replace(".pkl", "")
            norm_file = _normalise_video_id(file_stem)

            n_frames = len(data.get("frames", {}))
            n_preds = sum(
                len(fd.get("predictions", []))
                for fd in data.get("frames", {}).values()
            )

            log.debug(
                "  CORRECTION_PKL | file=%s | Current: raw_video_id=%s | "
                "Changed: norm_vid=%s, norm_file=%s | "
                "Final: indexed_keys=[%s, %s, %s] | frames=%d, predictions=%d",
                os.path.basename(pkl_path), raw_vid,
                norm_vid, norm_file,
                norm_vid, raw_vid, norm_file,
                n_frames, n_preds,
            )

            correction_map[norm_vid] = data
            correction_map[raw_vid] = data
            correction_map[file_stem] = data
            if norm_file not in correction_map:
                correction_map[norm_file] = data
            # Also index with leading zeros stripped
            stripped = norm_vid.lstrip("0") or "0"
            if stripped not in correction_map:
                correction_map[stripped] = data
            n_unique_corrections += 1
    else:
        log.warning("CORRECTIONS_DIR_MISSING | dir=%s", corrections_dir)

    log.info(
        "CORRECTIONS_LOADED | unique_files=%d | total_index_keys=%d",
        n_unique_corrections, len(correction_map),
    )

    # ---- Determine videos to process ------------------------------------
    # Only process videos that have correction PKLs (not all AG videos)
    ag_keys = set(ag_loader.video_id_annotations_map.keys())

    # Log sample IDs from both sides for diagnostics
    sample_ag = sorted(ag_keys)[:5]
    sample_corr_files = sorted(glob.glob(os.path.join(corrections_dir, "*.pkl")))[:5]
    sample_corr_stems = [os.path.basename(p).replace(".pkl", "") for p in sample_corr_files]
    log.info(
        "ID_DIAGNOSTIC | sample_ag_keys=%s | sample_correction_stems=%s",
        sample_ag, sample_corr_stems,
    )

    if video_id:
        video_ids = [_normalise_video_id(video_id)]
        if video_ids[0] not in ag_keys:
            log.error("VIDEO_NOT_IN_AG | requested=%s | normalised=%s", video_id, video_ids[0])
            return {video_ids[0]: False}
    else:
        # Build a lookup from multiple normalised forms → AG key
        ag_lookup: Dict[str, str] = {}
        for ak in ag_keys:
            ag_lookup[ak] = ak
            ag_lookup[ak.lstrip("0") or "0"] = ak               # stripped leading zeros
            ag_lookup[_normalise_video_id(ak)] = ak

        # Collect unique correction IDs and match against AG
        correction_vid_ids: Dict[str, str] = {}  # norm_id → raw_file_stem
        for pkl_path in sorted(glob.glob(os.path.join(corrections_dir, "*.pkl"))):
            file_stem = os.path.basename(pkl_path).replace(".pkl", "")
            norm = _normalise_video_id(file_stem)
            correction_vid_ids[norm] = file_stem

        video_ids = []
        n_not_in_ag = 0
        for norm_id, raw_stem in sorted(correction_vid_ids.items()):
            # Try multiple matching strategies
            ag_key = (
                ag_lookup.get(norm_id)
                or ag_lookup.get(raw_stem)
                or ag_lookup.get(norm_id.lstrip("0") or "0")
            )
            if ag_key:
                video_ids.append(ag_key)
            else:
                log.debug("CORRECTION_NOT_IN_AG | raw_stem=%s | norm=%s | skipped", raw_stem, norm_id)
                n_not_in_ag += 1

        if n_not_in_ag:
            log.warning(
                "CORRECTIONS_WITHOUT_AG | %d correction videos not found in AG dataset",
                n_not_in_ag,
            )

        if not video_ids:
            log.warning("NO_VIDEOS | no correction videos matched the AG dataset")
            return {}

    log.info(
        "PROCESSING | n_videos=%d (from %d correction PKLs)",
        len(video_ids), n_unique_corrections,
    )

    # ---- Process each video ----------------------------------------------
    results: Dict[str, bool] = {}
    total_gt = 0
    total_corr = 0
    n_with_corrections = 0
    n_unmatched = 0

    for i, vid in enumerate(video_ids):
        norm_vid = _normalise_video_id(vid)
        corrections = correction_map.get(norm_vid) or correction_map.get(vid)

        if corrections is None:
            log.debug(
                "CORRECTION_LOOKUP | video=%s | norm=%s | result=NOT_FOUND",
                vid, norm_vid,
            )
            n_unmatched += 1
        else:
            corr_vid = corrections.get("video_id", "?")
            log.debug(
                "CORRECTION_LOOKUP | video=%s | norm=%s | result=FOUND | corrections_video_id=%s",
                vid, norm_vid, corr_vid,
            )

        try:
            record = augment_video(vid, ag_loader, corrections)
            if record is None:
                results[vid] = False
                continue

            stats = record.pop("_stats", {})
            n_gt = stats.get("n_gt", 0)
            n_corr = stats.get("n_corr", 0)
            total_gt += n_gt
            total_corr += n_corr
            if n_corr > 0:
                n_with_corrections += 1

            save_path = os.path.join(output_dir, f"{vid}.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(record, f)

            log.debug("SAVED | video=%s | path=%s | gt=%d, corr=%d", vid, save_path, n_gt, n_corr)

            if (i + 1) % 100 == 0 or i == 0 or video_id:
                log.info(
                    "PROGRESS | [%d/%d] %s: %d GT + %d correction objects",
                    i + 1, len(video_ids), vid, n_gt, n_corr,
                )
            results[vid] = True
        except Exception as e:
            log.error("AUGMENT_ERROR | video=%s | error=%s", vid, e, exc_info=True)
            results[vid] = False

    # ---- Final summary ---------------------------------------------------
    ok = sum(v for v in results.values())
    fail = len(results) - ok

    log.info("=" * 70)
    log.info("AUGMENT_COMPLETE | timestamp=%s", datetime.now().isoformat())
    log.info("SUMMARY | videos_ok=%d | videos_failed=%d", ok, fail)
    log.info("SUMMARY | gt_objects=%d | correction_objects=%d | total=%d",
             total_gt, total_corr, total_gt + total_corr)
    log.info("SUMMARY | videos_with_corrections=%d | videos_without_corrections=%d",
             n_with_corrections, n_unmatched)
    log.info("=" * 70)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Augment AG GT annotations with human corrections for missing objects",
    )
    parser.add_argument(
        "--ag_root_directory", type=str,
        default="/data/rohith/ag",
        help="Root directory of the Action Genome dataset",
    )
    parser.add_argument(
        "--corrections_dir", type=str, default="/data/rohith/ag/wsg_corrections/",
        help="Directory containing downloaded correction PKL files",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/data/rohith/ag/wsg_2d_augmentations/",
        help="Directory to save augmented annotation PKL files",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Augment a single video (e.g. '00607.mp4'). Omit to augment all.",
    )
    args = parser.parse_args()

    augment_all(
        ag_root_directory=args.ag_root_directory,
        corrections_dir=args.corrections_dir,
        output_dir=args.output_dir,
        video_id=args.video,
    )


if __name__ == "__main__":
    main()
