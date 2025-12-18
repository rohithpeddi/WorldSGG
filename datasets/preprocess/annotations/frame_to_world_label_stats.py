#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
from tqdm import tqdm

# Safe for headless runs (saving plots)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from datasets.preprocess.annotations.frame_to_world_base import FrameToWorldBase
from dataloader.standard.action_genome.ag_dataset import StandardAG


# ======================================================================================
# Missing 3D Box Stats Estimator
# ======================================================================================

class Missing3DBoxStatsEstimator(FrameToWorldBase):

    def estimate_missing_3d_boxes(
        self,
        video_id: str,
        *,
        store_frame_label_stats: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Estimate missing 3D boxes (per-frame + per-video) using:
          Level-1: GT present but 3D missing
          Level-2: GT missing; use GDINO to split into:
            - gdino_detected => "annotation missing"
            - not_detected  => "likely occlusion/out-of-view"

        Adds:
          - labelwise_frame_stats: per frame, per label reason/presence (optional)
          - labelwise_video_stats: per video, per label aggregated counts

        Returns a dict with frame_stats + totals + labelwise summaries.
        """

        # -----------------------
        # helpers
        # -----------------------
        def _norm_label(lbl: Optional[str]) -> Optional[str]:
            if lbl is None:
                return None
            lbl = str(lbl).strip().lower()

            # normalize common aliases used elsewhere in your codebase
            if lbl == "closet/cabinet":
                return "closet"
            if lbl == "cup/glass/bottle":
                return "cup"
            if lbl == "paper/notebook":
                return "paper"
            if lbl == "sofa/couch":
                return "sofa"
            if lbl == "phone/camera":
                return "phone"

            # additional practical alias
            if lbl == "couch":
                return "sofa"
            return lbl

        def _norm_frame_name(k: Any) -> str:
            if isinstance(k, int):
                return f"{k:06d}.png"
            s = str(k)
            if s.endswith(".png"):
                return s
            try:
                v = int(s)
                return f"{v:06d}.png"
            except Exception:
                return s + ".png"

        def _get_gt_labels_for_frame(video_gt_bboxes: Dict[str, Any], frame_name: str) -> List[str]:
            rec = video_gt_bboxes.get(frame_name, None)
            if rec is not None:
                return rec.get("labels", []) or []

            stem = frame_name[:-4] if frame_name.endswith(".png") else frame_name
            rec = video_gt_bboxes.get(stem, None)
            if rec is not None:
                return rec.get("labels", []) or []

            try:
                v = int(stem)
                rec = video_gt_bboxes.get(f"{v:06d}.png", None)
                if rec is not None:
                    return rec.get("labels", []) or []
            except Exception:
                pass

            return []

        def _sorted_frame_keys(keys: List[str]) -> List[str]:
            def _key_fn(s: str) -> int:
                stem = s[:-4] if s.endswith(".png") else s
                try:
                    return int(stem)
                except Exception:
                    return 10**18
            return sorted(keys, key=_key_fn)

        # normalize video id
        if not video_id.endswith(".mp4"):
            video_id = video_id + ".mp4"

        # -----------------------
        # ensure active objects loaded
        # -----------------------
        if (
            video_id not in self.video_id_active_objects_annotations_map
            or video_id not in self.video_id_active_objects_b_reasoned_map
        ):
            self.fetch_stored_active_objects_in_video(video_id)

        # -----------------------
        # load 3D annotations
        # -----------------------
        video_3dgt = self.get_video_3d_annotations(video_id)
        if video_3dgt is None:
            return {"video_id": video_id, "error": "3D bbox annotations not found"}

        frame_3dbb_map_world = video_3dgt.get("frames", None)
        if frame_3dbb_map_world is None:
            return {"video_id": video_id, "error": "3D bbox annotations has no 'frames' key"}

        # normalize frame keys
        frame_3dbb_map_world_norm: Dict[str, Dict[str, Any]] = {}
        for k, v in frame_3dbb_map_world.items():
            frame_3dbb_map_world_norm[_norm_frame_name(k)] = v
        frame_3dbb_map_world = frame_3dbb_map_world_norm

        # -----------------------
        # collect label universe from 3D
        # -----------------------
        all_labels: set = set()
        num_frames_with_objects = 0
        num_total_objects = 0

        for frame_name, frame_rec in frame_3dbb_map_world.items():
            objects = frame_rec.get("objects", []) or []
            if not objects:
                continue
            num_frames_with_objects += 1
            num_total_objects += len(objects)
            for obj in objects:
                lbl = _norm_label(obj.get("label", None))
                if lbl:
                    all_labels.add(lbl)

        # -----------------------
        # Static vs dynamic label sets (based on your active-object reasoning)
        # -----------------------
        video_active_object_labels = [
            _norm_label(x) for x in self.video_id_active_objects_annotations_map.get(video_id, [])
        ]
        video_reasoned_active_object_labels = [
            _norm_label(x) for x in self.video_id_active_objects_b_reasoned_map.get(video_id, [])
        ]
        video_active_object_labels = [x for x in video_active_object_labels if x]
        video_reasoned_active_object_labels = [x for x in video_reasoned_active_object_labels if x]

        non_moving_objects = ["floor", "sofa", "couch", "bed", "doorway", "table", "chair"]
        non_moving_objects = [_norm_label(x) for x in non_moving_objects]

        video_dynamic_object_labels = [
            obj for obj in video_reasoned_active_object_labels
            if obj not in non_moving_objects
        ]
        video_static_object_labels = [
            obj for obj in video_active_object_labels
            if obj not in video_dynamic_object_labels
        ]

        expected_all_3d = set(all_labels)
        expected_dynamic_3d = set([lbl for lbl in video_dynamic_object_labels if lbl in expected_all_3d])

        static_labels_in_3d = [lbl for lbl in video_static_object_labels if lbl in expected_all_3d]
        expected_static_3d = set(static_labels_in_3d)

        dynamic_set = set(video_dynamic_object_labels)

        # -----------------------
        # Load GT and GDINO
        # -----------------------
        video_gt_bboxes, _video_gt_raw = self.get_video_gt_annotations(video_id)
        video_gdino = self.get_video_gdino_annotations(video_id)

        # normalize gdino keys/labels
        video_gdino_norm: Dict[str, Dict[str, Any]] = {}
        for k, v in video_gdino.items():
            fn = _norm_frame_name(k)
            labels = [_norm_label(x) for x in (v.get("labels", []) or [])]
            labels = [x for x in labels if x]
            video_gdino_norm[fn] = {
                "boxes": v.get("boxes", []) or [],
                "labels": labels,
                "scores": v.get("scores", []) or [],
            }
        video_gdino = video_gdino_norm

        # -----------------------
        # per-frame stats + labelwise stats
        # -----------------------
        frame_keys = _sorted_frame_keys(list(frame_3dbb_map_world.keys()))
        frame_stats: Dict[str, Dict[str, Any]] = {}

        # labelwise video aggregation
        labelwise_video_stats: Dict[str, Dict[str, Any]] = {}
        for lbl in expected_all_3d:
            labelwise_video_stats[lbl] = {
                "label": lbl,
                "is_dynamic": (lbl in dynamic_set),

                "frames_total": 0,

                "frames_present_3d": 0,
                "frames_missing_3d": 0,

                "frames_present_gt": 0,
                "frames_present_gdino": 0,

                "missing_L1_3d_missing_gt_present": 0,
                "missing_L2_gt_missing_gdino_detected": 0,
                "missing_L2_gt_missing_gdino_not_detected": 0,
            }

        totals = {
            "frames_total": len(frame_keys),
            "frames_with_3d_objects": num_frames_with_objects,
            "total_3d_objects": num_total_objects,

            "all_3d_labels": sorted(list(expected_all_3d)),
            "static_labels_in_3d": sorted(list(expected_static_3d)),
            "dynamic_labels_in_3d": sorted(list(expected_dynamic_3d)),

            "missing_3d_total": 0,
            "missing_static_3d_total": 0,
            "missing_dynamic_3d_total": 0,

            "L1_missing_3d_but_gt_present": 0,
            "L1_missing_static_3d_but_gt_present": 0,
            "L1_missing_dynamic_3d_but_gt_present": 0,

            "L2_missing_gt_but_gdino_detected": 0,
            "L2_missing_static_gt_but_gdino_detected": 0,
            "L2_missing_dynamic_gt_but_gdino_detected": 0,

            "L2_missing_gt_and_gdino_not_detected": 0,
            "L2_missing_static_gt_and_gdino_not_detected": 0,
            "L2_missing_dynamic_gt_and_gdino_not_detected": 0,

            "missing_per_label_frames": {},  # label -> number of frames missing (any reason)
        }

        for frame_name in frame_keys:
            rec_3d = frame_3dbb_map_world.get(frame_name, {}) or {}
            objects_3d = rec_3d.get("objects", []) or []

            labels_3d = set()
            for obj in objects_3d:
                lbl = _norm_label(obj.get("label", None))
                if lbl:
                    labels_3d.add(lbl)

            gt_labels = [_norm_label(x) for x in _get_gt_labels_for_frame(video_gt_bboxes, frame_name)]
            gt_labels = [x for x in gt_labels if x]
            labels_gt = set(gt_labels)

            det = video_gdino.get(frame_name, {"labels": []})
            labels_det = set(det.get("labels", []) or [])

            missing_all = expected_all_3d - labels_3d
            missing_static = expected_static_3d - labels_3d
            missing_dynamic = expected_dynamic_3d - labels_3d

            missing_by_reason = {
                "L1_3d_missing_but_gt_present": [],
                "L2_gt_missing_but_gdino_detected": [],
                "L2_gt_and_gdino_not_detected": [],
            }

            labelwise_frame_stats = {} if store_frame_label_stats else None

            for lbl in expected_all_3d:
                labelwise_video_stats[lbl]["frames_total"] += 1

                present_3d = (lbl in labels_3d)
                present_gt = (lbl in labels_gt)
                present_det = (lbl in labels_det)

                if present_3d:
                    labelwise_video_stats[lbl]["frames_present_3d"] += 1
                else:
                    labelwise_video_stats[lbl]["frames_missing_3d"] += 1

                if present_gt:
                    labelwise_video_stats[lbl]["frames_present_gt"] += 1
                if present_det:
                    labelwise_video_stats[lbl]["frames_present_gdino"] += 1

                if not present_3d:
                    totals["missing_per_label_frames"][lbl] = totals["missing_per_label_frames"].get(lbl, 0) + 1

                    if present_gt:
                        missing_by_reason["L1_3d_missing_but_gt_present"].append(lbl)
                        labelwise_video_stats[lbl]["missing_L1_3d_missing_gt_present"] += 1
                        reason = "L1_3d_missing_but_gt_present"
                    elif present_det:
                        missing_by_reason["L2_gt_missing_but_gdino_detected"].append(lbl)
                        labelwise_video_stats[lbl]["missing_L2_gt_missing_gdino_detected"] += 1
                        reason = "L2_gt_missing_but_gdino_detected"
                    else:
                        missing_by_reason["L2_gt_and_gdino_not_detected"].append(lbl)
                        labelwise_video_stats[lbl]["missing_L2_gt_missing_gdino_not_detected"] += 1
                        reason = "L2_gt_and_gdino_not_detected"

                    if store_frame_label_stats:
                        labelwise_frame_stats[lbl] = {
                            "present_3d": False,
                            "present_gt": bool(present_gt),
                            "present_gdino": bool(present_det),
                            "reason": reason,
                        }
                else:
                    if store_frame_label_stats:
                        labelwise_frame_stats[lbl] = {
                            "present_3d": True,
                            "present_gt": bool(present_gt),
                            "present_gdino": bool(present_det),
                            "reason": "present_3d",
                        }

            totals["missing_3d_total"] += len(missing_all)
            totals["missing_static_3d_total"] += len(missing_static)
            totals["missing_dynamic_3d_total"] += len(missing_dynamic)

            l1 = missing_by_reason["L1_3d_missing_but_gt_present"]
            totals["L1_missing_3d_but_gt_present"] += len(l1)
            totals["L1_missing_static_3d_but_gt_present"] += sum(1 for x in l1 if x not in dynamic_set)
            totals["L1_missing_dynamic_3d_but_gt_present"] += sum(1 for x in l1 if x in dynamic_set)

            l2a = missing_by_reason["L2_gt_missing_but_gdino_detected"]
            totals["L2_missing_gt_but_gdino_detected"] += len(l2a)
            totals["L2_missing_static_gt_but_gdino_detected"] += sum(1 for x in l2a if x not in dynamic_set)
            totals["L2_missing_dynamic_gt_but_gdino_detected"] += sum(1 for x in l2a if x in dynamic_set)

            l2b = missing_by_reason["L2_gt_and_gdino_not_detected"]
            totals["L2_missing_gt_and_gdino_not_detected"] += len(l2b)
            totals["L2_missing_static_gt_and_gdino_not_detected"] += sum(1 for x in l2b if x not in dynamic_set)
            totals["L2_missing_dynamic_gt_and_gdino_not_detected"] += sum(1 for x in l2b if x in dynamic_set)

            frame_stats[frame_name] = {
                "num_3d_objects": len(objects_3d),
                "labels_3d": sorted(list(labels_3d)),
                "labels_gt": sorted(list(labels_gt)),
                "labels_gdino": sorted(list(labels_det)),

                "missing_3d_all": sorted(list(missing_all)),
                "missing_3d_static": sorted(list(missing_static)),
                "missing_3d_dynamic": sorted(list(missing_dynamic)),

                "missing_by_reason": {
                    "L1_3d_missing_but_gt_present": sorted(missing_by_reason["L1_3d_missing_but_gt_present"]),
                    "L2_gt_missing_but_gdino_detected": sorted(missing_by_reason["L2_gt_missing_but_gdino_detected"]),
                    "L2_gt_and_gdino_not_detected": sorted(missing_by_reason["L2_gt_and_gdino_not_detected"]),
                },
                "counts": {
                    "missing_3d_all": len(missing_all),
                    "missing_3d_static": len(missing_static),
                    "missing_3d_dynamic": len(missing_dynamic),
                    "L1_3d_missing_but_gt_present": len(l1),
                    "L2_gt_missing_but_gdino_detected": len(l2a),
                    "L2_gt_and_gdino_not_detected": len(l2b),
                },
            }

            if store_frame_label_stats:
                frame_stats[frame_name]["labelwise_frame_stats"] = labelwise_frame_stats

        totals["missing_per_label_frames"] = dict(
            sorted(totals["missing_per_label_frames"].items(), key=lambda kv: kv[1], reverse=True)
        )

        labelwise_video_stats_sorted = dict(
            sorted(
                labelwise_video_stats.items(),
                key=lambda kv: kv[1]["frames_missing_3d"],
                reverse=True,
            )
        )

        out = {
            "video_id": video_id,
            "frame_stats": frame_stats,
            "totals": totals,
            "labelwise_video_stats": labelwise_video_stats_sorted,
            "sets": {
                "expected_all_3d": sorted(list(expected_all_3d)),
                "expected_static_3d": sorted(list(expected_static_3d)),
                "expected_dynamic_3d": sorted(list(expected_dynamic_3d)),

                "video_active_object_labels": sorted(list(set(video_active_object_labels))),
                "video_reasoned_active_object_labels": sorted(list(set(video_reasoned_active_object_labels))),
                "video_dynamic_object_labels": sorted(list(set(video_dynamic_object_labels))),
                "video_static_object_labels": sorted(list(set(video_static_object_labels))),
            },
        }

        if verbose:
            print(f"\n[missing3d][{video_id}] frames={totals['frames_total']}, "
                  f"missing_total={totals['missing_3d_total']}, "
                  f"L1={totals['L1_missing_3d_but_gt_present']}, "
                  f"L2det={totals['L2_missing_gt_but_gdino_detected']}, "
                  f"L2nodet={totals['L2_missing_gt_and_gdino_not_detected']}")

        return out


# ======================================================================================
# Plot helpers (operate on compiled split blobs)
# ======================================================================================

def _split_video_records(split_blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    for vid, out in split_blob.get("videos", {}).items():
        if "totals" not in out:
            continue
        t = out["totals"]
        sets = out.get("sets", {})
        recs.append({
            "video_id": vid,
            "frames_total": t.get("frames_total", 0),

            "expected_all": len(sets.get("expected_all_3d", [])),
            "expected_static": len(sets.get("expected_static_3d", [])),
            "expected_dynamic": len(sets.get("expected_dynamic_3d", [])),

            "missing_total": t.get("missing_3d_total", 0),
            "missing_static": t.get("missing_static_3d_total", 0),
            "missing_dynamic": t.get("missing_dynamic_3d_total", 0),

            "L1": t.get("L1_missing_3d_but_gt_present", 0),
            "L2_det": t.get("L2_missing_gt_but_gdino_detected", 0),
            "L2_nodet": t.get("L2_missing_gt_and_gdino_not_detected", 0),
        })
    return recs


def _split_frame_records(split_blob: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = []
    for vid, out in split_blob.get("videos", {}).items():
        fstats = out.get("frame_stats", {})
        for frame_name, fr in fstats.items():
            c = fr.get("counts", {})
            recs.append({
                "video_id": vid,
                "frame_name": frame_name,
                "missing_total": c.get("missing_3d_all", 0),
                "missing_static": c.get("missing_3d_static", 0),
                "missing_dynamic": c.get("missing_3d_dynamic", 0),
                "L1": c.get("L1_3d_missing_but_gt_present", 0),
                "L2_det": c.get("L2_gt_missing_but_gdino_detected", 0),
                "L2_nodet": c.get("L2_gt_and_gdino_not_detected", 0),
            })
    return recs


def _aggregate_labelwise_over_split(split_blob: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
    agg: Dict[str, Dict[str, int]] = {}
    for _vid, out in split_blob.get("videos", {}).items():
        lv = out.get("labelwise_video_stats", {}) or {}
        for lbl, st in lv.items():
            if lbl not in agg:
                agg[lbl] = {
                    "frames_missing_3d": 0,
                    "missing_L1": 0,
                    "missing_L2_det": 0,
                    "missing_L2_nodet": 0,
                    "is_dynamic_votes": 0,
                    "videos_seen": 0,
                }
            agg[lbl]["frames_missing_3d"] += int(st.get("frames_missing_3d", 0))
            agg[lbl]["missing_L1"] += int(st.get("missing_L1_3d_missing_gt_present", 0))
            agg[lbl]["missing_L2_det"] += int(st.get("missing_L2_gt_missing_gdino_detected", 0))
            agg[lbl]["missing_L2_nodet"] += int(st.get("missing_L2_gt_missing_gdino_not_detected", 0))
            agg[lbl]["is_dynamic_votes"] += 1 if bool(st.get("is_dynamic", False)) else 0
            agg[lbl]["videos_seen"] += 1
    return agg


def plot_framewise_histograms(
    split_blob: Dict[str, Any],
    *,
    split_name: str,
    save_dir: Optional[str] = None,
) -> None:
    frames = _split_frame_records(split_blob)
    if not frames:
        print(f"[plot] no frame records for {split_name}")
        return

    def _hist(key: str, title: str):
        vals = [r[key] for r in frames]
        plt.figure()
        plt.hist(vals, bins=50)
        plt.title(title)
        plt.xlabel(key)
        plt.ylabel("count")
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            out = Path(save_dir) / f"{split_name}_frame_hist_{key}.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[plot] saved {out}")
        else:
            plt.close()

    _hist("missing_total", f"{split_name}: frame-wise missing_total histogram")
    _hist("missing_static", f"{split_name}: frame-wise missing_static histogram")
    _hist("missing_dynamic", f"{split_name}: frame-wise missing_dynamic histogram")
    _hist("L1", f"{split_name}: frame-wise L1 histogram")
    _hist("L2_det", f"{split_name}: frame-wise L2_det histogram")
    _hist("L2_nodet", f"{split_name}: frame-wise L2_nodet histogram")


def plot_videowise_histograms(
    split_blob: Dict[str, Any],
    *,
    split_name: str,
    topk: int = 30,
    save_dir: Optional[str] = None,
) -> None:
    vids = _split_video_records(split_blob)
    if not vids:
        print(f"[plot] no video records for {split_name}")
        return

    def _hist(key: str, title: str):
        vals = [r[key] for r in vids]
        plt.figure()
        plt.hist(vals, bins=50)
        plt.title(title)
        plt.xlabel(key)
        plt.ylabel("count")
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            out = Path(save_dir) / f"{split_name}_video_hist_{key}.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"[plot] saved {out}")
        else:
            plt.close()

    _hist("missing_total", f"{split_name}: video-wise missing_total histogram")
    _hist("missing_static", f"{split_name}: video-wise missing_static histogram")
    _hist("missing_dynamic", f"{split_name}: video-wise missing_dynamic histogram")
    _hist("expected_all", f"{split_name}: expected_all labels per video histogram")

    vids_sorted = sorted(vids, key=lambda r: r["missing_total"], reverse=True)[:topk]
    if not vids_sorted:
        return

    x = np.arange(len(vids_sorted))
    ms = np.array([r["missing_static"] for r in vids_sorted], dtype=np.int64)
    md = np.array([r["missing_dynamic"] for r in vids_sorted], dtype=np.int64)
    labels = [r["video_id"] for r in vids_sorted]

    plt.figure(figsize=(max(10, 0.35 * len(labels)), 5))
    plt.bar(x, ms)
    plt.bar(x, md, bottom=ms)
    plt.title(f"{split_name}: top-{topk} videos missing_static vs missing_dynamic (stacked)")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("missing count (sum over frames)")
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"{split_name}_video_top{topk}_missing_static_dynamic_stacked.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {out}")
    else:
        plt.close()

    xa = [r["expected_all"] for r in vids]
    ya = [r["missing_total"] for r in vids]
    plt.figure()
    plt.scatter(xa, ya, s=10)
    plt.title(f"{split_name}: expected_all vs missing_total (per video)")
    plt.xlabel("expected_all (#labels in 3D universe)")
    plt.ylabel("missing_total (sum over frames)")
    if save_dir:
        out = Path(save_dir) / f"{split_name}_video_scatter_expected_all_vs_missing_total.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {out}")
    else:
        plt.close()


def plot_video_object_group_counts(
    split_blob: Dict[str, Any],
    *,
    split_name: str,
    topk: int = 30,
    save_dir: Optional[str] = None,
) -> None:
    vids = _split_video_records(split_blob)
    if not vids:
        print(f"[plot] no video records for {split_name}")
        return

    vids_sorted = sorted(vids, key=lambda r: r["expected_all"], reverse=True)[:topk]
    if not vids_sorted:
        return

    x = np.arange(len(vids_sorted))
    e_static = np.array([r["expected_static"] for r in vids_sorted], dtype=np.int64)
    e_dyn = np.array([r["expected_dynamic"] for r in vids_sorted], dtype=np.int64)
    labels = [r["video_id"] for r in vids_sorted]

    plt.figure(figsize=(max(10, 0.35 * len(labels)), 5))
    plt.bar(x, e_static)
    plt.bar(x, e_dyn, bottom=e_static)
    plt.title(f"{split_name}: top-{topk} videos by expected_all (static vs dynamic labels)")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("#labels")
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"{split_name}_video_top{topk}_expected_static_dynamic_stacked.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {out}")
    else:
        plt.close()


def plot_labelwise_topk(
    split_blob: Dict[str, Any],
    *,
    split_name: str,
    topk: int = 20,
    metric: str = "frames_missing_3d",  # or "missing_L1", "missing_L2_det", "missing_L2_nodet"
    save_dir: Optional[str] = None,
) -> None:
    agg = _aggregate_labelwise_over_split(split_blob)
    if not agg:
        print(f"[plot] no labelwise stats for {split_name}")
        return

    items = sorted(agg.items(), key=lambda kv: kv[1].get(metric, 0), reverse=True)[:topk]
    labels = [k for k, _ in items]
    vals = [v.get(metric, 0) for _, v in items]

    x = np.arange(len(labels))
    plt.figure(figsize=(max(10, 0.4 * len(labels)), 5))
    plt.bar(x, vals)
    plt.title(f"{split_name}: top-{topk} labels by {metric}")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel(metric)
    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        out = Path(save_dir) / f"{split_name}_label_top{topk}_{metric}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[plot] saved {out}")
    else:
        plt.close()


# ======================================================================================
# Dataset loaders + compilation (FIXED: dedupe video_ids from frame-level loader)
# ======================================================================================

def load_dataset(ag_root_directory: str):
    train_dataset = StandardAG(
        phase="train",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )
    test_dataset = StandardAG(
        phase="test",
        mode="sgdet",
        datasize="large",
        data_path=ag_root_directory,
        filter_nonperson_box_frame=True,
        filter_small_box=False,
    )

    # IMPORTANT: StandardAG yields frame-level samples => many repeats per video_id.
    # We set shuffle=False for stable ordering when extracting unique video ids.
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: b[0],
        pin_memory=False,
        num_workers=0,
    )
    return train_dataset, test_dataset, dataloader_train, dataloader_test


def _normalize_video_id(vid: str) -> str:
    vid = str(vid).strip()
    if not vid.endswith(".mp4"):
        vid = vid + ".mp4"
    return vid


def _iter_video_ids(source: Union[Iterable[str], Any]) -> Iterable[str]:
    """
    Accepts:
      - iterable of video_id strings, OR
      - DataLoader-like iterable yielding dicts containing video identifiers
    """
    for item in source:
        if isinstance(item, str):
            yield _normalize_video_id(item)
            continue

        if isinstance(item, dict):
            for k in ["video_id", "video", "video_name", "vid", "videoid"]:
                if k in item:
                    yield _normalize_video_id(item[k])
                    break
            else:
                raise KeyError(f"Cannot find video id key in item dict keys={list(item.keys())}")
            continue

        vid = getattr(item, "video_id", None)
        if vid is not None:
            yield _normalize_video_id(vid)
            continue

        raise TypeError(f"Unsupported item type from source: {type(item)}")


def collect_unique_video_ids(source) -> List[str]:
    seen = set()
    uniq = []
    for vid in _iter_video_ids(source):
        if vid in seen:
            continue
        seen.add(vid)
        uniq.append(vid)
    return uniq


def compile_missing3d_stats_for_split(
    estimator: Missing3DBoxStatsEstimator,
    source: Union[Iterable[str], Any],
    *,
    split_name: str,
    store_frame_label_stats: bool = False,
    verbose_per_video: bool = False,
    skip_errors: bool = True,
) -> Dict[str, Any]:
    videos: Dict[str, Any] = {}
    errors: Dict[str, str] = {}

    # ✅ crucial: dedupe (StandardAG is frame-level)
    video_ids = collect_unique_video_ids(source)

    for video_id in tqdm(video_ids, desc=f"[compile]{split_name}", total=len(video_ids)):
        try:
            estimator.fetch_stored_active_objects_in_video(video_id)

            out = estimator.estimate_missing_3d_boxes(
                video_id,
                store_frame_label_stats=store_frame_label_stats,
                verbose=verbose_per_video,
            )
            videos[video_id] = out
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            if not skip_errors:
                raise
            errors[video_id] = msg

    return {"split": split_name, "videos": videos, "errors": errors, "num_videos": len(video_ids)}


def compile_missing3d_stats_train_test(
    estimator: Missing3DBoxStatsEstimator,
    *,
    train_source: Union[Iterable[str], Any],
    test_source: Union[Iterable[str], Any],
    store_frame_label_stats: bool = False,
    verbose_per_video: bool = False,
    skip_errors: bool = True,
) -> Dict[str, Any]:
    out_train = compile_missing3d_stats_for_split(
        estimator,
        train_source,
        split_name="train",
        store_frame_label_stats=store_frame_label_stats,
        verbose_per_video=verbose_per_video,
        skip_errors=skip_errors,
    )
    out_test = compile_missing3d_stats_for_split(
        estimator,
        test_source,
        split_name="test",
        store_frame_label_stats=store_frame_label_stats,
        verbose_per_video=verbose_per_video,
        skip_errors=skip_errors,
    )
    return {"train": out_train, "test": out_test}


def save_compiled_stats_json(compiled: Dict[str, Any], out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(compiled, f, indent=2)
    print(f"[compile] wrote: {out_path}")


# ======================================================================================
# CLI / mains
# ======================================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Missing 3D box stats (frame + video, train/test compilation).")
    p.add_argument("--ag_root_directory", type=str, default="/data/rohith/ag")
    p.add_argument(
        "--dynamic_scene_dir_path",
        type=str,
        default="/data3/rohith/ag/ag4D/dynamic_scenes/pi3_dynamic",
    )

    # run modes
    p.add_argument("--run_single_video", action="store_true", help="run stats for one video_id")
    p.add_argument("--video_id", type=str, default="00T1E.mp4", help="video id for --run_single_video")
    p.add_argument("--run_train_test", action="store_true", help="compile stats over train+test splits")
    p.add_argument("--store_frame_label_stats", action="store_true", help="store per-frame per-label stats (heavy)")
    p.add_argument("--no_plots", action="store_true", help="skip generating plots for train/test compilation")

    # output
    p.add_argument("--out_dir", type=str, default="", help="override output dir (default: bbox_4d_root_dir/missing3d_stats)")
    return p.parse_args()


def main_missing3d_stats_single_video(args) -> None:
    stats_estimator = Missing3DBoxStatsEstimator(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    video_id = args.video_id
    stats_estimator.fetch_stored_active_objects_in_video(video_id)
    stats = stats_estimator.estimate_missing_3d_boxes(
        video_id,
        store_frame_label_stats=True,
        verbose=True,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (stats_estimator.bbox_4d_root_dir / "missing3d_stats")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_normalize_video_id(video_id)[:-4]}_missing3d_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[missing3d] Saved single-video stats to: {out_path}")


def main_missing3d_stats_train_test(args) -> None:
    stats_estimator = Missing3DBoxStatsEstimator(
        ag_root_directory=args.ag_root_directory,
        dynamic_scene_dir_path=args.dynamic_scene_dir_path,
    )

    _, _, dataloader_train, dataloader_test = load_dataset(args.ag_root_directory)

    compiled = compile_missing3d_stats_train_test(
        stats_estimator,
        train_source=dataloader_train,
        test_source=dataloader_test,
        store_frame_label_stats=args.store_frame_label_stats,
        verbose_per_video=False,
        skip_errors=True,
    )

    out_dir = Path(args.out_dir) if args.out_dir else (stats_estimator.bbox_4d_root_dir / "missing3d_stats")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "missing3d_compiled_train_test.json"
    save_compiled_stats_json(compiled, str(out_path))

    if not args.no_plots:
        plot_dir = out_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        plot_framewise_histograms(compiled["train"], split_name="train", save_dir=str(plot_dir))
        plot_framewise_histograms(compiled["test"], split_name="test", save_dir=str(plot_dir))

        plot_videowise_histograms(compiled["train"], split_name="train", save_dir=str(plot_dir))
        plot_videowise_histograms(compiled["test"], split_name="test", save_dir=str(plot_dir))

        plot_video_object_group_counts(compiled["train"], split_name="train", save_dir=str(plot_dir))
        plot_video_object_group_counts(compiled["test"], split_name="test", save_dir=str(plot_dir))

        plot_labelwise_topk(compiled["train"], split_name="train", metric="frames_missing_3d", save_dir=str(plot_dir))
        plot_labelwise_topk(compiled["test"], split_name="test", metric="frames_missing_3d", save_dir=str(plot_dir))

        plot_labelwise_topk(compiled["train"], split_name="train", metric="missing_L1", save_dir=str(plot_dir))
        plot_labelwise_topk(compiled["train"], split_name="train", metric="missing_L2_det", save_dir=str(plot_dir))
        plot_labelwise_topk(compiled["train"], split_name="train", metric="missing_L2_nodet", save_dir=str(plot_dir))

        plot_labelwise_topk(compiled["test"], split_name="test", metric="missing_L1", save_dir=str(plot_dir))
        plot_labelwise_topk(compiled["test"], split_name="test", metric="missing_L2_det", save_dir=str(plot_dir))
        plot_labelwise_topk(compiled["test"], split_name="test", metric="missing_L2_nodet", save_dir=str(plot_dir))

        print(f"[missing3d] wrote plots to: {plot_dir}")

    print(f"[missing3d] wrote compiled stats: {out_path}")


def main():
    args = parse_args()

    if args.run_single_video:
        main_missing3d_stats_single_video(args)

    if args.run_train_test:
        main_missing3d_stats_train_test(args)

    if (not args.run_single_video) and (not args.run_train_test):
        print("Nothing to run. Use --run_single_video and/or --run_train_test.")


if __name__ == "__main__":
    main()
