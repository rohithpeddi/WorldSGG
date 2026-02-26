"""
World Scene Graph Dataset
==========================

PyTorch Dataset that loads pre-built world scene graph PKLs (produced by
``world_scene_graph_generator.py``) for downstream training.

Each PKL contains merged augmented relationships + corrected 4D bboxes
per frame, with both visible (GT/GDino) and RAG-predicted objects.

Usage::

    from dataloader.world_ag_dataset import WorldAG, world_collate_fn
    from torch.utils.data import DataLoader

    dataset = WorldAG(
        phase="train",
        data_path="/data/rohith/ag",
        world_sg_dir="/data/rohith/ag/world_annotations/world_scene_graph",
    )
    loader = DataLoader(dataset, collate_fn=world_collate_fn, ...)

    for batch in loader:
        video_id = batch["video_id"]
        for frame_ann in batch["gt_annotations"]:
            person_entry = frame_ann[0]   # {"person_bbox": ..., "frame": ...}
            for obj in frame_ann[1:]:     # objects
                label = obj["class"]
                corners = obj["corners_final"]      # (8,3) or None
                att_rel = obj["attention_relationship"]  # LongTensor
                ...
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from constants import Constants as const


# ---------------------------------------------------------------------------
# Relationship label vocabulary (must match BaseAG.fetch_relationship_classes)
# ---------------------------------------------------------------------------

def _build_relationship_classes(data_path: str):
    """Load and normalize relationship class lists, matching BaseAG logic."""
    rel_path = os.path.join(data_path, const.ANNOTATIONS, const.RELATIONSHIP_CLASSES_FILE)
    classes = []
    with open(rel_path, "r") as f:
        for line in f:
            classes.append(line.strip("\n"))

    # Apply the same normalization as BaseAG
    classes[0] = "looking_at"
    classes[1] = "not_looking_at"
    classes[5] = "in_front_of"
    classes[7] = "on_the_side_of"
    classes[10] = "covered_by"
    classes[11] = "drinking_from"
    classes[13] = "have_it_on_the_back"
    classes[15] = "leaning_on"
    classes[16] = "lying_on"
    classes[17] = "not_contacting"
    classes[18] = "other_relationship"
    classes[19] = "sitting_on"
    classes[20] = "standing_on"
    classes[25] = "writing_on"

    attention = classes[0:3]
    spatial = classes[3:9]
    contacting = classes[9:]
    return classes, attention, spatial, contacting


def _build_object_classes(data_path: str) -> List[str]:
    """Load and normalize object class list, matching BaseAG logic."""
    obj_path = os.path.join(data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE)
    classes = [const.BACKGROUND]
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            classes.append(line.strip("\n"))

    # Same patches as BaseAG
    classes[9] = "closet/cabinet"
    classes[11] = "cup/glass/bottle"
    classes[23] = "paper/notebook"
    classes[24] = "phone/camera"
    classes[31] = "sofa/couch"
    return classes


# ---------------------------------------------------------------------------
# World Scene Graph Dataset
# ---------------------------------------------------------------------------

class WorldAG(Dataset):
    """
    PyTorch Dataset that loads world scene graph PKLs.

    Each item corresponds to one video and returns the same schema as
    ``StandardAG.__getitem__``, extended with 3D geometry:

    Returns::

        {
            "video_id": str,
            "frame_names": [str, ...],
            "gt_annotations": [
                [  # per frame
                    {"person_bbox": ..., "frame": ...},   # person entry
                    {                                       # per object
                        "class": int,
                        "bbox": np.ndarray | None,
                        "visible": bool,
                        "attention_relationship": LongTensor,
                        "contacting_relationship": LongTensor,
                        "spatial_relationship": LongTensor,
                        "corners_final": np.ndarray(8,3) | None,
                        "center_3d": np.ndarray(3,) | None,
                        "obb_floor_parallel_corners": np.ndarray(8,3) | None,
                        "obb_arbitrary_corners": np.ndarray(8,3) | None,
                        "source": str,
                        "world4d_filled": bool,
                    },
                    ...
                ],
                ...
            ]
        }
    """

    def __init__(
        self,
        phase: str,
        data_path: str,
        world_sg_dir: Optional[str] = None,
        filter_nonperson_box_frame: bool = True,
        include_invisible: bool = True,
    ):
        """
        Args:
            phase: "train" or "test"
            data_path: Root directory of Action Genome dataset
            world_sg_dir: Directory with world scene graph PKLs.
                Defaults to ``<data_path>/world_annotations/world_scene_graph``.
            filter_nonperson_box_frame: If True, skip frames without a person bbox.
            include_invisible: If True, include RAG-predicted (invisible) objects.
        """
        super().__init__()

        self._phase = phase
        self._data_path = data_path
        self._include_invisible = include_invisible
        self._filter_nonperson = filter_nonperson_box_frame

        if world_sg_dir:
            self._world_sg_dir = Path(world_sg_dir)
        else:
            self._world_sg_dir = Path(data_path) / "world_annotations" / "world_scene_graph"

        # Load class vocabularies
        self.object_classes = _build_object_classes(data_path)
        (
            self.relationship_classes,
            self.attention_relationships,
            self.spatial_relationships,
            self.contacting_relationships,
        ) = _build_relationship_classes(data_path)

        # Build video list
        self.video_list: List[str] = []          # video_id per entry
        self.frame_names_list: List[List[str]] = []  # frame keys per video
        self.gt_annotations: List[Any] = []      # parsed annotations per video

        self._build_dataset()

        print(f"[WorldAG][{phase}] {len(self.video_list)} videos, "
              f"{sum(len(f) for f in self.frame_names_list)} total frames")

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _build_dataset(self):
        """Discover and load all world scene graph PKLs."""
        if not self._world_sg_dir.exists():
            raise FileNotFoundError(
                f"World scene graph directory not found: {self._world_sg_dir}"
            )

        pkl_files = sorted(self._world_sg_dir.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(
                f"No PKL files found in {self._world_sg_dir}"
            )

        skipped = 0
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)

            video_id = data.get("video_id", pkl_path.stem)
            frames_map = data.get("frames", {})
            if not frames_map:
                skipped += 1
                continue

            # Sort frames
            sorted_keys = sorted(
                frames_map.keys(),
                key=lambda k: int(Path(k).stem) if Path(k).stem.isdigit() else k,
            )

            # Parse per-frame annotations
            frame_names = []
            gt_annotations_video = []

            for frame_key in sorted_keys:
                frame_data = frames_map[frame_key]
                person_bbox = frame_data.get("person_bbox", None)

                if self._filter_nonperson and person_bbox is None:
                    continue

                # Ensure person_bbox is ndarray
                if person_bbox is not None:
                    person_bbox = np.asarray(person_bbox, dtype=np.float32)

                # Build frame annotation list (same shape as StandardAG)
                frame_ann = [
                    {
                        const.PERSON_BOUNDING_BOX: person_bbox,
                        const.FRAME: frame_key,
                    }
                ]

                for obj in frame_data.get("objects", []):
                    parsed = self._parse_object(obj)
                    if parsed is not None:
                        frame_ann.append(parsed)

                if len(frame_ann) > 1:  # has at least one object
                    frame_names.append(frame_key)
                    gt_annotations_video.append(frame_ann)

            if len(frame_names) > 1:  # at least 2 frames
                self.video_list.append(video_id)
                self.frame_names_list.append(frame_names)
                self.gt_annotations.append(gt_annotations_video)
            else:
                skipped += 1

        if skipped > 0:
            print(f"[WorldAG] Skipped {skipped} videos (no valid frames)")

    def _parse_object(self, obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse a single object dict from the world SG PKL into training format."""
        visible = obj.get("visible", True)
        source = obj.get("source", "unknown")
        rel_source = obj.get("rel_source", "unknown")

        # Skip invisible objects if not requested
        if not visible and not self._include_invisible:
            return None

        # Class index
        class_idx = obj.get("class", -1)
        if class_idx <= 0:
            return None

        # 2D bbox (may be None for RAG-predicted objects)
        bbox_2d = obj.get("bbox_2d", None)
        if bbox_2d is not None:
            bbox_2d = np.asarray(bbox_2d, dtype=np.float32)

        # Relationship tensors
        att_rel = self._rel_str_to_tensor(
            obj.get("attention_relationship", []),
            self.attention_relationships,
        )
        spa_rel = self._rel_str_to_tensor(
            obj.get("spatial_relationship", []),
            self.spatial_relationships,
        )
        con_rel = self._rel_str_to_tensor(
            obj.get("contacting_relationship", []),
            self.contacting_relationships,
        )

        # 3D geometry
        corners_final = obj.get("corners_final", None)
        if corners_final is not None:
            corners_final = np.asarray(corners_final, dtype=np.float32)
            if corners_final.shape != (8, 3):
                corners_final = None

        center_3d = obj.get("center_3d", None)
        if center_3d is not None:
            center_3d = np.asarray(center_3d, dtype=np.float32)

        obb_fp = obj.get("obb_floor_parallel_corners", None)
        if obb_fp is not None:
            obb_fp = np.asarray(obb_fp, dtype=np.float32)
            if obb_fp.shape != (8, 3):
                obb_fp = None

        obb_arb = obj.get("obb_arbitrary_corners", None)
        if obb_arb is not None:
            obb_arb = np.asarray(obb_arb, dtype=np.float32)
            if obb_arb.shape != (8, 3):
                obb_arb = None

        return {
            const.CLASS: class_idx,
            const.BOUNDING_BOX: bbox_2d,
            const.VISIBLE: visible,
            const.ATTENTION_RELATIONSHIP: att_rel,
            const.SPATIAL_RELATIONSHIP: spa_rel,
            const.CONTACTING_RELATIONSHIP: con_rel,
            "corners_final": corners_final,
            "center_3d": center_3d,
            "obb_floor_parallel_corners": obb_fp,
            "obb_arbitrary_corners": obb_arb,
            "source": source,
            "rel_source": rel_source,
            "world4d_filled": obj.get("world4d_filled", False),
        }

    def _rel_str_to_tensor(
        self,
        rel_strings: List[str],
        vocab: List[str],
    ) -> torch.Tensor:
        """Convert relationship string labels to a LongTensor of indices."""
        if not rel_strings:
            return torch.zeros(0, dtype=torch.long)

        indices = []
        for r in rel_strings:
            if r in vocab:
                indices.append(vocab.index(r))
        if not indices:
            return torch.zeros(0, dtype=torch.long)
        return torch.tensor(indices, dtype=torch.long)

    # ------------------------------------------------------------------
    # PyTorch Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.video_list)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        video_id = self.video_list[index]
        frame_names = self.frame_names_list[index]
        gt_annotations = self.gt_annotations[index]

        return {
            "video_id": video_id,
            "frame_names": frame_names,
            "gt_annotations": gt_annotations,
        }


def world_collate_fn(batch):
    """Simple collate that returns the first (and only) item in the batch."""
    return batch[0]
