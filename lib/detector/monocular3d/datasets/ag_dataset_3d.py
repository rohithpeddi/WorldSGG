import math
import os
import pickle
import random
import struct
import zipfile
from collections import defaultdict
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

try:
    from torchvision.io import read_image, ImageReadMode
    _USE_TORCHVISION_IO = True
except ImportError:
    from PIL import Image
    _USE_TORCHVISION_IO = False

from ..constants import Constants as const


# ---------------------------------------------------------------------------
# Numpy version compatibility (pkl saved with numpy 2.x, loaded on 1.x)
# ---------------------------------------------------------------------------
class _NumpyCompatUnpickler(pickle.Unpickler):
    """Handle numpy 2.x pickles on numpy 1.x (numpy._core -> numpy.core)."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def _load_pkl_compat(path):
    """Load a pickle file with numpy version compatibility."""
    with open(path, "rb") as f:
        try:
            return pickle.load(f)
        except ModuleNotFoundError:
            f.seek(0)
            return _NumpyCompatUnpickler(f).load()


# ---------------------------------------------------------------------------
# Fast image dimension reader (reads JPEG/PNG header only, no full decode)
# ---------------------------------------------------------------------------
def _read_image_dims_fast(path: str) -> Tuple[int, int]:
    """Read (width, height) from image header without decoding pixels.
    Supports JPEG and PNG. Falls back to PIL for other formats."""
    try:
        with open(path, 'rb') as f:
            header = f.read(32)

            # PNG: signature + IHDR chunk
            if header[:8] == b'\x89PNG\r\n\x1a\n':
                w = struct.unpack('>I', header[16:20])[0]
                h = struct.unpack('>I', header[20:24])[0]
                return w, h

            # JPEG: scan for SOF marker
            if header[:2] == b'\xff\xd8':
                f.seek(0)
                data = f.read()
                idx = 2
                while idx < len(data) - 9:
                    if data[idx] != 0xFF:
                        idx += 1
                        continue
                    marker = data[idx + 1]
                    if marker in (0xC0, 0xC1, 0xC2):  # SOF0, SOF1, SOF2
                        h = struct.unpack('>H', data[idx + 5:idx + 7])[0]
                        w = struct.unpack('>H', data[idx + 7:idx + 9])[0]
                        return w, h
                    length = struct.unpack('>H', data[idx + 2:idx + 4])[0]
                    idx += 2 + length
    except Exception:
        pass

    # Fallback: PIL
    from PIL import Image as _PILImage
    with _PILImage.open(path) as img:
        return img.size  # (w, h)


# ---------------------------------------------------------------------------
# Helper: extract 8x3 corners from a frame object
# ---------------------------------------------------------------------------
def _get_corners_world(frame_object: dict) -> Optional[np.ndarray]:
    """Get 8x3 corners from a frame object."""
    # Format 1: aabb_floor_aligned["corners_world"]
    if "aabb_floor_aligned" in frame_object:
        aligned = frame_object["aabb_floor_aligned"]
        if isinstance(aligned, dict) and "corners_world" in aligned:
            arr = np.array(aligned["corners_world"], dtype=np.float32)
            if arr.shape == (8, 3):
                return arr
    # Format 2: top-level "corners_world"
    if "corners_world" in frame_object:
        arr = np.array(frame_object["corners_world"], dtype=np.float32)
        if arr.shape == (8, 3):
            return arr
    # Format 3: obb_corners_final
    if "obb_corners_final" in frame_object:
        arr = np.array(frame_object["obb_corners_final"], dtype=np.float32)
        if arr.shape == (8, 3):
            return arr
    return None


class ActionGenomeDataset3D(Dataset):
    """
    Action Genome dataset with resolution bucketing for efficient batching.
    Pre-computes target dimensions at init so frames with the same resolution
    can be batched together across different videos.
    """

    def __init__(
            self,
            data_path: str,
            phase: str = 'train',
            datasize: str = 'full',
            filter_nonperson_box_frame: bool = True,
            filter_small_box: bool = False,
            pixel_limit: int = 255000,
            target_size: Optional[int] = None,
            patch_size: int = 14,
            world_3d_annotations_path: Optional[str] = "/data/rohith/ag/world_annotations/monocular3d_bbox_annotations",
            depth_maps_dir: Optional[str] = None,
    ):
        self.data_path = data_path
        self.phase = phase
        self.datasize = datasize
        self.filter_nonperson_box_frame = filter_nonperson_box_frame
        self.filter_small_box = filter_small_box
        self.pixel_limit = pixel_limit
        self.target_size = target_size  # If set, forces square resize (legacy)
        self.patch_size = patch_size    # Must match backbone patch_size
        self.frames_path = os.path.join(self.data_path, const.FRAMES)
        self.world_3d_annotations = world_3d_annotations_path

        # Pre-computed depth maps (optional, for V2 3D head)
        self.depth_maps_dir = depth_maps_dir

        print(f"\n{'='*60}")
        print(f"  ActionGenomeDataset3D [{phase.upper()}]")
        print(f"{'='*60}")

        # Step 1: Load annotation pickle files (person/object bboxes, class labels)
        print("  [1/3] Loading annotation files...")
        self._fetch_object_classes()
        self.person_bbox, self.object_bbox = self._fetch_object_person_bboxes(filter_small_box)

        # Step 2: Build frame index + resolution buckets for efficient batching
        print("  [2/3] Building dataset index...")
        self._build_dataset()

        # Step 3: DINOv2 normalization stats (hardcoded — avoids loading AutoImageProcessor)
        self.image_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
        self.image_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

        print(f"  [3/3] ✓ Dataset ready: {len(self):,} frames  |  {len(self.object_classes)} classes")
        print(f"{'='*60}\n")

    def _fetch_object_classes(self):
        """Load the 37 Action Genome object class names from file."""
        self.object_classes = [const.BACKGROUND]
        object_classes_path = os.path.join(self.data_path, const.ANNOTATIONS, const.OBJECT_CLASSES_FILE)
        with open(object_classes_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                self.object_classes.append(line.strip('\n'))
        self.object_classes[9] = 'closet/cabinet'
        self.object_classes[11] = 'cup/glass/bottle'
        self.object_classes[23] = 'paper/notebook'
        self.object_classes[24] = 'phone/camera'
        self.object_classes[31] = 'sofa/couch'

    def _fetch_object_person_bboxes(self, filter_small_box=False):
        """Load person and object bounding box pickles."""
        annotations_path = os.path.join(self.data_path, const.ANNOTATIONS)
        with open(os.path.join(annotations_path, const.PERSON_BOUNDING_BOX_PKL), 'rb') as f:
            person_bbox = pickle.load(f)
        with open(os.path.join(annotations_path, const.OBJECT_BOUNDING_BOX_RELATIONSHIP_PKL), 'rb') as f:
            object_bbox = pickle.load(f)
        return person_bbox, object_bbox

    def _build_dataset(self):
        """Build the frame-level sample index: 2D boxes + 3D annotations + resolution buckets."""
        self.samples: Dict[str, Dict] = {}
        self.valid_nums = 0
        self.non_gt_human_nums = 0

        # Per-video intrinsics lookup
        self.video_intrinsics: Dict[str, Dict[str, float]] = {}

        # ---- Phase 1: Build 2D sample index (person + object bboxes per frame) ----
        for frame_name in tqdm(self.person_bbox.keys(), desc=f"  [{self.phase}] Indexing 2D boxes", ascii=True):
            if self.object_bbox[frame_name][0][const.METADATA][const.SET] != self.phase:
                continue

            person_boxes = self.person_bbox[frame_name][const.BOUNDING_BOX]
            if self.filter_nonperson_box_frame and len(person_boxes) == 0:
                self.non_gt_human_nums += 1
                continue

            objects = []
            for obj in self.object_bbox[frame_name]:
                if obj[const.VISIBLE] and obj[const.BOUNDING_BOX] is not None:
                    class_idx = self.object_classes.index(obj[const.CLASS])
                    bbox = obj[const.BOUNDING_BOX]
                    bbox_xyxy = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                    objects.append({'bbox': bbox_xyxy, 'class': class_idx})

            if len(objects) > 0 or len(person_boxes) > 0:
                self.samples[frame_name] = {
                    'filename': frame_name,
                    'person_boxes': person_boxes,
                    'objects': objects,
                    'person_boxes_3d': None,
                    'object_boxes_3d': []
                }
                self.valid_nums += 1

        print(f"    ✓ {self.valid_nums:,} valid frames  |  {self.non_gt_human_nums:,} removed (no person bbox)")

        # ---- 3D Annotations ----
        def _iter_3d_pkls():
            if self.world_3d_annotations.lower().endswith('.zip') and os.path.isfile(self.world_3d_annotations):
                with zipfile.ZipFile(self.world_3d_annotations, 'r') as zf:
                    for name in zf.namelist():
                        if not name.endswith('.pkl'):
                            continue
                        with zf.open(name, 'r') as f:
                            yield os.path.basename(name), _NumpyCompatUnpickler(f).load()
            elif os.path.isdir(self.world_3d_annotations):
                for video_file in os.listdir(self.world_3d_annotations):
                    if not video_file.endswith('.pkl'):
                        continue
                    path = os.path.join(self.world_3d_annotations, video_file)
                    yield video_file, _load_pkl_compat(path)
            else:
                return

        # Load 3D annotations from pickle files into samples
        n_3d_videos = 0
        n_3d_frames = 0
        n_3d_objects = 0

        for pkl_name, video_data in tqdm(_iter_3d_pkls(), desc=f"  [{self.phase}] Loading 3D annotations", ascii=True):
            video_id = os.path.splitext(pkl_name)[0]  # e.g. "001YG.pkl" -> "001YG"
            n_3d_videos += 1

            # Extract intrinsics (fx, fy, cx, cy) for this video
            intr = video_data.get("intrinsics", None)
            if intr is not None:
                self.video_intrinsics[video_id] = {
                    "fx": float(intr.get("fx", 500.0)),
                    "fy": float(intr.get("fy", 500.0)),
                    "cx": float(intr.get("cx", 320.0)),
                    "cy": float(intr.get("cy", 240.0)),
                }

            # Extract per-frame 3D object corners
            frames_final = video_data.get("frames_final", {})
            bbox_frames = frames_final.get("bbox_frames", {})

            # Debug: print frame name formats for first video to diagnose mismatches
            if n_3d_videos == 1 and bbox_frames:
                pkl_keys = list(bbox_frames.keys())[:3]
                sample_keys = [k for k in self.samples.keys() if k.startswith(video_id)][:3]
                print(f"    [3D debug] pkl frame keys (first 3): {pkl_keys}")
                print(f"    [3D debug] sample keys for {video_id} (first 3): {sample_keys}")
                # Also show what keys the pkl objects have
                first_frame = next(iter(bbox_frames.values()))
                first_objs = first_frame.get("objects", [])
                if first_objs:
                    print(f"    [3D debug] first object keys: {list(first_objs[0].keys())}")

            for frame_name_bare, frame_data in bbox_frames.items():
                # Pickle uses bare frame names ('000063.png'), samples use 'video_id.mp4/000063.png'
                frame_name = f"{video_id}.mp4/{frame_name_bare}"
                if frame_name not in self.samples:
                    continue

                objects_3d = frame_data.get("objects", [])
                obj_3d_list = []
                person_3d = None

                # OBB generator shortens compound class names; map them back
                _LABEL_REMAP = {
                    "closet": "closet/cabinet",
                    "cup": "cup/glass/bottle",
                    "paper": "paper/notebook",
                    "sofa": "sofa/couch",
                    "phone": "phone/camera",
                }

                for obj in objects_3d:
                    label = obj.get("label", None)
                    # Normalize shortened PKL labels to full dataset class names
                    label = _LABEL_REMAP.get(label, label)
                    # Extract 3D corners
                    corners = np.array(obj["obb_corners_final"], dtype=np.float32).reshape(8, 3)

                    # Person class (label="person" or class index 1)
                    if label == "person" or label == "__person__":
                        if person_3d is None:
                            person_3d = corners
                    else:
                        # Map label string to class index
                        if label in self.object_classes:
                            cls_idx = self.object_classes.index(label)
                        else:
                            continue  # Unknown class, skip
                        obj_3d_list.append({'class': cls_idx, 'bbox_3d': corners})
                        n_3d_objects += 1

                if person_3d is not None or len(obj_3d_list) > 0:
                    self.samples[frame_name]['person_boxes_3d'] = person_3d
                    self.samples[frame_name]['object_boxes_3d'] = obj_3d_list
                    n_3d_frames += 1

        print(f"    ✓ 3D annotations: {n_3d_videos} videos, {n_3d_frames} frames, {n_3d_objects} objects")

        self.frame_names: List[str] = sorted(self.samples.keys())

        # Pre-compute target dimensions per VIDEO (all frames in a video share the same resolution).
        # Read one image header per video instead of every frame (~7K reads vs ~180K).
        print("  Pre-computing per-video resolutions for bucketing...")

        # Group frame indices by video_id
        video_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, frame_name in enumerate(self.frame_names):
            video_id = frame_name.split("/")[0] if "/" in frame_name else "__unknown__"
            video_to_indices[video_id].append(idx)

        # Read one sample frame per video to get native resolution
        video_target_size: Dict[str, Tuple[int, int]] = {}
        n_read_errors = 0
        for video_id, indices in tqdm(video_to_indices.items(), desc=f"  [{self.phase}] Reading resolutions", ascii=True):
            sample_frame = self.frame_names[indices[0]]
            img_path = os.path.join(self.frames_path, self.samples[sample_frame]['filename'])
            try:
                orig_w, orig_h = _read_image_dims_fast(img_path)
            except Exception:
                orig_w, orig_h = 640, 480
                n_read_errors += 1

            if self.target_size is not None:
                tw, th = self.target_size, self.target_size
            else:
                tw, th = self._compute_target_size(orig_w, orig_h, self.pixel_limit, self.patch_size)
            video_target_size[video_id] = (tw, th)

        # Assign target sizes to all frames and build resolution buckets
        self.frame_target_sizes: List[Tuple[int, int]] = [None] * len(self.frame_names)
        self.resolution_buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for video_id, indices in video_to_indices.items():
            tw, th = video_target_size[video_id]
            for idx in indices:
                self.frame_target_sizes[idx] = (tw, th)
                self.resolution_buckets[(tw, th)].append(idx)

        n_buckets = len(self.resolution_buckets)
        sizes_str = ", ".join(f"{w}x{h}({len(idxs)})" for (w, h), idxs in
                              sorted(self.resolution_buckets.items(), key=lambda x: -len(x[1]))[:8])
        print(f"  {len(video_to_indices)} videos → {n_buckets} resolution buckets (top: {sizes_str})")
        if n_read_errors:
            print(f"  WARNING: {n_read_errors} videos failed header read, using fallback dims")

    @staticmethod
    def _compute_target_size(orig_w: int, orig_h: int, pixel_limit: int = 255000,
                             patch_size: int = 14) -> Tuple[int, int]:
        """
        Compute annotation-consistent target size.
        Aspect-ratio preserving, dimensions rounded to multiples of patch_size,
        total pixels ≤ pixel_limit.
        """
        scale = math.sqrt(pixel_limit / (orig_w * orig_h)) if orig_w * orig_h > 0 else 1
        w_target, h_target = orig_w * scale, orig_h * scale
        k = round(w_target / patch_size)
        m = round(h_target / patch_size)
        while (k * patch_size) * (m * patch_size) > pixel_limit:
            if k / m > w_target / h_target:
                k -= 1
            else:
                m -= 1
        return max(1, k) * patch_size, max(1, m) * patch_size

    def __len__(self):
        return len(self.frame_names)

    def __getitem__(self, idx: int):
        """
        Load a single frame: decode image, resize, normalize, build GT targets.

        Pipeline: disk → JPEG decode → resize → float32/255 → DINOv2 normalize
                  + scale bboxes + match 3D annotations + build intrinsics

        Returns: (img_chw, target_dict)
        """
        frame_name = self.frame_names[idx]
        sample = self.samples[frame_name]
        target_w, target_h = self.frame_target_sizes[idx]

        img_path = os.path.join(self.frames_path, sample['filename'])

        # ---- Step 1: Load image from disk (JPEG/PNG decode) ----
        if _USE_TORCHVISION_IO:
            img_tensor = read_image(img_path, mode=ImageReadMode.RGB)  # Fast torchvision decode → uint8 CHW
            _, orig_h, orig_w = img_tensor.shape
        else:
            pil_image = Image.open(img_path).convert('RGB')  # Fallback: PIL decode
            orig_w, orig_h = pil_image.size

        # ---- Step 2: Resize to pre-computed target dims (deterministic, aspect-ratio preserving) ----
        if _USE_TORCHVISION_IO:
            img_tensor = TF.resize(img_tensor, [target_h, target_w],
                                   interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            img_chw = img_tensor.float() / 255.0
        else:
            resized = pil_image.resize((target_w, target_h), Image.BILINEAR)
            img_np = np.array(resized).astype(np.float32) / 255.0
            img_chw = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()

        # ---- Step 3: DINOv2 normalization (moved here from model transform for speed) ----
        img_chw = TF.normalize(img_chw, mean=list(self.image_mean), std=list(self.image_std))

        # ---- Step 4: Scale 2D bounding boxes to match resized image dims ----
        scale_x = target_w / float(orig_w)
        scale_y = target_h / float(orig_h)

        boxes: List[List[float]] = []
        labels: List[int] = []
        boxes_3d: List[np.ndarray] = []

        # ---- Step 5: Build GT targets (person + object boxes, labels, 3D corners) ----
        # Person bounding boxes
        person_boxes = np.array(sample['person_boxes'], dtype=np.float32).reshape(-1, 4)
        person_3d = sample.get('person_boxes_3d', None)

        for i, pb in enumerate(person_boxes):
            x1, y1, x2, y2 = pb.tolist()
            x1 = np.clip(x1 * scale_x, 0, target_w - 1)
            x2 = np.clip(x2 * scale_x, 0, target_w - 1)
            y1 = np.clip(y1 * scale_y, 0, target_h - 1)
            y2 = np.clip(y2 * scale_y, 0, target_h - 1)

            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                boxes.append([x1, y1, x2, y2])
                labels.append(1)
                if i == 0 and person_3d is not None:
                    boxes_3d.append(person_3d)
                else:
                    boxes_3d.append(np.zeros((8, 3), dtype=np.float32))

        # Object bounding boxes + 3D corner matching
        available_3d_objects = list(sample.get('object_boxes_3d', []))

        for obj in sample['objects']:
            x1, y1, x2, y2 = np.array(obj['bbox'], dtype=np.float32).tolist()
            x1 = np.clip(x1 * scale_x, 0, target_w - 1)
            x2 = np.clip(x2 * scale_x, 0, target_w - 1)
            y1 = np.clip(y1 * scale_y, 0, target_h - 1)
            y2 = np.clip(y2 * scale_y, 0, target_h - 1)

            if (x2 - x1) >= 1 and (y2 - y1) >= 1:
                cls_idx = obj['class']
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_idx)

                match_3d = None
                for i, obj3d in enumerate(available_3d_objects):
                    if obj3d['class'] == cls_idx:
                        match_3d = obj3d['bbox_3d']
                        available_3d_objects.pop(i)
                        break
                boxes_3d.append(match_3d if match_3d is not None else np.zeros((8, 3), dtype=np.float32))

        # ---- Step 6: Camera intrinsics (scaled to match resized image) ----
        video_id = frame_name.split("/")[0] if "/" in frame_name else ""
        vid_intr = self.video_intrinsics.get(video_id, None)

        if vid_intr is not None:
            fx_scaled = vid_intr["fx"] * scale_x
            fy_scaled = vid_intr["fy"] * scale_y
            cx_scaled = vid_intr["cx"] * scale_x
            cy_scaled = vid_intr["cy"] * scale_y
        else:
            fx_scaled = float(max(target_w, target_h))
            fy_scaled = fx_scaled
            cx_scaled = target_w / 2.0
            cy_scaled = target_h / 2.0

        # ---- Step 7: Pack into target dict ----
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64),
            'boxes_3d': torch.tensor(np.array(boxes_3d), dtype=torch.float32) if boxes_3d else torch.empty((0, 8, 3),
                                                                                                           dtype=torch.float32),
            'focal_lengths': torch.tensor([fx_scaled, fy_scaled], dtype=torch.float32),
            'principal_point': torch.tensor([cx_scaled, cy_scaled], dtype=torch.float32),
        }

        # ---- Step 8 (optional): Load pre-computed depth map for V2 3D head ----
        if self.depth_maps_dir is not None:
            # Frame name: "001YG.mp4/000063.png" → stem = "001YG.mp4/000063"
            stem = os.path.splitext(frame_name)[0]
            # Try UniDepth (.npz with 'depth' key) first, then DepthAnything (.npy) fallback
            depth_npz = os.path.join(self.depth_maps_dir, stem + '.npz')
            depth_npy = os.path.join(self.depth_maps_dir, stem + '.npy')
            depth_orig = None
            if os.path.exists(depth_npz):
                data = np.load(depth_npz)
                depth_orig = data['depth'].astype(np.float32)  # UniDepth: metric depth
            elif os.path.exists(depth_npy):
                depth_orig = np.load(depth_npy).astype(np.float32)  # DepthAnything: relative depth

            if depth_orig is not None:
                # Resize depth map to match target image dims
                depth_tensor = torch.from_numpy(depth_orig).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                depth_resized = torch.nn.functional.interpolate(
                    depth_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
                ).squeeze(0).squeeze(0)  # (target_h, target_w)
                target['depth_map'] = depth_resized
            else:
                target['depth_map'] = torch.zeros((target_h, target_w), dtype=torch.float32)

        return img_chw, target


# ---------------------------------------------------------------------------
# Resolution-bucketed batch sampler
# ---------------------------------------------------------------------------
class ResolutionBucketBatchSampler(Sampler):
    """
    Batch sampler that groups frames by resolution, then yields batches
    of `batch_size` frames from the same resolution bucket.

    This guarantees `torch.stack` works (all images in a batch are the same size)
    while allowing frames from DIFFERENT videos in the same batch, maximizing
    GPU utilization and gradient diversity.

    Each epoch: bucket order is shuffled, frames within each bucket are shuffled,
    then chunked into batches of `batch_size`.
    """

    def __init__(self, resolution_buckets: Dict[Tuple[int, int], List[int]],
                 batch_size: int, drop_last: bool = False):
        self.resolution_buckets = resolution_buckets
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        # Build all batches first, then shuffle for inter-resolution randomness
        all_batches = []

        for res, indices in self.resolution_buckets.items():
            perm = list(indices)
            random.shuffle(perm)

            for start in range(0, len(perm), self.batch_size):
                batch = perm[start: start + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                all_batches.append(batch)

        # Shuffle batch order so we don't always train on the same resolution first
        random.shuffle(all_batches)
        yield from all_batches

    def __len__(self):
        total = 0
        for indices in self.resolution_buckets.values():
            n = len(indices)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def collate_fn(batch):
    """Custom collate: keep images as list (varying sizes) + targets as list of dicts."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets
