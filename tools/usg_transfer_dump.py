"""
USG-Par frame-transfer baseline (Rung 1): dump WSGG-format predictions.
========================================================================

Runs an AG-trained USG-Par checkpoint (ChocoWu/USG + our usg_par/datasets/ag.py
adapter) per-frame over the ActionGenome4D test videos and converts its
query-level predictions into the same per-frame record format that
tools/dump_predictions.py emits, so tools/bucketed_breakdown.py and the
standard evaluator run unchanged.

Transfer protocol (the favorable-to-them variant, predcls-flavored):
  - USG-Par sees the raw RGB frames only (its native input).
  - Its predicted queries are matched to the benchmark's world slots by
    class-consistent IoU between query mask-extent boxes and the slots'
    original-space GT boxes (gt_bboxes_2d), visible slots only.
  - A (person, object) pair's 26 predicate scores come from the RPC pair
    (person_query, object_query) if it was proposed; otherwise zeros.
  - Slots unobserved at frame t can never be matched — occpair recall is 0
    BY CONSTRUCTION. That is the point of this rung: a pixels-only parser
    has no mechanism for out-of-FOV objects.

The 26-dim relation vector follows the canonical AG head order (3 att +
6 spa + 17 con) — guaranteed by the ag.py adapter's vocabulary — so it is
split positionally into the three head distributions.

Usage (server, usg conda env — needs BOTH repos on sys.path):
    python tools/usg_transfer_dump.py \
        --usg-repo /home/rxp190007/CODE/USG \
        --usg-config configs/ag.yaml --ckpt checkpoints/ag/epoch19.pt \
        --config configs/methods/predcls/w_usg_predcls_resnet50.yaml \
        --out results/bucket_dumps/usg_par_transfer_predcls__all.pkl
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FRAMES_ROOT_DEFAULT = "/data/rohith/ag/frames"


def masks_to_boxes_norm(pred_masks: torch.Tensor, thr: float = 0.0):
    """(N, H, W) mask logits -> (N, 4) [0,1]-normalized xyxy boxes (or NaN rows)."""
    n, h, w = pred_masks.shape
    boxes = torch.full((n, 4), float("nan"))
    binm = pred_masks > thr
    for i in range(n):
        ys, xs = torch.nonzero(binm[i], as_tuple=True)
        if ys.numel() == 0:
            continue
        boxes[i] = torch.tensor([
            xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h,
        ])
    return boxes


def iou_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(A,4) x (B,4) normalized xyxy -> (A,B) IoU. NaN rows give 0."""
    a = a.unsqueeze(1)  # (A,1,4)
    b = b.unsqueeze(0)  # (1,B,4)
    lt = torch.maximum(a[..., :2], b[..., :2])
    rb = torch.minimum(a[..., 2:], b[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    iou = inter / (area_a + area_b - inter + 1e-9)
    return torch.nan_to_num(iou, nan=0.0)


def match_queries_to_slots(
    q_boxes, q_cls, slot_boxes_norm, slot_cls_usg, visible, iou_floor=0.1,
):
    """Greedy per-slot best-IoU match, class-consistent first.

    Returns slot->query index dict (visible, matchable slots only).
    """
    out = {}
    for s in torch.nonzero(visible, as_tuple=True)[0].tolist():
        sb = slot_boxes_norm[s]
        if torch.isnan(sb).any() or (sb[2] - sb[0]) <= 0 or (sb[3] - sb[1]) <= 0:
            continue
        ious = iou_matrix(q_boxes, sb.unsqueeze(0)).squeeze(1)  # (Nq,)
        cls_ok = q_cls == slot_cls_usg[s]
        cand = ious * cls_ok.float()
        best = int(cand.argmax())
        if cand[best] >= iou_floor:
            out[s] = best
            continue
        best = int(ious.argmax())  # class-agnostic fallback, stricter floor
        if ious[best] >= 0.3:
            out[s] = best
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usg-repo", required=True)
    ap.add_argument("--usg-config", default="configs/ag.yaml")
    ap.add_argument("--ckpt", required=True, help="USG-Par .pt state_dict")
    ap.add_argument("--config", required=True, help="WSGG config (predcls, any method)")
    ap.add_argument("--frames-root", default=FRAMES_ROOT_DEFAULT)
    ap.add_argument("--chunk", type=int, default=8, help="frames per forward pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, args.usg_repo)
    import yaml
    from train import build_model, build_class_embeddings  # noqa: USG repo
    from usg_par.encoders.builders import build_openclip, get_tokenizer  # noqa: USG repo

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(args.usg_repo, args.usg_config)) as f:
        usg_cfg = yaml.safe_load(f)

    # --- USG-Par model + AG vocab ---
    from usg_par.datasets.ag import AGDataset
    ag_vocab = AGDataset(usg_cfg["dataset"]["ann_dir"], args.frames_root,
                         split="test", load_frames=False)
    clip_model, preprocess = build_openclip()
    tokenizer = get_tokenizer()
    model = build_model(usg_cfg, clip_model, "video", ag_vocab.num_predicates).to(device)
    state = torch.load(os.path.join(args.usg_repo, args.ckpt), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    class_emb = build_class_embeddings(model, tokenizer, ag_vocab.object_classes).to(device)

    n_att, n_spa, n_con = 3, 6, 17
    assert ag_vocab.num_predicates == n_att + n_spa + n_con

    # WSGG object id -> USG-Par class id (names differ: 'cup/glass/bottle' etc.)
    from dataloader.world_ag_dataset import OBJECT_CLASSES, LABEL_NORMALIZE_MAP

    def _usg_id(wsgg_name):
        short = LABEL_NORMALIZE_MAP.get(wsgg_name, wsgg_name)
        key = short.replace("_", "").replace(" ", "").lower()
        return ag_vocab._obj_to_id.get(key, -1)

    wsgg_to_usg = torch.tensor(
        [_usg_id(n) if n != "__background__" else -1 for n in OBJECT_CLASSES])

    # --- WSGG test set (predcls: GT slots + labels are the transfer targets) ---
    from wsgg_base import load_wsgg_config
    conf = load_wsgg_config(args.config)
    from dataloader.world_ag_dataset import WorldAG, world_collate_fn
    from torch.utils.data import DataLoader
    from wsgg_base import annot_dir_for
    ds = WorldAG(phase="test", data_path=conf.data_path, mode="predcls",
                 feature_model=getattr(conf, "feature_model", "resnet50"),
                 include_invisible=getattr(conf, "include_invisible", True),
                 max_objects=getattr(conf, "max_objects", 64),
                 annot_dir_name=annot_dir_for(conf, "test"))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                    collate_fn=world_collate_fn)
    n_videos = len(dl) if args.limit <= 0 else min(args.limit, len(dl))
    print(f"[usg-transfer] videos={n_videos} ckpt={args.ckpt}")

    records = []
    it = iter(dl)
    skipped_frames = 0
    with torch.no_grad():
        for _ in tqdm(range(n_videos), desc="USG transfer"):
            batch = next(it)
            vid = batch["video_id"]
            T = int(batch["T"])
            # Disk layout: <frames_root>/<vid>.mp4/<nnnnnn>.png; batch video_id
            # may lack the .mp4 suffix and frame names may carry a vid/ prefix.
            vid_dir = vid if vid.endswith(".mp4") else f"{vid}.mp4"
            frame_names = [os.path.basename(f) for f in batch["frame_names"]]

            # ---- per-frame USG-Par inference (chunked video forward) ----
            per_frame = {}  # t -> (cls_logits, pred_masks, sub_idx, obj_idx, rel_logits)
            t_all = list(range(T))
            for c0 in range(0, T, args.chunk):
                idxs = t_all[c0:c0 + args.chunk]
                imgs = []
                for t in idxs:
                    p = os.path.join(args.frames_root, vid_dir, frame_names[t])
                    if not os.path.isfile(p):
                        imgs.append(None)
                        continue
                    imgs.append(preprocess(Image.open(p).convert("RGB")))
                keep = [j for j, im in enumerate(imgs) if im is not None]
                if not keep:
                    skipped_frames += len(idxs)
                    continue
                frames = torch.stack([imgs[j] for j in keep]).unsqueeze(0).to(device)
                out = model({"video": frames}, {"video": class_emb})
                mo = out.per_modality["video"]
                for row, j in enumerate(keep):
                    t = idxs[j]
                    per_frame[t] = (
                        mo.cls_logits[row].float().cpu(),
                        mo.pred_masks[row].float().cpu()
                        if mo.pred_masks is not None else None,
                        mo.rpc_out.sub_idx[row].cpu(),
                        mo.rpc_out.obj_idx[row].cpu(),
                        mo.relation_logits[row].float().cpu(),
                    )

            # ---- convert each frame into a WSGG dump record ----
            for t in range(T):
                K_max = batch["pair_valid"].shape[1]
                att_dist = np.zeros((K_max, n_att), dtype=np.float32)
                spa_dist = np.zeros((K_max, n_spa), dtype=np.float32)
                con_dist = np.zeros((K_max, n_con), dtype=np.float32)

                if t in per_frame:
                    cls_logits, pred_masks, sub_idx, obj_idx, rel_logits = per_frame[t]
                    num_c = cls_logits.shape[-1] - 1
                    q_cls = cls_logits[:, :num_c].argmax(-1)
                    q_boxes = (masks_to_boxes_norm(pred_masks)
                               if pred_masks is not None
                               else torch.full((cls_logits.shape[0], 4), float("nan")))

                    # original-space GT boxes, normalized by the actual frame size
                    fp = os.path.join(args.frames_root, vid_dir, frame_names[t])
                    with Image.open(fp) as im:
                        img_w, img_h = im.size
                    slot_boxes = batch["gt_bboxes_2d"][t].clone().float()
                    slot_boxes[:, [0, 2]] /= img_w
                    slot_boxes[:, [1, 3]] /= img_h
                    empty = (batch["gt_bboxes_2d"][t].abs().sum(-1) == 0)
                    slot_boxes[empty] = float("nan")

                    slot_cls_usg = wsgg_to_usg[batch["object_classes"][t].long()]
                    visible = batch["visibility_mask"][t].bool() & batch["valid_mask"][t].bool()
                    s2q = match_queries_to_slots(q_boxes, q_cls, slot_boxes,
                                                 slot_cls_usg, visible)

                    pair_rows = {(int(s), int(o)): r for r, (s, o)
                                 in enumerate(zip(sub_idx.tolist(), obj_idx.tolist()))}
                    rel_prob = rel_logits.sigmoid()

                    for k in torch.nonzero(batch["pair_valid"][t], as_tuple=True)[0].tolist():
                        ps = int(batch["person_idx"][t][k])
                        os_ = int(batch["object_idx"][t][k])
                        if ps not in s2q or os_ not in s2q:
                            continue
                        row = pair_rows.get((s2q[ps], s2q[os_]))
                        if row is None:
                            continue
                        v = rel_prob[row].numpy()
                        a = v[:n_att]
                        att_dist[k] = a / max(a.sum(), 1e-9)  # attention head is a softmax
                        spa_dist[k] = v[n_att:n_att + n_spa]
                        con_dist[k] = v[n_att + n_spa:]

                rec = {
                    "video_id": vid,
                    "attention_distribution": att_dist,
                    "spatial_distribution": spa_dist,
                    "contacting_distribution": con_dist,
                    "gt_attention": batch["gt_attention"][t].numpy(),
                    "gt_spatial": batch["gt_spatial"][t].numpy(),
                    "gt_contacting": batch["gt_contacting"][t].numpy(),
                    "pair_valid": batch["pair_valid"][t].numpy(),
                    "person_idx": batch["person_idx"][t].numpy(),
                    "object_idx": batch["object_idx"][t].numpy(),
                    "object_classes": batch["object_classes"][t].numpy(),
                    "bboxes_2d": batch["bboxes_2d"][t].numpy(),
                    "valid_mask": batch["valid_mask"][t].numpy(),
                    "visibility_mask": batch["visibility_mask"][t].numpy(),
                    "frame_t": int(t),
                    "is_last": bool(t == T - 1),
                }
                records.append(rec)

    meta = {
        "experiment": "usg_par_transfer_predcls",
        "mode": "predcls", "method": "usg_par_transfer",
        "ckpt": args.ckpt, "frames": "all",
        "feature_model": "openclip_convnext_large (USG-Par native)",
        "n_videos": n_videos, "n_frames": len(records),
        "skipped_frames_missing_png": skipped_frames,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"meta": meta, "records": records}, f, protocol=4)
    print(f"[usg-transfer] wrote {len(records)} frames -> {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB, "
          f"{skipped_frames} frames missing PNGs)")


if __name__ == "__main__":
    main()
