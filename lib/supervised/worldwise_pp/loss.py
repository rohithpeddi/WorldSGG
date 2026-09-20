"""
WorldWise++ loss = WorldWiseLoss (identical settings and keys)
                 + lambda_det      * free-query detection loss (Hungarian + one-to-many aux)
                 + lambda_slot_box * slot box / corner refinement loss.

Detection targets per frame = valid & visible world slots with a non-zero
``gt_bboxes_2d`` (labels from ``object_classes``, corners from ``gt_corners``).
Predicted free-query corners live in the camera frame (pinhole back-projection);
``gt_corners`` live in the FINAL (world) frame, so when ``camera_poses`` (cam->FINAL)
are available the predictions are transformed before the L1.  The slot corner
refinement is a residual on the *input* corners: FINAL frame in predcls (no
transform) and camera frame in sgdet (transformed like the eval code does).

The returned dict keeps the standard keys the trainer sums / logs (``total`` is
extended in place) and adds ``det_cls``, ``det_box``, ``det_p3d``, ``det_n_matched``,
``slot_box``, ``slot_p3d``.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from lib.supervised.worldwise.loss import WorldWiseLoss


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    def area(x):
        return (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)

    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-6)


def xyxy_to_cxcywh(b: torch.Tensor) -> torch.Tensor:
    x0, y0, x1, y1 = b.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def cam_to_final(corners: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """corners (M, 8, 3) with pose (4, 4), or corners (T, N, 8, 3) with poses (T, 4, 4)
    (cam->FINAL) -> FINAL-frame corners."""
    if pose.dim() == 2:
        R, t = pose[:3, :3], pose[:3, 3]
        return corners @ R.transpose(0, 1) + t
    R, t = pose[:, :3, :3], pose[:, :3, 3]                                  # (T,3,3), (T,3)
    return corners @ R.transpose(1, 2)[:, None] + t[:, None, None]


@torch.no_grad()
def hungarian_match(logits: torch.Tensor, boxes: torch.Tensor, corners: Optional[torch.Tensor],
                    gt_labels: torch.Tensor, gt_boxes: torch.Tensor, gt_corners: Optional[torch.Tensor],
                    w_cls: float = 1.0, w_box: float = 5.0, w_3d: float = 1.0,
                    one_to_many_iou: float = 0.0, boxes_xyxy: Optional[torch.Tensor] = None,
                    gt_boxes_xyxy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """One frame.  Returns (query_idx, gt_idx).  With ``one_to_many_iou`` > 0 every
    unmatched query whose box IoU with some GT exceeds the threshold is additionally
    assigned to that GT (Hydra-SGG-style hybrid assignment, train time only)."""
    from scipy.optimize import linear_sum_assignment
    e = torch.zeros(0, dtype=torch.long, device=logits.device)
    if gt_labels.numel() == 0 or logits.shape[0] == 0:
        return e, e
    prob = logits.float().softmax(-1)[:, gt_labels]                                 # (Q, G)
    cost = -w_cls * prob + w_box * torch.cdist(boxes.float(), gt_boxes.float(), p=1)
    if corners is not None and gt_corners is not None:
        cost = cost + w_3d * torch.cdist(corners.float().flatten(1), gt_corners.float().flatten(1), p=1) / 24.0
    cost = torch.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=-1e4)
    qi, gi = linear_sum_assignment(cost.cpu().numpy())
    qi = torch.as_tensor(qi, device=logits.device, dtype=torch.long)
    gi = torch.as_tensor(gi, device=logits.device, dtype=torch.long)
    if one_to_many_iou > 0 and boxes_xyxy is not None and gt_boxes_xyxy is not None:
        iou = box_iou_xyxy(boxes_xyxy.float(), gt_boxes_xyxy.float())
        iou[qi] = -1
        best, arg = iou.max(1)
        extra = torch.nonzero(best > one_to_many_iou).squeeze(1)
        qi = torch.cat([qi, extra])
        gi = torch.cat([gi, arg[extra]])
    return qi, gi


class WorldWisePPLoss(WorldWiseLoss):
    def __init__(self, *, num_object_classes: int, lambda_det: float = 1.0, lambda_slot_box: float = 1.0,
                 det_one_to_many_iou: float = 0.5, no_object_weight: float = 0.1, w_det_cls: float = 1.0,
                 w_det_box: float = 5.0, w_det_p3d: float = 1.0, **worldwise_kwargs):
        super().__init__(**worldwise_kwargs)
        self.num_object_classes = int(num_object_classes)
        self.lambda_det = float(lambda_det)
        self.lambda_slot_box = float(lambda_slot_box)
        self.one_to_many_iou = float(det_one_to_many_iou)
        self.w_det = dict(cls=float(w_det_cls), box=float(w_det_box), p3d=float(w_det_p3d))
        w = torch.ones(self.num_object_classes + 1)
        w[self.num_object_classes] = float(no_object_weight)
        self.register_buffer("det_cls_weight", w)

    # ------------------------------------------------------------------
    @staticmethod
    def _frame_targets(t: int, valid, visible, gt_bboxes_2d, gt_corners, labels):
        """Detection targets of frame t: valid & visible slots with a non-zero GT box."""
        m = valid[t] & visible[t] & (gt_bboxes_2d[t].abs().sum(-1) > 0)
        idx = torch.nonzero(m).squeeze(1)
        gc = gt_corners[t][idx] if gt_corners is not None else None
        return idx, labels[t][idx], gt_bboxes_2d[t][idx].float(), gc

    def _detection_loss(self, det: dict, valid, visible, gt_bboxes_2d, gt_corners, labels, camera_poses, zero):
        logits, boxes, xyxy, corners = det["logits"], det["boxes"], det["boxes_xyxy"], det["corners"]
        T, Q = logits.shape[:2]
        H, W = det["image_hw"]
        scale = torch.tensor([W, H, W, H], device=logits.device, dtype=torch.float32)
        l_cls, l_box, l_p3d = zero, zero, zero
        n_matched = 0
        n_box = 0
        n_p3d = 0
        for t in range(T):
            idx, tl, tb_xyxy, tc = self._frame_targets(t, valid, visible, gt_bboxes_2d, gt_corners, labels)
            tb = xyxy_to_cxcywh(tb_xyxy / scale)
            pc = corners[t].float()
            if camera_poses is not None:
                pc = cam_to_final(pc, camera_poses[t].float())
            has_c = (tc.abs().flatten(1).sum(1) > 0) if tc is not None else None
            qi, gi = hungarian_match(
                logits[t], boxes[t], pc if tc is not None else None, tl, tb,
                tc.float() if tc is not None else None,
                w_cls=self.w_det["cls"], w_box=self.w_det["box"], w_3d=self.w_det["p3d"],
                one_to_many_iou=self.one_to_many_iou, boxes_xyxy=xyxy[t], gt_boxes_xyxy=tb_xyxy,
            )
            tgt = torch.full((Q,), self.num_object_classes, dtype=torch.long, device=logits.device)
            tgt[qi] = tl[gi]
            l_cls = l_cls + F.cross_entropy(logits[t].float(), tgt, weight=self.det_cls_weight)
            if qi.numel():
                l_box = l_box + F.l1_loss(boxes[t][qi].float(), tb[gi], reduction="sum")
                n_box += int(qi.numel())
                if tc is not None:
                    keep = has_c[gi]
                    if keep.any():
                        l_p3d = l_p3d + F.l1_loss(pc[qi][keep], tc[gi][keep].float(), reduction="sum") / 24.0
                        n_p3d += int(keep.sum())
                n_matched += int(qi.numel())
        return {
            "det_cls": l_cls / max(T, 1),
            "det_box": l_box / max(n_box, 1),
            "det_p3d": l_p3d / max(n_p3d, 1),
            "det_n_matched": torch.tensor(float(n_matched), device=logits.device),
        }

    def _slot_loss(self, det: dict, valid, visible, gt_bboxes_2d, gt_corners, camera_poses, zero):
        sb, sc = det["slot_boxes"].float(), det["slot_corners"].float()
        H, W = det["image_hw"]
        scale = torch.tensor([W, H, W, H], device=sb.device, dtype=torch.float32)
        m = valid & visible & (gt_bboxes_2d.abs().sum(-1) > 0)
        if not m.any():
            return {"slot_box": zero, "slot_p3d": zero}
        l_box = F.l1_loss(sb[m], gt_bboxes_2d[m].float() / scale)
        l_p3d = zero
        if gt_corners is not None:
            if self.mode == "sgdet" and camera_poses is not None and camera_poses.dim() == 3:
                sc = cam_to_final(sc, camera_poses.float())
            mc = m & (gt_corners.abs().flatten(2).sum(-1) > 0)
            if mc.any():
                l_p3d = F.l1_loss(sc[mc], gt_corners[mc].float())
        return {"slot_box": l_box, "slot_p3d": l_p3d}

    # ------------------------------------------------------------------
    def forward(self, predictions, *args, gt_bboxes_2d: Optional[torch.Tensor] = None,
                gt_corners: Optional[torch.Tensor] = None, camera_poses: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        losses = super().forward(predictions, *args, **kwargs)
        det = predictions.get("det")
        if det is None or gt_bboxes_2d is None:
            return losses
        valid = kwargs.get("valid_mask")
        visible = kwargs["visibility_mask"]
        labels = kwargs.get("gt_node_labels")
        if valid is None:
            valid = torch.ones_like(visible)
        zero = losses["total"] * 0.0
        extra = zero
        if self.lambda_det > 0 and det["logits"].shape[1] > 0 and labels is not None:
            d = self._detection_loss(det, valid, visible, gt_bboxes_2d, gt_corners, labels, camera_poses, zero)
            losses.update(d)
            extra = extra + self.lambda_det * (self.w_det["cls"] * d["det_cls"] + self.w_det["box"] * d["det_box"]
                                               + self.w_det["p3d"] * d["det_p3d"])
        if self.lambda_slot_box > 0:
            s = self._slot_loss(det, valid, visible, gt_bboxes_2d, gt_corners, camera_poses, zero)
            losses.update(s)
            extra = extra + self.lambda_slot_box * (s["slot_box"] + s["slot_p3d"])
        losses["total"] = losses["total"] + extra
        return losses
