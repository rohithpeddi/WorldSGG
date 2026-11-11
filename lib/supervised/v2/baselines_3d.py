"""V2: the baseline ladder (W-STTran, W-STTran++, W-DSGDetr, W-DSGDetr++, W-USG) under the V2 rules.

User, Nov 11 2026: V2 covers every V1 run, so the baselines get the same V2 treatment as WorldWise-3D
(lib/supervised/v2/worldwise_3d.py); everything else in each baseline is unchanged:
  * hidden slots (valid, not visible) and masked items (training only: visible slot-frames with probability
    ``p_mask_visible`` plus spans of up to ``max_span`` frames for a fraction ``p_span_mask`` of slots) get no box
    input: the geometry encoder, the object spatial (camera) encoder and the motion encoder never see their corners,
    their geometry token is the learned [GEO-MASK], and their pairs' union features are zero;
  * appearance keeps each baseline's own completion, the LKS buffer (nearest visible frame of the same slot); masked
    items are passed to it as not visible, so their own-frame features are never read;
  * after the last per-slot stage (tokenizer, motion fusion, temporal object encoder), g_hat = geo.infer(token) is
    injected into the slot token of hidden / masked slots and decoded into a Z-up world box; that box replaces their
    corners wherever the baseline used corners (the inter-object transformer's 3-D position encoding);
  * training: 3-D loss on masked items only (``geometry_losses``); unobserved objects get none.
The masked items are new for the baselines (V1 baselines mask nothing): without them the geometry head would have no
training signal. Their rate is WorldWise's (p_mask_visible 0.3, spans 0.3 x up to 10 frames).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from lib.supervised.baselines.lks_buffer.lks_memory import vectorized_lks_buffer
from lib.supervised.components import MotionFeatureEncoder as ObjectMotionEncoder   # name used by W-DSGDetr++
from lib.supervised.v2.geometry import (GeometryV2, geometry_losses, prepare_masks, to_camera, to_world,
                                        zero_masked_union)


class BaselineV2Mixin:
    """Shared V2 forward for the baseline ladder. Put before the baseline class in the bases."""

    def _init_v2(self, config, num_object_classes):
        self.geo = GeometryV2(config.d_model, config.d_struct, num_object_classes)
        self.mode = getattr(config, "mode", "predcls")
        self.p_span = float(getattr(config, "p_span_mask", 0.3))
        self.max_span = int(getattr(config, "max_span", 10))

    def _motion(self, corners_in, valid_geo, camera_pose_seq):
        """W-DSGDetr++ motion features (V1 formula) from the boxes the model may see only."""
        T = corners_in.shape[0]
        motion = self.object_motion_encoder(velocity=None, valid_mask=valid_geo)
        if T > 1:
            vel = ObjectMotionEncoder.compute_velocity(corners_in[1:], corners_in[:-1])
            vv = valid_geo[1:] & valid_geo[:-1]
            vel = vel * vv.unsqueeze(-1).to(vel.dtype)
            cam_R = camera_pose_seq[1:, :3, :3] if camera_pose_seq is not None else None
            m1 = self.object_motion_encoder(velocity=vel, camera_R=cam_R, valid_mask=valid_geo[1:])
            motion = torch.cat([motion[:1], m1], dim=0)
        return motion

    def forward(self, visual_features_seq, corners_seq, valid_mask_seq, visibility_mask_seq, person_idx_seq,
                object_idx_seq, pair_valid, camera_pose_seq=None, union_features_seq=None, node_labels_seq=None,
                p_mask_visible: float = 0.0, slot_classes_seq: Optional[torch.Tensor] = None,
                gt_corners_seq: Optional[torch.Tensor] = None, gt_box_valid_seq: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        T, N = corners_seq.shape[:2]
        device = corners_seq.device
        valid = valid_mask_seq.bool()
        H, M, G, valid_geo, vis_eff = prepare_masks(valid, visibility_mask_seq, self.training,
                                                    p_mask_visible if self.training else 0.0, self.p_span,
                                                    self.max_span)
        corners_in = corners_seq * valid_geo[..., None, None].to(corners_seq.dtype)

        # Step 1: LKS buffer; masked items count as not visible (their own-frame features are never read)
        buffer_all, staleness_all = vectorized_lks_buffer(raw_visual=visual_features_seq, visibility_mask=vis_eff,
                                                          valid_mask=valid)
        # Step 2: geometry, never from a hidden / masked box
        struct_all, _ = self.global_structural_encoder(corners_in, valid_geo)
        struct_all = torch.where(G.unsqueeze(-1), self.geo.geo_mask.to(struct_all.dtype).expand_as(struct_all),
                                 struct_all)
        cam_all = None
        if camera_pose_seq is not None and hasattr(self, "object_spatial_encoder"):
            _, cam_all = self.object_spatial_encoder(camera_pose=camera_pose_seq, corners=corners_in,
                                                     valid_mask=valid_geo)
            cam_all = cam_all * valid_geo.unsqueeze(-1).to(cam_all.dtype)
        # Step 3: tokenizer (+ motion fusion, + temporal object encoder where the baseline has them)
        tokens_all = self.tokenizer(geometry_tokens=struct_all, buffer_features=buffer_all, valid_mask=valid,
                                    cam_feats=cam_all, staleness=staleness_all)
        if hasattr(self, "object_motion_encoder"):
            motion = self._motion(corners_in, valid_geo, camera_pose_seq)
            tokens_all = self.motion_fusion(torch.cat([tokens_all, motion.to(tokens_all.dtype)], dim=-1))
            tokens_all = tokens_all * valid.unsqueeze(-1).to(tokens_all.dtype)
        if hasattr(self, "temporal_obj_encoder"):
            tokens_all = self.temporal_obj_encoder(tokens=tokens_all, valid_mask=valid)

        # V2: explicit geometry token + decoded box
        g_hat = self.geo.infer(tokens_all)
        tokens_all = tokens_all + G.unsqueeze(-1).to(tokens_all.dtype) * self.geo.inject(g_hat).to(tokens_all.dtype)
        tokens_all = tokens_all * valid.unsqueeze(-1).to(tokens_all.dtype)
        cam_frame = self.mode == "sgdet" and camera_pose_seq is not None
        corners_w_in = to_world(corners_in.float(), camera_pose_seq.float()) if cam_frame else corners_in.float()
        scale = self.geo.video_scale(corners_w_in, valid_geo)
        cam_c = camera_pose_seq[:, :3, 3].float() if camera_pose_seq is not None else torch.zeros(T, 3, device=device)
        classes = slot_classes_seq if slot_classes_seq is not None else (
            node_labels_seq if node_labels_seq is not None else torch.zeros(T, N, dtype=torch.long, device=device))
        corners_pred_w, centre_pred_w = self.geo.decode(g_hat, cam_c, scale, classes)
        corners_pred_in = to_camera(corners_pred_w, camera_pose_seq.float()) if cam_frame else corners_pred_w
        corners_eff = torch.where(G[..., None, None], corners_pred_in.to(corners_seq.dtype), corners_in)
        union_features_seq = zero_masked_union(union_features_seq, G, person_idx_seq, object_idx_seq)

        # Step 4: object context (inferred geometry for hidden / masked slots), nodes, relations
        if hasattr(self, "inter_object_encoder"):
            enriched_all = self.inter_object_encoder(tokens=tokens_all, corners=corners_eff, valid_mask=valid)
        else:                                                   # W-USG: object context without 3-D PE
            enriched_all = self.object_context_encoder(tokens_all, valid)
        node_logits_all = self.node_predictor(enriched_all)
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq, pair_valid, union_features_seq,
            node_class_override=node_labels_seq)
        out = {}
        if hasattr(self, "relation_decoder"):                  # W-USG
            refined = self.relation_decoder(rel_tokens, enriched_all, pair_valid_out, valid)
            edge_out = self.rel_predictor.batched_predict(refined, pair_valid_out)
            out["align_logits"] = self._text_alignment_logits(enriched_all, valid)
        else:
            enriched_rel = self.temporal_edge_attn(rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq)
            edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out)
        out.update({
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
            "v2": {"H": H, "M": M, "G": G, "g_hat": g_hat, "corners_pred_world": corners_pred_w,
                   "corners_pred_input": corners_pred_in, "scale": scale},
        })
        if self.training and gt_corners_seq is not None:
            out["v2_losses"] = self._v2_losses(g_hat, corners_pred_w, centre_pred_w, gt_corners_seq,
                                               gt_box_valid_seq, M, valid, camera_pose_seq, cam_frame, scale)
        return out

    def _v2_losses(self, g_hat, corners_pred_w, centre_pred_w, gt_corners_seq, gt_box_valid_seq, M, valid,
                   camera_pose_seq, cam_frame, scale):
        """Identical to WorldWise3D._v2_losses: masked items with an annotated box only."""
        gt_w = gt_corners_seq.float()
        ok = M & valid & (gt_w.abs().flatten(2).sum(-1) > 0)
        if gt_box_valid_seq is not None:
            ok = ok & gt_box_valid_seq.bool()
        with torch.no_grad():
            gt_in = to_camera(gt_w, camera_pose_seq.float()) if cam_frame else gt_w
            g_target, _ = self.global_structural_encoder(gt_in.to(g_hat.dtype), valid)
        latent, box = geometry_losses(g_hat, g_target, corners_pred_w, centre_pred_w, gt_w, ok, scale)
        return {"geo_latent": latent, "geo_box": box, "geo_n_items": ok.sum().detach().float()}


def _make(base_cls, name):
    class _V2(BaselineV2Mixin, base_cls):
        def __init__(self, config, num_object_classes: int = 37, attention_class_num: int = 3,
                     spatial_class_num: int = 6, contact_class_num: int = 17):
            base_cls.__init__(self, config, num_object_classes, attention_class_num, spatial_class_num,
                              contact_class_num)
            self._init_v2(config, num_object_classes)
    _V2.__name__ = _V2.__qualname__ = name
    return _V2


def baseline_v2_class(method_name: str):
    """V2 model class for ``w_sttran_3d`` | ``w_sttran_pp_3d`` | ``w_dsgdetr_3d`` | ``w_dsgdetr_pp_3d`` | ``w_usg_3d``."""
    if method_name == "w_sttran_3d":
        from lib.supervised.baselines.w_sttran.w_sttran import WSTTran as B
    elif method_name == "w_sttran_pp_3d":
        from lib.supervised.baselines.w_sttran.w_sttran_pp import WSTTranPP as B
    elif method_name == "w_dsgdetr_3d":
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr import WDSGDetr as B
    elif method_name == "w_dsgdetr_pp_3d":
        from lib.supervised.baselines.w_dsgdetr.w_dsgdetr_pp import WDSGDetrPP as B
    elif method_name == "w_usg_3d":
        from lib.supervised.baselines.w_usg.w_usg import WUSG as B
    else:
        raise ValueError(f"not a V2 baseline: {method_name}")
    return _make(B, B.__name__ + "3D")


BASELINE_V2 = {"w_sttran_3d": "w_sttran", "w_sttran_pp_3d": "w_sttran_pp", "w_dsgdetr_3d": "w_dsgdetr",
               "w_dsgdetr_pp_3d": "w_dsgdetr_pp", "w_usg_3d": "w_usg"}
