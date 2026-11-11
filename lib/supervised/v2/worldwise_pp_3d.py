"""V2: WorldWise++-3D — WorldWise++ with an explicit, inferred geometry token for hidden and masked slots.

Same V2 rules as ``worldwise_3d`` (no box input for hidden slots and masked items; [GEO-MASK]; masked encoders;
g_hat inferred and injected; decoded Z-up world box used as the slot's geometry downstream; 3-D loss on masked items
only). In WorldWise++ the completion is the entity decoder (temporal self-attention per slot, spatial attention with
the free queries, cross-attention to the DINOv3 + Pi3 grid), so g_hat is inferred from the decoder's slot output and
injected before the visibility embedding and the relation readout. The free-query detection heads and the slot
box-refinement heads are unchanged (their losses stay on visible slots, as in V1).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from lib.supervised.worldwise_pp.model import WorldWisePP
from lib.supervised.v2.geometry import (GeometryV2, geometry_losses, masked_motion, prepare_masks, to_camera,
                                        to_world)


class WorldWisePP3D(WorldWisePP):
    def __init__(self, config, num_object_classes: int = 37, attention_class_num: int = 3,
                 spatial_class_num: int = 6, contact_class_num: int = 17):
        super().__init__(config, num_object_classes, attention_class_num, spatial_class_num, contact_class_num)
        self.geo = GeometryV2(config.d_model, config.d_struct, num_object_classes)
        self.mode = getattr(config, "mode", "predcls")
        self.p_span = float(getattr(config, "p_span_mask", 0.3))
        self.max_span = int(getattr(config, "max_span", 10))

    def forward(self, visual_features_seq, corners_seq, valid_mask_seq, visibility_mask_seq, person_idx_seq,
                object_idx_seq, pair_valid, p_mask_visible: float = 0.0, camera_pose_seq=None, union_features_seq=None,
                node_labels_seq=None, gt_contacting_seq=None, grid_dino_seq=None, grid_pi3_seq=None, image_hw=None,
                bboxes_2d_seq=None, slot_classes_seq: Optional[torch.Tensor] = None,
                gt_corners_seq: Optional[torch.Tensor] = None, gt_box_valid_seq: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        if grid_dino_seq is None or grid_pi3_seq is None or image_hw is None:
            raise ValueError("WorldWisePP3D needs grid_dino_seq, grid_pi3_seq and image_hw")
        T, N = corners_seq.shape[:2]
        device = corners_seq.device
        if T > self.max_frames:
            raise ValueError(f"video has {T} frames > max_frames={self.max_frames}")
        valid = valid_mask_seq.bool()
        H, M, G, valid_geo, vis_eff = prepare_masks(valid, visibility_mask_seq, self.training,
                                                    p_mask_visible if self.training else 0.0, self.p_span,
                                                    self.max_span)
        corners_in = corners_seq * valid_geo[..., None, None].to(corners_seq.dtype)

        struct_all, _ = self.global_structural_encoder(corners_in, valid_geo)
        struct_all = torch.where(G.unsqueeze(-1), self.geo.geo_mask.to(struct_all.dtype).expand_as(struct_all),
                                 struct_all)
        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            if self.use_object_spatial_encoder:
                _, cam_all = self.object_spatial_encoder(camera_pose_seq, corners_in, valid_geo)
                cam_all = cam_all * valid_geo.unsqueeze(-1).to(cam_all.dtype)
            if self.use_camera_temporal:
                ego_tokens_all = self.camera_temporal_encoder(camera_pose_seq)
        motion_all = None
        if self.use_object_motion_encoder:
            vel, acc = masked_motion(corners_in, valid_geo)
            camera_R_all = camera_pose_seq[:, :3, :3] if camera_pose_seq is not None else None
            motion_all = self.object_motion_encoder(velocity=vel, acceleration=acc, camera_R=camera_R_all,
                                                    valid_mask=valid_geo)
        hybrid_tokens, is_masked_all, original_visual_all, _ = self.scaffold_tokenizer(
            geometry_tokens=struct_all, visual_features=visual_features_seq, visibility_mask=vis_eff,
            valid_mask=valid, p_mask_visible=0.0, cam_feats=cam_all, motion_feats=motion_all,
            ego_tokens=ego_tokens_all)
        artificially_masked_all = M

        mem = self.grid_fusion(grid_dino_seq, grid_pi3_seq).to(hybrid_tokens.dtype)
        Q = self.n_free_queries
        slot_pos = self.slot_pos(struct_all).masked_fill(~valid.unsqueeze(-1), 0.0)
        if Q > 0:
            q = torch.cat([self.free_queries[None].expand(T, -1, -1).to(hybrid_tokens.dtype), hybrid_tokens], dim=1)
            q_pos = torch.cat([self.free_pos[None].expand(T, -1, -1).to(slot_pos.dtype), slot_pos], dim=1)
            pad = torch.cat([torch.zeros(T, Q, dtype=torch.bool, device=device), ~valid], dim=1)
        else:
            q, q_pos, pad = hybrid_tokens, slot_pos, ~valid
        frame_pos = self.frame_pos(torch.arange(T, device=device)).to(q.dtype)
        attn_maps = []
        for layer in self.decoder:
            q, w = layer(q, mem, q_pos, pad, Q, valid, frame_pos)
            attn_maps.append(w)
        q = self.out_norm(q).masked_fill(pad.unsqueeze(-1), 0.0)
        free_q, slots = q[:, :Q], q[:, Q:]

        # V2: explicit geometry token + decoded box
        g_hat = self.geo.infer(slots)
        slots = slots + G.unsqueeze(-1).to(slots.dtype) * self.geo.inject(g_hat).to(slots.dtype)
        slots = slots * valid.unsqueeze(-1).to(slots.dtype)
        cam_frame = self.mode == "sgdet" and camera_pose_seq is not None
        corners_w_in = to_world(corners_in.float(), camera_pose_seq.float()) if cam_frame else corners_in.float()
        scale = self.geo.video_scale(corners_w_in, valid_geo)
        cam_c = camera_pose_seq[:, :3, 3].float() if camera_pose_seq is not None else torch.zeros(T, 3, device=device)
        classes = slot_classes_seq if slot_classes_seq is not None else (
            node_labels_seq if node_labels_seq is not None else torch.zeros(T, N, dtype=torch.long, device=device))
        corners_pred_w, centre_pred_w = self.geo.decode(g_hat, cam_c, scale, classes)
        corners_pred_in = to_camera(corners_pred_w, camera_pose_seq.float()) if cam_frame else corners_pred_w
        corners_eff = torch.where(G[..., None, None], corners_pred_in.to(corners_seq.dtype), corners_in)

        completed_tokens = slots + self.visibility_emb(vis_eff.long())
        completed_tokens = completed_tokens * valid.unsqueeze(-1).to(completed_tokens.dtype)
        node_logits_all = self.node_predictor(completed_tokens)
        recon_pred_all = self.reconstruction_proj(completed_tokens)

        K_max = person_idx_seq.shape[1] if person_idx_seq.dim() > 1 else 0
        pair_feat = None
        if K_max > 0:
            p_safe = person_idx_seq.clamp(0, N - 1)
            o_safe = object_idx_seq.clamp(0, N - 1)
            attn_feat = self._pair_attention_features(attn_maps, p_safe, o_safe).to(slots.dtype)
            dexp = slots.shape[-1]
            q_p = torch.gather(slots, 1, p_safe.unsqueeze(-1).expand(T, K_max, dexp))
            q_o = torch.gather(slots, 1, o_safe.unsqueeze(-1).expand(T, K_max, dexp))
            pair_feat = torch.cat([attn_feat, q_p * q_o], dim=-1)
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            completed_tokens, node_logits_all, person_idx_seq, object_idx_seq, pair_valid, pair_feat,
            node_class_override=node_labels_seq, corners=corners_eff)
        if self.use_temporal_edge_attn:
            enriched_rel = self.temporal_edge_attn(rel_tokens, pair_valid_out, person_idx_seq, object_idx_seq)
        else:
            enriched_rel = rel_tokens
        con_tokens = None
        if self.use_predicate_prototypes:
            if self.training and gt_contacting_seq is not None:
                self.prototype_memory.update(enriched_rel, gt_contacting_seq, pair_valid_out)
            con_tokens = self.prototype_memory(enriched_rel, pair_valid_out)
        edge_out = self.rel_predictor.batched_predict(enriched_rel, pair_valid_out, rel_tokens_contacting=con_tokens)

        Himg, Wimg = int(image_hw[0]), int(image_hw[1])
        if Q > 0:
            det_logits = self.det_cls_head(free_q)
            det_boxes = torch.sigmoid(self.det_box_head(free_q).float())
            det_xyxy = self.cxcywh_to_xyxy(det_boxes, (Himg, Wimg))
            det_corners = self._free_corners(free_q.reshape(T * Q, -1), det_xyxy.reshape(T * Q, 4), (Himg, Wimg))
            det_corners = det_corners.view(T, Q, 8, 3)
        else:
            det_logits = slots.new_zeros(T, 0, self.num_object_classes + 1)
            det_boxes = slots.new_zeros(T, 0, 4, dtype=torch.float32)
            det_xyxy = det_boxes
            det_corners = slots.new_zeros(T, 0, 8, 3, dtype=torch.float32)
        if bboxes_2d_seq is None:
            bboxes_2d_seq = corners_seq.new_zeros(T, N, 4)
        scale_px = torch.tensor([Wimg, Himg, Wimg, Himg], device=device, dtype=torch.float32)
        slot_boxes = bboxes_2d_seq.float() / scale_px + self.slot_box_head(slots).float()
        # V1 slot refinement kept as is (input corners + delta); it feeds only WorldWise++'s slot loss on visible
        # slots and is never used downstream or exported by the V2 dump (which exports the decoded boxes)
        slot_corners = corners_seq.float() + self.slot_corner_head(slots).float().view(T, N, 8, 3)

        out = {
            "node_logits": node_logits_all,
            "attention_logits": edge_out["attention_logits"],
            "attention_distribution": edge_out["attention_distribution"],
            "spatial_distribution": edge_out["spatial_distribution"],
            "contacting_distribution": edge_out["contacting_distribution"],
            "spatial_logits": edge_out["spatial_logits"],
            "contacting_logits": edge_out["contacting_logits"],
            "is_masked": is_masked_all,
            "artificially_masked": artificially_masked_all,
            "original_visual": original_visual_all,
            "reconstruction_predictions": recon_pred_all,
            "reconstruction_targets": original_visual_all,
            "det": {"logits": det_logits, "boxes": det_boxes, "boxes_xyxy": det_xyxy, "corners": det_corners,
                    "slot_boxes": slot_boxes, "slot_corners": slot_corners, "image_hw": (Himg, Wimg)},
            "v2": {"H": H, "M": M, "G": G, "g_hat": g_hat, "corners_pred_world": corners_pred_w,
                   "corners_pred_input": corners_pred_in, "scale": scale},
        }
        if self.training and gt_corners_seq is not None:
            gt_w = gt_corners_seq.float()
            ok = M & valid & (gt_w.abs().flatten(2).sum(-1) > 0)
            if gt_box_valid_seq is not None:
                ok = ok & gt_box_valid_seq.bool()
            with torch.no_grad():
                gt_in = to_camera(gt_w, camera_pose_seq.float()) if cam_frame else gt_w
                g_target, _ = self.global_structural_encoder(gt_in.to(g_hat.dtype), valid)
            latent, box = geometry_losses(g_hat, g_target, corners_pred_w, centre_pred_w, gt_w, ok, scale)
            out["v2_losses"] = {"geo_latent": latent, "geo_box": box, "geo_n_items": ok.sum().detach().float()}
        return out
