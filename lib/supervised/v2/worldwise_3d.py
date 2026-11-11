"""V2: WorldWise-3D — WorldWise with an explicit, inferred geometry token for hidden and masked slots.

Differences from WorldWise (V1), everything else identical:
  * hidden slots (valid, not visible) and masked items (training only: random visible slot-frames with probability
    p_mask_visible plus spans of up to ``max_span`` frames for a fraction ``p_span_mask`` of slots) get no box input:
    the geometry encoder, camera-position encoder and motion encoder never see their corners; their geometry token is
    the learned [GEO-MASK]; their appearance is [MASK] (scaffold tokenizer) and their pairs' union features are zero;
  * after the associative retriever, g_hat = geo.infer(completed token) is injected into the slot token and decoded
    into a Z-up world box; hidden / masked slots use that decoded box (in the input frame) wherever WorldWise used
    their corners (inter-object 3-D position encoding, relation head geometry);
  * training: 3-D loss on masked items only (``geometry_losses``); unobserved objects get none.
Extra forward inputs: ``slot_classes_seq`` (T, N) class ids (PredCls annotation, SGDet detector labels: model inputs),
``gt_corners_seq`` / ``gt_box_valid_seq`` (training targets only; never passed at test).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from lib.supervised.worldwise.worldwise import WorldWise
from lib.supervised.v2.geometry import (GeometryV2, geometry_losses, masked_motion, prepare_masks, to_camera,
                                        to_world, zero_masked_union)


class WorldWise3D(WorldWise):
    def __init__(self, config, num_object_classes: int = 37, attention_class_num: int = 3,
                 spatial_class_num: int = 6, contact_class_num: int = 17):
        super().__init__(config, num_object_classes, attention_class_num, spatial_class_num, contact_class_num)
        self.geo = GeometryV2(config.d_model, config.d_struct, num_object_classes)
        self.mode = getattr(config, "mode", "predcls")
        self.p_span = float(getattr(config, "p_span_mask", 0.3))
        self.max_span = int(getattr(config, "max_span", 10))

    def forward(self, visual_features_seq, corners_seq, valid_mask_seq, visibility_mask_seq, person_idx_seq,
                object_idx_seq, pair_valid, p_mask_visible: float = 0.0, camera_pose_seq=None, union_features_seq=None,
                node_labels_seq=None, gt_contacting_seq=None, slot_classes_seq: Optional[torch.Tensor] = None,
                gt_corners_seq: Optional[torch.Tensor] = None, gt_box_valid_seq: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        T, N = corners_seq.shape[:2]
        device = corners_seq.device
        valid = valid_mask_seq.bool()
        H, M, G, valid_geo, vis_eff = prepare_masks(valid, visibility_mask_seq, self.training,
                                                    p_mask_visible if self.training else 0.0, self.p_span,
                                                    self.max_span)
        corners_in = corners_seq * valid_geo[..., None, None].to(corners_seq.dtype)

        # Step 1: geometry, never from a hidden / masked box
        struct_all, _ = self.global_structural_encoder(corners_in, valid_geo)
        struct_all = torch.where(G.unsqueeze(-1), self.geo.geo_mask.to(struct_all.dtype).expand_as(struct_all),
                                 struct_all)
        # Step 2
        cam_all = None
        ego_tokens_all = None
        if camera_pose_seq is not None:
            if self.use_object_spatial_encoder:
                _, cam_all = self.object_spatial_encoder(camera_pose_seq, corners_in, valid_geo)
                cam_all = cam_all * valid_geo.unsqueeze(-1).to(cam_all.dtype)
            if self.use_camera_temporal:
                ego_tokens_all = self.camera_temporal_encoder(camera_pose_seq)
        # Step 3
        motion_all = None
        if self.use_object_motion_encoder:
            vel, acc = masked_motion(corners_in, valid_geo)
            camera_R_all = camera_pose_seq[:, :3, :3] if camera_pose_seq is not None else None
            motion_all = self.object_motion_encoder(velocity=vel, acceleration=acc, camera_R=camera_R_all,
                                                    valid_mask=valid_geo)
        # Step 4: scaffold tokenizer; masked items are passed as not visible so they get [MASK]
        hybrid_tokens, is_masked_all, original_visual_all, _ = self.scaffold_tokenizer(
            geometry_tokens=struct_all, visual_features=visual_features_seq, visibility_mask=vis_eff,
            valid_mask=valid, p_mask_visible=0.0, cam_feats=cam_all, motion_feats=motion_all,
            ego_tokens=ego_tokens_all, hold_masked=(self.completion == "hold_last_visible"))
        artificially_masked_all = M
        # Step 5
        if self.completion == "mwae":
            completed = self.retriever(tokens=hybrid_tokens, visibility_mask=vis_eff, valid_mask=valid,
                                       cam_feats=cam_all)
        else:
            completed = hybrid_tokens
        # Step 6
        completed = completed + self.visibility_emb(vis_eff.long())
        completed = completed * valid.unsqueeze(-1).to(completed.dtype)

        # V2: explicit geometry token + decoded box
        g_hat = self.geo.infer(completed)
        completed = completed + G.unsqueeze(-1).to(completed.dtype) * self.geo.inject(g_hat).to(completed.dtype)
        completed = completed * valid.unsqueeze(-1).to(completed.dtype)
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

        # Step 7: inter-object transformer with inferred geometry for hidden / masked slots
        enriched_all = self.inter_object_encoder(tokens=completed, corners=corners_eff, valid_mask=valid)
        h_prev_all = None
        if self.use_energy_refinement:
            enriched_all, h_prev_all = self.energy_refiner(tokens=enriched_all, corners=corners_eff, valid_mask=valid)
        node_logits_all = self.node_predictor(enriched_all)
        recon_pred_all = self.reconstruction_proj(completed)
        rel_tokens, pair_valid_out = self.rel_predictor.batched_form_and_attend(
            enriched_all, node_logits_all, person_idx_seq, object_idx_seq, pair_valid, union_features_seq,
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
            "v2": {"H": H, "M": M, "G": G, "g_hat": g_hat, "corners_pred_world": corners_pred_w,
                   "corners_pred_input": corners_pred_in, "scale": scale},
        }
        if self.use_energy_refinement:
            out["enriched"] = enriched_all
            out["h_prev"] = h_prev_all
        if self.training and gt_corners_seq is not None:
            out["v2_losses"] = self._v2_losses(g_hat, corners_pred_w, centre_pred_w, gt_corners_seq,
                                               gt_box_valid_seq, M, valid, camera_pose_seq, cam_frame, scale)
        return out

    def _v2_losses(self, g_hat, corners_pred_w, centre_pred_w, gt_corners_seq, gt_box_valid_seq, M, valid,
                   camera_pose_seq, cam_frame, scale):
        gt_w = gt_corners_seq.float()                                   # annotation world frame in both modes
        ok = M & valid & (gt_w.abs().flatten(2).sum(-1) > 0)
        if gt_box_valid_seq is not None:
            ok = ok & gt_box_valid_seq.bool()
        with torch.no_grad():
            gt_in = to_camera(gt_w, camera_pose_seq.float()) if cam_frame else gt_w
            g_target, _ = self.global_structural_encoder(gt_in.to(g_hat.dtype), valid)
        latent, box = geometry_losses(g_hat, g_target, corners_pred_w, centre_pred_w, gt_w, ok, scale)
        return {"geo_latent": latent, "geo_box": box, "geo_n_items": ok.sum().detach().float()}
