"""V2 trainers: WorldWise-3D and WorldWise++-3D (``method_name: worldwise_3d`` / ``worldwise_pp_3d``).

Registered from ``train_wsgg_methods.py`` through ``build_v2_methods`` (base classes passed in, so there is no
circular import). Training and validation follow V1 exactly; the forward additionally receives the slot classes
(model inputs) and, in training only, ``gt_corners`` / ``gt_box_valid`` as targets of the masked-item 3-D loss. The
loss is the V1 loss plus ``lambda_geo`` x (geo_latent + geo_box).
"""
import torch
import torch.nn as nn


class LossV2(nn.Module):
    def __init__(self, base: nn.Module, lambda_geo: float = 1.0):
        super().__init__()
        self.base = base
        self.lambda_geo = float(lambda_geo)

    def forward(self, predictions, *args, **kwargs):
        losses = self.base(predictions, *args, **kwargs)
        v2 = predictions.get("v2_losses")
        if v2:
            extra = losses["total"] * 0.0
            for k in ("geo_latent", "geo_box"):
                if k in v2:
                    losses[k] = self.lambda_geo * v2[k]
                    extra = extra + losses[k]
            losses["geo_n_items"] = v2.get("geo_n_items", extra.detach() * 0.0)
            losses["total"] = losses["total"] + extra
        return losses


def build_v2_methods(TrainWorldWise, TrainWorldWisePP, _to_device, _predcls_labels, _pp_grid_kwargs):
    def _v2_kwargs(b, train):
        kw = {"slot_classes_seq": b.get("object_classes")}
        if train:
            kw.update(gt_corners_seq=b.get("gt_corners"), gt_box_valid_seq=b.get("gt_box_valid"))
        return kw

    class TrainWorldWise3D(TrainWorldWise):
        def init_model(self):
            from lib.supervised.v2.worldwise_3d import WorldWise3D
            self._model = WorldWise3D(
                config=self._conf, num_object_classes=len(self._object_classes),
                attention_class_num=len(self._train_dataset.attention_relationships),
                spatial_class_num=len(self._train_dataset.spatial_relationships),
                contact_class_num=len(self._train_dataset.contacting_relationships)).to(self._device)

        def init_loss_fn(self):
            super().init_loss_fn()
            self._loss_fn = LossV2(self._loss_fn, getattr(self._conf, "lambda_geo", 1.0)).to(self._device)

        def _forward(self, b, train):
            return self._model.forward(
                visual_features_seq=b["visual_features"], corners_seq=b["corners"], valid_mask_seq=b["valid_mask"],
                visibility_mask_seq=b["visibility_mask"], person_idx_seq=b["person_idx"],
                object_idx_seq=b["object_idx"], pair_valid=b["pair_valid"],
                p_mask_visible=getattr(self._conf, "p_mask_visible", 0.3) if train else 0.0,
                camera_pose_seq=b.get("camera_poses"), union_features_seq=b.get("union_features"),
                node_labels_seq=_predcls_labels(self._conf, b),
                gt_contacting_seq=b.get("gt_contacting") if train else None, **_v2_kwargs(b, train))

        def process_train_video(self, batch) -> dict:
            b = _to_device(batch, self._device)
            pred = self._forward(b, True)
            return self._loss_fn(
                predictions=pred, gt_attention=b["gt_attention"], gt_spatial=b["gt_spatial"],
                gt_contacting=b["gt_contacting"], pair_valid=b["pair_valid"], visibility_mask=b["visibility_mask"],
                person_idx=b["person_idx"], object_idx=b["object_idx"], valid_mask=b.get("valid_mask"),
                corners=b.get("corners"), gt_node_labels=b.get("object_classes"),
                vlm_confidence=b.get("vlm_confidence"))

        def process_test_video(self, batch) -> dict:
            b = _to_device(batch, self._device)
            pred = self._forward(b, False)
            if b["visual_features"].shape[0] > 0:
                return {"attention_distribution": pred["attention_distribution"][-1],
                        "spatial_distribution": pred["spatial_distribution"][-1],
                        "contacting_distribution": pred["contacting_distribution"][-1]}
            return None

    class TrainWorldWisePP3D(TrainWorldWisePP):
        def init_model(self):
            from lib.supervised.v2.worldwise_pp_3d import WorldWisePP3D
            self._model = WorldWisePP3D(
                config=self._conf, num_object_classes=len(self._object_classes),
                attention_class_num=len(self._train_dataset.attention_relationships),
                spatial_class_num=len(self._train_dataset.spatial_relationships),
                contact_class_num=len(self._train_dataset.contacting_relationships)).to(self._device)

        def init_loss_fn(self):
            super().init_loss_fn()
            self._loss_fn = LossV2(self._loss_fn, getattr(self._conf, "lambda_geo", 1.0)).to(self._device)

        def _forward(self, b, train):
            return self._model.forward(
                visual_features_seq=b["visual_features"], corners_seq=b["corners"], valid_mask_seq=b["valid_mask"],
                visibility_mask_seq=b["visibility_mask"], person_idx_seq=b["person_idx"],
                object_idx_seq=b["object_idx"], pair_valid=b["pair_valid"],
                p_mask_visible=getattr(self._conf, "p_mask_visible", 0.3) if train else 0.0,
                camera_pose_seq=b.get("camera_poses"), node_labels_seq=_predcls_labels(self._conf, b),
                gt_contacting_seq=b.get("gt_contacting") if train else None, **_pp_grid_kwargs(b),
                **_v2_kwargs(b, train))

        def process_train_video(self, batch) -> dict:
            b = _to_device(batch, self._device)
            pred = self._forward(b, True)
            return self._loss_fn(
                predictions=pred, gt_attention=b["gt_attention"], gt_spatial=b["gt_spatial"],
                gt_contacting=b["gt_contacting"], pair_valid=b["pair_valid"], visibility_mask=b["visibility_mask"],
                person_idx=b["person_idx"], object_idx=b["object_idx"], valid_mask=b.get("valid_mask"),
                corners=b.get("corners"), gt_node_labels=b.get("object_classes"),
                vlm_confidence=b.get("vlm_confidence"), gt_bboxes_2d=b.get("gt_bboxes_2d"),
                gt_corners=b.get("gt_corners"), camera_poses=b.get("camera_poses"))

        def process_test_video(self, batch) -> dict:
            b = _to_device(batch, self._device)
            pred = self._forward(b, False)
            if b["visual_features"].shape[0] > 0:
                return {"attention_distribution": pred["attention_distribution"][-1],
                        "spatial_distribution": pred["spatial_distribution"][-1],
                        "contacting_distribution": pred["contacting_distribution"][-1]}
            return None

    return {"worldwise_3d": TrainWorldWise3D, "worldwise_pp_3d": TrainWorldWisePP3D}


def _v2_targets(b, train):
    kw = {"slot_classes_seq": b.get("object_classes")}
    if train:
        kw.update(gt_corners_seq=b.get("gt_corners"), gt_box_valid_seq=b.get("gt_box_valid"))
    return kw


def build_v2_more(v2_map, base_map, _to_device, _predcls_labels):
    """V2 trainers for the rest of the V1 rows (Nov 11): WorldWise+-3D and the five baselines.

    ``v2_map`` holds the WorldWise-3D trainer (WorldWise+-3D reuses it with WorldWise+'s token projectors);
    ``base_map`` is the V1 METHOD_MAP (each baseline trainer keeps its own dataset, loss and logit adjustment; the
    loss gains ``lambda_geo`` x (geo_latent + geo_box) through LossV2)."""
    from lib.supervised.v2.baselines_3d import BASELINE_V2, baseline_v2_class

    class TrainWorldWisePlus3D(v2_map["worldwise_3d"]):
        def init_model(self):
            from lib.supervised.v2.worldwise_plus_3d import WorldWisePlus3D
            self._model = WorldWisePlus3D(
                config=self._conf, num_object_classes=len(self._object_classes),
                attention_class_num=len(self._train_dataset.attention_relationships),
                spatial_class_num=len(self._train_dataset.spatial_relationships),
                contact_class_num=len(self._train_dataset.contacting_relationships)).to(self._device)

    def baseline_trainer(method):
        Base = base_map[BASELINE_V2[method]]

        class TrainBaseline3D(Base):
            def init_model(self):
                C = baseline_v2_class(method)
                self._model = C(
                    config=self._conf, num_object_classes=len(self._object_classes),
                    attention_class_num=len(self._train_dataset.attention_relationships),
                    spatial_class_num=len(self._train_dataset.spatial_relationships),
                    contact_class_num=len(self._train_dataset.contacting_relationships)).to(self._device)

            def init_loss_fn(self):
                super().init_loss_fn()
                self._loss_fn = LossV2(self._loss_fn, getattr(self._conf, "lambda_geo", 1.0)).to(self._device)

            def _forward(self, b, train):
                return self._model.forward(
                    visual_features_seq=b["visual_features"], corners_seq=b["corners"],
                    valid_mask_seq=b["valid_mask"], visibility_mask_seq=b["visibility_mask"],
                    person_idx_seq=b["person_idx"], object_idx_seq=b["object_idx"], pair_valid=b["pair_valid"],
                    camera_pose_seq=b.get("camera_poses"), union_features_seq=b.get("union_features"),
                    node_labels_seq=_predcls_labels(self._conf, b),
                    p_mask_visible=getattr(self._conf, "p_mask_visible", 0.3) if train else 0.0,
                    **_v2_targets(b, train))

            def process_train_video(self, batch) -> dict:
                b = _to_device(batch, self._device)
                pred = self._forward(b, True)
                return self._loss_fn(
                    predictions=pred, gt_attention=b["gt_attention"], gt_spatial=b["gt_spatial"],
                    gt_contacting=b["gt_contacting"], pair_valid=b["pair_valid"],
                    visibility_mask=b["visibility_mask"], person_idx=b["person_idx"], object_idx=b["object_idx"],
                    valid_mask=b.get("valid_mask"), gt_node_labels=b.get("object_classes"))

            def process_test_video(self, batch) -> dict:
                b = _to_device(batch, self._device)
                pred = self._forward(b, False)
                if b["visual_features"].shape[0] > 0:
                    return {"attention_distribution": pred["attention_distribution"][-1],
                            "spatial_distribution": pred["spatial_distribution"][-1],
                            "contacting_distribution": pred["contacting_distribution"][-1]}
                return None

        TrainBaseline3D.__name__ = TrainBaseline3D.__qualname__ = f"Train{baseline_v2_class(method).__name__}"
        return TrainBaseline3D

    out = {"worldwise_plus_3d": TrainWorldWisePlus3D}
    out.update({m: baseline_trainer(m) for m in BASELINE_V2})
    return out


def build_v2_model(conf, ds, device):
    """Model for a V2 config (dump / gates), mirroring tools/reeval_test.build_model."""
    nobj, natt = len(ds.object_classes), len(ds.attention_relationships)
    nspa, ncon = len(ds.spatial_relationships), len(ds.contacting_relationships)
    if conf.method_name == "worldwise_3d":
        from lib.supervised.v2.worldwise_3d import WorldWise3D as C
    elif conf.method_name == "worldwise_pp_3d":
        from lib.supervised.v2.worldwise_pp_3d import WorldWisePP3D as C
    elif conf.method_name == "worldwise_plus_3d":
        from lib.supervised.v2.worldwise_plus_3d import WorldWisePlus3D as C
    else:
        from lib.supervised.v2.baselines_3d import BASELINE_V2, baseline_v2_class
        if conf.method_name not in BASELINE_V2:
            raise ValueError(f"not a V2 method: {conf.method_name}")
        C = baseline_v2_class(conf.method_name)
    return C(conf, nobj, natt, nspa, ncon).to(device)


def forward_v2(model, conf, b):
    """Test-time forward of a V2 model on a device batch (no targets)."""
    kw = dict(visual_features_seq=b["visual_features"], corners_seq=b["corners"], valid_mask_seq=b["valid_mask"],
              visibility_mask_seq=b["visibility_mask"], person_idx_seq=b["person_idx"],
              object_idx_seq=b["object_idx"], pair_valid=b["pair_valid"], p_mask_visible=0.0,
              camera_pose_seq=b.get("camera_poses"),
              node_labels_seq=b.get("object_classes") if conf.mode == "predcls" else None,
              slot_classes_seq=b.get("object_classes"))
    if conf.method_name == "worldwise_pp_3d":
        kw.update(grid_dino_seq=b["grid_dino"], grid_pi3_seq=b["grid_pi3"], image_hw=b["image_hw"],
                  bboxes_2d_seq=b.get("bboxes_2d"))
    else:
        kw["union_features_seq"] = b.get("union_features")
    return model(**kw)
