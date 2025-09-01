from lib.supervised.config import Config
from train_sgg_base import TrainSGGBase
from lib.supervised.sgg.sttran import STTran
from lib.supervised.sgg.dsgdetr.dsgdetr import DsgDETR


# -----------------------------------------------------------------------------------------------------
# --------------------------------------------- BASELINE METHODS ---------------------------------------
# -----------------------------------------------------------------------------------------------------


class TrainSTTran(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainDsgDetr(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher

        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred


# -----------------------------------------------------------------------------------------------------
# ----------------------------------- CURRICULUM LEARNING METHODS -------------------------------------
# -----------------------------------------------------------------------------------------------------

class TrainCurriculumSTTran(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainCurriculumDsgDetr(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher

        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred


# -----------------------------------------------------------------------------------------------------
# --------------------------------------------- STL BASED METHODS -------------------------------------
# -----------------------------------------------------------------------------------------------------

# --------------------------------------------- STTRAN METHODS -------------------------------------

class TrainSTTranSTLBase(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        raise NotImplementedError

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        self.get_sequence_no_tracking(entry, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainSTTranSTLGeneric(TrainSTTranSTLBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._enable_stl_loss = True
        self._enable_generic_loss = True
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = False


class TrainSTTranSTLDS(TrainSTTranSTLBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._enable_stl_loss = True
        self._enable_generic_loss = False
        self._enable_dataset_specific_loss = True
        self._enable_time_conditioned_dataset_specific_loss = False


class TrainSTTranSTLTimeCondDS(TrainSTTranSTLBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._enable_stl_loss = True
        self._enable_generic_loss = False
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = True


class TrainSTTranSTLComb(TrainSTTranSTLBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._enable_stl_loss = True
        self._enable_generic_loss = True
        self._enable_dataset_specific_loss = True
        self._enable_time_conditioned_dataset_specific_loss = False


class TrainSTTranSTLTimeCondComb(TrainSTTranSTLBase):

    def __init__(self, conf):
        super().__init__(conf)

    def init_model(self):
        self._model = STTran(mode=self._conf.mode,
                             attention_class_num=len(self._train_dataset.attention_relationships),
                             spatial_class_num=len(self._train_dataset.spatial_relationships),
                             contact_class_num=len(self._train_dataset.contacting_relationships),
                             obj_classes=self._train_dataset.object_classes,
                             enc_layer_num=self._conf.enc_layer,
                             dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._enable_stl_loss = True
        self._enable_generic_loss = True
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = True


# --------------------------------------------- DSGDETR METHODS -------------------------------------

class TrainDsgDetrSTLBase(TrainSGGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        raise NotImplementedError

    def process_train_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred

    def process_test_video(self, entry, frame_size, gt_annotation) -> dict:
        from lib.supervised.sgg.dsgdetr.track import get_sequence_with_tracking
        get_sequence_with_tracking(entry, gt_annotation, self._matcher, frame_size, self._conf.mode)
        pred = self._model(entry)
        return pred


class TrainDsgDetrSTLGeneric(TrainDsgDetrSTLBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

        self._enable_stl_loss = True
        self._enable_generic_loss = True
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = False


class TrainDsgDetrSTLDS(TrainDsgDetrSTLBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

        self._enable_stl_loss = True
        self._enable_generic_loss = False
        self._enable_dataset_specific_loss = True
        self._enable_time_conditioned_dataset_specific_loss = False


class TrainDsgDetrSTLTimeCondDS(TrainDsgDetrSTLBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

        self._enable_stl_loss = True
        self._enable_generic_loss = False
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = True


class TrainDsgDetrSTLComb(TrainDsgDetrSTLBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

        self._enable_stl_loss = True
        self._enable_generic_loss = True
        self._enable_dataset_specific_loss = True
        self._enable_time_conditioned_dataset_specific_loss = False


class TrainDsgDetrSTLTimeCondComb(TrainDsgDetrSTLBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._matcher = None

    def init_model(self):
        from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher
        self._model = DsgDETR(mode=self._conf.mode,
                              attention_class_num=len(self._train_dataset.attention_relationships),
                              spatial_class_num=len(self._train_dataset.spatial_relationships),
                              contact_class_num=len(self._train_dataset.contacting_relationships),
                              obj_classes=self._train_dataset.object_classes,
                              enc_layer_num=self._conf.enc_layer,
                              dec_layer_num=self._conf.dec_layer).to(device=self._device)

        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)

        self._enable_stl_loss = True
        self._enable_generic_loss = True
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = True


def main():
    conf = Config()
    is_curriculum = False
    if conf.method_name == "sttran":
        train_class = TrainSTTran(conf)
    elif conf.method_name == "dsgdetr":
        train_class = TrainDsgDetr(conf)
    elif conf.method_name == "sttran_curriculum":
        train_class = TrainCurriculumSTTran(conf)
        is_curriculum = True
    elif conf.method_name == "dsgdetr_curriculum":
        train_class = TrainCurriculumDsgDetr(conf)
        is_curriculum = True
    elif conf.method_name == "sttran_stl_generic":
        train_class = TrainSTTranSTLGeneric(conf)
    elif conf.method_name == "dsgdetr_stl_generic":
        train_class = TrainDsgDetrSTLGeneric(conf)
    elif conf.method_name == "sttran_stl_ds":
        train_class = TrainSTTranSTLDS(conf)
    elif conf.method_name == "dsgdetr_stl_ds":
        train_class = TrainDsgDetrSTLDS(conf)
    elif conf.method_name == "sttran_stl_time_cond_ds":
        train_class = TrainSTTranSTLTimeCondDS(conf)
    elif conf.method_name == "dsgdetr_stl_time_cond_ds":
        train_class = TrainDsgDetrSTLTimeCondDS(conf)
    elif conf.method_name == "sttran_stl_comb":
        train_class = TrainSTTranSTLComb(conf)
    elif conf.method_name == "dsgdetr_stl_comb":
        train_class = TrainDsgDetrSTLComb(conf)
    elif conf.method_name == "sttran_stl_time_cond_comb":
        train_class = TrainSTTranSTLTimeCondComb(conf)
    elif conf.method_name == "dsgdetr_stl_time_cond_comb":
        train_class = TrainDsgDetrSTLTimeCondComb(conf)
    else:
        raise NotImplementedError

    print(f"-------------------------------------------------------")
    print(f"Training method: {conf.method_name}-{'Curriculum' if is_curriculum else 'No Curriculum'}")
    print(f"-------------------------------------------------------")

    train_class.init_method_training(is_curriculum=is_curriculum)


if __name__ == "__main__":
    main()
