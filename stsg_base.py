import os
from abc import abstractmethod

import torch
import wandb
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from constants import Constants as const
from lib_b.AdamW import AdamW
from lib.supervised import BasicSceneGraphEvaluator
from lib.supervised.sgg.dsgdetr.matcher import HungarianMatcher


class STSGBase:

    def __init__(self, conf):
        self._train_dataset = None
        self._evaluator = None
        self._model = None
        self._conf = None
        self._device = None

        self._conf = conf

        # Load checkpoint name
        self._checkpoint_name = None
        self._checkpoint_save_dir_path = None

        # Init Wandb
        self._enable_wandb = self._conf.use_wandb

        # Init STL filename parameters
        directory_path = os.path.dirname(os.path.abspath(__file__))
        self._stl_generic_text_file_path = os.path.join(directory_path, "lib/stl/data/rules/generic.txt")
        self._stl_dataset_specific_file_path = os.path.join(directory_path, "lib/stl/data/rules/dataset_specific.json")


    def _init_config(self, is_train=True):
        print('The CKPT saved here:', self._conf.save_path)
        os.makedirs(self._conf.save_path, exist_ok=True)

        # Set the preferred device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._conf.ckpt is not None:
            if self._conf.task_name == const.SGG:
                # Checkpoint name is of the format for model trained with full annotations: sttran_sgdet_epoch_1.tar, dsgdetr_sgdet_epoch_1.tar
                # Checkpoint name is of the format for model trained with partial annotations: sttran_partial_10_sgdet_epoch_1.tar
                # Checkpoint name is of the format for model trained with label noise: sttran_label_noise_10_sgdet_epoch_1.tar
                self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
                self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])

                if "_sgdet" in self._checkpoint_name:
                    self._conf.mode = const.SGDET
                elif "_sgcls" in self._checkpoint_name:
                    self._conf.mode = const.SGCLS
                elif "_predcls" in self._checkpoint_name:
                    self._conf.mode = const.PREDCLS

                # self._conf.mode = self._checkpoint_name.split('_')[-1]
                print("--------------------------------------------------------")
                print(f"Loading checkpoint with name: {self._checkpoint_name}")
                print(f"Mode: {self._conf.mode}")
                print("--------------------------------------------------------")
            elif self._conf.task_name == const.SGA:
                # Checkpoint name format for full annotations: sttran_ant_sgdet_future_3_epoch_1.tar
                # Checkpoint name format for partial annotations: sttran_ant_partial_10_sgdet_future_3_epoch_1.tar
                # Checkpoint name format for label noise: sttran_ant_label_noise_10_sgdet_future_3_epoch_1.tar
                self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
                self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])
                self._conf.max_window = int(self._checkpoint_name.split('_')[-1])
                self._conf.mode = self._checkpoint_name.split('_')[-3]
                print("--------------------------------------------------------")
                print(f"Loading checkpoint with name: {self._checkpoint_name}")
                print(f"Mode: {self._conf.mode}")
                print(f"Max Window: {self._conf.max_window}")
                print("--------------------------------------------------------")
        else:
            # Set the checkpoint name and save path details
            if self._conf.task_name == const.SGG:
                if self._conf.use_partial_annotations:
                    self._checkpoint_name = f"{self._conf.method_name}_partial_{self._conf.partial_percentage}_{self._conf.mode}"
                elif self._conf.use_label_noise:
                    self._checkpoint_name = f"{self._conf.method_name}_label_noise_{self._conf.label_noise_percentage}_{self._conf.mode}"
                else:
                    self._checkpoint_name = f"{self._conf.method_name}_{self._conf.mode}"
                print("--------------------------------------------------------")
                print(f"Training model with name: {self._checkpoint_name}")
                print("--------------------------------------------------------")
            elif self._conf.task_name == const.SGA:
                if self._conf.use_partial_annotations:
                    self._checkpoint_name = f"{self._conf.method_name}_partial_{self._conf.partial_percentage}_{self._conf.mode}_future_{self._conf.max_window}"
                elif self._conf.use_label_noise:
                    self._checkpoint_name = f"{self._conf.method_name}_label_noise_{self._conf.label_noise_percentage}_{self._conf.mode}_future_{self._conf.max_window}"
                else:
                    self._checkpoint_name = f"{self._conf.method_name}_{self._conf.mode}_future_{self._conf.max_window}"
                print("--------------------------------------------------------")
                print(f"Training model with name: {self._checkpoint_name}")
                print("--------------------------------------------------------")

        self._checkpoint_save_dir_path = os.path.join(self._conf.save_path, self._conf.task_name, self._conf.method_name)
        os.makedirs(self._checkpoint_save_dir_path, exist_ok=True)

        # --------------------------- W&B CONFIGURATION ---------------------------
        if self._enable_wandb:
            wandb.init(project=self._checkpoint_name, config=self._conf)

        print("-------------------- CONFIGURATION DETAILS ------------------------")
        for i in self._conf.args:
            print(i, ':', self._conf.args[i])
        print("-------------------------------------------------------------------")

    def _init_optimizer(self):
        if self._conf.optimizer == const.ADAMW:
            self._optimizer = AdamW(self._model.parameters(), lr=self._conf.lr)
        elif self._conf.optimizer == const.ADAM:
            self._optimizer = optim.Adam(self._model.parameters(), lr=self._conf.lr)
        elif self._conf.optimizer == const.SGD:
            self._optimizer = optim.SGD(self._model.parameters(), lr=self._conf.lr, momentum=0.9, weight_decay=0.01)
        else:
            raise NotImplementedError

    def _prepare_scenesayer_ant_labels_and_distribution(self, pred, count, distribution_type, label_type, max_len):
        pred_distribution = pred[distribution_type][count-1]

        # Intersection between the objects in the last context frame and the objects in each future frame flattened
        window_mask_curr = pred["mask_curr_" + str(count)]
        window_mask_gt = pred["mask_gt_" + str(count)]

        assert window_mask_curr.shape[0] == window_mask_gt.shape[0]

        # Total number of objects in this window
        window_total_objects = window_mask_curr.shape[0]

        # We need to filter out the labels and the specific type of distribution based on the masks
        window_gt_labels = []
        window_gt_labels_mask = []
        for i in range(window_mask_gt.shape[0]):
            # 1. Take ground truth labels for the distribution type for all frames
            # 2. Filter them to only include the labels that are common with last context frame (Ordered*)
            window_gt_labels.append(pred[label_type][window_mask_gt[i]])
            # 1. Take ground truth labels for loss masks for all frames
            # 2. Filter them to only include the labels that are common with last context frame (Ordered*)
            window_gt_labels_mask.append(pred[f'{label_type}_mask'][window_mask_gt[i]])

        # 2. Filter pred distribution to include that are present in the last context frame (Ordered*)
        window_pred_distribution = pred_distribution[window_mask_curr]

        assert len(window_gt_labels) == len(window_gt_labels_mask) == window_pred_distribution.shape[0]

        # Filter out the distribution based on masks
        filtered_window_pred_distribution = []
        filtered_window_gt_labels = []
        if self._conf.bce_loss:
            # For Binary Cross Entropy Loss (BCE)
            for i in range(window_total_objects):
                gt = torch.tensor(window_gt_labels[i], device=self._device)
                loss_mask = torch.tensor(window_gt_labels_mask[i], device=self._device)

                assert gt.shape[0] == loss_mask.shape[0]

                gt_masked = gt[loss_mask == 1]
                pred_distribution_i = window_pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = torch.zeros([max_len], dtype=torch.float32, device=self._device)
                    label[gt_masked] = 1
                    filtered_window_gt_labels.append(label)
                    filtered_window_pred_distribution.append(pred_distribution_i)
        else:
            # For Multi Label Margin Loss (MLM)
            for i in range(window_total_objects):
                gt = torch.tensor(window_gt_labels[i], device=self._device)
                loss_mask = torch.tensor(window_gt_labels_mask[i], device=self._device)

                assert gt.shape[0] == loss_mask.shape[0]

                gt_masked = gt[loss_mask == 1]
                pred_distribution_i = window_pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = -torch.ones([max_len], dtype=torch.long, device=self._device)
                    label[:gt_masked.size(0)] = gt_masked
                    filtered_window_gt_labels.append(label)
                    filtered_window_pred_distribution.append(pred_distribution_i)

        if len(filtered_window_gt_labels) == 0 and len(filtered_window_pred_distribution) == 0:
            return None, None

        filtered_labels = torch.stack(filtered_window_gt_labels)
        filtered_distribution = torch.stack(filtered_window_pred_distribution)

        assert filtered_labels.shape[0] == filtered_distribution.shape[0]

        return filtered_distribution, filtered_labels

    def _prepare_transformer_ant_labels_and_distribution(self, pred, count, distribution_type, label_type, max_len):
        window_pred = pred["output"][count]
        pred_distribution = window_pred[distribution_type]

        # Intersection between the objects in the last context frame and the objects in each future frame flattened
        window_mask_ant = window_pred["mask_ant"].cpu().numpy()
        window_mask_gt = window_pred["mask_gt"].cpu().numpy()

        assert window_mask_ant.shape[0] == window_mask_gt.shape[0]

        # Total number of objects in this window
        window_total_objects = window_mask_ant.shape[0]

        # We need to filter out the labels and the specific type of distribution based on the masks
        window_gt_labels = []
        window_gt_labels_mask = []
        for i in range(window_mask_gt.shape[0]):
            # 1. Take ground truth labels for the distribution type for all frames
            # 2. Filter them to only include the labels that are common with last context frame (Ordered*)
            window_gt_labels.append(pred[label_type][window_mask_gt[i]])
            # 1. Take ground truth labels for loss masks for all frames
            # 2. Filter them to only include the labels that are common with last context frame (Ordered*)
            window_gt_labels_mask.append(pred[f'{label_type}_mask'][window_mask_gt[i]])

        # 2. Filter pred distribution to include that are present in the last context frame (Ordered*)
        window_pred_distribution = pred_distribution[window_mask_ant]

        assert len(window_gt_labels) == len(window_gt_labels_mask) == window_pred_distribution.shape[0]

        # Filter out the distribution based on masks
        filtered_window_pred_distribution = []
        filtered_window_gt_labels = []
        if self._conf.bce_loss:
            # For Binary Cross Entropy Loss (BCE)
            for i in range(window_total_objects):
                gt = torch.tensor(window_gt_labels[i], device=self._device)
                loss_mask = torch.tensor(window_gt_labels_mask[i], device=self._device)

                assert gt.shape[0] == loss_mask.shape[0]

                gt_masked = gt[loss_mask == 1]
                pred_distribution_i = window_pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = torch.zeros([max_len], dtype=torch.float32, device=self._device)
                    label[gt_masked] = 1
                    filtered_window_gt_labels.append(label)
                    filtered_window_pred_distribution.append(pred_distribution_i)
        else:
            # For Multi Label Margin Loss (MLM)
            for i in range(window_total_objects):
                gt = torch.tensor(window_gt_labels[i], device=self._device)
                loss_mask = torch.tensor(window_gt_labels_mask[i], device=self._device)

                assert gt.shape[0] == loss_mask.shape[0]

                gt_masked = gt[loss_mask == 1]
                pred_distribution_i = window_pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = -torch.ones([max_len], dtype=torch.long, device=self._device)
                    label[:gt_masked.size(0)] = gt_masked
                    filtered_window_gt_labels.append(label)
                    filtered_window_pred_distribution.append(pred_distribution_i)

        if len(filtered_window_gt_labels) == 0 and len(filtered_window_pred_distribution) == 0:
            return None, None

        filtered_labels = torch.stack(filtered_window_gt_labels)
        filtered_distribution = torch.stack(filtered_window_pred_distribution)

        assert filtered_labels.shape[0] == filtered_distribution.shape[0]

        return filtered_distribution, filtered_labels

    def _prepare_labels_and_distribution(self, pred, distribution_type, label_type, max_len):
        total_labels = len(pred[label_type])
        pred_distribution = pred[distribution_type]

        # Filter out both the distribution and labels if all the labels are masked
        # Note: Loss should not include items if all the labels are masked
        filtered_labels = []
        filtered_distribution = []
        if not self._conf.bce_loss:
            # For Multi Label Margin Loss (MLM)
            for i in range(total_labels):
                gt = torch.tensor(pred[label_type][i], device=self._device)
                mask = torch.tensor(pred[f'{label_type}_mask'][i], device=self._device)
                gt_masked = gt[mask == 1]
                pred_distribution_i = pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = -torch.ones([max_len], dtype=torch.long, device=self._device)
                    label[:gt_masked.size(0)] = gt_masked
                    filtered_labels.append(label)
                    filtered_distribution.append(pred_distribution_i)
        else:
            # For Binary Cross Entropy Loss (BCE)
            for i in range(total_labels):
                gt = torch.tensor(pred[label_type][i], device=self._device)
                mask = torch.tensor(pred[f'{label_type}_mask'][i], device=self._device)
                gt_masked = gt[mask == 1]
                pred_distribution_i = pred_distribution[i]

                if gt_masked.shape[0] == 0:
                    continue
                else:
                    label = torch.zeros([max_len], dtype=torch.float32, device=self._device)
                    label[gt_masked] = 1
                    filtered_labels.append(label)
                    filtered_distribution.append(pred_distribution_i)

        if len(filtered_labels) == 0 and len(filtered_distribution) == 0:
            return None, None, pred_distribution

        filtered_labels = torch.stack(filtered_labels)
        filtered_distribution = torch.stack(filtered_distribution)

        return filtered_distribution, filtered_labels, pred_distribution

    def _init_scheduler(self):
        self._scheduler = ReduceLROnPlateau(self._optimizer, "max", patience=1, factor=0.5, verbose=True,
                                            threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    def _init_matcher(self):
        self._matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        self._matcher.eval()

    def _load_checkpoint(self):
        if self._model is None:
            raise ValueError("Model is not initialized")

        if self._conf.ckpt:
            if os.path.exists(self._conf.ckpt) is False:
                raise ValueError(f"Checkpoint file {self._conf.ckpt} does not exist")

            try:
                # Load checkpoint to the specified device
                ckpt = torch.load(self._conf.ckpt, map_location=self._device)

                # Determine the key for the state_dict based on availability
                state_dict_key = 'state_dict' if 'state_dict' in ckpt else f'{self._conf.method_name}_state_dict'

                # Load the state dictionary
                self._model.load_state_dict(ckpt[state_dict_key], strict=False)
                print(f"Loaded model from checkpoint {self._conf.ckpt}")

            except FileNotFoundError:
                print(f"Error: Checkpoint file {self._conf.ckpt} not found.")
            except KeyError:
                print(f"Error: Appropriate state_dict not found in the checkpoint.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")

    @staticmethod
    def _save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, method_name):
        print("*" * 40)
        os.makedirs(checkpoint_save_file_path, exist_ok=True)
        torch.save({f"{method_name}_state_dict": model.state_dict()},
                   os.path.join(checkpoint_save_file_path, f"{checkpoint_name}_epoch_{epoch}.tar"))
        print(f"Saved {method_name} checkpoint after {epoch} epochs")
        print("*" * 40)

    def _init_evaluators(self):
        # For VidSGG set iou_threshold=0.5
        # For SGA set iou_threshold=0
        iou_threshold = 0.5 if self._conf.task_name == 'sgg' else 0.0

        self._evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._train_dataset.object_classes,
            AG_all_predicates=self._train_dataset.relationship_classes,
            AG_attention_predicates=self._train_dataset.attention_relationships,
            AG_spatial_predicates=self._train_dataset.spatial_relationships,
            AG_contacting_predicates=self._train_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            save_file=os.path.join(self._conf.save_path, const.PROGRESS_TEXT_FILE),
            constraint='with'
        )

    @abstractmethod
    def _init_object_detector(self):
        pass

    @abstractmethod
    def init_model(self):
        pass

    @staticmethod
    def get_sequence_no_tracking(entry, task="sgcls"):
        if task == "predcls":
            indices = []
            for i in entry["labels"].unique():
                indices.append(torch.where(entry["labels"] == i)[0])
            entry["indices"] = indices
            return

        if task == "sgdet" or task == "sgcls":
            # for sgdet, use the predicted object classes, as a special case of
            # the proposed method, comment this out for general coase tracking.
            indices = [[]]
            # indices[0] store single-element sequence, to save memory
            pred_labels = torch.argmax(entry["distribution"], 1)
            for i in pred_labels.unique():
                index = torch.where(pred_labels == i)[0]
                if len(index) == 1:
                    indices[0].append(index)
                else:
                    indices.append(index)
            if len(indices[0]) > 0:
                indices[0] = torch.cat(indices[0])
            else:
                indices[0] = torch.tensor([])
            entry["indices"] = indices
            return