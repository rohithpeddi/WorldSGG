import copy
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import Constants as const

from dataloader.label_noise.action_genome.ag_dataset import LabelNoiseAG
from dataloader.partial.action_genome.ag_dataset import PartialAG
from dataloader.standard.action_genome.ag_dataset import StandardAG
from dataloader.standard.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
from lib_b.object_detector import Detector
from stsg_base import STSGBase


class TrainSGGBase(STSGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._model = None

        # Load while initializing the object detector
        self._object_detector = None

        # Load while initializing the dataset
        self._train_dataset = None
        self._dataloader_train = None
        self._dataloader_test = None
        self._test_dataset = None
        self._object_classes = None

        # Load checkpoint name
        self._checkpoint_name = None
        self._checkpoint_save_dir_path = None

        # Enable STL constraint loss
        self._enable_stl_loss = False
        self._enable_generic_loss = False
        self._enable_dataset_specific_loss = False
        self._enable_time_conditioned_dataset_specific_loss = False

    def _init_loss_functions(self):
        self._bce_loss = nn.BCELoss()
        self._ce_loss = nn.CrossEntropyLoss()
        self._mlm_loss = nn.MultiLabelMarginLoss()
        self._bbox_loss = nn.SmoothL1Loss()
        self._abs_loss = nn.L1Loss()
        self._mse_loss = nn.MSELoss()

        if self._enable_stl_loss:
            if self._enable_generic_loss or self._enable_dataset_specific_loss:
                # self._stl_rule_parser = Parser()
                self._stl_rule_parser = IterativeParser()
                self._stl_tokenizer = Tokenizer()


    def _init_object_detector(self):
        self._object_detector = Detector(
            train=True,
            object_classes=self._object_classes,
            use_SUPPLY=True,
            mode=self._conf.mode
        ).to(device=self._device)
        self._object_detector.eval()

    # ----------------- Load the dataset -------------------------
    # Three main settings:
    # (a) Standard Dataset: Where full annotations are used
    # (b) Partial Annotations: Where partial object and relationship annotations are used
    # (c) Label Noise: Where label noise is added to the dataset
    # -------------------------------------------------------------
    def init_dataset(self):

        if self._conf.use_partial_annotations:
            print("-----------------------------------------------------")
            print("Loading the partial annotations dataset with percentage:", self._conf.partial_percentage)
            print("-----------------------------------------------------")
            self._train_dataset = PartialAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                partial_percentage=self._conf.partial_percentage,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
        elif self._conf.use_label_noise:
            print("-----------------------------------------------------")
            print("Loading the dataset with label noise percentage:", self._conf.label_noise_percentage)
            print("-----------------------------------------------------")
            self._train_dataset = LabelNoiseAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                noise_percentage=self._conf.label_noise_percentage,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
        else:
            print("-----------------------------------------------------")
            print("Loading the standard dataset")
            print("-----------------------------------------------------")
            self._train_dataset = StandardAG(
                phase="train",
                mode=self._conf.mode,
                datasize=self._conf.datasize,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True
            )

        self._test_dataset = StandardAG(
            phase="test",
            mode=self._conf.mode,
            datasize=self._conf.datasize,
            data_path=self._conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if self._conf.mode == 'predcls' else True
        )

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=False
        )

        self._object_classes = self._train_dataset.object_classes


    def init_curriculum_dataset(self, epoch):
        print("-----------------------------------------------------")
        print("Loading the curriculum dataset")
        print("-----------------------------------------------------")

        if epoch == 0:
            self._train_dataset = PartialAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                partial_percentage=40,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
            self._conf.partial_percentage = 40
        elif epoch == 1:
            self._train_dataset = PartialAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                partial_percentage=10,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
            self._conf.partial_percentage = 10
        elif epoch > 1:
            self._train_dataset = PartialAG(
                phase="train",
                mode=self._conf.mode,
                maintain_distribution=self._conf.maintain_distribution,
                datasize=self._conf.datasize,
                partial_percentage=10,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
            )
            self._conf.partial_percentage = 10

        assert self._train_dataset is not None
        self._conf.use_partial_annotations = True

        self._dataloader_train = DataLoader(
            self._train_dataset,
            shuffle=True,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=True,
            num_workers=0
        )

        self._object_classes = self._train_dataset.object_classes


    def _calculate_generic_stl_loss(self, predictions):
        # Predictions should be a tensor of shape [total_num_objects, num_relationships]
        # The total number of objects here can be considered as the batch size

        # 0. Construct value map for each relationship expression used to make an expression to a predicate
        num_objects, num_relationships = predictions.shape
        # Threshold for sigmoid = 0.5, Threshold for logits = 0
        constants = torch.zeros(num_relationships).reshape((num_relationships, 1)).to(device=self._device)
        constants[3:] += 0.5
        # Set requires_grad to False for the constant tensor
        constants.requires_grad = False

        # 1. Construct STL Expressions for each relationship and their corresponding constants
        relationship_classes = self._train_dataset.relationship_classes
        rel_exp_list = []
        const_exp_list = []
        for rel_idx, relation in enumerate(relationship_classes):
            rel_exp = Expression(f"{relation}_generic", predictions[:, rel_idx].reshape(1, -1, 1))
            const_exp = Expression(f"{relation}_const", constants[rel_idx])
            rel_exp_list.append(rel_exp)
            const_exp_list.append(const_exp)

        # 2. Construct STL Predicates for each relationship type prediction
        # In our case, predicates are of the form: GreaterThan(relation_expression, constant_expression)
        # For attention relationship, predictions[:, :3] --> Each of processed logit should be greater than 0.0
        # For spatial relationship, predictions[:, 3:9] --> Each of processed sigmoid should be greater than 0.5
        # For contacting relationship, predictions[:, 9:] --> Each of processed sigmoid should be greater than 0.5
        relation_to_stl_predicate_map = {}
        relation_to_prediction_map = {}
        for rel_idx, relation in enumerate(relationship_classes):
            rel_exp = rel_exp_list[rel_idx]
            stl_predicate = GreaterThan(rel_exp, const_exp_list[rel_idx])
            relation_to_stl_predicate_map[relation] = stl_predicate
            relation_to_prediction_map[relation] = predictions[:, rel_idx].reshape(1, -1, 1)

        # 4. Construct STL Formulas from generic.text file
        try:
            with open(self._stl_generic_text_file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"Error: The file '{self._stl_generic_text_file_path}' was not found.")
            return

        stl_formula_list = []
        stl_formula_identifiers = []
        for idx, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            tokens = self._stl_tokenizer.tokenize(line)
            self._stl_rule_parser.init_tokens(tokens, relation_to_stl_predicate_map)
            formula = self._stl_rule_parser.parse_formula()
            formula_identifiers = self._stl_rule_parser.identifiers
            # Ensure all tokens are consumed
            if self._stl_rule_parser.current_token()[0] != 'EOF':
                raise RuntimeError(f'Unexpected token {self._stl_rule_parser.current_token()} at the end of formula')
            stl_formula_list.append(formula)
            stl_formula_identifiers.append(formula_identifiers)

        # 5. Calculate the loss for each formula
        stl_loss = 0
        for formula_idx, formula in enumerate(stl_formula_list):
            identifier_list = stl_formula_identifiers[formula_idx]
            inputs = (relation_to_prediction_map[identifier_list[0]], relation_to_prediction_map[identifier_list[1]])
            stl_loss += formula.robustness(inputs=inputs)

        normalized_stl_loss = stl_loss / len(stl_formula_list)
        return normalized_stl_loss


    def _calculate_dataset_specific_stl_loss(self, predictions):
        pass


    def _calculate_stl_loss(self, predictions):
        loss = 0
        if self._enable_generic_loss:
            loss += self._calculate_generic_stl_loss(predictions)
        if self._enable_dataset_specific_loss:
            loss += self._calculate_dataset_specific_stl_loss(predictions)
        return loss

    def _construct_inputs_for_expression(self, attention_distribution, spatial_distribution, contact_distribution):
        # 1. Convert the attention distribution to predicate friendly form
        # For each class (column tensor of att distribution), estimate logsumexp of other class tensors
        # Create a copy for attention distribution tensor
        att_dist = attention_distribution.clone()
        log_sum_exp_att = torch.zeros(att_dist.shape)
        log_sum_exp_att = log_sum_exp_att.to(device=self._device)
        for i in range(att_dist.shape[1]):
            log_sum_exp_att[:, i] = torch.logsumexp(att_dist[:, [j for j in range(att_dist.shape[1]) if j != i]], dim=1)
        attention_predicate = att_dist - log_sum_exp_att

        # 2. Convert the spatial distribution, contact distribution to predicate friendly form
        # We don't have to explicitly change anything as they are sigmoid outputs!
        spatial_predicate = spatial_distribution.clone()
        contact_predicate = contact_distribution.clone()

        # 3. Concatenate the three predicates
        # Output shape: [ batch_size(total number of objects detected), num_relationships (total number of relationships)]
        predicates_exp_input = torch.cat((attention_predicate, spatial_predicate, contact_predicate), dim=1)

        return predicates_exp_input


    def _calculate_losses_for_partial_annotations(self, pred):
        losses = {}

        # 1. Object Loss
        if self._conf.mode in [const.SGCLS, const.SGDET]:
            losses[const.OBJECT_LOSS] = self._ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

        # 2. Attention Loss
        attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
        attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(device=self._device).squeeze()
        attention_label_mask = torch.tensor(pred[const.ATTENTION_GT_MASK], dtype=torch.float32).to(
            device=self._device).squeeze()
        assert attention_label.shape == attention_label_mask.shape
        # Change to shape [1] if the tensor defaults to a single value
        if len(attention_label.shape) == 0:
            attention_label = attention_label.unsqueeze(0)
            attention_label_mask = attention_label_mask.unsqueeze(0)

        # Filter attention distribution and attention label based on the attention label mask
        filtered_attention_distribution = attention_distribution[attention_label_mask == 1]
        filtered_attention_label = attention_label[attention_label_mask == 1]

        assert filtered_attention_distribution.shape[0] == filtered_attention_label.shape[0]
        losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(filtered_attention_distribution, filtered_attention_label)

        # --------------------------------------------------------------------------------------------
        # For both spatial and contacting relations, if all the annotations are masked then the loss is not calculated

        # 3. Spatial Loss
        (filtered_spatial_distribution,
         filtered_spatial_labels, spatial_distribution) = self._prepare_labels_and_distribution(
            pred=pred,
            distribution_type=const.SPATIAL_DISTRIBUTION,
            label_type=const.SPATIAL_GT,
            max_len=6
        )

        if filtered_spatial_distribution is not None and filtered_spatial_labels is not None:
            if not self._conf.bce_loss:
                losses[const.SPATIAL_RELATION_LOSS] = self._mlm_loss(filtered_spatial_distribution, filtered_spatial_labels)
            else:
                losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(filtered_spatial_distribution, filtered_spatial_labels)

        # 4. Contacting Loss
        (filtered_contact_distribution,
         filtered_contact_labels, contact_distribution) = self._prepare_labels_and_distribution(
            pred=pred,
            distribution_type=const.CONTACTING_DISTRIBUTION,
            label_type=const.CONTACTING_GT,
            max_len=17
        )

        if filtered_contact_distribution is not None and filtered_contact_labels is not None:
            if not self._conf.bce_loss:
                losses[const.CONTACTING_RELATION_LOSS] = self._mlm_loss(filtered_contact_distribution, filtered_contact_labels)
            else:
                losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(filtered_contact_distribution, filtered_contact_labels)

        # 0. Enable STL Loss
        if self._enable_stl_loss:
            predicate_exp_input = self._construct_inputs_for_expression(
                attention_distribution,
                spatial_distribution,
                contact_distribution
            )

            # Calculate STL Loss
            losses[const.STL_LOSS] = self._calculate_stl_loss(predicate_exp_input)

        return losses


    def _calculate_losses_for_full_annotations(self, pred):
        attention_distribution = pred[const.ATTENTION_DISTRIBUTION]
        spatial_distribution = pred[const.SPATIAL_DISTRIBUTION]
        contact_distribution = pred[const.CONTACTING_DISTRIBUTION]

        attention_label = torch.tensor(pred[const.ATTENTION_GT], dtype=torch.long).to(device=self._device).squeeze()
        if not self._conf.bce_loss:
            # Adjust Labels for MLM Loss
            spatial_label = -torch.ones([len(pred[const.SPATIAL_GT]), 6], dtype=torch.long).to(device=self._device)
            contact_label = -torch.ones([len(pred[const.CONTACTING_GT]), 17], dtype=torch.long).to(device=self._device)
            for i in range(len(pred[const.SPATIAL_GT])):
                spatial_label[i, : len(pred[const.SPATIAL_GT][i])] = torch.tensor(pred[const.SPATIAL_GT][i])
                contact_label[i, : len(pred[const.CONTACTING_GT][i])] = torch.tensor(pred[const.CONTACTING_GT][i])
        else:
            # Adjust Labels for BCE Loss
            spatial_label = torch.zeros([len(pred[const.SPATIAL_GT]), 6], dtype=torch.float32).to(device=self._device)
            contact_label = torch.zeros([len(pred[const.CONTACTING_GT]), 17], dtype=torch.float32).to(device=self._device)
            for i in range(len(pred[const.SPATIAL_GT])):
                spatial_label[i, pred[const.SPATIAL_GT][i]] = 1
                contact_label[i, pred[const.CONTACTING_GT][i]] = 1

        losses = {}

        # 0. Enable STL Loss
        if self._enable_stl_loss:
            predicate_exp_input = self._construct_inputs_for_expression(
                attention_distribution,
                spatial_distribution,
                contact_distribution
            )

            # Calculate STL Loss
            losses[const.STL_LOSS] = self._calculate_stl_loss(predicate_exp_input)

        # 1. Object Loss
        if self._conf.mode == const.SGCLS or self._conf.mode == const.SGDET:
            losses[const.OBJECT_LOSS] = self._ce_loss(pred[const.DISTRIBUTION], pred[const.LABELS])

        # 2. Attention Loss
        losses[const.ATTENTION_RELATION_LOSS] = self._ce_loss(attention_distribution, attention_label)

        # 3. Spatial Loss and Contacting Loss
        if not self._conf.bce_loss:
            losses[const.SPATIAL_RELATION_LOSS] = self._mlm_loss(spatial_distribution, spatial_label)
            losses[const.CONTACTING_RELATION_LOSS] = self._mlm_loss(contact_distribution, contact_label)
        else:
            losses[const.SPATIAL_RELATION_LOSS] = self._bce_loss(spatial_distribution, spatial_label)
            losses[const.CONTACTING_RELATION_LOSS] = self._bce_loss(contact_distribution, contact_label)
        return losses

    def _train_model(self, is_curriculum=False):
        tr = []
        for epoch in range(self._conf.nepoch):
            self._model.train()

            if is_curriculum:
                self.init_curriculum_dataset(epoch)

            train_iter = iter(self._dataloader_train)
            counter = 0
            start_time = time.time()
            self._object_detector.is_train = True
            for train_idx in tqdm(range(len(self._dataloader_train))):
                data = next(train_iter)
                im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                video_index = data[4]

                gt_annotation = self._train_dataset.gt_annotations[video_index]
                gt_annotation_mask = None
                if self._conf.use_partial_annotations:
                    gt_annotation_mask = self._train_dataset.gt_annotations_mask[video_index]

                if len(gt_annotation) == 0:
                    print(f'No annotations found in the video {video_index}. Skipping...')
                    continue

                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data
                with torch.no_grad():
                    entry = self._object_detector(
                        im_data,
                        im_info,
                        gt_boxes,
                        num_boxes,
                        gt_annotation,
                        im_all=None,
                        gt_annotation_mask=gt_annotation_mask
                    )

                # ----------------- Process the video (Method Specific)-----------------
                pred = self.process_train_video(entry, frame_size, gt_annotation)
                # ----------------------------------------------------------------------

                if self._conf.use_partial_annotations:
                    losses = self._calculate_losses_for_partial_annotations(pred)
                else:
                    losses = self._calculate_losses_for_full_annotations(pred)


                self._optimizer.zero_grad()
                loss = sum(losses.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5, norm_type=2)
                self._optimizer.step()

                if self._conf.use_wandb:
                    wandb.log(losses)

                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))

                if counter % 1000 == 0 and counter >= 1000:
                    time_per_batch = (time.time() - start_time) / 1000
                    print(
                        "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, counter,
                                                                                      len(self._dataloader_train),
                                                                                      time_per_batch,
                                                                                      len(self._dataloader_train) * time_per_batch / 60))

                    mn = pd.concat(tr[-1000:], axis=1).mean(1)
                    print(mn)
                    start_time = time.time()
                counter += 1

            self._save_model(
                model=self._model,
                epoch=epoch,
                checkpoint_save_file_path=self._checkpoint_save_dir_path,
                checkpoint_name=self._checkpoint_name,
                method_name=self._conf.method_name
            )

            test_iter = iter(self._dataloader_test)
            self._model.eval()
            self._object_detector.is_train = False
            with torch.no_grad():
                for b in tqdm(range(len(self._dataloader_test))):
                    data = next(test_iter)
                    im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                    gt_annotation = self._test_dataset.gt_annotations[data[4]]
                    frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                    entry = self._object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)

                    # ----------------- Process the video (Method Specific)-----------------
                    pred = self.process_test_video(entry, frame_size, gt_annotation)
                    # ----------------------------------------------------------------------

                    self._evaluator.evaluate_scene_graph(gt_annotation, pred)
                print('-----------------------------------------------------------------------------------', flush=True)
            score = np.mean(self._evaluator.result_dict[self._conf.mode + "_recall"][20])
            self._evaluator.print_stats()
            self._evaluator.reset_result()
            self._scheduler.step(score)

    @abstractmethod
    def process_train_video(self, video, frame_size, gt_annotation) -> dict:
        pass

    @abstractmethod
    def process_test_video(self, video, frame_size, gt_annotation) -> dict:
        pass

    def init_method_training(self, is_curriculum=False):
        # 0. Initialize the config
        self._init_config()

        # 1. Initialize the dataset
        self.init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pre-trained models
        self.init_model()
        self._init_loss_functions()
        self._load_checkpoint()
        self._init_object_detector()
        self._init_optimizer()
        self._init_scheduler()

        # 4. Initialize model training
        print("-----------------------------------------------------")
        print(f"Initialized the training with the following settings:{is_curriculum}")
        print("-----------------------------------------------------")
        self._train_model(is_curriculum=is_curriculum)