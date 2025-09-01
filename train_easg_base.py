import random
import time
from abc import abstractmethod

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from constants import EgoConstants as const
from dataloader.label_noise.easg.easg_dataset import LabelNoiseEASG
from dataloader.partial.easg.easg_dataset import PartialEASG
from dataloader.standard.easg.easg_dataset import StandardEASG
from easg_base import EASGBase


class TrainEASGBase(EASGBase):

    def __init__(self, conf):
        super().__init__(conf)

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
            self._train_dataset = PartialEASG(conf=self._conf, split=const.TRAIN)
        elif self._conf.use_label_noise:
            print("-----------------------------------------------------")
            print("Loading the dataset with label noise percentage:", self._conf.label_noise_percentage)
            print("-----------------------------------------------------")
            self._train_dataset = LabelNoiseEASG(conf=self._conf, split=const.TRAIN)
        else:
            print("-----------------------------------------------------")
            print("Loading the standard dataset")
            print("-----------------------------------------------------")
            self._train_dataset = StandardEASG(conf=self._conf, split=const.TRAIN)

        self._val_dataset = StandardEASG(conf=self._conf, split=const.VAL)
        self._dataloader_train = DataLoader(self._train_dataset, shuffle=True)
        self._dataloader_val = DataLoader(self._val_dataset, shuffle=False)

    @abstractmethod
    def init_model(self):
        pass

    def _init_loss_functions(self):
        self._criterion_rel = nn.BCEWithLogitsLoss()
        self._criterion = nn.CrossEntropyLoss()

    def _init_optimizer(self):
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._conf.lr)

    def _init_scheduler(self):
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, T_max=self._conf.sch_param)

    def _train_model(self):
        tr = []
        for epoch in range(self._conf.num_epoch):
            self._model.train()

            # train_iter = iter(self._dataloader_train)
            counter = 0
            start_time = time.time()

            list_index = list(range(len(self._train_dataset)))
            random.shuffle(list_index)
            # self._object_detector.is_train = True
            for train_idx in tqdm(range(len(list_index)), desc=f"Training Epoch {epoch}"):
                graph = self._train_dataset[list_index[train_idx]]
                self._optimizer.zero_grad()

                clip_feat = graph['clip_feat'].unsqueeze(0).to(self._device)
                obj_feats = graph['obj_feats'].to(self._device)
                out_verb, out_objs, out_rels = self._model(clip_feat, obj_feats)

                verb_idx = graph['verb_idx'].to(self._device)
                obj_indices = graph['obj_indices'].to(self._device)
                rels_vecs = graph['rels_vecs'].to(self._device)

                losses = {
                    'verb_loss': self._criterion(out_verb, verb_idx),
                    'obj_loss': self._criterion(out_objs, obj_indices)
                }

                if self._conf.use_partial_annotations:
                    rel_mask = graph['rel_mask'].to(self._device)
                    out_rels = out_rels[rel_mask == 1, :]
                    rels_vecs = rels_vecs[rel_mask == 1, :]
                    if not len(out_rels) == 0:
                        losses['rel_loss'] = self._criterion_rel(out_rels, rels_vecs)
                else:
                    losses['rel_loss'] = self._criterion_rel(out_rels, rels_vecs)

                loss = sum(losses.values())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=5, norm_type=2)
                self._optimizer.step()

                tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
                counter += 1

            # val_iter = iter(self._dataloader_val)
            list_index = list(range(len(self._val_dataset)))
            self._model.eval()
            # self._object_detector.is_train = False
            with torch.no_grad():
                for val_idx in tqdm(range(len(list_index)), desc=f"Validation Epoch {epoch}"):
                    graph = self._val_dataset[list_index[val_idx]]

                    clip_feat = graph['clip_feat'].unsqueeze(0).to(self._device)
                    obj_feats = graph['obj_feats'].to(self._device)
                    out_verb, out_objs, out_rels = self._model(clip_feat, obj_feats)

                    self._evaluator.evaluate_scene_graph(out_verb, out_objs, out_rels, graph)

            if epoch % 9 == 0:
                self._evaluator.print_stats()
                self._save_model(
                    model=self._model,
                    epoch=epoch,
                    checkpoint_save_file_path=self._checkpoint_save_dir_path,
                    checkpoint_name=self._checkpoint_name,
                    method_name=self._conf.method_name
                )

            self._evaluator.reset_result()
            self._scheduler.step()

    def init_method_training(self):
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
        # self._init_object_detector()
        self._init_optimizer()
        self._init_scheduler()

        # 4. Initialize model training
        self._train_model()
