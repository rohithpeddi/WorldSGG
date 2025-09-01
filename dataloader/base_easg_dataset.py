import os
import pickle
from abc import abstractmethod

import torch
from torch.utils.data import Dataset


class BaseEASGData(Dataset):

    def __init__(self, conf, split):
        self._conf = conf
        self._path_to_annotations = self._conf.path_to_annotations
        self._path_to_data = self._conf.path_to_data
        self._split = split

        print(f"[{self._conf.method_name}_{self._split}] LOADING ANNOTATION DATA")

        annotations_file_path = os.path.join(self._path_to_annotations, f'easg_{self._split}.pkl')
        with open(annotations_file_path, 'rb') as f:
            annotations = pickle.load(f)

        verbs_file_path = os.path.join(self._path_to_annotations, 'verbs.txt')
        with open(verbs_file_path) as f:
            verbs = [l.strip() for l in f.readlines()]

        objs_file_path = os.path.join(self._path_to_annotations, 'objects.txt')
        with open(objs_file_path) as f:
            objs = [l.strip() for l in f.readlines()]

        rels_file_path = os.path.join(self._path_to_annotations, 'relationships.txt')
        with open(rels_file_path) as f:
            rels = [l.strip() for l in f.readlines()]

        print(f"[{self._conf.method_name}_{self._split}] LOADING FEATURES DATA ")

        roi_feats_file_path = os.path.join(self._path_to_data, f'roi_feats_{self._split}.pkl')
        with open(roi_feats_file_path, 'rb') as f:
            roi_feats = pickle.load(f)

        clip_feats_file_path = os.path.join(self._path_to_data, f'verb_features.pt')
        clip_feats = torch.load(clip_feats_file_path)

        # Making these things accessible for functions in other methods.
        self.roi_feats = roi_feats
        self.clip_feats = clip_feats
        self.verbs = verbs
        self.objs = objs
        self.rels = rels
        self.annotations = annotations

        self._init_graphs()

    @abstractmethod
    def _init_graphs(self):
        pass

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
