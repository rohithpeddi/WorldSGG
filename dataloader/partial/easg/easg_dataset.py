import random

import torch

from dataloader.base_easg_dataset import BaseEASGData


class PartialEASG(BaseEASGData):

    def __init__(self, conf, split):
        super().__init__(conf, split)

    def _init_graphs(self):
        """
        graph:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['objs']: dict of obj_idx
                dict[obj_idx]: dict
                    dict['obj_feat']: 1024-D ROI feature vector
                    dict['rels_vec']: multi-hot vector of relationships

        graph_batch:
            dict['verb_idx']: index of its verb
            dict['clip_feat']: 2304-D clip-wise feature vector
            dict['obj_indices']: batched version of obj_idx
            dict['obj_feats']: batched version of obj_feat
            dict['rels_vecs']: batched version of rels_vec
            dict['triplets']: all the triplets consisting of (verb, obj, rel)
        """

        print(f"[{self._conf.method_name}_{self._split}] PREPARING GT GRAPH DATA AND FEATURES ")

        graphs = []
        for graph_uid in self.annotations:
            graph = {}
            for aid in self.annotations[graph_uid]['annotations']:
                for i, annt in enumerate(self.annotations[graph_uid]['annotations'][aid]):
                    verb_idx = self.verbs.index(annt['verb'])
                    if verb_idx not in graph:
                        graph[verb_idx] = {}
                        graph[verb_idx]['verb_idx'] = verb_idx
                        graph[verb_idx]['objs'] = {}

                    graph[verb_idx]['clip_feat'] = self.clip_feats[aid]

                    obj_idx = self.objs.index(annt['obj'])
                    if obj_idx not in graph[verb_idx]['objs']:
                        graph[verb_idx]['objs'][obj_idx] = {}
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.zeros((0, 1024), dtype=torch.float32)
                        graph[verb_idx]['objs'][obj_idx]['rels_vec'] = torch.zeros(len(self.rels), dtype=torch.float32)

                    rel_idx = self.rels.index(annt['rel'])
                    graph[verb_idx]['objs'][obj_idx]['rels_vec'][rel_idx] = 1

                    for frameType in self.roi_feats[graph_uid][aid][i]:
                        graph[verb_idx]['objs'][obj_idx]['obj_feat'] = torch.cat(
                            (graph[verb_idx]['objs'][obj_idx]['obj_feat'],
                             self.roi_feats[graph_uid][aid][i][frameType]), dim=0)

            for verb_idx in graph:
                for obj_idx in graph[verb_idx]['objs']:
                    graph[verb_idx]['objs'][obj_idx]['obj_feat'] = graph[verb_idx]['objs'][obj_idx]['obj_feat'].mean(
                        dim=0)

                graphs.append(graph[verb_idx])

        print(f"[{self._conf.method_name}_{self._split}] PREPARING GT GRAPH BATCH DATA ")

        # The dataset assumes the following structure:
        # For each object there is a single relation annotated and that corresponds to a single triplet
        if self._conf.use_partial_annotations:
            total_num_objs = 0
            for graph in graphs:
                total_num_objs += len(graph['objs'])
            total_num_obj_idx_changes = int(self._conf.partial_percentage * 0.01 * total_num_objs)

            # Step 1: Create a new graph_masks object
            graph_masks = []  # List of masks for each graph
            for graph in graphs:
                mask = {}  # Mask for objs in the current graph
                for obj_idx in graph['objs']:
                    mask[obj_idx] = False  # Initialize all masks to False
                graph_masks.append(mask)

            # Step 2: Create a graph_object_mask that masks the given total number among all graph objects
            # Collect all (graph index, object index) pairs
            all_graph_obj_indices = []
            for graph_idx, graph in enumerate(graphs):
                for obj_idx in graph['objs']:
                    all_graph_obj_indices.append((graph_idx, obj_idx))

            # Randomly select objects to change based on total_num_obj_idx_changes
            selected_indices = random.sample(all_graph_obj_indices, total_num_obj_idx_changes)

            # Update the masks for the selected objects
            for graph_idx, obj_idx in selected_indices:
                graph_masks[graph_idx][obj_idx] = True
        else:
            graph_masks = None  # No label noise; masks are not needed

        self.graphs = []
        for graph_idx, graph in enumerate(graphs):
            graph_batch = {}
            verb_idx = graph['verb_idx']
            graph_batch['verb_idx'] = torch.tensor([verb_idx], dtype=torch.long)
            graph_batch['clip_feat'] = graph['clip_feat']
            graph_batch['obj_indices'] = torch.zeros(0, dtype=torch.long)
            graph_batch['obj_feats'] = torch.zeros((0, 1024), dtype=torch.float32)
            graph_batch['rels_vecs'] = torch.zeros((0, len(self.rels)), dtype=torch.float32)
            graph_batch['triplets'] = torch.zeros((0, 3), dtype=torch.long)

            if graph_masks is not None:
                graph_batch["rel_mask"] = torch.zeros(0, dtype=torch.long)

            for obj_idx in graph['objs']:
                graph_batch['obj_indices'] = torch.cat(
                    (graph_batch['obj_indices'], torch.tensor([obj_idx], dtype=torch.long)), dim=0)
                graph_batch['obj_feats'] = torch.cat(
                    (graph_batch['obj_feats'], graph['objs'][obj_idx]['obj_feat'].unsqueeze(0)), dim=0)

                rels_vec = graph['objs'][obj_idx]['rels_vec']
                graph_batch['rels_vecs'] = torch.cat((graph_batch['rels_vecs'], rels_vec.unsqueeze(0)), dim=0)

                if graph_masks is not None:
                    if graph_masks[graph_idx][obj_idx]:
                        graph_batch["rel_mask"] = torch.cat(
                            (graph_batch["rel_mask"], torch.tensor([1], dtype=torch.long)), dim=0)
                    else:
                        graph_batch["rel_mask"] = torch.cat(
                            (graph_batch["rel_mask"], torch.tensor([0], dtype=torch.long)), dim=0)

                triplets = []
                for rel_idx in torch.where(rels_vec)[0]:
                    triplets.append((verb_idx, obj_idx, rel_idx.item()))
                graph_batch['triplets'] = torch.cat((graph_batch['triplets'], torch.tensor(triplets, dtype=torch.long)),
                                                    dim=0)
            self.graphs.append(graph_batch)

        print(f"[{self._conf.method_name}_{self._split}] Finished processing graph data ")

    def __getitem__(self, idx):
        return self.graphs[idx]
