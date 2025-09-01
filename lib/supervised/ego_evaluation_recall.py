from math import ceil

import torch


class BasicEgoActionSceneGraphEvaluator:

    def __init__(self, conf):
        self._conf = conf

        self.num_top_verb = 5
        self.num_top_rel_with = 1
        self.num_top_rel_no = 5
        self.num_rel = 13
        self.list_k = [10, 20, 50, 100]

        self.mode_setting_keys = [
            "predcls_with",
            "predcls_no",
            "sgcls_with",
            "sgcls_no",
            "easgcls_with",
            "easgcls_no"
        ]

        self.recall_result_dict = {}
        self.mean_recall_result_dict = {}

        self._init_recall_dicts()

    def _init_recall_dicts(self):
        for key in self.mode_setting_keys:
            self.recall_result_dict[key] = {k: [] for k in self.list_k}
            self.mean_recall_result_dict[key] = {k: [[] for _ in range(self.num_rel)] for k in self.list_k}

    def reset_result(self):
        for key in self.mode_setting_keys:
            for k in self.list_k:
                self.recall_result_dict[key][k] = []
                self.mean_recall_result_dict[key][k] = [[] for _ in range(self.num_rel)]

    def fetch_stats_json(self):
        recall_dict = {}
        mean_recall_dict = {}
        harmonic_mean_recall_dict = {}

        for key, mode_setting_dict in self.recall_result_dict.items():
            recall_dict[key] = {}
            for k, score_list in mode_setting_dict.items():
                recall_value = sum(score_list) / len(score_list) * 100
                recall_dict[key][k] = recall_value

        for key, mode_setting_dict in self.mean_recall_result_dict.items():
            mean_recall_dict[key] = {}
            for k, all_rel_score_list in mode_setting_dict.items():
                sum_recall = sum(
                    [sum(rel_score_list) / len(rel_score_list) if rel_score_list else 0.0 for rel_score_list in
                     all_rel_score_list])
                mean_recall_value = (sum_recall / float(self.num_rel)) * 100
                mean_recall_dict[key][k] = mean_recall_value

        for mode_key, mode_recall_dict in recall_dict.items():
            mode_mean_recall_dict = mean_recall_dict[mode_key]
            harmonic_mean_recall_dict[mode_key] = {}
            for k, mode_k_recall_value in mode_recall_dict.items():
                mode_k_mean_recall_value = mode_mean_recall_dict[k]
                harmonic_mean = 2 * mode_k_mean_recall_value * mode_k_recall_value / (
                        mode_k_mean_recall_value + mode_k_recall_value)
                harmonic_mean_recall_dict[mode_key][k] = harmonic_mean

        results_dict = {
            "recall": recall_dict,
            "mean_recall": mean_recall_dict,
            "harmonic_mean_recall": harmonic_mean_recall_dict
        }

        return results_dict

    def print_stats(self):
        results_dict = self.fetch_stats_json()
        print("-----------------------------------------------")
        print("Results:")
        for k, score_dict in results_dict["recall"].items():
            print(f"Recall@{k}:")
            for key, score in score_dict.items():
                print(f"  {key}: {score:.2f}")

        for k, score_dict in results_dict["mean_recall"].items():
            print(f"Mean Recall@{k}:")
            for key, score in score_dict.items():
                print(f"  {key}: {score:.2f}")

    @staticmethod
    def intersect_2d(out, gt):
        return (out[..., None] == gt.T[None, ...]).all(1)

    def evaluate_scene_graph(self, out_verb, out_objs, out_rels, gt_graph):
        scores_verb = out_verb[0].detach().cpu().softmax(dim=0)
        scores_objs = out_objs.detach().cpu().softmax(dim=1)
        scores_rels = out_rels.detach().cpu().sigmoid()

        if self._conf.random_guess:
            scores_verb = torch.rand(scores_verb.shape)
            scores_verb /= scores_verb.sum()
            scores_objs = torch.rand(scores_objs.shape)
            scores_objs /= scores_objs.sum()
            scores_rels = torch.rand(scores_rels.shape)

        verb_idx = gt_graph['verb_idx']
        obj_indices = gt_graph['obj_indices']
        rels_vecs = gt_graph['rels_vecs']
        triplets_gt = gt_graph['triplets']
        num_obj = obj_indices.shape[0]

        # make triplets for precls
        triplets_pred_with = []
        scores_pred_with = []
        triplets_pred_no = []
        scores_pred_no = []
        for obj_idx, scores_rel in zip(obj_indices, scores_rels):
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for ri in sorted_scores_rel[:self.num_top_rel_with]:
                triplets_pred_with.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_with.append(scores_rel[ri].item())

            for ri in sorted_scores_rel[:ceil(max(self.list_k) / num_obj)]:
                triplets_pred_no.append((verb_idx.item(), obj_idx.item(), ri.item()))
                scores_pred_no.append(scores_rel[ri].item())

        # make triplets for sgcls
        triplets_sg_with = []
        scores_sg_with = []
        triplets_sg_no = []
        scores_sg_no = []
        num_top_obj_with = ceil(max(self.list_k) / (self.num_top_rel_with * num_obj))
        num_top_obj_no = ceil(max(self.list_k) / (self.num_top_rel_no * num_obj))
        for scores_obj, scores_rel in zip(scores_objs, scores_rels):
            sorted_scores_obj = scores_obj.argsort(descending=True)
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for oi in sorted_scores_obj[:num_top_obj_with]:
                for ri in sorted_scores_rel[:self.num_top_rel_with]:
                    triplets_sg_with.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_with.append((scores_obj[oi] + scores_rel[ri]).item())
            for oi in sorted_scores_obj[:num_top_obj_no]:
                for ri in sorted_scores_rel[:self.num_top_rel_no]:
                    triplets_sg_no.append((verb_idx.item(), oi.item(), ri.item()))
                    scores_sg_no.append((scores_obj[oi] + scores_rel[ri]).item())

        # make triplets for easgcls
        triplets_easg_with = []
        scores_easg_with = []
        triplets_easg_no = []
        scores_easg_no = []
        num_top_obj_with = ceil(max(self.list_k) / (self.num_top_verb * self.num_top_rel_with * num_obj))
        num_top_obj_no = ceil(max(self.list_k) / (self.num_top_verb * self.num_top_rel_no * num_obj))
        for vi in scores_verb.argsort(descending=True)[:self.num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:self.num_top_rel_with]:
                        triplets_easg_with.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_with.append((scores_verb[vi] + scores_obj[oi] + scores_rel[ri]).item())
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:self.num_top_rel_no]:
                        triplets_easg_no.append((vi.item(), oi.item(), ri.item()))
                        scores_easg_no.append((scores_verb[vi] + scores_obj[oi] + scores_rel[ri]).item())

        triplets_pred_with = torch.tensor(triplets_pred_with, dtype=torch.long)
        triplets_pred_no = torch.tensor(triplets_pred_no, dtype=torch.long)
        triplets_sg_with = torch.tensor(triplets_sg_with, dtype=torch.long)
        triplets_sg_no = torch.tensor(triplets_sg_no, dtype=torch.long)
        triplets_easg_with = torch.tensor(triplets_easg_with, dtype=torch.long)
        triplets_easg_no = torch.tensor(triplets_easg_no, dtype=torch.long)

        # sort the triplets using the averaged scores
        triplets_pred_with = triplets_pred_with[torch.argsort(torch.tensor(scores_pred_with), descending=True)]
        triplets_pred_no = triplets_pred_no[torch.argsort(torch.tensor(scores_pred_no), descending=True)]
        triplets_sg_with = triplets_sg_with[torch.argsort(torch.tensor(scores_sg_with), descending=True)]
        triplets_sg_no = triplets_sg_no[torch.argsort(torch.tensor(scores_sg_no), descending=True)]
        triplets_easg_with = triplets_easg_with[torch.argsort(torch.tensor(scores_easg_with), descending=True)]
        triplets_easg_no = triplets_easg_no[torch.argsort(torch.tensor(scores_easg_no), descending=True)]

        num_gt = triplets_gt.shape[0]
        out_to_gt_pred_with = self.intersect_2d(triplets_gt, triplets_pred_with)
        out_to_gt_pred_no = self.intersect_2d(triplets_gt, triplets_pred_no)
        out_to_gt_sg_with = self.intersect_2d(triplets_gt, triplets_sg_with)
        out_to_gt_sg_no = self.intersect_2d(triplets_gt, triplets_sg_no)
        out_to_gt_easg_with = self.intersect_2d(triplets_gt, triplets_easg_with)
        out_to_gt_easg_no = self.intersect_2d(triplets_gt, triplets_easg_no)

        # Mask creation logic to identify which of the output triplets correspond to a particular relationship triplet
        # This mask is later added to the output to calculate relationship-wise mean recall
        num_gt = triplets_gt.shape[0]
        num_pred_with = triplets_pred_with.shape[0]
        num_pred_no = triplets_pred_no.shape[0]
        num_sg_with = triplets_sg_with.shape[0]
        num_sg_no = triplets_sg_no.shape[0]
        num_easg_with = triplets_easg_with.shape[0]
        num_easg_no = triplets_easg_no.shape[0]

        assert out_to_gt_pred_with.shape == (num_gt, num_pred_with)
        assert out_to_gt_pred_no.shape == (num_gt, num_pred_no)
        assert out_to_gt_sg_with.shape == (num_gt, num_sg_with)
        assert out_to_gt_sg_no.shape == (num_gt, num_sg_no)
        assert out_to_gt_easg_with.shape == (num_gt, num_easg_with)
        assert out_to_gt_easg_no.shape == (num_gt, num_easg_no)

        mask_out_to_gt_pred_with = torch.zeros((self.num_rel, num_gt, num_pred_with), dtype=torch.bool)
        mask_out_to_gt_pred_no = torch.zeros((self.num_rel, num_gt, num_pred_no), dtype=torch.bool)
        mask_out_to_gt_sg_with = torch.zeros((self.num_rel, num_gt, num_sg_with), dtype=torch.bool)
        mask_out_to_gt_sg_no = torch.zeros((self.num_rel, num_gt, num_sg_no), dtype=torch.bool)
        mask_out_to_gt_easg_with = torch.zeros((self.num_rel, num_gt, num_easg_with), dtype=torch.bool)
        mask_out_to_gt_easg_no = torch.zeros((self.num_rel, num_gt, num_easg_no), dtype=torch.bool)

        # Mask updation logic
        for rel_idx in range(self.num_rel):
            # Ground truth relationship mask
            gt_rel_mask = (triplets_gt[:, 2] == rel_idx)  # Shape: (num_gt,)

            # Predicted relationship masks for different cases
            pred_with_rel_mask = (triplets_pred_with[:, 2] == rel_idx)  # Shape: (num_pred_with,)
            pred_no_rel_mask = (triplets_pred_no[:, 2] == rel_idx)  # Shape: (num_pred_no,)
            sg_with_rel_mask = (triplets_sg_with[:, 2] == rel_idx)  # Shape: (num_sg_with,)
            sg_no_rel_mask = (triplets_sg_no[:, 2] == rel_idx)  # Shape: (num_sg_no,)
            easg_with_rel_mask = (triplets_easg_with[:, 2] == rel_idx)  # Shape: (num_easg_with,)
            easg_no_rel_mask = (triplets_easg_no[:, 2] == rel_idx)  # Shape: (num_easg_no,)

            # Compute masks via outer product to align ground truth and predictions
            mask_out_to_gt_pred_with[rel_idx] = gt_rel_mask[:, None] & pred_with_rel_mask[None, :]
            mask_out_to_gt_pred_no[rel_idx] = gt_rel_mask[:, None] & pred_no_rel_mask[None, :]
            mask_out_to_gt_sg_with[rel_idx] = gt_rel_mask[:, None] & sg_with_rel_mask[None, :]
            mask_out_to_gt_sg_no[rel_idx] = gt_rel_mask[:, None] & sg_no_rel_mask[None, :]
            mask_out_to_gt_easg_with[rel_idx] = gt_rel_mask[:, None] & easg_with_rel_mask[None, :]
            mask_out_to_gt_easg_no[rel_idx] = gt_rel_mask[:, None] & easg_no_rel_mask[None, :]

        for k in self.list_k:
            self.recall_result_dict["predcls_with"][k].append(
                out_to_gt_pred_with[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_result_dict["predcls_no"][k].append(out_to_gt_pred_no[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_result_dict["sgcls_with"][k].append(out_to_gt_sg_with[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_result_dict["sgcls_no"][k].append(out_to_gt_sg_no[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_result_dict["easgcls_with"][k].append(
                out_to_gt_easg_with[:, :k].any(dim=1).sum().item() / num_gt)
            self.recall_result_dict["easgcls_no"][k].append(out_to_gt_easg_no[:, :k].any(dim=1).sum().item() / num_gt)

            for rel_idx in range(self.num_rel):
                rel_out_to_gt_pred_with = out_to_gt_pred_with * mask_out_to_gt_pred_with[rel_idx]
                rel_out_to_gt_pred_no = out_to_gt_pred_no * mask_out_to_gt_pred_no[rel_idx]
                rel_out_to_gt_sg_with = out_to_gt_sg_with * mask_out_to_gt_sg_with[rel_idx]
                rel_out_to_gt_sg_no = out_to_gt_sg_no * mask_out_to_gt_sg_no[rel_idx]
                rel_out_to_gt_easg_with = out_to_gt_easg_with * mask_out_to_gt_easg_with[rel_idx]
                rel_out_to_gt_easg_no = out_to_gt_easg_no * mask_out_to_gt_easg_no[rel_idx]

                self.mean_recall_result_dict["predcls_with"][k][rel_idx].append(
                    rel_out_to_gt_pred_with[:, :k].any(dim=1).sum().item() / num_gt)
                self.mean_recall_result_dict["predcls_no"][k][rel_idx].append(
                    rel_out_to_gt_pred_no[:, :k].any(dim=1).sum().item() / num_gt)
                self.mean_recall_result_dict["sgcls_with"][k][rel_idx].append(
                    rel_out_to_gt_sg_with[:, :k].any(dim=1).sum().item() / num_gt)
                self.mean_recall_result_dict["sgcls_no"][k][rel_idx].append(
                    rel_out_to_gt_sg_no[:, :k].any(dim=1).sum().item() / num_gt)
                self.mean_recall_result_dict["easgcls_with"][k][rel_idx].append(
                    rel_out_to_gt_easg_with[:, :k].any(dim=1).sum().item() / num_gt)
                self.mean_recall_result_dict["easgcls_no"][k][rel_idx].append(
                    rel_out_to_gt_easg_no[:, :k].any(dim=1).sum().item() / num_gt)
