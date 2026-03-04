import csv
import os
from functools import reduce

import numpy as np
import torch.nn as nn


# ============================================================================
# Self-contained replacements for lib_b utilities
# ============================================================================

def bbox_overlaps(boxes1, boxes2):
    """
    Compute IoU between two sets of bounding boxes.

    Args:
        boxes1: (M, 4) numpy array of [x1, y1, x2, y2] boxes.
        boxes2: (N, 4) numpy array of [x1, y1, x2, y2] boxes.

    Returns:
        overlaps: (M, N) numpy array of IoU values.
    """
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)  # each (M, 1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)  # each (N, 1)

    # Intersection
    xi1 = np.maximum(x11, x21.T)  # (M, N)
    yi1 = np.maximum(y11, y21.T)
    xi2 = np.minimum(x12, x22.T)
    yi2 = np.minimum(y12, y22.T)
    inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)

    # Union
    area1 = (x12 - x11) * (y12 - y11)  # (M, 1)
    area2 = (x22 - x21) * (y22 - y21)  # (N, 1)
    union = area1 + area2.T - inter

    return inter / np.maximum(union, 1e-8)


def intersect_2d(x1, x2):
    """
    Check row-wise equality between two 2D arrays.

    Args:
        x1: (M, D) numpy array.
        x2: (N, D) numpy array.

    Returns:
        (M, N) bool array where [i, j] = True iff x1[i] == x2[j] (all columns).
    """
    return (x1[:, None] == x2[None, :]).all(axis=2)


def argsort_desc(scores):
    """
    Return (row, col) indices that sort a 2D array in descending order.

    Args:
        scores: (M, N) numpy array.

    Returns:
        (K, 2) array of [row, col] indices sorted by descending score.
    """
    flat_order = np.argsort(-scores.ravel())
    rows = flat_order // scores.shape[1]
    cols = flat_order % scores.shape[1]
    return np.column_stack((rows, cols))


class BasicSceneGraphEvaluator:
    def __init__(
            self,
            mode,
            AG_object_classes,
            AG_all_predicates,
            AG_attention_predicates,
            AG_spatial_predicates,
            AG_contacting_predicates,
            iou_threshold=0.5,
            save_file="tmp",
            constraint=False,
            semi_threshold=None
    ):
        self.result_dict = {}
        self.mode = mode
        self.num_rel = len(AG_all_predicates)
        self.result_dict[self.mode + '_recall'] = {
            10: [], 20: [], 50: [], 100: []
        }
        self.result_dict[self.mode + '_mean_recall_collect'] = {
            k: [[] for _ in range(self.num_rel)] for k in (10, 20, 50, 100)
        }
        
        self.constraint = constraint  # semi constraint if True
        self.iou_threshold = iou_threshold
        self.AG_object_classes = AG_object_classes
        self.AG_all_predicates = AG_all_predicates
        self.AG_attention_predicates = AG_attention_predicates
        self.AG_spatial_predicates = AG_spatial_predicates
        self.AG_contacting_predicates = AG_contacting_predicates
        self.semi_threshold = semi_threshold
        self.save_file = save_file
    
    def reset_result(self):
        self.result_dict[self.mode + '_recall'] = {10: [], 20: [], 50: [], 100: []}
        self.result_dict[self.mode + '_mean_recall_collect'] = {
            k: [[] for _ in range(self.num_rel)] for k in (10, 20, 50, 100)
        }
    
    def fetch_stats_json(self, save_file_path=None):
        recall_dict = {}
        mean_recall_dict = {}
        harmonic_mean_recall_dict = {}
        
        for k, v in self.result_dict[self.mode + '_recall'].items():
            recall_value = np.mean(v)
            recall_dict[k] = recall_value
        
        for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
            sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
            mean_recall_value = sum_recall / float(self.num_rel)
            mean_recall_dict[k] = mean_recall_value
        
        for k, recall_value in recall_dict.items():
            mean_recall_value = mean_recall_dict[k]
            harmonic_mean = 2 * mean_recall_value * recall_value / (mean_recall_value + recall_value)
            harmonic_mean_recall_dict[k] = harmonic_mean

        if save_file_path is not None:
            # Save the results corresponding to mean recall collection for each k in to the file
            # Construct a matrix - k x num_rel
            recall_matrix = np.zeros((len(self.result_dict[self.mode + '_mean_recall_collect']), self.num_rel))
            for i, (k, v) in enumerate(self.result_dict[self.mode + '_mean_recall_collect'].items()):
                recall_matrix[i, :] = np.array([np.mean(v[id]) for id in range(len(v))])

            # Print the recall matrix to the csv file for further analysis
            with open(save_file_path, "w") as stats_file:
                writer = csv.writer(stats_file, quoting=csv.QUOTE_NONNUMERIC)
                for i in range(len(recall_matrix)):
                    writer.writerow(recall_matrix[i])

        results = {
            "recall": recall_dict,
            "mean_recall": mean_recall_dict,
            "harmonic_mean_recall": harmonic_mean_recall_dict
        }
        
        return results
    
    def print_stats(self):
        def print_and_write(message):
            print(message)
            stats_file.write(message + '\n')
        
        with open(self.save_file, "a") as stats_file:
            header = f'======================{self.mode}======================'
            print_and_write(header)
            
            recall_dict = {}
            mean_recall_dict = {}
            harmonic_mean_recall_dict = {}
            
            for k, v in self.result_dict[self.mode + '_recall'].items():
                recall_value = np.mean(v)
                recall_dict[k] = recall_value
                print_and_write(f'R@{k}: {recall_value:.6f}')
            
            for k, v in self.result_dict[self.mode + '_mean_recall_collect'].items():
                sum_recall = np.sum([np.mean(vi) if vi else 0.0 for vi in v])
                mean_recall_value = sum_recall / float(self.num_rel)
                mean_recall_dict[k] = mean_recall_value
                print_and_write(f'mR@{k}: {mean_recall_value:.6f}')
            
            for k, recall_value in recall_dict.items():
                mean_recall_value = mean_recall_dict[k]
                harmonic_mean = 2 * mean_recall_value * recall_value / (mean_recall_value + recall_value)
                harmonic_mean_recall_dict[k] = harmonic_mean
                print_and_write(f'hR@{k}: {harmonic_mean:.6f}')
    
    def fetch_pred_tuples(self, gt, pred):
        idx_pred_triplets_map = {}
        # attention_distribution is already softmax'd from the model
        for idx, frame_gt in enumerate(gt):
            frame_idx = frame_gt[0]['frame'].split('/')[-1].split('.')[0]
            # first part for attention and contact, second for spatial
            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),  # attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:, ::-1],  # spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()),
                                    axis=0)  # contacting
            
            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                 np.zeros(
                     [pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])),
                axis=1)
            pred_scores_3 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                 pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)
            
            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            
            pred_rel_inds = pred_entry['pred_rel_inds']
            rel_scores = pred_entry['rel_scores']
            
            pred_boxes = pred_entry['pred_boxes'].astype(float)
            pred_classes = pred_entry['pred_classes']
            obj_scores = pred_entry['obj_scores']
            
            if self.constraint == 'no':
                obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
                overall_scores = obj_scores_per_rel[:, None] * rel_scores
                score_inds = argsort_desc(overall_scores)[:100]
                pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
                predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
            else:
                pred_rels = np.column_stack(
                    (pred_rel_inds, rel_scores.argmax(1)))  # 1+  dont add 1 because no dummy 'no relations'
                predicate_scores = rel_scores.max(1)
            
            if pred_rels.size == 0:
                continue
            
            pred_triplets, pred_triplet_boxes, relation_scores = \
                _triplet(pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes,
                         predicate_scores, obj_scores)
            
            sorted_scores = relation_scores.prod(1)
            pred_triplets = pred_triplets[sorted_scores.argsort()[::-1], :]
            
            # Subject Object Relationship Class
            idx_pred_triplets_map[frame_idx] = pred_triplets[:, [0, 2, 1]]
        
        return idx_pred_triplets_map
    
    def evaluate_scene_graph(self, gt, pred):
        """collect the ground truth and prediction"""
        # attention_distribution is already softmax'd from the model
        for idx, frame_gt in enumerate(gt):
            # generate the ground truth
            gt_boxes = np.zeros([len(frame_gt), 4])  # now there is no person box! we assume that person box index == 0
            gt_classes = np.zeros(len(frame_gt))
            gt_relations = []
            human_idx = 0
            gt_classes[human_idx] = 1
            gt_boxes[human_idx] = frame_gt[0]['person_bbox']
            for m, n in enumerate(frame_gt[1:]):
                # each pair
                gt_boxes[m + 1, :] = n['bbox']
                gt_classes[m + 1] = n['class']
                gt_relations.append([human_idx, m + 1, self.AG_all_predicates.index(self.AG_attention_predicates[n[
                    'attention_relationship']])])  # for attention triplet <human-object-predicate>_
                # spatial and contacting relationship could be multiple
                for spatial in n['spatial_relationship'].numpy().tolist():
                    gt_relations.append([m + 1, human_idx, self.AG_all_predicates.index(
                        self.AG_spatial_predicates[spatial])])  # for spatial triplet <object-human-predicate>
                for contact in n['contacting_relationship'].numpy().tolist():
                    gt_relations.append([human_idx, m + 1, self.AG_all_predicates.index(
                        self.AG_contacting_predicates[contact])])  # for contact triplet <human-object-predicate>
            
            gt_entry = {
                'gt_classes': gt_classes,
                'gt_relations': np.array(gt_relations),
                'gt_boxes': gt_boxes,
            }
            
            # first part for attention and contact, second for spatial
            rels_i = np.concatenate((pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy(),  # attention
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()[:, ::-1],  # spatial
                                     pred['pair_idx'][pred['im_idx'] == idx].cpu().clone().numpy()),
                                    axis=0)  # contacting
            
            pred_scores_1 = np.concatenate((pred['attention_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['spatial_distribution'].shape[1]]),
                                            np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0],
                                                      pred['contacting_distribution'].shape[1]])), axis=1)
            pred_scores_2 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 pred['spatial_distribution'][pred['im_idx'] == idx].cpu().numpy(),
                 np.zeros(
                     [pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['contacting_distribution'].shape[1]])),
                axis=1)
            pred_scores_3 = np.concatenate(
                (np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['attention_distribution'].shape[1]]),
                 np.zeros([pred['pair_idx'][pred['im_idx'] == idx].shape[0], pred['spatial_distribution'].shape[1]]),
                 pred['contacting_distribution'][pred['im_idx'] == idx].cpu().numpy()), axis=1)
            
            if self.mode == 'predcls':
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            else:
                pred_entry = {
                    'pred_boxes': pred['boxes'][:, 1:].cpu().clone().numpy(),
                    'pred_classes': pred['pred_labels'].cpu().clone().numpy(),
                    'pred_rel_inds': rels_i,
                    'obj_scores': pred['pred_scores'].cpu().clone().numpy(),
                    'rel_scores': np.concatenate((pred_scores_1, pred_scores_2, pred_scores_3), axis=0)
                }
            
            evaluate_from_dict(gt_entry, pred_entry, self.mode, self.result_dict,
                               iou_thresh=self.iou_threshold, method=self.constraint, threshold=self.semi_threshold,
                               num_rel=self.num_rel)


def evaluate_from_dict(gt_entry, pred_entry, mode, result_dict, method=None, threshold=0.9, num_rel=26, **kwargs):
    """
    Shortcut to doing evaluate_recall from dict
    :param mode:
    :param num_rel:
    :param threshold:
    :param method:
    :param gt_entry: Dictionary containing gt_relations, gt_boxes, gt_classes
    :param pred_entry: Dictionary containing pred_rels, pred_boxes (if detection), pred_classes
    :param result_dict:
    :param kwargs:
    :return:
    """
    gt_rels = gt_entry['gt_relations']
    gt_boxes = gt_entry['gt_boxes'].astype(float)
    gt_classes = gt_entry['gt_classes']
    
    pred_rel_inds = pred_entry['pred_rel_inds']
    rel_scores = pred_entry['rel_scores']
    
    pred_boxes = pred_entry['pred_boxes'].astype(float)
    pred_classes = pred_entry['pred_classes']
    obj_scores = pred_entry['obj_scores']
    
    if method == 'semi':
        pred_rels = []
        predicate_scores = []
        for i, j in enumerate(pred_rel_inds):
            if rel_scores[i, 0] + rel_scores[i, 1] > 0:
                # this is the attention distribution
                pred_rels.append(np.append(j, rel_scores[i].argmax()))
                predicate_scores.append(rel_scores[i].max())
            elif rel_scores[i, 3] + rel_scores[i, 4] > 0:
                # this is the spatial distribution
                for k in np.where(rel_scores[i] > threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])
            elif rel_scores[i, 9] + rel_scores[i, 10] > 0:
                # this is the contact distribution
                for k in np.where(rel_scores[i] > threshold)[0]:
                    pred_rels.append(np.append(j, k))
                    predicate_scores.append(rel_scores[i, k])
        pred_rels = np.array(pred_rels)
        predicate_scores = np.array(predicate_scores)
    elif method == 'no':
        obj_scores_per_rel = obj_scores[pred_rel_inds].prod(1)
        overall_scores = obj_scores_per_rel[:, None] * rel_scores
        score_inds = argsort_desc(overall_scores)[:100]
        pred_rels = np.column_stack((pred_rel_inds[score_inds[:, 0]], score_inds[:, 1]))
        predicate_scores = rel_scores[score_inds[:, 0], score_inds[:, 1]]
    else:
        pred_rels = np.column_stack(
            (pred_rel_inds, rel_scores.argmax(1)))  # 1+  dont add 1 because no dummy 'no relations'
        predicate_scores = rel_scores.max(1)
    
    pred_to_gt, pred_5ples, rel_scores = evaluate_recall(
        gt_rels, gt_boxes, gt_classes,
        pred_rels, pred_boxes, pred_classes,
        predicate_scores, obj_scores, phrdet=mode == 'phrdet', **kwargs)
    
    for k in result_dict[mode + '_recall']:
        match = reduce(np.union1d, pred_to_gt[:k])
        recall_hit = [0] * num_rel
        recall_count = [0] * num_rel
        
        for idx in range(gt_rels.shape[0]):
            local_label = gt_rels[idx, 2]
            recall_count[int(local_label)] += 1
        
        for idx in range(len(match)):
            local_label = gt_rels[int(match[idx]), 2]
            recall_hit[int(local_label)] += 1
        
        for n in range(num_rel):
            if recall_count[n] > 0:
                result_dict[mode + '_mean_recall_collect'][k][n].append(float(recall_hit[n] / recall_count[n]))
        
        rec_i = float(len(match)) / float(gt_rels.shape[0])
        result_dict[mode + '_recall'][k].append(rec_i)
    return pred_to_gt, pred_5ples, rel_scores


###########################
def evaluate_recall(
        gt_rels,
        gt_boxes,
        gt_classes,
        pred_rels,
        pred_boxes,
        pred_classes,
        rel_scores=None,
        cls_scores=None,
        iou_thresh=0.5,
        phrdet=False
):
    """
    Evaluates the recall
    :param cls_scores:
    :param rel_scores:
    :param iou_thresh:
    :param phrdet:
    :param gt_rels: [#gt_rel, 3] array of GT relations
    :param gt_boxes: [#gt_box, 4] array of GT boxes
    :param gt_classes: [#gt_box] array of GT classes
    :param pred_rels: [#pred_rel, 3] array of pred rels. Assumed these are in sorted order
                      and refer to IDs in pred classes / pred boxes
                      (id0, id1, rel)
    :param pred_boxes:  [#pred_box, 4] array of pred boxes
    :param pred_classes: [#pred_box] array of predicted classes for these boxes
    :return: pred_to_gt: Matching from predicate to GT
             pred_5ples: the predicted (id0, id1, cls0, cls1, rel)
             rel_scores: [cls_0score, cls1_score, relscore]
                   """
    if pred_rels.size == 0:
        return [[]], np.zeros((0, 5)), np.zeros(0)
    
    num_gt_boxes = gt_boxes.shape[0]
    num_gt_relations = gt_rels.shape[0]
    assert num_gt_relations != 0
    
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels[:, 2],
                                                gt_rels[:, :2],
                                                gt_classes,
                                                gt_boxes)
    num_boxes = pred_boxes.shape[0]
    
    try:
        assert pred_rels[:, :2].max() < pred_classes.shape[0]
    except AssertionError:
        print("assert error ")
    # pdb.set_trace()
    
    # Exclude self rels
    # assert np.all(pred_rels[:,0] != pred_rels[:,ĺeftright])
    # assert np.all(pred_rels[:,2] > 0)
    
    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(pred_rels[:, 2], pred_rels[:, :2], pred_classes, pred_boxes,
                 rel_scores, cls_scores)
    
    # sorted_scores = relation_scores.prod(1)
    # pred_triplets = pred_triplets[sorted_scores.argsort()[::-1], :]
    # pred_triplet_boxes = pred_triplet_boxes[sorted_scores.argsort()[::-1], :]
    # relation_scores = relation_scores[sorted_scores.argsort()[::-1], :]
    # scores_overall = relation_scores.prod(1)
    
    # if not np.all(scores_overall[1:] <= scores_overall[:-1] + 1e-5):
    #     print("Somehow the relations weren't sorted properly: \n{}".format(scores_overall))
    # pdb.set_trace()
    # raise ValueError("Somehow the relations weren't sorted properly")
    
    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_triplet_boxes,
        pred_triplet_boxes,
        iou_thresh,
        phrdet=phrdet,
    )
    
    # Contains some extra stuff for visualization. Not needed.
    pred_5ples = np.column_stack((
        pred_rels[:, :2],
        pred_triplets[:, [0, 2, 1]],
    ))
    
    return pred_to_gt, pred_5ples, relation_scores


def _triplet(
        predicates,
        relations,
        classes,
        boxes,
        predicate_scores=None,
        class_scores=None
):
    """
    format predictions into triplets
    :param predicates: A 1d numpy array of num_boxes*(num_boxes-ĺeftright) predicates, corresponding to
                       each pair of possibilities
    :param relations: A (num_boxes*(num_boxes-ĺeftright), 2.0) array, where each row represents the boxes
                      in that relation
    :param classes: A (num_boxes) array of the classes for each thing.
    :param boxes: A (num_boxes,4) array of the bounding boxes for everything.
    :param predicate_scores: A (num_boxes*(num_boxes-ĺeftright)) array of the scores for each predicate
    :param class_scores: A (num_boxes) array of the likelihood for each object.
    :return: Triplets: (num_relations, 3) array of class, relation, class
             Triplet boxes: (num_relation, 8) array of boxes for the parts
             Triplet scores: num_relation array of the scores overall for the triplets
    """
    assert (predicates.shape[0] == relations.shape[0])
    
    sub_ob_classes = classes[relations[:, :2]]
    triplets = np.column_stack((sub_ob_classes[:, 0], predicates, sub_ob_classes[:, 1]))
    triplet_boxes = np.column_stack((boxes[relations[:, 0]], boxes[relations[:, 1]]))
    
    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[relations[:, 0]],
            class_scores[relations[:, 1]],
            predicate_scores,
        ))
    
    return triplets, triplet_boxes, triplet_scores


def _compute_pred_matches(
        gt_triplets,
        pred_triplets,
        gt_boxes,
        pred_boxes,
        iou_thresh,
        phrdet=False
):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    :param gt_triplets:
    :param pred_triplets:
    :param gt_boxes:
    :param pred_boxes:
    :param iou_thresh:
    :return:
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)
            
            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:, :2], box_union.max(1)[:, 2:]), 1)
            
            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thresh
        else:
            sub_iou = bbox_overlaps(gt_box[None, :4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None, 4:], boxes[:, 4:])[0]
            
            inds = (sub_iou >= iou_thresh) & (obj_iou >= iou_thresh)
        
        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt


# ============================================================================
# 3D IoU Utility
# ============================================================================

def bbox_overlaps_3d(corners1, corners2):
    """
    Approximate 3D IoU between two sets of oriented bounding boxes.

    Uses axis-aligned bounding box (AABB) approximation of the 8-corner
    OBBs for efficiency.

    Args:
        corners1: (M, 8, 3) numpy array of 3D OBB corners.
        corners2: (N, 8, 3) numpy array of 3D OBB corners.

    Returns:
        overlaps: (M, N) numpy array of approximate IoU values.
    """
    # Convert OBB corners to AABB: min/max along each axis
    min1 = corners1.min(axis=1)  # (M, 3)
    max1 = corners1.max(axis=1)  # (M, 3)
    min2 = corners2.min(axis=1)  # (N, 3)
    max2 = corners2.max(axis=1)  # (N, 3)

    # Intersection
    inter_min = np.maximum(min1[:, None, :], min2[None, :, :])  # (M, N, 3)
    inter_max = np.minimum(max1[:, None, :], max2[None, :, :])  # (M, N, 3)
    inter_dims = np.maximum(inter_max - inter_min, 0)           # (M, N, 3)
    inter_vol = inter_dims.prod(axis=2)                         # (M, N)

    # Volumes
    dims1 = max1 - min1  # (M, 3)
    dims2 = max2 - min2  # (N, 3)
    vol1 = dims1.prod(axis=1)[:, None]  # (M, 1)
    vol2 = dims2.prod(axis=1)[None, :]  # (1, N)

    union_vol = vol1 + vol2 - inter_vol
    return inter_vol / np.maximum(union_vol, 1e-8)


# ============================================================================
# WSGG Adapter: convert padded-tensor format → evaluate_from_dict format
# ============================================================================

# Short-form → full AG name mapping (for GT label denormalization)
_LABEL_DENORMALIZE = {
    "closet": "closet/cabinet",
    "cup": "cup/glass/bottle",
    "paper": "paper/notebook",
    "sofa": "sofa/couch",
    "phone": "phone/camera",
}


def _denormalize_label(label):
    """Short-form label → full AG name. E.g. 'closet' → 'closet/cabinet'."""
    return _LABEL_DENORMALIZE.get(label, label)


def evaluate_wsgg_video(
    gt_annot,
    pred_pkl,
    evaluator,
    mode="predcls",
    verbose=True,
):
    """
    Evaluate a single video's WSGG predictions against GT annotations.

    Converts the padded-tensor / PKL format into gt_entry / pred_entry dicts
    and feeds them to the existing ``evaluate_from_dict``.

    Args:
        gt_annot: GT annotation dict for one frame (from combine_world4d_relationships
                  output). Must contain ``person_info`` and ``object_info_list``.
                  Uses unnormalized (full AG) class names for GT labels.
        pred_pkl: Prediction dict for one video, as saved by ``dump_predictions.py``.
        evaluator: A ``BasicSceneGraphEvaluator`` instance.
        mode: "predcls" or "sgdet".
        verbose: If True, log detailed diagnostics.
    """
    import logging
    _log = logging.getLogger("evaluate_wsgg_video")

    video_id = pred_pkl.get("video_id", "?")

    person_info = gt_annot.get("person_info", {})
    object_info_list = gt_annot.get("object_info_list", [])

    if not object_info_list:
        if verbose:
            _log.info(f"[{video_id}] SKIP: empty object_info_list")
        return

    # ---- Build GT entry ----
    n_objects = len(object_info_list) + 1  # +1 for person (index 0)
    gt_boxes = np.zeros((n_objects, 4), dtype=np.float32)
    gt_classes = np.zeros(n_objects, dtype=np.int64)

    # Person is always index 0, class 1 ("person" in OBJECT_CLASSES)
    person_bbox = person_info.get("bbox_2d", None)
    if person_bbox is not None:
        gt_boxes[0] = np.asarray(person_bbox, dtype=np.float32)[:4]
    gt_classes[0] = 1  # person

    gt_relations = []
    human_idx = 0

    # Object class name → index lookup (use full/unnormalized AG names)
    from dataloader.world_ag_dataset import NAME_TO_IDX

    if verbose:
        _log.info(f"[{video_id}] GT: {n_objects} objects (1 person + {len(object_info_list)} objs)")
        _log.info(f"[{video_id}]   person bbox: {person_bbox}")

    for m, obj in enumerate(object_info_list):
        obj_idx = m + 1

        # Bbox
        bbox_2d = obj.get("bbox_2d", None)
        if bbox_2d is not None:
            gt_boxes[obj_idx] = np.asarray(bbox_2d, dtype=np.float32)[:4]

        # Class — use full AG name (unnormalized)
        cls_full = obj.get("class", _denormalize_label(obj.get("label", "")))
        cls_short = obj.get("label", "")
        cls_id = NAME_TO_IDX.get(cls_full, NAME_TO_IDX.get(cls_short, 0))
        gt_classes[obj_idx] = cls_id

        if verbose:
            _log.info(
                f"[{video_id}]   obj[{obj_idx}]: class='{cls_full}' "
                f"(short='{cls_short}', id={cls_id}), "
                f"bbox={bbox_2d is not None}, visible={obj.get('visible', '?')}"
            )

        # Attention relationship (single-label)
        att_rels = obj.get("attention_relationship", [])
        for att_str in att_rels:
            if att_str in evaluator.AG_attention_predicates:
                att_idx = evaluator.AG_attention_predicates.index(att_str)
                global_idx = evaluator.AG_all_predicates.index(
                    evaluator.AG_attention_predicates[att_idx]
                )
                gt_relations.append([human_idx, obj_idx, global_idx])
                if verbose:
                    _log.info(
                        f"[{video_id}]     att: '{att_str}' → "
                        f"local={att_idx}, global={global_idx}, "
                        f"triple=[{human_idx}, {obj_idx}, {global_idx}]"
                    )
            else:
                if verbose:
                    _log.warning(
                        f"[{video_id}]     att: '{att_str}' NOT IN predicates "
                        f"{evaluator.AG_attention_predicates}"
                    )
            break  # attention is single-label

        # Spatial relationships (multi-label)
        spa_rels = obj.get("spatial_relationship", [])
        for spa_str in spa_rels:
            if spa_str in evaluator.AG_spatial_predicates:
                spa_idx = evaluator.AG_spatial_predicates.index(spa_str)
                global_idx = evaluator.AG_all_predicates.index(
                    evaluator.AG_spatial_predicates[spa_idx]
                )
                gt_relations.append([obj_idx, human_idx, global_idx])
                if verbose:
                    _log.info(
                        f"[{video_id}]     spa: '{spa_str}' → "
                        f"local={spa_idx}, global={global_idx}, "
                        f"triple=[{obj_idx}, {human_idx}, {global_idx}]"
                    )
            else:
                if verbose:
                    _log.warning(
                        f"[{video_id}]     spa: '{spa_str}' NOT IN predicates "
                        f"{evaluator.AG_spatial_predicates}"
                    )

        # Contacting relationships (multi-label)
        con_rels = obj.get("contacting_relationship", [])
        for con_str in con_rels:
            if con_str in evaluator.AG_contacting_predicates:
                con_idx = evaluator.AG_contacting_predicates.index(con_str)
                global_idx = evaluator.AG_all_predicates.index(
                    evaluator.AG_contacting_predicates[con_idx]
                )
                gt_relations.append([human_idx, obj_idx, global_idx])
                if verbose:
                    _log.info(
                        f"[{video_id}]     con: '{con_str}' → "
                        f"local={con_idx}, global={global_idx}, "
                        f"triple=[{human_idx}, {obj_idx}, {global_idx}]"
                    )
            else:
                if verbose:
                    _log.warning(
                        f"[{video_id}]     con: '{con_str}' NOT IN predicates "
                        f"{evaluator.AG_contacting_predicates}"
                    )

    if not gt_relations:
        if verbose:
            _log.info(f"[{video_id}] SKIP: no GT relations found")
        return

    gt_entry = {
        "gt_classes": gt_classes,
        "gt_relations": np.array(gt_relations, dtype=np.int64),
        "gt_boxes": gt_boxes,
    }

    if verbose:
        _log.info(
            f"[{video_id}] GT entry: {len(gt_relations)} relations, "
            f"classes={gt_classes.tolist()}, "
            f"boxes_nonzero={np.any(gt_boxes != 0, axis=1).sum()}"
        )

    # ---- Build pred entry ----
    att_dist = pred_pkl["attention_distribution"]   # (K_valid, 3)
    spa_dist = pred_pkl["spatial_distribution"]     # (K_valid, 6)
    con_dist = pred_pkl["contacting_distribution"]  # (K_valid, 17)
    person_idx = pred_pkl["person_idx"]             # (K_valid,)
    object_idx = pred_pkl["object_idx"]             # (K_valid,)

    if isinstance(att_dist, np.ndarray):
        pass  # already numpy
    else:
        att_dist = att_dist.cpu().numpy()
        spa_dist = spa_dist.cpu().numpy()
        con_dist = con_dist.cpu().numpy()
        person_idx = person_idx.cpu().numpy()
        object_idx = object_idx.cpu().numpy()

    K = att_dist.shape[0]
    if K == 0:
        if verbose:
            _log.info(f"[{video_id}] SKIP: K=0 valid pairs in predictions")
        return

    if verbose:
        _log.info(
            f"[{video_id}] Pred: K={K} valid pairs, "
            f"person_idx={person_idx.tolist()}, "
            f"object_idx={object_idx.tolist()}"
        )
        _log.info(
            f"[{video_id}]   att_dist shape={att_dist.shape}, "
            f"range=[{att_dist.min():.4f}, {att_dist.max():.4f}], "
            f"argmax={att_dist.argmax(axis=1).tolist()}"
        )
        _log.info(
            f"[{video_id}]   spa_dist shape={spa_dist.shape}, "
            f"range=[{spa_dist.min():.4f}, {spa_dist.max():.4f}]"
        )
        _log.info(
            f"[{video_id}]   con_dist shape={con_dist.shape}, "
            f"range=[{con_dist.min():.4f}, {con_dist.max():.4f}]"
        )

    n_att = att_dist.shape[1]
    n_spa = spa_dist.shape[1]
    n_con = con_dist.shape[1]

    # Build pair indices: attention (h→o), spatial (o→h), contacting (h→o)
    pair_idx_att = np.stack([person_idx, object_idx], axis=1)          # (K, 2)
    pair_idx_spa = np.stack([object_idx, person_idx], axis=1)          # (K, 2) reversed
    pair_idx_con = np.stack([person_idx, object_idx], axis=1)          # (K, 2)
    rels_i = np.concatenate([pair_idx_att, pair_idx_spa, pair_idx_con], axis=0)  # (3K, 2)

    # Construct block-diagonal score matrix: (3K, n_att+n_spa+n_con)
    total_preds = n_att + n_spa + n_con
    zeros_spa_con = np.zeros((K, n_spa + n_con))
    zeros_att_con = np.zeros((K, n_att + n_con))
    zeros_att_spa = np.zeros((K, n_att + n_spa))

    scores_att = np.concatenate([att_dist, zeros_spa_con], axis=1)   # (K, total)
    scores_spa = np.concatenate([np.zeros((K, n_att)), spa_dist,
                                  np.zeros((K, n_con))], axis=1)     # (K, total)
    scores_con = np.concatenate([zeros_att_spa, con_dist], axis=1)   # (K, total)

    rel_scores = np.concatenate([scores_att, scores_spa, scores_con], axis=0)  # (3K, total)

    # Boxes and classes
    if mode == "predcls":
        pred_boxes = gt_boxes.copy()
        pred_classes = gt_classes.copy()
        obj_scores = np.ones(n_objects, dtype=np.float32)
    else:
        # SGDet: use predicted boxes, labels, scores
        pred_bboxes_2d = pred_pkl.get("bboxes_2d", gt_boxes)
        if not isinstance(pred_bboxes_2d, np.ndarray):
            pred_bboxes_2d = pred_bboxes_2d.cpu().numpy()
        pred_boxes = pred_bboxes_2d.astype(np.float32)

        pred_labels = pred_pkl.get("pred_labels", gt_classes)
        if not isinstance(pred_labels, np.ndarray):
            pred_labels = pred_labels.cpu().numpy()
        pred_classes = pred_labels

        pred_det_scores = pred_pkl.get("pred_scores", np.ones(len(pred_classes)))
        if not isinstance(pred_det_scores, np.ndarray):
            pred_det_scores = pred_det_scores.cpu().numpy()
        obj_scores = pred_det_scores

        # Coupled 2D+3D IoU object scores (if 3D boxes available)
        pred_3d = pred_pkl.get("bboxes_3d", None)
        gt_3d_list = []
        for i, obj in enumerate(object_info_list):
            c = obj.get("corners_final", None)
            if c is not None:
                gt_3d_list.append(np.asarray(c, dtype=np.float32))
            else:
                gt_3d_list.append(np.zeros((8, 3), dtype=np.float32))
        # Add person 3D
        person_3d = person_info.get("corners_final", None)
        if person_3d is not None:
            person_3d = np.asarray(person_3d, dtype=np.float32)
        else:
            person_3d = np.zeros((8, 3), dtype=np.float32)
        gt_3d_all = np.stack([person_3d] + gt_3d_list)  # (N, 8, 3)

        if pred_3d is not None:
            if not isinstance(pred_3d, np.ndarray):
                pred_3d = pred_3d.cpu().numpy()
            # Compute per-object coupled 2D*3D IoU
            iou_2d_diag = np.array([
                bbox_overlaps(gt_boxes[i:i+1], pred_boxes[i:i+1])[0, 0]
                if i < pred_boxes.shape[0] else 0.0
                for i in range(n_objects)
            ])
            iou_3d_diag = np.array([
                bbox_overlaps_3d(gt_3d_all[i:i+1], pred_3d[i:i+1])[0, 0]
                if i < pred_3d.shape[0] else 0.0
                for i in range(n_objects)
            ])
            obj_scores = iou_2d_diag * iou_3d_diag

    if verbose:
        _log.info(
            f"[{video_id}] Pred entry: "
            f"pred_boxes shape={pred_boxes.shape}, "
            f"pred_classes={pred_classes.tolist()}, "
            f"obj_scores={obj_scores.tolist()[:10]}, "
            f"rels_i shape={rels_i.shape}, "
            f"rel_scores shape={rel_scores.shape}"
        )
        # Show which predicate the model predicts for each pair
        for k_i in range(min(K, 5)):
            att_pred = att_dist[k_i].argmax()
            spa_preds = np.where(spa_dist[k_i] > 0.5)[0]
            con_preds = np.where(con_dist[k_i] > 0.5)[0]
            _log.info(
                f"[{video_id}]   pair[{k_i}]: person={person_idx[k_i]}, "
                f"obj={object_idx[k_i]} → "
                f"att_pred={att_pred}({evaluator.AG_attention_predicates[att_pred] if att_pred < len(evaluator.AG_attention_predicates) else '?'}), "
                f"spa_preds={[evaluator.AG_spatial_predicates[s] if s < len(evaluator.AG_spatial_predicates) else '?' for s in spa_preds]}, "
                f"con_preds={[evaluator.AG_contacting_predicates[c] if c < len(evaluator.AG_contacting_predicates) else '?' for c in con_preds]}"
            )

        # Show the GT relations as triplets for comparison
        _log.info(f"[{video_id}] GT triplets (subj_cls, pred, obj_cls):")
        for rel in gt_relations:
            subj_cls = gt_classes[rel[0]]
            obj_cls = gt_classes[rel[1]]
            pred_name = evaluator.AG_all_predicates[rel[2]] if rel[2] < len(evaluator.AG_all_predicates) else "?"
            _log.info(
                f"[{video_id}]   [{rel[0]}]{subj_cls} --{pred_name}({rel[2]})--> [{rel[1]}]{obj_cls}"
            )

        # Check: are pred pair indices within bounds of pred_classes?
        max_pair_idx = rels_i.max()
        _log.info(
            f"[{video_id}] Index check: max pair idx={max_pair_idx}, "
            f"pred_classes len={len(pred_classes)}, "
            f"in-bounds={'YES' if max_pair_idx < len(pred_classes) else '*** NO ***'}"
        )

    pred_entry = {
        "pred_boxes": pred_boxes,
        "pred_classes": pred_classes,
        "pred_rel_inds": rels_i,
        "obj_scores": obj_scores,
        "rel_scores": rel_scores,
    }

    evaluate_from_dict(
        gt_entry, pred_entry, evaluator.mode, evaluator.result_dict,
        iou_thresh=evaluator.iou_threshold,
        method=evaluator.constraint,
        threshold=evaluator.semi_threshold,
        num_rel=evaluator.num_rel,
    )
