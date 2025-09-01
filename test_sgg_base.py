import copy
import csv
import os
import time

import networkx as nx
import torch
from abc import abstractmethod

from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from torch.utils.data import DataLoader
from tqdm import tqdm
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result, ResultDetails, Metrics

from lib_b.object_detector import Detector
from lib.supervised import BasicSceneGraphEvaluator
from stsg_base import STSGBase


class TestSGGBase(STSGBase):

    def __init__(self, conf):
        super().__init__(conf)
        self._model = None

        # Load while initializing the object detector
        self._object_detector = None

        # Load while initializing the dataset
        self._dataloader_test = None
        self._test_dataset = None
        self._object_classes = None

        # Load checkpoint name
        self._checkpoint_name = None

        # Load the evaluators
        self._evaluators = []

        # Define the corruption function name
        # (Dataset - Video - Corruption Type) Possible Values
        # (Fixed - Fixed - {corruption_name})
        # (Mixed - Fixed), (Mixed - Mixed)
        # [{fixed_fixed_rain}, {mixed_fixed}, {mixed_mixed}]
        self._corruption_name = None

    def _init_evaluators(self):
        print("Replaced default single evaluator with multiple evaluators")
        # Evaluators order - [With Constraint, No Constraint, Semi Constraint]
        iou_threshold = 0.5 if self._conf.task_name == 'sgg' else 0.0
        self._with_constraint_evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes,
            AG_all_predicates=self._test_dataset.relationship_classes,
            AG_attention_predicates=self._test_dataset.attention_relationships,
            AG_spatial_predicates=self._test_dataset.spatial_relationships,
            AG_contacting_predicates=self._test_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            constraint='with')

        self._no_constraint_evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes,
            AG_all_predicates=self._test_dataset.relationship_classes,
            AG_attention_predicates=self._test_dataset.attention_relationships,
            AG_spatial_predicates=self._test_dataset.spatial_relationships,
            AG_contacting_predicates=self._test_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            constraint='no')

        self._semi_constraint_evaluator = BasicSceneGraphEvaluator(
            mode=self._conf.mode,
            AG_object_classes=self._test_dataset.object_classes,
            AG_all_predicates=self._test_dataset.relationship_classes,
            AG_attention_predicates=self._test_dataset.attention_relationships,
            AG_spatial_predicates=self._test_dataset.spatial_relationships,
            AG_contacting_predicates=self._test_dataset.contacting_relationships,
            iou_threshold=iou_threshold,
            constraint='semi', semi_threshold=0.9)

        self._evaluators.append(self._with_constraint_evaluator)  # 0
        self._evaluators.append(self._no_constraint_evaluator)  # 1
        self._evaluators.append(self._semi_constraint_evaluator)  # 2

    def _init_object_detector(self):
        self._object_detector = Detector(
            train=True,
            object_classes=self._object_classes,
            use_SUPPLY=True,
            mode=self._conf.mode
        ).to(device=self._device)

        # Set the object detector to eval mode to avoid backpropagation
        self._object_detector.eval()
        self._object_detector.is_train = False

    def _collate_evaluation_stats(self):
        with_constraint_evaluator_stats = self._with_constraint_evaluator.fetch_stats_json(
            save_file_path=f"with_constraint_{self._checkpoint_name}.csv")
        no_constraint_evaluator_stats = self._no_constraint_evaluator.fetch_stats_json(
            save_file_path=f"no_constraint_{self._checkpoint_name}.csv")
        semi_constraint_evaluator_stats = self._semi_constraint_evaluator.fetch_stats_json(
            save_file_path=f"semi_constraint_{self._checkpoint_name}.csv")

        collated_stats = [
            self._conf.method_name,
            with_constraint_evaluator_stats["recall"][10],
            with_constraint_evaluator_stats["recall"][20],
            with_constraint_evaluator_stats["recall"][50],
            with_constraint_evaluator_stats["recall"][100],
            with_constraint_evaluator_stats["mean_recall"][10],
            with_constraint_evaluator_stats["mean_recall"][20],
            with_constraint_evaluator_stats["mean_recall"][50],
            with_constraint_evaluator_stats["mean_recall"][100],
            with_constraint_evaluator_stats["harmonic_mean_recall"][10],
            with_constraint_evaluator_stats["harmonic_mean_recall"][20],
            with_constraint_evaluator_stats["harmonic_mean_recall"][50],
            with_constraint_evaluator_stats["harmonic_mean_recall"][100],
            no_constraint_evaluator_stats["recall"][10],
            no_constraint_evaluator_stats["recall"][20],
            no_constraint_evaluator_stats["recall"][50],
            no_constraint_evaluator_stats["recall"][100],
            no_constraint_evaluator_stats["mean_recall"][10],
            no_constraint_evaluator_stats["mean_recall"][20],
            no_constraint_evaluator_stats["mean_recall"][50],
            no_constraint_evaluator_stats["mean_recall"][100],
            no_constraint_evaluator_stats["harmonic_mean_recall"][10],
            no_constraint_evaluator_stats["harmonic_mean_recall"][20],
            no_constraint_evaluator_stats["harmonic_mean_recall"][50],
            no_constraint_evaluator_stats["harmonic_mean_recall"][100],
            semi_constraint_evaluator_stats["recall"][10],
            semi_constraint_evaluator_stats["recall"][20],
            semi_constraint_evaluator_stats["recall"][50],
            semi_constraint_evaluator_stats["recall"][100],
            semi_constraint_evaluator_stats["mean_recall"][10],
            semi_constraint_evaluator_stats["mean_recall"][20],
            semi_constraint_evaluator_stats["mean_recall"][50],
            semi_constraint_evaluator_stats["mean_recall"][100],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][10],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][20],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][50],
            semi_constraint_evaluator_stats["harmonic_mean_recall"][100]
        ]
        return collated_stats

    def _evaluate_predictions(self, gt_annotation, prediction):
        for evaluator in self._evaluators:
            evaluator.evaluate_scene_graph(gt_annotation, prediction)

    def _publish_evaluation_results(self):
        # 1. Collate the evaluation statistics
        self._collated_stats = self._collate_evaluation_stats()
        # 2. Write to the CSV File
        self._write_evaluation_statistics()
        # 3. Publish the results to Firebase
        # self._publish_results_to_firebase()

    def _write_evaluation_statistics(self):
        # Create the results directory
        results_dir = os.path.join(os.getcwd(), 'results')
        task_dir = os.path.join(results_dir, "sgg")

        if self._conf.use_input_corruptions:
            scenario_dir = os.path.join(task_dir, "corruptions")
            file_name = f'{self._checkpoint_name}_{self._corruption_name}.csv'
        else:
            if "partial" in self._checkpoint_name:
                scenario_dir = os.path.join(task_dir, "partial")
            elif "label" in self._checkpoint_name:
                scenario_dir = os.path.join(task_dir, "labelnoise")
            else:
                scenario_dir = os.path.join(task_dir, "full")
            file_name = f'{self._checkpoint_name}.csv'

        assert scenario_dir is not None, "Scenario directory is not set"
        mode_results_dir = os.path.join(scenario_dir, self._conf.mode)
        os.makedirs(mode_results_dir, exist_ok=True)
        results_file_path = os.path.join(mode_results_dir, file_name)

        with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
            writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
            # Write the header
            if not os.path.isfile(results_file_path):
                writer.writerow([
                    "Method Name",
                    "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                    "hR@100",
                    "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                    "hR@100",
                    "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                    "hR@100"
                ])
                # Write the results row
            writer.writerow(self._collated_stats)

    @staticmethod
    def _prepare_metrics_from_stats(evaluator_stats):
        metrics = Metrics(
            evaluator_stats["recall"][10],
            evaluator_stats["recall"][20],
            evaluator_stats["recall"][50],
            evaluator_stats["recall"][100],
            evaluator_stats["mean_recall"][10],
            evaluator_stats["mean_recall"][20],
            evaluator_stats["mean_recall"][50],
            evaluator_stats["mean_recall"][100],
            evaluator_stats["harmonic_mean_recall"][10],
            evaluator_stats["harmonic_mean_recall"][20],
            evaluator_stats["harmonic_mean_recall"][50],
            evaluator_stats["harmonic_mean_recall"][100]
        )

        return metrics

    def _publish_results_to_firebase(self):
        db_service = FirebaseService()

        if self._conf.use_input_corruptions:
            scenario_name = "corruption"
        else:
            if "partial" in self._checkpoint_name:
                scenario_name = "partial"
            elif "label" in self._checkpoint_name:
                scenario_name = "labelnoise"
            else:
                scenario_name = "full"

        result = Result(
            task_name=self._conf.task_name,
            scenario_name=scenario_name,
            method_name=self._conf.method_name,
            mode=self._conf.mode
        )

        if self._conf.use_input_corruptions:
            result.dataset_corruption_type = self._corruption_name

        result_details = ResultDetails()
        with_constraint_metrics = self._prepare_metrics_from_stats(self._evaluators[0].fetch_stats_json())
        no_constraint_metrics = self._prepare_metrics_from_stats(self._evaluators[1].fetch_stats_json())
        semi_constraint_metrics = self._prepare_metrics_from_stats(self._evaluators[2].fetch_stats_json())

        result_details.add_with_constraint_metrics(with_constraint_metrics)
        result_details.add_no_constraint_metrics(no_constraint_metrics)
        result_details.add_semi_constraint_metrics(semi_constraint_metrics)

        result.add_result_details(result_details)

        print("Saving result: ", result.result_id)
        db_service.update_result(result.result_id, result.to_dict())
        print("Saved result: ", result.result_id)

        return result

    def _test_model(self):
        start_time = time.time()
        print('-----------------------------------------------------------------------------------', flush=True)
        print(f"Testing the model: {self._conf.method_name} on {self._conf.mode} mode", flush=True)
        print(f"Starting testing at: {start_time}", flush=True)
        if self._conf.use_input_corruptions:
            print(f"Testing the model on corruption: {self._corruption_name}", flush=True)
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        self._object_detector.is_train = False
        with torch.no_grad():
            for num_video_id in tqdm(range(len(self._dataloader_test)), desc="Testing Progress", ascii=True):
                data = next(test_iter)
                im_data, im_info, gt_boxes, num_boxes = [copy.deepcopy(d.cuda(0)) for d in data[:4]]
                gt_annotation = self._test_dataset.gt_annotations[data[4]]
                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                entry = self._object_detector(
                    im_data,
                    im_info,
                    gt_boxes,
                    num_boxes,
                    gt_annotation,
                    im_all=None,
                    gt_annotation_mask=None,
                )

                # ----------------- Process the video (Method Specific) -----------------
                prediction = self.process_test_video(entry, frame_size, gt_annotation)
                # ----------------------------------------------------------------------

                self._evaluate_predictions(gt_annotation, prediction)

        end_time = time.time()
        print(f"Time taken for testing: {end_time - start_time} seconds")
        print('-----------------------------------------------------------------------------------', flush=True)

    def _init_dataset(self):
        if self._conf.use_input_corruptions:
            from dataloader.corrupted.image_based.ag_dataset import ImageCorruptedAG
            from dataloader.corrupted.image_based.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn
            # Using the parameters set in the configuration file, initialize the corrupted dataset
            self._test_dataset = ImageCorruptedAG(
                phase='test',
                mode=self._conf.mode,
                datasize=self._conf.datasize,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True,
                dataset_corruption_mode=self._conf.dataset_corruption_mode,
                video_corruption_mode=self._conf.video_corruption_mode,
                dataset_corruption_type=self._conf.dataset_corruption_type,
                corruption_severity_level=self._conf.corruption_severity_level
            )

            self._corruption_name = (f"{self._conf.dataset_corruption_mode}_{self._conf.video_corruption_mode}_"
                                     f"{self._conf.dataset_corruption_type}_{self._conf.corruption_severity_level}")

            print("---------------------------------------------------------------------------")
            print(f"Initializing corrupted dataset with corruption name: {self._corruption_name}")
            print("---------------------------------------------------------------------------")

        else:
            from dataloader.standard.action_genome.ag_dataset import StandardAG
            from dataloader.standard.action_genome.ag_dataset import cuda_collate_fn as ag_data_cuda_collate_fn

            # Use Standard AG test dataset if no corruptions are used
            self._test_dataset = StandardAG(
                phase='test',
                mode=self._conf.mode,
                datasize=self._conf.datasize,
                data_path=self._conf.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if self._conf.mode == 'predcls' else True
            )

            print("---------------------------------------------------------------------------")
            print("Initializing standard dataset")
            print("---------------------------------------------------------------------------")

        self._object_classes = self._test_dataset.object_classes

        self._dataloader_test = DataLoader(
            self._test_dataset,
            shuffle=False,
            collate_fn=ag_data_cuda_collate_fn,
            pin_memory=False
        )

    def init_method_evaluation(self):
        # 0. Init config
        self._init_config()

        # 1. Initialize the dataset
        self._init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pretrained model
        self.init_model()
        self._load_checkpoint()
        self._init_object_detector()

        # 4. Test the model
        self._test_model()

        # 5. Publish the evaluation results
        self._publish_evaluation_results()

    def _prepare_prediction_graph(
            self,
            predictions_map,
            dataset,
            video_id,
            constraint_type
    ):
        # Loop through each frame in the video
        for frame_idx, pred_numpy_array in predictions_map.items():
            graph = nx.MultiGraph()
            pred_set = set()
            for pred in pred_numpy_array:
                # Convert numpy arrays to tuples for hashing
                pred_set.add(tuple(pred))

            # Process each prediction
            for pred_tuple in pred_set:
                subject_class = dataset.object_classes[pred_tuple[0]]
                object_class = dataset.object_classes[pred_tuple[1]]
                predicate_class = dataset.relationship_classes[pred_tuple[2]]

                # Add nodes and edge
                graph.add_node(subject_class, label=subject_class)
                graph.add_node(object_class, label=object_class)
                graph.add_edge(subject_class, object_class, label=predicate_class)

            self._draw_and_save_graph(graph, video_id, frame_idx, constraint_type)

    def _draw_and_save_graph(
            self,
            graph,
            video_id,
            frame_idx,
            constraint_type
    ):
        plt.figure(figsize=(12, 12))

        pos = nx.spring_layout(graph, seed=42)  # positions for all nodes, with a fixed layout

        # Draw nodes and labels
        nx.draw_networkx_nodes(graph, pos, node_size=700)
        nx.draw_networkx_labels(graph, pos)

        # Custom drawing of the edges using FancyArrowPatch
        for u, v, key, data in graph.edges(keys=True, data=True):
            # Determine if there are multiple edges and calculate offset
            num_edges = graph.number_of_edges(u, v)
            edge_count = sum(1 for _ in graph[u][v])
            offset = 0.13 * (key - edge_count // 2)  # Offset for curvature

            # Parameters for the FancyArrowPatch
            arrow_options = {
                'arrowstyle': '-',
                'connectionstyle': f"arc3,rad={offset}",
                'color': 'black',
                'linewidth': 1
            }

            # Draw the edge with curvature
            edge = FancyArrowPatch(pos[u], pos[v], **arrow_options)
            plt.gca().add_patch(edge)

            # Improved calculation for the position of the edge label
            label_pos_x = (pos[u][0] + pos[v][0]) / 2 + offset * 0.75 * (pos[v][1] - pos[u][1])
            label_pos_y = (pos[u][1] + pos[v][1]) / 2 - offset * 0.75 * (pos[v][0] - pos[u][0])
            plt.text(label_pos_x, label_pos_y, str(data['label']), color='blue', fontsize=10, ha='center', va='center')

        # Save graph
        file_name = "{}_{}.png".format(video_id, frame_idx)
        file_directory_path = os.path.join(
            os.path.dirname(__file__),
            "analysis",
            "docs",
            "qualitative_results",
            self._checkpoint_name,
            video_id,
            self._conf.mode,
            constraint_type
        )

        os.makedirs(file_directory_path, exist_ok=True)
        file_path = os.path.join(file_directory_path, file_name)
        plt.savefig(file_path)

    def _generate_graphs(self):
        start_time = time.time()
        print('-----------------------------------------------------------------------------------', flush=True)
        print(f"Testing the model: {self._conf.method_name} on {self._conf.mode} mode", flush=True)
        print(f"Starting testing at: {start_time}", flush=True)
        if self._conf.use_input_corruptions:
            print(f"Testing the model on corruption: {self._corruption_name}", flush=True)
        test_iter = iter(self._dataloader_test)
        self._model.eval()
        self._object_detector.is_train = False

        video_id_index_map = {}
        for index, video_gt_annotation in enumerate(self._test_dataset.gt_annotations):
            video_id = video_gt_annotation[0][0]['frame'].split(".")[0]
            video_id_index_map[video_id] = index

        video_id_list = ["21F9H", "X95D0", "M18XP", "0A8CF", "LUQWY", "QE4YE", "ENOLD"]
        with torch.no_grad():
            for video_id in video_id_list:
                d_im_data, d_im_info, d_gt_boxes, d_num_boxes, d_index = self._test_dataset.fetch_video_data(
                    video_id_index_map[video_id])
                im_data = copy.deepcopy(d_im_data.cuda(0))
                im_info = copy.deepcopy(d_im_info.cuda(0))
                gt_boxes = copy.deepcopy(d_gt_boxes.cuda(0))
                num_boxes = copy.deepcopy(d_num_boxes.cuda(0))
                gt_annotation = self._test_dataset.gt_annotations[video_id_index_map[video_id]]
                frame_size = (im_info[0][:2] / im_info[0, 2]).cpu().data

                entry = self._object_detector(
                    im_data,
                    im_info,
                    gt_boxes,
                    num_boxes,
                    gt_annotation,
                    im_all=None,
                    gt_annotation_mask=None,
                )

                # ----------------- Process the video (Method Specific) -----------------
                prediction = self.process_test_video(entry, frame_size, gt_annotation)
                # ----------------------------------------------------------------------

                # 0 - With Constraint, 1 - No Constraint, 2 - Semi Constraint
                with_constraint_predictions_map = self._evaluators[0].fetch_pred_tuples(gt_annotation, prediction)
                no_constraint_prediction_map = self._evaluators[1].fetch_pred_tuples(gt_annotation, prediction)
                semi_constraint_prediction_map = self._evaluators[2].fetch_pred_tuples(gt_annotation, prediction)

                self._prepare_prediction_graph(
                    predictions_map=with_constraint_predictions_map,
                    dataset=self._test_dataset,
                    video_id=video_id,
                    constraint_type="with_constraints"
                )

                self._prepare_prediction_graph(
                    predictions_map=no_constraint_prediction_map,
                    dataset=self._test_dataset,
                    video_id=video_id,
                    constraint_type="no_constraints"
                )

                self._prepare_prediction_graph(
                    predictions_map=semi_constraint_prediction_map,
                    dataset=self._test_dataset,
                    video_id=video_id,
                    constraint_type="semi_constraints"
                )

        end_time = time.time()
        print(f"Time taken for testing: {end_time - start_time} seconds")
        print('-----------------------------------------------------------------------------------', flush=True)

    def store_qualitative_results(self):
        # 0. Init config
        self._init_config()

        # 1. Initialize the dataset
        self._init_dataset()

        # 2. Initialize evaluators
        self._init_evaluators()

        # 3. Initialize and load pretrained model
        self.init_model()
        self._load_checkpoint()
        self._init_object_detector()

        # 4. Generate qualitative results
        self._generate_graphs()

    @abstractmethod
    def init_model(self):
        pass

    @abstractmethod
    def process_test_video(self, video_entry, frame_size, gt_annotation) -> dict:
        pass
