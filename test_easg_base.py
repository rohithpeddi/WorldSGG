import csv
import os
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result, ResultDetails, Metrics
from dataloader.standard.easg.easg_dataset import StandardEASG
from easg_base import EASGBase
from constants import EgoConstants as const


class TestEASGBase(EASGBase):

    def __init__(self, conf):
        super().__init__(conf)

    # ----------------- Load the dataset -------------------------
    # Three main settings:
    # (a) Standard Dataset: Where full annotations are used
    # (b) Partial Annotations: Where partial object and relationship annotations are used
    # (c) Label Noise: Where label noise is added to the dataset
    # -------------------------------------------------------------
    def _init_dataset(self):
        self._val_dataset = StandardEASG(conf=self._conf, split=const.VAL)
        self._dataloader_val = DataLoader(self._val_dataset, shuffle=False)

    def _test_model(self):
        list_index = list(range(len(self._val_dataset)))
        self._model.eval()
        with torch.no_grad():
            for val_idx in tqdm(range(len(list_index))):
                graph = self._val_dataset[list_index[val_idx]]

                clip_feat = graph['clip_feat'].unsqueeze(0).to(self._device)
                obj_feats = graph['obj_feats'].to(self._device)
                out_verb, out_objs, out_rels = self._model(clip_feat, obj_feats)

                self._evaluator.evaluate_scene_graph(out_verb, out_objs, out_rels, graph)

    def _collate_evaluation_stats(self):
        stats_json = self._evaluator.fetch_stats_json()

        # Stats json has the following syntax:
        # 1. recall: { predcls_with, predcls_no, sgcls_with, sgcls_no, easg_with, easg_no }
        # 2. mean_recall: { predcls_with, predcls_no, sgcls_with, sgcls_no, easg_with, easg_no }
        # 3. harmonic_mean_recall: { predcls_with, predcls_no, sgcls_with, sgcls_no, easg_with, easg_no }
        # Invert the structure to have the following:
        # 1. predcls_with : { recall, mean_recall, harmonic_mean_recall }
        # 2. predcls_no : { recall, mean_recall, harmonic_mean_recall }
        # 3. sgcls_with : { recall, mean_recall, harmonic_mean_recall }
        # 4. sgcls_no : { recall, mean_recall, harmonic_mean_recall }
        # 5. easgcls_with : { recall, mean_recall, harmonic_mean_recall }
        # 6. easgcls_no : { recall, mean_recall, harmonic_mean_recall }

        inverted_stats_json = {}
        for eval_type in stats_json.keys():
            for mode in stats_json[eval_type].keys():
                if mode not in inverted_stats_json:
                    inverted_stats_json[mode] = {}
                inverted_stats_json[mode][eval_type] = stats_json[eval_type][mode]

        def evaluator_stats_to_list(with_constraint_evaluator_stats, no_constraint_evaluator_stats):
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
            ]

            return collated_stats

        mode_evaluator_stats_dict = {
            "predcls": evaluator_stats_to_list(inverted_stats_json["predcls_with"], inverted_stats_json["predcls_no"]),
            "sgcls": evaluator_stats_to_list(inverted_stats_json["sgcls_with"], inverted_stats_json["sgcls_no"]),
            "easgcls": evaluator_stats_to_list(inverted_stats_json["easgcls_with"], inverted_stats_json["easgcls_no"])
        }

        return mode_evaluator_stats_dict, inverted_stats_json

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

    def _publish_evaluation_results(self):
        mode_evaluator_stats_dict, inverted_stats_json = self._collate_evaluation_stats()
        self._write_evaluation_statistics(mode_evaluator_stats_dict)
        result_dict = self._publish_results_to_firebase(inverted_stats_json)
        return result_dict

    def _publish_results_to_firebase(self, inverted_stats_json):
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

        mode_list = ["predcls", "sgcls", "easgcls"]
        constraint_list = ["with", "no"]

        result_dict = {}

        for mode in mode_list:
            result = Result(
                task_name=self._conf.task_name,
                scenario_name=scenario_name,
                method_name=self._conf.method_name,
                mode=mode
            )

            if self._conf.use_input_corruptions:
                result.dataset_corruption_type = self._corruption_name

            if "partial" in self._checkpoint_name:
                if "10" in self._checkpoint_name:
                    result.partial_percentage = 10
                elif "40" in self._checkpoint_name:
                    result.partial_percentage = 40
                elif "70" in self._checkpoint_name:
                    result.partial_percentage = 70
            elif "label" in self._checkpoint_name:
                if "10" in self._checkpoint_name:
                    result.label_noise_percentage = 10
                elif "20" in self._checkpoint_name:
                    result.label_noise_percentage = 20
                elif "30" in self._checkpoint_name:
                    result.label_noise_percentage = 30

            result_details = ResultDetails()
            for constraint in constraint_list:
                constraint_metrics = self._prepare_metrics_from_stats(inverted_stats_json[f"{mode}_{constraint}"])
                if constraint == "with":
                    result_details.add_with_constraint_metrics(constraint_metrics)
                else:
                    result_details.add_no_constraint_metrics(constraint_metrics)

            result.add_result_details(result_details)

            print("Saving result: ", result.result_id)
            db_service.update_result_to_db("results_2_11_easg", result.result_id, result.to_dict())
            print("Saved result: ", result.result_id)

            result_dict[mode] = result

        return result_dict

    def _write_evaluation_statistics(self, mode_evaluator_stats_dict):
        # Create the results directory
        results_dir = os.path.join(os.getcwd(), 'results')
        task_dir = os.path.join(results_dir, "easg")

        for mode in mode_evaluator_stats_dict.keys():
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

                file_name = f'{self._checkpoint_name}_{mode}.csv'

            assert scenario_dir is not None, "Scenario directory is not set"
            mode_results_dir = os.path.join(scenario_dir, mode)
            os.makedirs(mode_results_dir, exist_ok=True)
            results_file_path = os.path.join(mode_results_dir, file_name)

            with open(results_file_path, "a", newline='') as activity_idx_step_idx_annotation_csv_file:
                writer = csv.writer(activity_idx_step_idx_annotation_csv_file, quoting=csv.QUOTE_NONNUMERIC)
                # Use this as the reference knowing what we write in the csv file

                # if not os.path.isfile(results_file_path):
                #     writer.writerow([
                #         "Method Name",
                #         "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                #         "hR@100",
                #         "R@10", "R@20", "R@50", "R@100", "mR@10", "mR@20", "mR@50", "mR@100", "hR@10", "hR@20", "hR@50",
                #         "hR@100"
                #     ])
                writer.writerow(mode_evaluator_stats_dict[mode])

    @abstractmethod
    def init_model(self):
        pass

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
        # self._init_object_detector()

        # 4. Test the model
        self._test_model()

        # 5. Publish the evaluation results
        self._publish_evaluation_results()
