import csv
import os

import numpy as np

from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value, fetch_rounded_value


class PreparePaperResultSGG(PrepareResultsBase):
	
	def __init__(self):
		super(PreparePaperResultSGG, self).__init__()
		self.scenario_list = ["full", "partial"]
		self.mode_list = ["sgcls", "sgdet", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		self.latex_method_list = [
			"sttran_full", "sttran_partial",
			"dsgdetr_full", "dsgdetr_partial"
		]
		self.partial_percentages = [10]
		self.task_name = "sgg"
	
	def fetch_sgg_recall_results_json_csv(self):
		db_results = self.fetch_db_sgg_results()
		
		sgg_results_json = {}
		# Has the following structure:
		# {
		# 	"sttran": {
		# 		"partial": {
		#			"10": {
		# 			    "with_constraint": {
		# 				    "R@10": 0.0,
		# 				    "R@20": 0.0,
		# 				    "R@50": 0.0,
		# 				    "R@100": 0.0
		# 			    },
		# 			    "no_constraint": {
		# 				    "R@10": 0.0,
		# 				    "R@20": 0.0,
		# 				    "R@50": 0.0,
		# 				    "R@100": 0.0
		# 			    },
		# 			    "semi_constraint": {
		# 				    "R@10": 0.0,
		# 				    "R@20": 0.0,
		# 				    "R@50": 0.0,
		# 				    "R@100": 0.0
		# 			    }
		# 			}
		# 		},
		# 		"full": {....}
		# 	},
		# 	"dsgdetr": {....}
		# }
		
		for method in self.method_list:
			sgg_results_json[method] = {}
			for scenario_name in self.scenario_list:
				sgg_results_json[method][scenario_name] = {}
				if scenario_name == "full":
					for mode in self.mode_list:
						sgg_results_json[method][scenario_name][
							mode] = self.fetch_paper_recall_empty_metrics_json()
				else:
					percentage_list = self.partial_percentages
					for percentage_num in percentage_list:
						sgg_results_json[method][scenario_name][percentage_num] = {}
						for mode in self.mode_list:
							sgg_results_json[method][scenario_name][percentage_num][
								mode] = self.fetch_paper_recall_empty_metrics_json()
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][mode] = completed_recall_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][percentage_num][mode] = completed_recall_metrics_json
		return sgg_results_json
	
	def fetch_sgg_mean_recall_results_json_csv(self):
		db_results = self.fetch_db_sgg_results()
		
		sgg_results_json = {}
		# Has the following structure:
		# {
		# 	"sttran": {
		# 		"partial": {
		#			"10": {
		# 			    "with_constraint": {
		# 				    "mR@10": 0.0,
		# 				    "mR@20": 0.0,
		# 				    "mR@50": 0.0,
		# 				    "mR@100": 0.0
		# 			    },
		# 			    "no_constraint": {
		# 				    "mR@10": 0.0,
		# 				    "mR@20": 0.0,
		# 				    "mR@50": 0.0,
		# 				    "mR@100": 0.0
		# 			    },
		# 			    "semi_constraint": {
		# 				    "mR@10": 0.0,
		# 				    "mR@20": 0.0,
		# 				    "mR@50": 0.0,
		# 				    "mR@100": 0.0
		# 			    }
		# 			}
		# 		},
		# 		"full": {....}
		# 	},
		# 	"dsgdetr": {....}
		# }
		
		for method in self.method_list:
			sgg_results_json[method] = {}
			for scenario_name in self.scenario_list:
				sgg_results_json[method][scenario_name] = {}
				if scenario_name == "full":
					for mode in self.mode_list:
						sgg_results_json[method][scenario_name][
							mode] = self.fetch_paper_mean_recall_empty_metrics_json()
				else:
					percentage_list = self.partial_percentages
					for percentage_num in percentage_list:
						sgg_results_json[method][scenario_name][percentage_num] = {}
						for mode in self.mode_list:
							sgg_results_json[method][scenario_name][percentage_num][
								mode] = self.fetch_paper_mean_recall_empty_metrics_json()
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][mode] = completed_mean_recall_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[method_name][scenario_name][percentage_num][mode] = completed_mean_recall_metrics_json
		return sgg_results_json
	
	def fetch_sgg_mean_recall_results_json_latex(self):
		db_results = self.fetch_db_sgg_results()
		sgg_results_json = {}
		for method in self.latex_method_list:
			sgg_results_json[method] = {}
			if "full" in method:
				for mode in self.mode_list:
					sgg_results_json[method][mode] = self.fetch_paper_mean_recall_empty_metrics_json()
			elif "partial" in method:
				percentage_list = self.partial_percentages
				for percentage_num in percentage_list:
					sgg_results_json[method][percentage_num] = {}
					for mode in self.mode_list:
						sgg_results_json[method][percentage_num][
							mode] = self.fetch_paper_mean_recall_empty_metrics_json()
		
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				paper_method_name = method_name + "_full"
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[paper_method_name][mode] = completed_mean_recall_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				paper_method_name = method_name + "_partial"
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[paper_method_name][percentage_num][mode] = completed_mean_recall_metrics_json
		return sgg_results_json
	
	# ---------------------------------------------------------------------------------------------
	# --------------------------------- CSV Generation Methods ------------------------------------
	# ---------------------------------------------------------------------------------------------
	
	def generate_sgg_mean_recall_results_csvs_method_wise(self, sgg_mean_recall_results_json):
		csv_file_name = "sgg_mean_recall.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_sgg_results_csvs",
		                             csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Method Name", "Scenario Name", "Partial Percentage",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100"
			])
			
			for method_name in self.method_list:
				for scenario_name in self.scenario_list:
					if scenario_name == "full":
						writer.writerow([
							method_name,
							scenario_name,
							"-",
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][0]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgdet"][2]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][0]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["sgcls"][2]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][0]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][1]["mR@100"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@10"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@20"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@50"],
							sgg_mean_recall_results_json[method_name][scenario_name]["predcls"][2]["mR@100"]
						])
						continue
					else:
						percentage_list = self.partial_percentages
						for percentage_num in percentage_list:
							writer.writerow([
								method_name,
								scenario_name,
								percentage_num,
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"mR@100"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@10"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@20"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@50"],
								sgg_mean_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"mR@100"]
							])
	
	def generate_sgg_recall_results_csvs_method_wise(self, sgg_recall_results_json):
		csv_file_name = "sgg_recall.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_sgg_results_csvs",
		                             csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Method Name", "Scenario Name", "Partial Percentage",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100",
			])
			
			for method_name in self.method_list:
				for scenario_name in self.scenario_list:
					if scenario_name == "full":
						writer.writerow([
							method_name,
							scenario_name,
							"-",
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][0]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgdet"][2]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][0]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["sgcls"][2]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][0]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][1]["R@100"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@10"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@20"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@50"],
							sgg_recall_results_json[method_name][scenario_name]["predcls"][2]["R@100"]
						])
						continue
					else:
						percentage_list = self.partial_percentages
						for percentage_num in percentage_list:
							writer.writerow([
								method_name,
								scenario_name,
								percentage_num,
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][0][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgdet"][2][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][0][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2]["R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2]["R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2]["R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["sgcls"][2][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][0][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][1][
									"R@100"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@10"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@20"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@50"],
								sgg_recall_results_json[method_name][scenario_name][percentage_num]["predcls"][2][
									"R@100"]
							])
	
	def compile_sgg_method_wise_results(self):
		sgg_mean_recall_results_json = self.fetch_sgg_mean_recall_results_json_csv()
		self.generate_sgg_mean_recall_results_csvs_method_wise(sgg_mean_recall_results_json)
		
		sgg_recall_results_json = self.fetch_sgg_recall_results_json_csv()
		self.generate_sgg_recall_results_csvs_method_wise(sgg_recall_results_json)
	
	# ---------------------------------------------------------------------------------------------
	# --------------------------------- Latex Generation Methods ----------------------------------
	# ---------------------------------------------------------------------------------------------
	
	@staticmethod
	def fill_sgg_combined_values_matrix_mean_recall(values_matrix, mean_recall_results_json,
	                                                idx, comb_method_name, partial_percentage=10):
		
		if "full" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][1]["mR@10"])
			values_matrix[idx, 4] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][1]["mR@20"])
			values_matrix[idx, 5] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][1]["mR@50"])
			values_matrix[idx, 6] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][2]["mR@10"])
			values_matrix[idx, 7] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][2]["mR@20"])
			values_matrix[idx, 8] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][2]["mR@50"])
			values_matrix[idx, 9] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][0]["mR@10"])
			values_matrix[idx, 10] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][0]["mR@20"])
			values_matrix[idx, 11] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][0]["mR@50"])
			values_matrix[idx, 12] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][1]["mR@10"])
			values_matrix[idx, 13] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][1]["mR@20"])
			values_matrix[idx, 14] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][1]["mR@50"])
			values_matrix[idx, 15] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][2]["mR@10"])
			values_matrix[idx, 16] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][2]["mR@20"])
			values_matrix[idx, 17] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][2]["mR@50"])
			values_matrix[idx, 18] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][0]["mR@10"])
			values_matrix[idx, 19] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][0]["mR@20"])
			values_matrix[idx, 20] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][0]["mR@50"])
			values_matrix[idx, 21] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][1]["mR@10"])
			values_matrix[idx, 22] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][1]["mR@20"])
			values_matrix[idx, 23] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][1]["mR@50"])
			values_matrix[idx, 24] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][2]["mR@10"])
			values_matrix[idx, 25] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][2]["mR@20"])
			values_matrix[idx, 26] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][2]["mR@50"])
		elif "partial" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][1]["mR@10"])
			values_matrix[idx, 4] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][1]["mR@20"])
			values_matrix[idx, 5] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][1]["mR@50"])
			values_matrix[idx, 6] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][2]["mR@10"])
			values_matrix[idx, 7] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][2]["mR@20"])
			values_matrix[idx, 8] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][2]["mR@50"])
			values_matrix[idx, 9] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][0]["mR@10"])
			values_matrix[idx, 10] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][0]["mR@20"])
			values_matrix[idx, 11] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][0]["mR@50"])
			values_matrix[idx, 12] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][1]["mR@10"])
			values_matrix[idx, 13] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][1]["mR@20"])
			values_matrix[idx, 14] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][1]["mR@50"])
			values_matrix[idx, 15] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][2]["mR@10"])
			values_matrix[idx, 16] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][2]["mR@20"])
			values_matrix[idx, 17] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][2]["mR@50"])
			values_matrix[idx, 18] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][0]["mR@10"])
			values_matrix[idx, 19] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][0]["mR@20"])
			values_matrix[idx, 20] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][0]["mR@50"])
			values_matrix[idx, 21] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][1]["mR@10"])
			values_matrix[idx, 22] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][1]["mR@20"])
			values_matrix[idx, 23] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][1]["mR@50"])
			values_matrix[idx, 24] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][2]["mR@10"])
			values_matrix[idx, 25] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][2]["mR@20"])
			values_matrix[idx, 26] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][2]["mR@50"])
		
		return values_matrix
	
	@staticmethod
	def generate_sgg_combined_paper_latex_header():
		latex_header = "\\begin{table*}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Results for SGG.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|ccccccccc|ccccccccc|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         & \\multicolumn{9}{c}{\\textbf{SGDET}} & \\multicolumn{9}{c}{\\textbf{SGCLS}} & \\multicolumn{9}{c}{\\textbf{PREDCLS}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1-1}\\cmidrule(lr){2-10} \\cmidrule(lr){11-19} \\cmidrule(lr){20-28} \n "
		latex_header += "         & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1-1} \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \\cmidrule(lr){17-19} \\cmidrule(lr){20-22} \\cmidrule(lr){23-25} \\cmidrule(lr){26-28} \n "
		
		latex_header += (
				"        \\textbf{Method} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & " + " \\\\ \\hline\n")
		
		return latex_header
	
	@staticmethod
	def generate_sgg_sgcls_predcls_paper_latex_header():
		latex_header = "\\begin{table*}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{SGCLS and PREDCLS Mean Recall Results for SGG.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|ccccccccc|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         \\multirow{3}{*}{Method} & \\multicolumn{9}{c}{\\textbf{SGCLS}} & \\multicolumn{9}{c}{\\textbf{PREDCLS}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){2-10} \\cmidrule(lr){11-19} \n "
		latex_header += "         & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10} \\cmidrule(lr){11-13} \\cmidrule(lr){14-16} \\cmidrule(lr){17-19} \n "
		
		latex_header += (
				"         & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} " + " \\\\ \\hline\n")
		
		return latex_header
	
	@staticmethod
	def generate_sgg_sgdet_paper_latex_header():
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{SGDET Mean Recall Results for VidSGG.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\linewidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         \\multirow{3}{*}{Method} & \\multicolumn{9}{c}{\\textbf{SGDET}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){2-10} \n "
		latex_header += "         & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}  \n "
		
		latex_header += (
				"         & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} " + " \\\\ \\hline\n")
		
		return latex_header
	
	def generate_paper_sgg_mean_recall_combined_latex_table(self, sgg_mean_recall_results_json):
		latex_file_name = f"sgg_mean_recall.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "latex_tables", latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sgg_combined_paper_latex_header()
		
		values_matrix = np.zeros((4, 27), dtype=np.float32)
		
		row_counter = 0
		for method in self.method_list:
			for scenario in self.scenario_list:
				comb_method_name = f"{method}_{scenario}"
				values_matrix = self.fill_sgg_combined_values_matrix_mean_recall(values_matrix,
				                                                                 sgg_mean_recall_results_json,
				                                                                 row_counter, comb_method_name)
				row_counter += 1
		
		row_counter = 0
		for method in self.method_list:
			for scenario in self.scenario_list:
				comb_method_name = f"{method}_{scenario}"
				latex_method_name = self.fetch_method_name_latex(comb_method_name)
				if row_counter % 2 == 1:
					latex_row = f"        {latex_method_name}"
					for col_idx in range(27):
						latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
					latex_row += "  \\\\ \n"
					latex_row += "          \\hline \n"
				else:
					latex_row = f"        {latex_method_name}"
					for col_idx in range(27):
						latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
					latex_row += "  \\\\ \n"
				latex_table += latex_row
				row_counter += 1
		latex_footer = self.generate_full_width_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)
	
	def generate_paper_sgg_mean_recall_sgcls_predcls_latex_table(self, sgg_mean_recall_results_json):
		latex_file_name = f"sgg_mean_recall_sgcls_predcls.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_latex_tables", latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sgg_sgcls_predcls_paper_latex_header()
		
		values_matrix = np.zeros((4, 27), dtype=np.float32)
		
		row_counter = 0
		for method in self.method_list:
			for scenario in self.scenario_list:
				comb_method_name = f"{method}_{scenario}"
				values_matrix = self.fill_sgg_combined_values_matrix_mean_recall(values_matrix,
				                                                                 sgg_mean_recall_results_json,
				                                                                 row_counter, comb_method_name)
				row_counter += 1
		
		row_counter = 0
		for method in self.method_list:
			for scenario in self.scenario_list:
				comb_method_name = f"{method}_{scenario}"
				latex_method_name = self.fetch_method_name_latex(comb_method_name)
				if row_counter % 2 == 1:
					latex_row = f"        {latex_method_name}"
					for col_idx in range(9, 27):
						latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
					latex_row += "  \\\\ \n"
					latex_row += "          \\hline \n"
				else:
					latex_row = f"        {latex_method_name}"
					for col_idx in range(9, 27):
						latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
					latex_row += "  \\\\ \n"
				latex_table += latex_row
				row_counter += 1
		latex_footer = self.generate_full_width_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)
	
	def generate_paper_sgg_mean_recall_sgdet_latex_table(self, sgg_mean_recall_results_json):
		latex_file_name = f"sgg_mean_recall_sgdet.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_latex_tables", latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sgg_sgdet_paper_latex_header()
		
		values_matrix = np.zeros((4, 27), dtype=np.float32)
		
		row_counter = 0
		for method in self.method_list:
			for scenario in self.scenario_list:
				comb_method_name = f"{method}_{scenario}"
				values_matrix = self.fill_sgg_combined_values_matrix_mean_recall(values_matrix,
				                                                                 sgg_mean_recall_results_json,
				                                                                 row_counter, comb_method_name)
				row_counter += 1
		
		row_counter = 0
		for method in self.method_list:
			for scenario in self.scenario_list:
				comb_method_name = f"{method}_{scenario}"
				latex_method_name = self.fetch_method_name_latex(comb_method_name)
				if row_counter % 2 == 1:
					latex_row = f"        {latex_method_name}"
					for col_idx in range(9):
						latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
					latex_row += "  \\\\ \n"
					latex_row += "          \\hline \n"
				else:
					latex_row = f"        {latex_method_name}"
					for col_idx in range(9):
						latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
					latex_row += "  \\\\ \n"
				latex_table += latex_row
				row_counter += 1
		latex_footer = self.generate_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)


def prepare_paper_sgg_latex_tables():
	prepare_paper_results_sgg = PreparePaperResultSGG()
	sgg_mean_recall_results_json = prepare_paper_results_sgg.fetch_sgg_mean_recall_results_json_latex()
	# prepare_paper_results_sgg.generate_paper_sgg_mean_recall_latex_table(sgg_mean_recall_results_json)
	prepare_paper_results_sgg.generate_paper_sgg_mean_recall_sgcls_predcls_latex_table(sgg_mean_recall_results_json)
	prepare_paper_results_sgg.generate_paper_sgg_mean_recall_sgdet_latex_table(sgg_mean_recall_results_json)


def main():
	prepare_paper_results_sgg = PreparePaperResultSGG()
	prepare_paper_results_sgg.compile_sgg_method_wise_results()


def combine_results():
	prepare_paper_results_sgg = PreparePaperResultSGG()
	prepare_paper_results_sgg.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\paper_sgg_results_csvs",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\paper_sgg_results_csvs\sgg_combined_results.xlsx"
	)


if __name__ == '__main__':
	# main()
	# combine_results()
	prepare_paper_sgg_latex_tables()
