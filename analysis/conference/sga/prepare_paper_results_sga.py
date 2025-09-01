import csv
import os

import numpy as np

from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value, fetch_rounded_value


class PreparePaperResultsSGA(PrepareResultsBase):
	
	def __init__(self):
		super(PreparePaperResultsSGA, self).__init__()
		self.scenario_list = ["full", "partial"]
		self.mode_list = ["sgcls", "sgdet", "predcls"]
		self.method_list = ["sttran_ant", "dsgdetr_ant", "sttran_gen_ant", "dsgdetr_gen_ant", "ode", "sde"]
		self.partial_percentages = ['10']
		self.task_name = "sga"
		# self.latex_method_list = [
		# 	"sttran_ant_full", "sttran_ant_partial",
		# 	"dsgdetr_ant_full", "dsgdetr_ant_partial",
		# 	"sttran_gen_ant_full", "sttran_gen_ant_partial",
		# 	"dsgdetr_gen_ant_full", "dsgdetr_gen_ant_partial",
		# 	"ode_full", "ode_partial",
		# 	"sde_full", "sde_partial"
		# ]
		
		self.latex_method_list = [
			"sttran_gen_ant_full", "sttran_gen_ant_partial",
			"dsgdetr_gen_ant_full", "dsgdetr_gen_ant_partial",
			"ode_full", "ode_partial",
			"sde_full", "sde_partial"
		]
		
		self.context_fraction_list = ['0.3', '0.5', '0.7', '0.9']
		self.latex_context_fraction_list = ['0.5', '0.7']
	
	def fetch_sga_recall_results_json_csv(self):
		db_results = self.fetch_db_sga_results()
		sga_results_json = {}
		for method in self.method_list:
			sga_results_json[method] = {}
			for scenario_name in self.scenario_list:
				sga_results_json[method][scenario_name] = {}
				if scenario_name == "full":
					for mode in self.mode_list:
						sga_results_json[method][scenario_name][mode] = {}
						for cf in self.context_fraction_list:
							sga_results_json[method][scenario_name][mode][
								cf] = self.fetch_paper_recall_empty_metrics_json()
				else:
					percentage_list = self.partial_percentages
					for percentage_num in percentage_list:
						sga_results_json[method][scenario_name][percentage_num] = {}
						for mode in self.mode_list:
							sga_results_json[method][scenario_name][percentage_num][mode] = {}
							for cf in self.context_fraction_list:
								sga_results_json[method][scenario_name][percentage_num][mode][
									cf] = self.fetch_paper_recall_empty_metrics_json()
		for sga_result in db_results:
			mode = sga_result.mode
			method_name = sga_result.method_name
			scenario_name = sga_result.scenario_name
			if scenario_name == "full":
				# Take only the context fraction results and ignore the future frames results
				if sga_result.context_fraction is None:
					continue
				
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				cf = str(sga_result.context_fraction)
				sga_results_json[method_name][scenario_name][mode][cf] = completed_recall_metrics_json
				continue
			elif scenario_name == "partial":
				
				# Take only the context fraction results and ignore the future frames results
				if sga_result.context_fraction is None:
					continue
				
				percentage_num = str(sga_result.partial_percentage)
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				
				if percentage_num not in self.partial_percentages:
					continue
				
				cf = str(sga_result.context_fraction)
				sga_results_json[method_name][scenario_name][percentage_num][mode][cf] = completed_recall_metrics_json
		return sga_results_json
	
	def full_sga_results_json(self):
		db_results = self.fetch_db_sga_results()
		sga_results_json = {}
		for method in self.method_list:
			sga_results_json[method] = {}
			for mode in self.mode_list:
				sga_results_json[method][mode] = {}
				for cf in self.context_fraction_list:
					sga_results_json[method][mode][cf] = self.fetch_empty_metrics_json()
		for sga_result in db_results:
			mode = sga_result.mode
			method_name = sga_result.method_name
			scenario_name = sga_result.scenario_name
			if scenario_name == "full":
				# Take only the context fraction results and ignore the future frames results
				if sga_result.context_fraction is None:
					continue
				
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				cf = str(sga_result.context_fraction)
				sga_results_json[method_name][mode][cf] = completed_recall_metrics_json
				continue
			elif scenario_name == "partial":
				continue
		
		return sga_results_json
	
	def fetch_sga_mean_recall_results_json_csv(self):
		db_results = self.fetch_db_sga_results()
		sga_results_json = {}
		for method in self.method_list:
			sga_results_json[method] = {}
			for scenario_name in self.scenario_list:
				sga_results_json[method][scenario_name] = {}
				if scenario_name == "full":
					for mode in self.mode_list:
						sga_results_json[method][scenario_name][mode] = {}
						for cf in self.context_fraction_list:
							sga_results_json[method][scenario_name][mode][
								cf] = self.fetch_paper_mean_recall_empty_metrics_json()
				else:
					percentage_list = self.partial_percentages
					for percentage_num in percentage_list:
						sga_results_json[method][scenario_name][percentage_num] = {}
						for mode in self.mode_list:
							sga_results_json[method][scenario_name][percentage_num][mode] = {}
							for cf in self.context_fraction_list:
								sga_results_json[method][scenario_name][percentage_num][mode][
									cf] = self.fetch_paper_mean_recall_empty_metrics_json()
		for sga_result in db_results:
			mode = sga_result.mode
			method_name = sga_result.method_name
			scenario_name = sga_result.scenario_name
			if scenario_name == "full":
				# Take only the context fraction results and ignore the future frames results
				if sga_result.context_fraction is None:
					continue
				
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				cf = str(sga_result.context_fraction)
				sga_results_json[method_name][scenario_name][mode][cf] = completed_recall_metrics_json
				continue
			elif scenario_name == "partial":
				
				# Take only the context fraction results and ignore the future frames results
				if sga_result.context_fraction is None:
					continue
				
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				cf = str(sga_result.context_fraction)
				percentage_num = str(sga_result.partial_percentage)
				
				if percentage_num not in self.partial_percentages:
					continue
				
				sga_results_json[method_name][scenario_name][percentage_num][mode][cf] = completed_recall_metrics_json
		return sga_results_json
	
	def fetch_sga_mean_recall_results_json_latex(self):
		db_results = self.fetch_db_sga_results()
		sga_results_json = {}
		for method in self.latex_method_list:
			sga_results_json[method] = {}
			if "full" in method:
				for mode in self.mode_list:
					sga_results_json[method][mode] = {}
					for cf in self.context_fraction_list:
						sga_results_json[method][mode][cf] = self.fetch_paper_mean_recall_empty_metrics_json()
			elif "partial" in method:
				percentage_list = self.partial_percentages
				for percentage_num in percentage_list:
					sga_results_json[method][percentage_num] = {}
					for mode in self.mode_list:
						sga_results_json[method][percentage_num][mode] = {}
						for cf in self.context_fraction_list:
							sga_results_json[method][percentage_num][mode][
								cf] = self.fetch_paper_mean_recall_empty_metrics_json()
		
		for sga_result in db_results:
			mode = sga_result.mode
			method_name = sga_result.method_name
			
			scenario_name = sga_result.scenario_name
			if scenario_name == "full":
				paper_method_name = method_name + "_full"
				
				if sga_result.context_fraction is None:
					continue
				
				if paper_method_name not in self.latex_method_list:
					continue
				
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				
				cf = str(sga_result.context_fraction)
				sga_results_json[paper_method_name][mode][cf] = completed_mean_recall_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = str(sga_result.partial_percentage)
				paper_method_name = method_name + "_partial"
				
				if sga_result.context_fraction is None:
					continue
				
				if paper_method_name not in self.latex_method_list:
					continue
				
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_mean_recall_metrics_json = self.fetch_paper_completed_mean_recall_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				
				if percentage_num not in self.partial_percentages:
					continue
				
				cf = str(sga_result.context_fraction)
				sga_results_json[paper_method_name][percentage_num][mode][cf] = completed_mean_recall_metrics_json
		return sga_results_json
	
	# ---------------------------------------------------------------------------------------------
	# --------------------------------- CSV Generation Methods ------------------------------------
	# ---------------------------------------------------------------------------------------------
	
	def generate_sga_mean_recall_results_csvs_method_wise(self, sga_mean_recall_results_json):
		csv_file_name = "sga_mean_recall.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_sga_results_csvs",
		                             csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Context Fraction", "Method Name", "Scenario Name", "Partial Percentage",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100",
				"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50",
				"mR@100"
			])
			
			for cf in self.context_fraction_list:
				for method_name in self.method_list:
					for scenario_name in self.scenario_list:
						if scenario_name == "full":
							writer.writerow([
								cf,
								method_name,
								scenario_name,
								"-",
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["mR@100"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["mR@10"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["mR@20"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["mR@50"],
								sga_mean_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["mR@100"]
							])
						elif scenario_name == "partial":
							for partial_num in self.partial_percentages:
								writer.writerow([
									cf,
									method_name,
									scenario_name,
									partial_num,
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										0][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										0][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										0][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										0][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										1][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										1][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										1][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										1][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										2][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										2][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										2][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][
										2][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										0][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										0][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										0][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										0][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										1][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										1][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										1][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										1][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										2][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										2][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										2][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][
										2][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][0][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][0][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][0][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][0][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][1][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][1][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][1][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][1][
										"mR@100"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][2][
										"mR@10"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][2][
										"mR@20"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][2][
										"mR@50"],
									sga_mean_recall_results_json[method_name][scenario_name][partial_num]["predcls"][
										cf][2][
										"mR@100"]
								])
	
	def generate_sga_recall_results_csvs_method_wise(self, sga_recall_results_json):
		csv_file_name = "sga_recall.csv"
		csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_sga_results_csvs",
		                             csv_file_name)
		os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
		with open(csv_file_path, "a", newline='') as csv_file:
			writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
			writer.writerow([
				"Context Fraction", "Method Name", "Scenario Name", "Partial Percentage",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50",
				"R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50",
				"R@100",
				"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50",
				"R@100"
			])
			
			for cf in self.context_fraction_list:
				for method_name in self.method_list:
					for scenario_name in self.scenario_list:
						if scenario_name == "full":
							writer.writerow([
								cf,
								method_name,
								scenario_name,
								"-",
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][0]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][1]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["sgdet"][cf][2]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][0]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][1]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["sgcls"][cf][2]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][0]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][1]["R@100"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["R@10"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["R@20"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["R@50"],
								sga_recall_results_json[method_name][scenario_name]["predcls"][cf][2]["R@100"]
							])
						elif scenario_name == "partial":
							for partial_num in self.partial_percentages:
								writer.writerow([
									cf,
									method_name,
									scenario_name,
									partial_num,
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][0][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][0][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][0][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][0][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][1][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][1][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][1][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][1][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][2][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][2][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][2][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgdet"][cf][2][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][0][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][0][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][0][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][0][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][1][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][1][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][1][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][1][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][2][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][2][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][2][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["sgcls"][cf][2][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][0][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][0][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][0][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][0][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][1][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][1][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][1][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][1][
										"R@100"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][2][
										"R@10"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][2][
										"R@20"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][2][
										"R@50"],
									sga_recall_results_json[method_name][scenario_name][partial_num]["predcls"][cf][2][
										"R@100"]
								])
	
	def compile_sga_method_wise_results_csvs(self):
		sga_mean_recall_results_json = self.fetch_sga_mean_recall_results_json_csv()
		self.generate_sga_mean_recall_results_csvs_method_wise(sga_mean_recall_results_json)
		
		sga_recall_results_json = self.fetch_sga_recall_results_json_csv()
		self.generate_sga_recall_results_csvs_method_wise(sga_recall_results_json)
	
	# ---------------------------------------------------------------------------------------------
	# --------------------------------- Latex Generation Methods ------------------------------------
	# ---------------------------------------------------------------------------------------------
	
	@staticmethod
	def fill_full_sga_values_matrix(values_matrix, results_json, idx, method_name, mode, cf):
		values_matrix[idx, 0] = fetch_value(results_json[method_name][mode][cf][0]["R@10"])
		values_matrix[idx, 1] = fetch_value(results_json[method_name][mode][cf][0]["R@20"])
		values_matrix[idx, 2] = fetch_value(results_json[method_name][mode][cf][0]["R@50"])
		values_matrix[idx, 3] = fetch_value(results_json[method_name][mode][cf][1]["R@10"])
		values_matrix[idx, 4] = fetch_value(results_json[method_name][mode][cf][1]["R@20"])
		values_matrix[idx, 5] = fetch_value(results_json[method_name][mode][cf][1]["R@50"])
		values_matrix[idx, 6] = fetch_value(results_json[method_name][mode][cf][0]["mR@10"])
		values_matrix[idx, 7] = fetch_value(results_json[method_name][mode][cf][0]["mR@20"])
		values_matrix[idx, 8] = fetch_value(results_json[method_name][mode][cf][0]["mR@50"])
		values_matrix[idx, 9] = fetch_value(results_json[method_name][mode][cf][1]["mR@10"])
		values_matrix[idx, 10] = fetch_value(results_json[method_name][mode][cf][1]["mR@20"])
		values_matrix[idx, 11] = fetch_value(results_json[method_name][mode][cf][1]["mR@50"])
		return values_matrix
	
	@staticmethod
	def fill_sga_combined_values_matrix_mean_recall(values_matrix, mean_recall_results_json,
	                                                idx, comb_method_name, cf, partial_percentage='10'):
		if "full" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][1]["mR@10"])
			values_matrix[idx, 4] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][1]["mR@20"])
			values_matrix[idx, 5] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][1]["mR@50"])
			values_matrix[idx, 6] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][2]["mR@10"])
			values_matrix[idx, 7] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][2]["mR@20"])
			values_matrix[idx, 8] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][2]["mR@50"])
			values_matrix[idx, 9] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][0]["mR@10"])
			values_matrix[idx, 10] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][0]["mR@20"])
			values_matrix[idx, 11] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][0]["mR@50"])
			values_matrix[idx, 12] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][1]["mR@10"])
			values_matrix[idx, 13] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][1]["mR@20"])
			values_matrix[idx, 14] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][1]["mR@50"])
			values_matrix[idx, 15] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][2]["mR@10"])
			values_matrix[idx, 16] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][2]["mR@20"])
			values_matrix[idx, 17] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][2]["mR@50"])
			values_matrix[idx, 18] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][0]["mR@10"])
			values_matrix[idx, 19] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][0]["mR@20"])
			values_matrix[idx, 20] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][0]["mR@50"])
			values_matrix[idx, 21] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][1]["mR@10"])
			values_matrix[idx, 22] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][1]["mR@20"])
			values_matrix[idx, 23] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][1]["mR@50"])
			values_matrix[idx, 24] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][2]["mR@10"])
			values_matrix[idx, 25] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][2]["mR@20"])
			values_matrix[idx, 26] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][2]["mR@50"])
		elif "partial" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][1]["mR@10"])
			values_matrix[idx, 4] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][1]["mR@20"])
			values_matrix[idx, 5] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][1]["mR@50"])
			values_matrix[idx, 6] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][2]["mR@10"])
			values_matrix[idx, 7] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][2]["mR@20"])
			values_matrix[idx, 8] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][2]["mR@50"])
			values_matrix[idx, 9] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][0]["mR@10"])
			values_matrix[idx, 10] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][0]["mR@20"])
			values_matrix[idx, 11] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][0]["mR@50"])
			values_matrix[idx, 12] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][1]["mR@10"])
			values_matrix[idx, 13] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][1]["mR@20"])
			values_matrix[idx, 14] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][1]["mR@50"])
			values_matrix[idx, 15] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][2]["mR@10"])
			values_matrix[idx, 16] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][2]["mR@20"])
			values_matrix[idx, 17] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][2]["mR@50"])
			values_matrix[idx, 18] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][0]["mR@10"])
			values_matrix[idx, 19] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][0]["mR@20"])
			values_matrix[idx, 20] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][0]["mR@50"])
			values_matrix[idx, 21] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][1]["mR@10"])
			values_matrix[idx, 22] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][1]["mR@20"])
			values_matrix[idx, 23] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][1]["mR@50"])
			values_matrix[idx, 24] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][2]["mR@10"])
			values_matrix[idx, 25] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][2]["mR@20"])
			values_matrix[idx, 26] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][2]["mR@50"])
		return values_matrix
	
	@staticmethod
	def fill_sga_combined_values_matrix_mean_recall_no_sc(values_matrix, mean_recall_results_json,
	                                                      idx, comb_method_name, cf, partial_percentage='10'):
		if "full" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][1]["mR@10"])
			values_matrix[idx, 4] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][1]["mR@20"])
			values_matrix[idx, 5] = fetch_value(mean_recall_results_json[comb_method_name]["sgdet"][cf][1]["mR@50"])
			values_matrix[idx, 6] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][0]["mR@10"])
			values_matrix[idx, 7] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][0]["mR@20"])
			values_matrix[idx, 8] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][0]["mR@50"])
			values_matrix[idx, 9] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][1]["mR@10"])
			values_matrix[idx, 10] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][1]["mR@20"])
			values_matrix[idx, 11] = fetch_value(mean_recall_results_json[comb_method_name]["sgcls"][cf][1]["mR@50"])
			values_matrix[idx, 12] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][0]["mR@10"])
			values_matrix[idx, 13] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][0]["mR@20"])
			values_matrix[idx, 14] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][0]["mR@50"])
			values_matrix[idx, 15] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][1]["mR@10"])
			values_matrix[idx, 16] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][1]["mR@20"])
			values_matrix[idx, 17] = fetch_value(mean_recall_results_json[comb_method_name]["predcls"][cf][1]["mR@50"])
		elif "partial" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][1]["mR@10"])
			values_matrix[idx, 4] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][1]["mR@20"])
			values_matrix[idx, 5] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgdet"][cf][1]["mR@50"])
			values_matrix[idx, 6] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][0]["mR@10"])
			values_matrix[idx, 7] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][0]["mR@20"])
			values_matrix[idx, 8] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][0]["mR@50"])
			values_matrix[idx, 9] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][1]["mR@10"])
			values_matrix[idx, 10] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][1]["mR@20"])
			values_matrix[idx, 11] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["sgcls"][cf][1]["mR@50"])
			values_matrix[idx, 12] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][0]["mR@10"])
			values_matrix[idx, 13] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][0]["mR@20"])
			values_matrix[idx, 14] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][0]["mR@50"])
			values_matrix[idx, 15] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][1]["mR@10"])
			values_matrix[idx, 16] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][1]["mR@20"])
			values_matrix[idx, 17] = fetch_value(
				mean_recall_results_json[comb_method_name][partial_percentage]["predcls"][cf][1]["mR@50"])
		
		return values_matrix
	
	@staticmethod
	def generate_sga_combined_paper_latex_header():
		latex_header = "\\begin{table*}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Mean Recall Results for SGA.}\n"
		latex_header += "    \\label{tab:sga_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|l|cccccc|cccccc|cccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         \\multirow{3}{*}{$\\mathcal{F}$} & \\multirow{3}{*}{Method} & \\multicolumn{6}{c}{\\textbf{SGDET}} & \\multicolumn{6}{c}{\\textbf{SGCLS}} & \\multicolumn{6}{c}{\\textbf{PREDCLS}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \\cmidrule(lr){15-20} \n "
		latex_header += "         & & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}}  & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}}  \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-20} \n "
		
		latex_header += (
				"      &   & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} " + " \\\\ \\hline\n")
		
		return latex_header
	
	@staticmethod
	def generate_combined_wn_recalls_latex_header(setting_name, mode):
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Results for " + setting_name + ", when trained using anticipatory horizon of 3 future frames.}\n"
		latex_header += "    \\label{tab:anticipation_results_" + mode + "}\n"
		latex_header += "    \\setlength{\\tabcolsep}{5pt} \n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{ll|cccccc|cccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         & & \\multicolumn{6}{c|}{\\textbf{Recall (R)}} & \\multicolumn{6}{c}{\\textbf{Mean Recall (mR)}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-8} \\cmidrule(lr){9-14} \n "
		latex_header += "        \\multicolumn{2}{c|}{\\textbf{" + setting_name + "}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c|}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}}\\\\ \n"
		latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){3-5} \\cmidrule(lr){6-8}\\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \n "
		latex_header += ("        $\\mathcal{F}$ & \\textbf{Method} & \\textbf{10} & \\textbf{20} & \\textbf{50} & "
		                 "\\textbf{10} & \\textbf{20} & \\textbf{50} & "
		                 "\\textbf{10} & \\textbf{20} & \\textbf{50}  & "
		                 "\\textbf{10} & \\textbf{20} & \\textbf{50}   \\\\ \\hline\n")
		return latex_header
	
	@staticmethod
	def generate_sga_sgcls_predcls_paper_latex_header():
		latex_header = "\\begin{table*}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{SGCLS and PREDCLS Mean Recall Results for SGA.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|l|ccccccccc|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "         \\multirow{3}{*}{$\\mathcal{F}$} & \\multirow{3}{*}{Method} & \\multicolumn{9}{c}{\\textbf{SGCLS}} & \\multicolumn{9}{c}{\\textbf{PREDCLS}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-11} \\cmidrule(lr){12-20} \n "
		latex_header += "         & & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-20} \n "
		
		latex_header += (
				"      &   & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} " + " \\\\ \\hline\n")
		
		return latex_header
	
	@staticmethod
	def generate_sga_sgdet_paper_latex_header():
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{SGDET Mean Recall Results for SGA.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\linewidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|l|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "        \\multirow{3}{*}{$\\mathcal{F}$} & \\multirow{3}{*}{Method} & \\multicolumn{9}{c}{\\textbf{SGDET}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-11} \n "
		latex_header += "        & & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{3}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11}  \n "
		
		latex_header += (
				"    &    & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50}  & "
				"\\textbf{@10} & \\textbf{@20} & \\textbf{@50} " + " \\\\ \\hline\n")
		
		return latex_header
	
	def generate_paper_sga_mean_recall_latex_table(self, sga_mean_recall_results_json):
		latex_file_name = f"sga_mean_recall.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_latex_tables",
		                               latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sga_combined_paper_latex_header()
		
		num_methods = len(self.latex_method_list)
		
		total_rows = num_methods * len(self.latex_context_fraction_list)
		values_matrix = np.zeros((total_rows, 18), dtype=np.float32)
		
		row_counter = 0
		for cf in self.latex_context_fraction_list:
			for method in self.method_list:
				for scenario in self.scenario_list:
					comb_method_name = f"{method}_{scenario}"
					
					if comb_method_name not in self.latex_method_list:
						continue
					
					values_matrix = self.fill_sga_combined_values_matrix_mean_recall_no_sc(
						values_matrix=values_matrix,
						mean_recall_results_json=sga_mean_recall_results_json,
						idx=row_counter,
						comb_method_name=comb_method_name,
						cf=cf)
					row_counter += 1
		
		max_value_boolean_matrx = np.zeros(values_matrix.shape, dtype=np.bool)
		# For every column, take two rows at a time and find the max value
		# find the column-wise max value and make the corresponding row, column index True in the max_value_boolean_matrix
		for col_idx in range(18):
			for row_idx in range(0, total_rows, 2):
				if values_matrix[row_idx, col_idx] > values_matrix[row_idx + 1, col_idx]:
					max_value_boolean_matrx[row_idx, col_idx] = True
				else:
					max_value_boolean_matrx[row_idx + 1, col_idx] = True
		
		row_counter = 0
		for cf in self.latex_context_fraction_list:
			for method in self.method_list:
				for scenario in self.scenario_list:
					comb_method_name = f"{method}_{scenario}"
					
					if comb_method_name not in self.latex_method_list:
						continue
					
					latex_method_name = self.fetch_method_name_latex(comb_method_name)
					
					if row_counter % num_methods == 0:
						latex_row = f"        \\multirow{{{num_methods}}}{{*}}{{{cf}}} &"
					else:
						latex_row = "        &"
					
					latex_row += f"        {latex_method_name}"
					
					for col_idx in range(18):
						if max_value_boolean_matrx[row_counter, col_idx]:
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
						else:
							latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
					
					if row_counter % 2 == 1:
						latex_row += "  \\\\ \n"
						if (row_counter % num_methods) == (num_methods - 1):
							latex_row += "          \\hline \n"
						else:
							latex_row += "          \\cmidrule(lr){2-11} \\cmidrule(lr){12-20} \n"
					else:
						latex_row += "  \\\\ \n"
					
					latex_table += latex_row
					row_counter += 1
		
		latex_footer = self.generate_full_width_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)
	
	def generate_paper_sga_mean_recall_sgcls_predcls_latex_table(self, sga_mean_recall_results_json):
		latex_file_name = f"sga_mean_recall_sgcls_predcls.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_latex_tables",
		                               latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sga_sgcls_predcls_paper_latex_header()
		
		values_matrix = np.zeros((48, 27), dtype=np.float32)
		
		row_counter = 0
		for cf in self.context_fraction_list:
			for method in self.method_list:
				for scenario in self.scenario_list:
					comb_method_name = f"{method}_{scenario}"
					values_matrix = self.fill_sga_combined_values_matrix_mean_recall(
						values_matrix=values_matrix,
						mean_recall_results_json=sga_mean_recall_results_json,
						idx=row_counter,
						comb_method_name=comb_method_name,
						cf=cf)
					row_counter += 1
		
		row_counter = 0
		for cf in self.context_fraction_list:
			for method in self.method_list:
				for scenario in self.scenario_list:
					comb_method_name = f"{method}_{scenario}"
					latex_method_name = self.fetch_method_name_latex(comb_method_name)
					
					if row_counter % 12 == 0:
						latex_row = f"        \\multirow{{12}}{{*}}{{{cf}}} &"
					else:
						latex_row = "        &"
					
					if row_counter % 2 == 1:
						latex_row += f"        {latex_method_name}"
						for col_idx in range(9, 27):
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
						latex_row += "  \\\\ \n"
						latex_row += "          \\cmidrule(lr){3-11} \\cmidrule(lr){12-20} \n"
					else:
						latex_row += f"        {latex_method_name}"
						for col_idx in range(9, 27):
							latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
						latex_row += "  \\\\ \n"
					latex_table += latex_row
					row_counter += 1
		
		latex_footer = self.generate_full_width_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)
	
	def generate_paper_sga_mean_recall_sgdet_latex_table(self, sga_mean_recall_results_json):
		latex_file_name = f"sga_mean_recall_sgdet.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_latex_tables",
		                               latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sga_sgdet_paper_latex_header()
		
		values_matrix = np.zeros((48, 27), dtype=np.float32)
		
		row_counter = 0
		for cf in self.context_fraction_list:
			for method in self.method_list:
				for scenario in self.scenario_list:
					comb_method_name = f"{method}_{scenario}"
					values_matrix = self.fill_sga_combined_values_matrix_mean_recall(
						values_matrix=values_matrix,
						mean_recall_results_json=sga_mean_recall_results_json,
						idx=row_counter,
						comb_method_name=comb_method_name,
						cf=cf)
					row_counter += 1
		
		row_counter = 0
		for cf in self.context_fraction_list:
			for method in self.method_list:
				for scenario in self.scenario_list:
					comb_method_name = f"{method}_{scenario}"
					latex_method_name = self.fetch_method_name_latex(comb_method_name)
					
					if row_counter % 12 == 0:
						latex_row = f"        \\multirow{{12}}{{*}}{{{cf}}} &"
					else:
						latex_row = "        &"
					
					if row_counter % 2 == 1:
						latex_row += f"        {latex_method_name}"
						for col_idx in range(9):
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
						latex_row += "  \\\\ \n"
						latex_row += "          \\cmidrule(lr){3-11} \n"
					else:
						latex_row += f"        {latex_method_name}"
						for col_idx in range(9):
							latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
						latex_row += "  \\\\ \n"
					latex_table += latex_row
					row_counter += 1
		latex_footer = self.generate_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)
	
	def generate_paper_full_sga_latex_table(self, sga_results_json):
		for mode in ["sgdet", "sgcls", "predcls"]:
			latex_file_name = f"sga_results_{mode}.tex"
			latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs",
			                               "paper_full_sga_latex_tables",
			                               latex_file_name)
			os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
			setting_name = f"SGA of {self.fetch_sga_mode_name_latex(mode)}"
			latex_table = self.generate_combined_wn_recalls_latex_header(setting_name, mode)
			values_matrix = np.zeros((24, 12), dtype=np.float32)
			
			counter = 0
			for cf in self.context_fraction_list:
				for method in self.method_list:
					values_matrix = self.fill_full_sga_values_matrix(values_matrix, sga_results_json, counter, method,
					                                                 mode, cf)
					counter += 1
			
			max_boolean_matrix = np.zeros(values_matrix.shape, dtype=np.bool)
			for col_idx in range(12):
				for row_idx in range(0, 24, 6):
					method_metric_values_matrix = values_matrix[row_idx:row_idx + 6, col_idx]
					max_idx = np.argmax(method_metric_values_matrix)
					max_boolean_matrix[row_idx + max_idx, col_idx] = True
			
			row_counter = 0
			for cf in self.context_fraction_list:
				# Multirow for each context fraction
				for method in self.method_list:
					latex_method_name = self.fetch_full_sga_method_name_latex(method)
					# For starting row of each method in a context fraction
					if row_counter % 6 == 0:
						latex_row = f"        \\multirow{{6}}{{*}}{{{cf}}} &"
					else:
						latex_row = "        &"
					
					latex_row += f"        {latex_method_name}"
					
					for col_idx in range(12):
						if max_boolean_matrix[row_counter, col_idx]:
							latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
						else:
							latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
					
					if row_counter % 6 == 5:
						latex_row += "  \\\\ \n"
						if (row_counter % 24) == 23:
							latex_row += "          \\hline \n"
						else:
							latex_row += "          \\cmidrule(lr){2-7} \\cmidrule(lr){8-13} \n"
					else:
						latex_row += "  \\\\ \n"
					
					latex_table += latex_row
					
					row_counter += 1
			
			latex_footer = self.generate_latex_footer()
			
			latex_table += latex_footer
			
			with open(latex_file_path, "a", newline='') as latex_file:
				latex_file.write(latex_table)


def prepare_paper_full_sga_latex_tables():
	prepare_paper_results_sga = PreparePaperResultsSGA()
	sga_results_json = prepare_paper_results_sga.full_sga_results_json()
	prepare_paper_results_sga.generate_paper_full_sga_latex_table(sga_results_json)


def prepare_paper_sga_latex_tables():
	prepare_paper_results_sga = PreparePaperResultsSGA()
	sga_mean_recall_results_json = prepare_paper_results_sga.fetch_sga_mean_recall_results_json_latex()
	# prepare_paper_results_sga.generate_paper_sga_mean_recall_sgcls_predcls_latex_table(sga_mean_recall_results_json)
	# prepare_paper_results_sga.generate_paper_sga_mean_recall_sgdet_latex_table(sga_mean_recall_results_json)
	prepare_paper_results_sga.generate_paper_sga_mean_recall_latex_table(sga_mean_recall_results_json)


def main():
	prepare_paper_results_sga = PreparePaperResultsSGA()
	prepare_paper_results_sga.compile_sga_method_wise_results_csvs()


def combine_results():
	prepare_paper_results_sga = PreparePaperResultsSGA()
	prepare_paper_results_sga.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\paper_sga_results_csvs",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\paper_sga_results_csvs\sga_combined_results.xlsx"
	)


if __name__ == '__main__':
	# main()
	# combine_results()
	# prepare_paper_sga_latex_tables()
	prepare_paper_full_sga_latex_tables()