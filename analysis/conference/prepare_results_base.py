import os

import numpy as np
import pandas as pd

from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


def formatted_metric_num(metric_num):
	if metric_num == "-":
		return metric_num
	else:
		return round(float(metric_num) * 100, 2)


def formatted_easg_metric_num(metric_num):
	if metric_num == "-":
		return metric_num
	else:
		return round(float(metric_num), 2)


def fetch_value(value_string):
	if value_string == "-":
		return 0.0
	else:
		return round(float(value_string), 1)


def fetch_rounded_value(value):
	return round(float(value), 1)


class PrepareResultsBase:
	
	def __init__(self):
		self.db_service = FirebaseService()
		self.database_name = "results_31_10"
		
		self.sga_database_name = "results_21_11_sga"
		self.sgg_database_name = "results_21_11_sgg"
		self.easg_database_name = "results_2_11_easg"
		self.sgg_corruptions_database_name = "results_14_11_sgg_corruptions"
		self.sga_corruptions_database_name = "results_14_11_sga_corruptions"
		
		self.sgg_mode_list = ["sgcls", "sgdet", "predcls"]
		self.sga_mode_list = ["sgcls", "sgdet", "predcls"]
		
		self.proposed_method_name = "ImparTail"
	
	def fetch_db_sgg_results(self):
		results_dict = self.db_service.fetch_results_from_db(self.sgg_database_name)
		sgg_db_result_list = []
		for result_id, result_dict in results_dict.items():
			result = Result.from_dict(result_dict)
			if result.task_name == const.SGG:
				sgg_db_result_list.append(result)
		return sgg_db_result_list
	
	def fetch_db_sgg_corruptions_results(self):
		results_dict = self.db_service.fetch_results_from_db(self.sgg_corruptions_database_name)
		sgg_db_result_list = []
		for result_id, result_dict in results_dict.items():
			result = Result.from_dict(result_dict)
			if result.task_name == const.SGG:
				sgg_db_result_list.append(result)
		return sgg_db_result_list
	
	def fetch_db_sga_corruptions_results(self):
		results_dict = self.db_service.fetch_results_from_db(self.sga_corruptions_database_name)
		sgg_db_result_list = []
		for result_id, result_dict in results_dict.items():
			result = Result.from_dict(result_dict)
			if result.task_name == const.SGG:
				sgg_db_result_list.append(result)
		return sgg_db_result_list
	
	def fetch_db_sga_results(self):
		results_dict = self.db_service.fetch_results_from_db(self.sga_database_name)
		sgg_db_result_list = []
		for result_id, result_dict in results_dict.items():
			result = Result.from_dict(result_dict)
			if result.task_name == const.SGA:
				sgg_db_result_list.append(result)
		return sgg_db_result_list
	
	def fetch_db_easg_results(self):
		results_dict = self.db_service.fetch_results_from_db(self.easg_database_name)
		easg_db_result_list = []
		for result_id, result_dict in results_dict.items():
			result = Result.from_dict(result_dict)
			if result.task_name == const.EASG:
				easg_db_result_list.append(result)
		return easg_db_result_list
	
	@staticmethod
	def fetch_paper_mean_recall_empty_metrics_json():
		metrics_json = {}
		for i in range(3):
			metrics_json[i] = {
				"mR@10": "-",
				"mR@20": "-",
				"mR@50": "-",
				"mR@100": "-"
			}
		return metrics_json
	
	@staticmethod
	def fetch_paper_recall_empty_metrics_json():
		metrics_json = {}
		for i in range(3):
			metrics_json[i] = {
				"R@10": "-",
				"R@20": "-",
				"R@50": "-",
				"R@100": "-"
			}
		return metrics_json
	
	# 0 - With constraint evaluator
	# 1 - No constraint evaluator
	# 2 - Semi Constraint evaluator
	@staticmethod
	def fetch_empty_metrics_json():
		metrics_json = {}
		for i in range(3):
			metrics_json[i] = {
				"R@10": "-",
				"R@20": "-",
				"R@50": "-",
				"R@100": "-",
				"mR@10": "-",
				"mR@20": "-",
				"mR@50": "-",
				"mR@100": "-",
				"hR@10": "-",
				"hR@20": "-",
				"hR@50": "-",
				"hR@100": "-"
			}
		return metrics_json
	
	@staticmethod
	def fetch_easg_empty_metrics_json():
		metrics_json = {}
		for i in range(2):
			metrics_json[i] = {
				"R@10": "-",
				"R@20": "-",
				"R@50": "-",
				"R@100": "-",
				"mR@10": "-",
				"mR@20": "-",
				"mR@50": "-",
				"mR@100": "-",
				"hR@10": "-",
				"hR@20": "-",
				"hR@50": "-",
				"hR@100": "-"
			}
		return metrics_json
	
	@staticmethod
	def fetch_corruption_name(corruption_type):
		words = corruption_type.split("_")
		dataset_corruption_type = words[0]
		video_corruption_type = words[1]
		severity_level = words[-1]
		
		if dataset_corruption_type == "fixed" and video_corruption_type == "fixed":
			corruption_name = "_".join(words[2:-1])
		else:
			corruption_name = "_".join(words[0:-2])
		
		return corruption_name, severity_level
	
	@staticmethod
	def fetch_paper_completed_mean_recall_metrics_json(
			with_constraint_metrics,
			no_constraint_metrics,
			semi_constraint_metrics
	):
		metrics_json = {
			0: {
				"mR@10": formatted_metric_num(with_constraint_metrics.mean_recall_10),
				"mR@20": formatted_metric_num(with_constraint_metrics.mean_recall_20),
				"mR@50": formatted_metric_num(with_constraint_metrics.mean_recall_50),
				"mR@100": formatted_metric_num(with_constraint_metrics.mean_recall_100)
			},
			1: {
				"mR@10": formatted_metric_num(no_constraint_metrics.mean_recall_10),
				"mR@20": formatted_metric_num(no_constraint_metrics.mean_recall_20),
				"mR@50": formatted_metric_num(no_constraint_metrics.mean_recall_50),
				"mR@100": formatted_metric_num(no_constraint_metrics.mean_recall_100)
			},
			2: {
				"mR@10": formatted_metric_num(semi_constraint_metrics.mean_recall_10),
				"mR@20": formatted_metric_num(semi_constraint_metrics.mean_recall_20),
				"mR@50": formatted_metric_num(semi_constraint_metrics.mean_recall_50),
				"mR@100": formatted_metric_num(semi_constraint_metrics.mean_recall_100)
			}
		}
		
		return metrics_json
	
	@staticmethod
	def fetch_paper_completed_recall_metrics_json(
			with_constraint_metrics,
			no_constraint_metrics,
			semi_constraint_metrics
	):
		metrics_json = {
			0: {
				"R@10": formatted_metric_num(with_constraint_metrics.recall_10),
				"R@20": formatted_metric_num(with_constraint_metrics.recall_20),
				"R@50": formatted_metric_num(with_constraint_metrics.recall_50),
				"R@100": formatted_metric_num(with_constraint_metrics.recall_100)
			},
			1: {
				"R@10": formatted_metric_num(no_constraint_metrics.recall_10),
				"R@20": formatted_metric_num(no_constraint_metrics.recall_20),
				"R@50": formatted_metric_num(no_constraint_metrics.recall_50),
				"R@100": formatted_metric_num(no_constraint_metrics.recall_100)
			},
			2: {
				"R@10": formatted_metric_num(semi_constraint_metrics.recall_10),
				"R@20": formatted_metric_num(semi_constraint_metrics.recall_20),
				"R@50": formatted_metric_num(semi_constraint_metrics.recall_50),
				"R@100": formatted_metric_num(semi_constraint_metrics.recall_100)
			}
		}
		
		return metrics_json
	
	@staticmethod
	def fetch_completed_metrics_json(
			with_constraint_metrics,
			no_constraint_metrics,
			semi_constraint_metrics
	):
		metrics_json = {
			0: {
				"R@10": formatted_metric_num(with_constraint_metrics.recall_10),
				"R@20": formatted_metric_num(with_constraint_metrics.recall_20),
				"R@50": formatted_metric_num(with_constraint_metrics.recall_50),
				"R@100": formatted_metric_num(with_constraint_metrics.recall_100),
				"mR@10": formatted_metric_num(with_constraint_metrics.mean_recall_10),
				"mR@20": formatted_metric_num(with_constraint_metrics.mean_recall_20),
				"mR@50": formatted_metric_num(with_constraint_metrics.mean_recall_50),
				"mR@100": formatted_metric_num(with_constraint_metrics.mean_recall_100),
				"hR@10": formatted_metric_num(with_constraint_metrics.harmonic_recall_10),
				"hR@20": formatted_metric_num(with_constraint_metrics.harmonic_recall_20),
				"hR@50": formatted_metric_num(with_constraint_metrics.harmonic_recall_50),
				"hR@100": formatted_metric_num(with_constraint_metrics.harmonic_recall_100)
			},
			1: {
				"R@10": formatted_metric_num(no_constraint_metrics.recall_10),
				"R@20": formatted_metric_num(no_constraint_metrics.recall_20),
				"R@50": formatted_metric_num(no_constraint_metrics.recall_50),
				"R@100": formatted_metric_num(no_constraint_metrics.recall_100),
				"mR@10": formatted_metric_num(no_constraint_metrics.mean_recall_10),
				"mR@20": formatted_metric_num(no_constraint_metrics.mean_recall_20),
				"mR@50": formatted_metric_num(no_constraint_metrics.mean_recall_50),
				"mR@100": formatted_metric_num(no_constraint_metrics.mean_recall_100),
				"hR@10": formatted_metric_num(no_constraint_metrics.harmonic_recall_10),
				"hR@20": formatted_metric_num(no_constraint_metrics.harmonic_recall_20),
				"hR@50": formatted_metric_num(no_constraint_metrics.harmonic_recall_50),
				"hR@100": formatted_metric_num(no_constraint_metrics.harmonic_recall_100)
			},
			2: {
				"R@10": formatted_metric_num(semi_constraint_metrics.recall_10),
				"R@20": formatted_metric_num(semi_constraint_metrics.recall_20),
				"R@50": formatted_metric_num(semi_constraint_metrics.recall_50),
				"R@100": formatted_metric_num(semi_constraint_metrics.recall_100),
				"mR@10": formatted_metric_num(semi_constraint_metrics.mean_recall_10),
				"mR@20": formatted_metric_num(semi_constraint_metrics.mean_recall_20),
				"mR@50": formatted_metric_num(semi_constraint_metrics.mean_recall_50),
				"mR@100": formatted_metric_num(semi_constraint_metrics.mean_recall_100),
				"hR@10": formatted_metric_num(semi_constraint_metrics.harmonic_recall_10),
				"hR@20": formatted_metric_num(semi_constraint_metrics.harmonic_recall_20),
				"hR@50": formatted_metric_num(semi_constraint_metrics.harmonic_recall_50),
				"hR@100": formatted_metric_num(semi_constraint_metrics.harmonic_recall_100)
			}
		}
		
		return metrics_json
	
	@staticmethod
	def fetch_easg_completed_metrics_json(
			with_constraint_metrics,
			no_constraint_metrics
	):
		metrics_json = {
			0: {
				"R@10": formatted_easg_metric_num(with_constraint_metrics.recall_10),
				"R@20": formatted_easg_metric_num(with_constraint_metrics.recall_20),
				"R@50": formatted_easg_metric_num(with_constraint_metrics.recall_50),
				"R@100": formatted_easg_metric_num(with_constraint_metrics.recall_100),
				"mR@10": formatted_easg_metric_num(with_constraint_metrics.mean_recall_10),
				"mR@20": formatted_easg_metric_num(with_constraint_metrics.mean_recall_20),
				"mR@50": formatted_easg_metric_num(with_constraint_metrics.mean_recall_50),
				"mR@100": formatted_easg_metric_num(with_constraint_metrics.mean_recall_100),
				"hR@10": formatted_easg_metric_num(with_constraint_metrics.harmonic_recall_10),
				"hR@20": formatted_easg_metric_num(with_constraint_metrics.harmonic_recall_20),
				"hR@50": formatted_easg_metric_num(with_constraint_metrics.harmonic_recall_50),
				"hR@100": formatted_easg_metric_num(with_constraint_metrics.harmonic_recall_100)
			},
			1: {
				"R@10": formatted_easg_metric_num(no_constraint_metrics.recall_10),
				"R@20": formatted_easg_metric_num(no_constraint_metrics.recall_20),
				"R@50": formatted_easg_metric_num(no_constraint_metrics.recall_50),
				"R@100": formatted_easg_metric_num(no_constraint_metrics.recall_100),
				"mR@10": formatted_easg_metric_num(no_constraint_metrics.mean_recall_10),
				"mR@20": formatted_easg_metric_num(no_constraint_metrics.mean_recall_20),
				"mR@50": formatted_easg_metric_num(no_constraint_metrics.mean_recall_50),
				"mR@100": formatted_easg_metric_num(no_constraint_metrics.mean_recall_100),
				"hR@10": formatted_easg_metric_num(no_constraint_metrics.harmonic_recall_10),
				"hR@20": formatted_easg_metric_num(no_constraint_metrics.harmonic_recall_20),
				"hR@50": formatted_easg_metric_num(no_constraint_metrics.harmonic_recall_50),
				"hR@100": formatted_easg_metric_num(no_constraint_metrics.harmonic_recall_100)
			}
		}
		
		return metrics_json
	
	@staticmethod
	def fetch_ref_tab_name(mode, eval_horizon, train_future_frame):
		if mode == "sgdet":
			ref_tab_name = f"sgdet_{eval_horizon}_{train_future_frame}"
		elif mode == "sgcls":
			ref_tab_name = f"sgcls_{eval_horizon}_{train_future_frame}"
		elif mode == "predcls":
			ref_tab_name = f"predcls_{eval_horizon}_{train_future_frame}"
		return ref_tab_name
	
	@staticmethod
	def fetch_method_name_json(method_name):
		if method_name == "NeuralODE" or method_name == "ode":
			method_name = "SceneSayerODE"
		elif method_name == "NeuralSDE" or method_name == "sde":
			method_name = "SceneSayerSDE"
		elif method_name == "sttran_ant":
			method_name = "STTran+"
		elif method_name == "sttran_gen_ant":
			method_name = "STTran++"
		elif method_name == "dsgdetr_ant":
			method_name = "DSGDetr+"
		elif method_name == "dsgdetr_gen_ant":
			method_name = "DSGDetr++"
		elif method_name == "sde_wo_bb":
			method_name = "SceneSayerSDE(w/oBB)"
		elif method_name == "sde_wo_recon":
			method_name = "SceneSayerSDE(w/oRecon)"
		elif method_name == "sde_wo_gen":
			method_name = "SceneSayerSDE(w/oGenLoss)"
		elif method_name == "sttran":
			method_name = "STTran"
		elif method_name == "dsgdetr":
			method_name = "DSGDetr"
		elif method_name == "tempura":
			method_name = "Tempura"
		return method_name
	
	@staticmethod
	def combine_csv_to_excel(folder_path, output_file):
		# Create a Pandas Excel writer using openpyxl as the engine
		writer = pd.ExcelWriter(output_file, engine='openpyxl')
		
		# Iterate over all CSV files in the folder
		for csv_file in os.listdir(folder_path):
			if csv_file.endswith('.csv'):
				# Read the CSV file
				df = pd.read_csv(os.path.join(folder_path, csv_file))
				
				# Write the data frame to a sheet named after the CSV file
				sheet_name = os.path.splitext(csv_file)[0]
				df.to_excel(writer, sheet_name=sheet_name, index=False)
		
		# Save the Excel file
		writer.save()
	
	@staticmethod
	def fetch_eval_setting_name_nocap_latex(eval_setting_name):
		if eval_setting_name == "with_constraint":
			eval_setting_name = "With Constraint"
		elif eval_setting_name == "no_constraint":
			eval_setting_name = "No Constraint"
		elif eval_setting_name == "semi_constraint":
			eval_setting_name = "Semi Constraint"
		
		return eval_setting_name
	
	@staticmethod
	def fetch_eval_setting_id_latex(eval_setting_name):
		if eval_setting_name == "with_constraint":
			eval_setting_id = 0
		elif eval_setting_name == "no_constraint":
			eval_setting_id = 1
		elif eval_setting_name == "semi_constraint":
			eval_setting_id = 2
		
		return eval_setting_id
	
	@staticmethod
	def fetch_eval_setting_name_cap_latex(eval_setting_name):
		if eval_setting_name == "with_constraint":
			eval_setting_name = "WITH CONSTRAINT"
		elif eval_setting_name == "no_constraint":
			eval_setting_name = "NO CONSTRAINT"
		elif eval_setting_name == "semi_constraint":
			eval_setting_name = "SEMI CONSTRAINT"
		
		return eval_setting_name
	
	@staticmethod
	def calculate_percentage_changes(base_values, cur_values):
		percentage_change = np.zeros(base_values.shape, dtype=np.float32)
		for i in range(base_values.shape[0]):
			if base_values[i] != 0:
				percentage_change[i] = ((cur_values[i] - base_values[i]) / base_values[i]) * 100
			else:
				percentage_change[i] = 0.0
		return percentage_change
	
	@staticmethod
	def fetch_sgg_mode_name_latex(mode):
		if mode == "sgdet":
			setting_name = "\\textbf{SGDET}"
		elif mode == "sgcls":
			setting_name = "\\textbf{SGCLS}"
		elif mode == "predcls":
			setting_name = "\\textbf{PREDCLS}"
		return setting_name
	
	@staticmethod
	def fetch_sga_mode_name_latex(mode):
		if mode == "sgdet":
			setting_name = "\\textbf{AGS}"
		elif mode == "sgcls":
			setting_name = "\\textbf{PGAGS}"
		elif mode == "predcls":
			setting_name = "\\textbf{GAGS}"
		return setting_name
	
	@staticmethod
	def fetch_sga_cf_name_latex(cf):
		if cf == "0.3" or cf == 0.3:
			cf = str(30)
		elif cf == "0.5" or cf == 0.5:
			cf = str(50)
		elif cf == "0.7" or cf == 0.7:
			cf = str(70)
		elif cf == "0.9" or cf == 0.9:
			cf = str(90)
		else:
			raise ValueError(f"Invalid context fraction: {cf}")
		return cf
	
	def fetch_method_name_latex(self, method_name):
		"""
		method_name will be of the form: sttran_partial, dsgdetr_partial, ode_partial, sde_partial, sttran_ant_partial
		"""
		method_name = method_name.lower()
		if method_name in ["sttran", "sttran_full"]:
			method_name = "STTran~\cite{cong_et_al_sttran_2021}"
		elif method_name in ["dsgdetr", "dsgdetr_full"]:
			method_name = "DSGDetr~\cite{Feng_2021}"
		elif method_name == "sttran_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "dsgdetr_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name in ["sttran_ant", "sttran_ant_full"]:
			method_name = "STTran+~\cite{peddi_et_al_scene_sayer_2024}"
		elif method_name in ["dsgdetr_ant", "dsgdetr_ant_full"]:
			method_name = "DSGDetr+~\cite{peddi_et_al_scene_sayer_2024}"
		elif method_name in ["sttran_gen_ant", "sttran_gen_ant_full"]:
			method_name = "STTran++~\cite{peddi_et_al_scene_sayer_2024}"
		elif method_name in ["dsgdetr_gen_ant", "dsgdetr_gen_ant_full"]:
			method_name = "DSGDetr++~\cite{peddi_et_al_scene_sayer_2024}"
		elif method_name in ["ode", "ode_full"]:
			method_name = "SceneSayerODE~\cite{peddi_et_al_scene_sayer_2024}"
		elif method_name in ["sde", "sde_full"]:
			method_name = "SceneSayerSDE~\cite{peddi_et_al_scene_sayer_2024}"
		elif method_name == "sttran_ant_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "dsgdetr_ant_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "sttran_gen_ant_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "dsgdetr_gen_ant_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "ode_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "sde_partial":
			method_name = f"\\quad+\\textbf{{\\methodname(Ours)}}"
		elif method_name == "tempura":
			method_name = "Tempura~\cite{tempura_2021}"
		return method_name
	
	def fetch_full_sga_method_name_latex(self, method_name):
		if method_name == "NeuralODE" or method_name == "ode":
			method_name = "\\textbf{SceneSayerODE (Ours)}"
		elif method_name == "NeuralSDE" or method_name == "sde":
			method_name = "\\textbf{SceneSayerSDE (Ours)}"
		elif method_name == "sttran_ant":
			method_name = "STTran+ \cite{cong_et_al_sttran_2021}"
		elif method_name == "sttran_gen_ant":
			method_name = "STTran++ \cite{cong_et_al_sttran_2021}"
		elif method_name == "dsgdetr_ant":
			method_name = "DSGDetr+ \cite{Feng_2021}"
		elif method_name == "dsgdetr_gen_ant":
			method_name = "DSGDetr++ \cite{Feng_2021}"
		elif method_name == "sde_wo_bb":
			method_name = "\\textbf{SceneSayerSDE (w/o BB)}"
		elif method_name == "sde_wo_recon":
			method_name = "\\textbf{SceneSayerSDE (w/o Recon)}"
		elif method_name == "sde_wo_gen":
			method_name = "\\textbf{SceneSayerSDE (w/o GenLoss)}"
		return method_name
	
	@staticmethod
	def generate_latex_footer():
		latex_footer = "    \\end{tabular}\n"
		latex_footer += "    }\n"
		latex_footer += "\\end{table}\n"
		return latex_footer
	
	@staticmethod
	def generate_full_width_latex_footer():
		latex_footer = "    \\end{tabular}\n"
		latex_footer += "    }\n"
		latex_footer += "\\end{table*}\n"
		return latex_footer
