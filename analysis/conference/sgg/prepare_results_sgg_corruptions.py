import csv
import os

import numpy as np

from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value, fetch_rounded_value
from constants import CorruptionConstants as const


class PrepareResultsSGGCorruptions(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGGCorruptions, self).__init__()
		self.mode_list = ["sgcls", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		self.scenario_list = ["full", "partial"]
		self.partial_percentages = ["10"]
		
		self.corruption_types = [
			const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
			const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.FOG, const.FROST,
			const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.PIXELATE,
			const.JPEG_COMPRESSION, const.SUN_GLARE, const.DUST, const.SATURATE
		]
		self.dataset_corruption_modes = [const.FIXED, const.MIXED]
		self.video_corruption_modes = [const.FIXED, const.MIXED]
		self.severity_levels = ["5"]
		
		self.latex_mode_list = ["predcls"]
		self.latex_method_list = ["sttran"]
		self.latex_scenario_list = ["full", "partial"]
		
		self.latex_corruption_types = [
			const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
			const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.FOG, const.FROST,
			const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.PIXELATE,
			const.JPEG_COMPRESSION, const.SUN_GLARE, const.DUST, const.SATURATE
		]
		
		# self.latex_corruption_types = [
		# 	const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
		# 	const.FOG, const.FROST, const.BRIGHTNESS, const.SUN_GLARE
		# ]
		
		self.corruption_type_latex_name_map = {
			const.GAUSSIAN_NOISE: "Gaussian Noise",
			const.SHOT_NOISE: "Shot Noise",
			const.IMPULSE_NOISE: "Impulse Noise",
			const.SPECKLE_NOISE: "Speckle Noise",
			const.GAUSSIAN_BLUR: "Gaussian Blur",
			const.DEFOCUS_BLUR: "Defocus Blur",
			const.FOG: "Fog",
			const.FROST: "Frost",
			const.SPATTER: "Spatter",
			const.CONTRAST: "Contrast",
			const.BRIGHTNESS: "Brightness",
			const.PIXELATE: "Pixelate",
			const.JPEG_COMPRESSION: "Compression",
			const.SUN_GLARE: "Sun Glare",
			const.DUST: "Dust",
			const.SATURATE: "Saturate"
		}
		
		self.task_name = "sgg"
	
	# -----------------------------------------------------------------------------------------------
	# ----------------------------- DATA GENERATION FILES -----------------------------------------
	# -----------------------------------------------------------------------------------------------
	
	def fetch_sgg_results_json(self):
		db_results = self.fetch_db_sgg_corruptions_results()
		sgg_results_json = {}
		for mode in self.mode_list:
			sgg_results_json[mode] = {}
			for method_name in self.method_list:
				sgg_results_json[mode][method_name] = {}
				for scenario in self.scenario_list:
					sgg_results_json[mode][method_name][scenario] = {}
					if scenario == "full":
						for dataset_corruption_mode in self.dataset_corruption_modes:
							sgg_results_json[mode][method_name][scenario][dataset_corruption_mode] = {}
							if dataset_corruption_mode == const.FIXED:
								for dataset_corruption_type in self.corruption_types:
									sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
										dataset_corruption_type] = {}
									for severity_level in self.severity_levels:
										sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
											dataset_corruption_type][
											severity_level] = self.fetch_empty_metrics_json()
							elif dataset_corruption_mode == const.MIXED:
								for video_corruption_mode in self.video_corruption_modes:
									sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
										video_corruption_mode] = {}
									for severity_level in self.severity_levels:
										sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
											video_corruption_mode][
											severity_level] = self.fetch_empty_metrics_json()
					elif scenario == "partial":
						for partial_num in self.partial_percentages:
							sgg_results_json[mode][method_name][scenario][partial_num] = {}
							for dataset_corruption_mode in self.dataset_corruption_modes:
								sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode] = {}
								if dataset_corruption_mode == const.FIXED:
									for dataset_corruption_type in self.corruption_types:
										sgg_results_json[mode][method_name][scenario][partial_num][
											dataset_corruption_mode][dataset_corruption_type] = {}
										for severity_level in self.severity_levels:
											sgg_results_json[mode][method_name][scenario][partial_num][
												dataset_corruption_mode][dataset_corruption_type][
												severity_level] = self.fetch_empty_metrics_json()
								elif dataset_corruption_mode == const.MIXED:
									for video_corruption_mode in self.video_corruption_modes:
										sgg_results_json[mode][method_name][scenario][partial_num][
											dataset_corruption_mode][video_corruption_mode] = {}
										for severity_level in self.severity_levels:
											sgg_results_json[mode][method_name][scenario][partial_num][
												dataset_corruption_mode][video_corruption_mode][
												severity_level] = self.fetch_empty_metrics_json()
		
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			dataset_corruption_type = sgg_result.dataset_corruption_type
			dataset_corruption_mode = sgg_result.dataset_corruption_mode
			video_corruption_mode = sgg_result.video_corruption_mode
			severity_level = sgg_result.corruption_severity_level
			scenario = sgg_result.scenario_name
			
			if dataset_corruption_type not in self.corruption_types:
				print(f"Skipping corruption type: {dataset_corruption_type}")
				continue
			
			if str(severity_level) not in self.severity_levels:
				print(f"Skipping severity level: {severity_level}")
				continue
			
			if scenario == "full":
				if dataset_corruption_mode == const.FIXED:
					sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][dataset_corruption_type][
						severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				elif dataset_corruption_mode == const.MIXED:
					sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][video_corruption_mode][
						severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				else:
					print(f"Skipping dataset corruption mode: {dataset_corruption_mode} under full scenario")
			elif scenario == "partial":
				partial_num = str(sgg_result.partial_percentage)
				if dataset_corruption_mode == const.FIXED:
					sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][
						dataset_corruption_type][severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				elif dataset_corruption_mode == const.MIXED:
					sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][
						video_corruption_mode][severity_level] = self.fetch_completed_metrics_json(
						sgg_result.result_details.with_constraint_metrics,
						sgg_result.result_details.no_constraint_metrics,
						sgg_result.result_details.semi_constraint_metrics
					)
				else:
					print(f"Skipping dataset corruption mode: {dataset_corruption_mode} under partial scenario")
			else:
				print(f"Skipping scenario: {scenario}")
		
		return sgg_results_json
	
	def generate_sgg_combined_results_csvs_method_wise(self, sgg_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sgg_corruptions_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs",
			                             "mode_results_sgg_corruptions", csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario", "Partial Percentage", "Dataset Corruption Mode", "Video Corruption Mode",
					"Corruption Type", "Severity Level",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100"
				])
				for method_name in self.method_list:
					for scenario in self.scenario_list:
						
						if scenario == "full":
							for dataset_corruption_mode in self.dataset_corruption_modes:
								if dataset_corruption_mode == const.FIXED:
									for corruption_type in self.corruption_types:
										for severity_level in self.severity_levels:
											writer.writerow([
												method_name,
												scenario,
												"-",
												dataset_corruption_mode,
												"Fixed",
												corruption_type,
												severity_level,
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][0]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][1]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													corruption_type][severity_level][2]["mR@100"]
											])
								
								elif dataset_corruption_mode == const.MIXED:
									for video_corruption_mode in self.video_corruption_modes:
										for severity_level in self.severity_levels:
											writer.writerow([
												method_name,
												scenario,
												"-",
												dataset_corruption_mode,
												video_corruption_mode,
												"Mixed",
												severity_level,
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][0]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][1]["mR@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["R@100"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@10"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@20"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@50"],
												sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][
													video_corruption_mode][severity_level][2]["mR@100"]
											])
						
						elif scenario == "partial":
							for partial_num in self.partial_percentages:
								for dataset_corruption_mode in self.dataset_corruption_modes:
									if dataset_corruption_mode == const.FIXED:
										for corruption_type in self.corruption_types:
											for severity_level in self.severity_levels:
												writer.writerow([
													method_name,
													scenario,
													partial_num,
													dataset_corruption_mode,
													"Fixed",
													corruption_type,
													severity_level,
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][0][
														"mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][1][
														"mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][corruption_type][severity_level][2][
														"mR@100"]
												])
									
									elif dataset_corruption_mode == const.MIXED:
										for video_corruption_mode in self.video_corruption_modes:
											for severity_level in self.severity_levels:
												writer.writerow([
													method_name,
													scenario,
													partial_num,
													dataset_corruption_mode,
													video_corruption_mode,
													"Mixed",
													severity_level,
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														0]["mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														1]["mR@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["R@100"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@10"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@20"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@50"],
													sgg_results_json[mode][method_name][scenario][partial_num][
														dataset_corruption_mode][video_corruption_mode][severity_level][
														2]["mR@100"]
												])
	
	def compile_sgg_method_wise_results(self):
		sgg_results_json = self.fetch_sgg_results_json()
		self.generate_sgg_combined_results_csvs_method_wise(sgg_results_json)
	
	# -----------------------------------------------------------------------------------------------
	# ----------------------------- LATEX GENERATION FILES -----------------------------------------
	# -----------------------------------------------------------------------------------------------
	
	@staticmethod
	def fill_sgg_paper_combined_values_matrix(values_matrix, sgg_results_json, idx, mode,
	                                          comb_method_name, corruption_type, severity_level,
	                                          partial_percentage="10"):
		if "full" in comb_method_name:
			method_name = comb_method_name.split("_")[0]
			scenario = "full"
			corruption_mode = "fixed"
			# sgg_results_json[mode][method_name][scenario][dataset_corruption_mode][dataset_corruption_type][severity_level]
			values_matrix[idx, 0] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][0][
					"mR@10"])
			values_matrix[idx, 1] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][0][
					"mR@20"])
			values_matrix[idx, 2] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][0][
					"mR@50"])
			values_matrix[idx, 3] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][1][
					"R@10"])
			values_matrix[idx, 4] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][1][
					"R@20"])
			values_matrix[idx, 5] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][1][
					"R@50"])
			values_matrix[idx, 6] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][1][
					"mR@10"])
			values_matrix[idx, 7] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][1][
					"mR@20"])
			values_matrix[idx, 8] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][1][
					"mR@50"])
			values_matrix[idx, 9] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][2][
					"mR@10"])
			values_matrix[idx, 10] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][2][
					"mR@20"])
			values_matrix[idx, 11] = fetch_value(
				sgg_results_json[mode][method_name][scenario][corruption_mode][corruption_type][severity_level][2][
					"mR@50"])
		elif "partial" in comb_method_name:
			method_name = comb_method_name.split("_")[0]
			scenario = "partial"
			corruption_mode = "fixed"
			
			# sgg_results_json[mode][method_name][scenario][partial_num][dataset_corruption_mode][dataset_corruption_type][severity_level]
			values_matrix[idx, 0] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][0]["mR@10"])
			values_matrix[idx, 1] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][0]["mR@20"])
			values_matrix[idx, 2] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][0]["mR@50"])
			values_matrix[idx, 3] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][1]["R@10"])
			values_matrix[idx, 4] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][1]["R@20"])
			values_matrix[idx, 5] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][1]["R@50"])
			values_matrix[idx, 6] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][1]["mR@10"])
			values_matrix[idx, 7] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][1]["mR@20"])
			values_matrix[idx, 8] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][1]["mR@50"])
			values_matrix[idx, 9] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][2]["mR@10"])
			values_matrix[idx, 10] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][2]["mR@20"])
			values_matrix[idx, 11] = fetch_value(
				sgg_results_json[mode][method_name][scenario][partial_percentage][corruption_mode][corruption_type][
					severity_level][2]["mR@50"])
		return values_matrix
	
	@staticmethod
	def generate_sgg_corruptions_paper_latex_header():
		latex_header = "\\begin{table*}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Robustness Evaluation Results for SGG.}\n"
		latex_header += "    \\label{tab:sgg_corruptions_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|l|l|l|ccc|cccccc|ccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "      \\multirow{2}{*}{Severity} & \\multirow{2}{*}{Mode} & \\multirow{2}{*}{Corruption} & \\multirow{2}{*}{Method} & \\multicolumn{3}{c}{\\textbf{With Constraint}} & \\multicolumn{6}{c}{\\textbf{No Constraint}} & \\multicolumn{3}{c}{\\textbf{Semi Constraint}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){5-7} \\cmidrule(lr){8-13} \\cmidrule(lr){14-16} \n "
		latex_header += (
				" & & & & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50} & "
				"\\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50} " + " \\\\ \\hline\n")
		return latex_header
	
	def generate_paper_sgg_corruptions_latex_table(self, sgg_results_json):
		severity_level = self.severity_levels[0]
		latex_file_name = f"sgg_corruptions_{severity_level}.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "paper_latex_tables",
		                               latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sgg_corruptions_paper_latex_header()
		
		num_rows = 2 * len(self.latex_method_list) * len(self.latex_corruption_types) * len(self.latex_mode_list)
		values_matrix = np.zeros((num_rows, 12), dtype=np.float32)
		
		row_counter = 0
		for mode in self.latex_mode_list:
			for corruption_type in self.latex_corruption_types:
				for method in self.latex_method_list:
					for scenario in self.latex_scenario_list:
						comb_method_name = f"{method}_{scenario}"
						values_matrix = self.fill_sgg_paper_combined_values_matrix(values_matrix,
						                                                           sgg_results_json,
						                                                           row_counter,
						                                                           mode,
						                                                           comb_method_name,
						                                                           corruption_type,
						                                                           severity_level,
						                                                           partial_percentage="10")
						row_counter += 1
		
		max_value_boolean_matrx = np.zeros(values_matrix.shape, dtype=np.bool)
		percentage_values = np.zeros(values_matrix.shape, dtype=np.float32)
		# For every column, take two rows at a time and find the max value
		# find the column-wise max value and make the corresponding row, column index True in the max_value_boolean_matrix
		for col_idx in range(12):
			for row_idx in range(0, num_rows, 2):
				if values_matrix[row_idx, col_idx] > values_matrix[row_idx + 1, col_idx]:
					max_value_boolean_matrx[row_idx, col_idx] = True
				else:
					max_value_boolean_matrx[row_idx + 1, col_idx] = True
				
				# Calculate the percentage values for the row_idx and row_idx + 1 and put it in the percentage_values matrix of row_idx+1
				percentage_values[row_idx + 1, col_idx] = (
						                                          (values_matrix[row_idx + 1, col_idx] - values_matrix[
							                                          row_idx, col_idx]) / values_matrix[
							                                          row_idx, col_idx]) * 100
		
		row_counter = 0
		for mode in self.latex_mode_list:
			for corruption_type in self.latex_corruption_types:
				for method in self.latex_method_list:
					for scenario in self.latex_scenario_list:
						comb_method_name = f"{method}_{scenario}"
						latex_method_name = self.fetch_method_name_latex(comb_method_name)
						
						# Start Line for each mode
						if row_counter % (len(self.latex_corruption_types) * 2 * len(self.latex_method_list)) == 0:
							latex_row = f"   \\multirow{{{len(self.latex_corruption_types) * 2 * len(self.latex_method_list)}}}{{*}}{{{severity_level}}} &      \\multirow{{{len(self.latex_corruption_types) * 2 * len(self.latex_method_list)}}}{{*}}{{{mode}}} & "
						else:
							latex_row = "    &    &"
						
						corruption_name = self.corruption_type_latex_name_map[corruption_type]
						
						# Start Line for each corruption type
						if row_counter % (2 * len(self.latex_method_list)) == 0:
							latex_row += f"\\multirow{{{2 * len(self.latex_method_list)}}}{{*}}{{{corruption_name}}} & "
						else:
							latex_row += " & "
						
						latex_row += f"        {latex_method_name}"
						
						for col_idx in range(12):
							if max_value_boolean_matrx[row_counter, col_idx]:
								latex_row += f" & \\cellcolor{{highlightColor}} \\textbf{{{fetch_rounded_value(values_matrix[row_counter, col_idx])}}}"
							else:
								latex_row += f" & {fetch_rounded_value(values_matrix[row_counter, col_idx])}"
							
							# Append the percentage values for every alternate row
							if row_counter % 2 != 0:
								percentage_change = fetch_rounded_value(percentage_values[row_counter, col_idx])
								if percentage_change > 0:
									latex_row += f" (+{percentage_change}\%)"
								else:
									latex_row += f" ({percentage_change}\%)"
								# latex_row += f" ({fetch_rounded_value(percentage_values[row_counter, col_idx])})"
						
						latex_row += "  \\\\ \n"
						
						# End Line for each corruption type
						if (row_counter + 1) % (
								2 * len(self.latex_method_list)) == 0 and row_counter < (
								num_rows - 1):  # At the end of each corruption type group
							latex_row += " \\cmidrule(lr){4-16}  \n "
						
						# End Line for each mode
						if (row_counter + 1) % (
								len(self.latex_corruption_types) * 2 * len(
							self.latex_method_list)) == 0:  # End of mode group
							latex_row += "          \\hline \n"
						
						latex_table += latex_row
						row_counter += 1
		
		latex_footer = self.generate_full_width_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)


def prepare_paper_sgg_latex_tables():
	prepare_paper_results_sgg_corruptions = PrepareResultsSGGCorruptions()
	sgg_results_json = prepare_paper_results_sgg_corruptions.fetch_sgg_results_json()
	prepare_paper_results_sgg_corruptions.generate_paper_sgg_corruptions_latex_table(sgg_results_json)


def main():
	prepare_results_sgg = PrepareResultsSGGCorruptions()
	prepare_results_sgg.compile_sgg_method_wise_results()


def combine_results():
	prepare_results_sgg = PrepareResultsSGGCorruptions()
	prepare_results_sgg.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_sgg_corruptions",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_sgg_corruptions\sgg_corruptions_modes_combined_results.xlsx"
	)


if __name__ == '__main__':
	prepare_paper_sgg_latex_tables()
# main()
# combine_results()
