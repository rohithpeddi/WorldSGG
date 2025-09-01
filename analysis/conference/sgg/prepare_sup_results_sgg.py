import os

import numpy as np

from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value


class PrepareSupResultsSGG(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareSupResultsSGG, self).__init__()
		self.scenario_list = ["full", "partial"]
		self.mode_list = ["sgcls", "sgdet", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		
		self.latex_mode_list = ["sgcls", "sgdet", "predcls"]
		self.latex_method_list = [
			"sttran_full", "sttran_partial",
			"dsgdetr_full", "dsgdetr_partial"
		]
		self.partial_percentages = [70, 40, 10]
		self.task_name = "sgg"
	
	def fetch_sgg_combined_results_json_latex(self):
		db_results = self.fetch_db_sgg_results()
		sgg_results_json = {}
		for method in self.latex_method_list:
			sgg_results_json[method] = {}
			if "full" in method:
				for mode in self.mode_list:
					sgg_results_json[method][mode] = self.fetch_empty_metrics_json()
			elif "partial" in method:
				percentage_list = self.partial_percentages
				for percentage_num in percentage_list:
					sgg_results_json[method][percentage_num] = {}
					for mode in self.mode_list:
						sgg_results_json[method][percentage_num][mode] = self.fetch_empty_metrics_json()
		
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				paper_method_name = method_name + "_full"
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[paper_method_name][mode] = completed_metrics_json
				continue
			elif scenario_name == "partial":
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				paper_method_name = method_name + "_partial"
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				completed_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[paper_method_name][percentage_num][mode] = completed_metrics_json
		return sgg_results_json
	
	# ---------------------------------------------------------------------------------------------
	# --------------------------------- Latex Generation Methods ----------------------------------
	# ---------------------------------------------------------------------------------------------
	
	def fill_sgg_combined_values_matrix(self, values_matrix, results_json, eval_setting_name, mode_name,
	                                    idx, comb_method_name, partial_per=10):
		
		setting_id = self.fetch_eval_setting_id_latex(eval_setting_name)
		if "full" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["R@10"])
			values_matrix[idx, 1] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["R@20"])
			values_matrix[idx, 2] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["R@50"])
			values_matrix[idx, 3] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["R@100"])
			values_matrix[idx, 4] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["mR@10"])
			values_matrix[idx, 5] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["mR@20"])
			values_matrix[idx, 6] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["mR@50"])
			values_matrix[idx, 7] = fetch_value(results_json[comb_method_name][mode_name][setting_id]["mR@100"])
		elif "partial" in comb_method_name:
			values_matrix[idx, 0] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["R@10"])
			values_matrix[idx, 1] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["R@20"])
			values_matrix[idx, 2] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["R@50"])
			values_matrix[idx, 3] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["R@100"])
			values_matrix[idx, 4] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["mR@10"])
			values_matrix[idx, 5] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["mR@20"])
			values_matrix[idx, 6] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["mR@50"])
			values_matrix[idx, 7] = fetch_value(
				results_json[comb_method_name][partial_per][mode_name][setting_id]["mR@100"])
		return values_matrix
	
	def generate_sgg_latex_header(self, eval_setting_name):
		"""
		eval_setting_name: str ---> Name of the evaluation setting
		eval_setting : [WITH CONSTRAINT, NO CONSTRAINT, SEMI-CONSTRAINT]
		"""
		nocap_setting_name = self.fetch_eval_setting_name_nocap_latex(eval_setting_name)
		cap_setting_name = self.fetch_eval_setting_name_cap_latex(eval_setting_name)
		
		latex_header = "\\begin{table*}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += f"    \\caption{{{nocap_setting_name} Results for VidSGG.}}\n"
		latex_header += f"    \\label{{tab:sup_sgg_{eval_setting_name}}}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{l|l|l|cccc|cccc}\n"
		latex_header += "    \\hline\n"
		latex_header += f"        \\multirow{{2}}{{*}}{{Mode}} & \\multirow{{2}}{{*}}{{Method}} & \\multirow{{2}}{{*}}{{$\\mathcal{{S}}$}} &  \\multicolumn{{8}}{{c}}{{\\textbf{{{cap_setting_name}}}}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){4-7} \\cmidrule(lr){8-11} \n "
		latex_header += (
				"        & & & "
				"\\textbf{R@10} & \\textbf{R@20} & \\textbf{R@50} & \\textbf{R@100} &"
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & \\textbf{mR@100}" + " \\\\ \\hline\n")
		
		return latex_header
	
	@staticmethod
	def generate_sup_sgg_latex_row(values_matrix, percentage_changes, max_values, row_counter):
		latex_row = ""
		for column_id in range(8):
			# If max value is True, then add (1) Bold and (2) \\cellcolor{{highlightColor}} to the latex row
			# Add the percentage change to the latex row, if percentage change is + add a + sign and if it is - add a - sign
			if max_values[row_counter, column_id]:
				latex_row += f" & \\cellcolor{{highlightColor}}\\textbf{{{values_matrix[row_counter, column_id]:.2f}}} "
			else:
				latex_row += f" & {values_matrix[row_counter, column_id]:.2f} "
			
			if percentage_changes[row_counter, column_id] > 0:
				latex_row += f" (+{percentage_changes[row_counter, column_id]:.2f}\%)"
			elif percentage_changes[row_counter, column_id] < 0:
				latex_row += f" ({percentage_changes[row_counter, column_id]:.2f}\%)"
		
		latex_row += " \\\\ \n"
		
		return latex_row
	
	def generate_eval_setting_latex_file(self, eval_setting_name):
		latex_file_name = f"sup_sgg_{eval_setting_name}.tex"
		latex_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "sup_latex_tables",
		                               latex_file_name)
		os.makedirs(os.path.dirname(latex_file_path), exist_ok=True)
		latex_table = self.generate_sgg_latex_header(eval_setting_name)
		values_matrix = np.zeros((24, 8), dtype=np.float32)
		results_json = self.fetch_sgg_combined_results_json_latex()
		
		row_counter = 0
		for mode in self.latex_mode_list:
			for method_name in self.latex_method_list:
				if "full" in method_name:
					values_matrix = self.fill_sgg_combined_values_matrix(
						values_matrix, results_json, eval_setting_name, mode, row_counter, method_name)
					row_counter += 1
				elif "partial" in method_name:
					for partial_per in self.partial_percentages:
						values_matrix = self.fill_sgg_combined_values_matrix(
							values_matrix, results_json, eval_setting_name, mode, row_counter, method_name, partial_per)
						row_counter += 1
		
		# Calculate percentage changes for each method compared to the original training method.
		percentage_changes = np.zeros(values_matrix.shape, dtype=np.float32)
		row_counter = 0
		for mode in self.latex_mode_list:
			for method_name in self.latex_method_list:
				if "full" in method_name:
					base_values = values_matrix[row_counter]
					row_counter += 1
				elif "partial" in method_name:
					for partial_per in self.partial_percentages:
						cur_values = values_matrix[row_counter]
						assert base_values is not None
						percentage_changes[row_counter] = self.calculate_percentage_changes(base_values, cur_values)
						row_counter += 1
					base_values = None
		
		# Calculate Max value for each column compared to the original training method
		# For each method, calculate the max percentage change compared to the original training method
		max_values = np.zeros(values_matrix.shape, dtype=np.bool)
		# Each method has 4 rows, so we need to calculate the max value for each method column-wise
		for mode_method_setting_id in range(6):
			start_idx = mode_method_setting_id * 4
			end_idx = start_idx + 4
			num_columns = values_matrix.shape[1]
			for column_id in range(num_columns):
				column_values = values_matrix[start_idx:end_idx, column_id]
				arg_max_column_id = np.argmax(column_values)
				max_values[start_idx + arg_max_column_id, column_id] = True
		
		# Generate the latex table
		row_counter = 0
		for mode in self.latex_mode_list:
			for method_name in self.latex_method_list:
				latex_method_name = self.fetch_method_name_latex(method_name)
				if "full" in method_name:
					if row_counter % 8 == 0:
						latex_row = f"   \\multirow{{8}}{{*}}{{{self.fetch_sgg_mode_name_latex(mode)}}} & {latex_method_name}"
					else:
						latex_row = f"  &  {latex_method_name}"
					# Add this as it does not apply for the percentage parameter for the full row in latex
					latex_row += f"& - "
					latex_row += self.generate_sup_sgg_latex_row(values_matrix, percentage_changes, max_values,
					                                             row_counter)
					latex_table += latex_row
					row_counter += 1
				elif "partial" in method_name:
					for partial_per in self.partial_percentages:
						if row_counter % 8 == 0:
							latex_row = f"   \\multirow{{8}}{{*}}{{{self.fetch_sgg_mode_name_latex(mode)}}} & {latex_method_name}"
						else:
							latex_row = f"  &  {latex_method_name}"
							
						latex_row += f"& {partial_per} "
						latex_row += self.generate_sup_sgg_latex_row(values_matrix, percentage_changes, max_values,
						                                             row_counter)
						row_counter += 1
						
						latex_table += latex_row
					
					# Add the midrule for methods after every 4 rows (4 rows for each method)
					if row_counter % 4 == 0 and row_counter > 0:
						latex_table += "    \\cmidrule(lr){2-11}\n"
			
			# Add the midrule for methods after every 4 rows (4 rows for each method)
			if row_counter % 8 == 0 and row_counter > 0:
				latex_table += "    \\cmidrule(lr){1-11}\n"
		
		latex_table += "    \\hline\n"
		latex_footer = self.generate_full_width_latex_footer()
		latex_table += latex_footer
		
		with open(latex_file_path, "a", newline='') as latex_file:
			latex_file.write(latex_table)
		
		print(f"Generated {latex_file_name} file.")


def main():
	sup_results_sgg = PrepareSupResultsSGG()
	eval_setting_list = ["with_constraint", "no_constraint", "semi_constraint"]
	for eval_setting in eval_setting_list:
		sup_results_sgg.generate_eval_setting_latex_file(eval_setting)


if __name__ == "__main__":
	main()
