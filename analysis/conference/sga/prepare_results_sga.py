import csv
import os

from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value


class PrepareResultsSGA(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGA, self).__init__()
		self.scenario_list = ["full", "partial", "labelnoise"]
		self.mode_list = ["sgcls", "sgdet", "predcls"]
		self.method_list = ["sttran_ant", "dsgdetr_ant", "sttran_gen_ant", "dsgdetr_gen_ant", "ode", "sde"]
		self.partial_percentages = [40]
		self.label_noise_percentages = [20]
		self.task_name = "sga"
		
		self.context_fraction_list = ['0.3', '0.5', '0.7', '0.9']
	
	def fetch_sga_results_json(self):
		db_results = self.fetch_db_sga_results()
		sga_results_json = {}
		for mode in self.mode_list:
			sga_results_json[mode] = {}
			for method_name in self.method_list:
				sga_results_json[mode][method_name] = {}
				for scenario_name in self.scenario_list:
					sga_results_json[mode][method_name][scenario_name] = {}
					for cf in self.context_fraction_list:
						if scenario_name == "full":
							sga_results_json[mode][method_name][scenario_name][cf] = self.fetch_empty_metrics_json()
						else:
							sga_results_json[mode][method_name][scenario_name][cf] = {}
							percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
							for percentage_num in percentage_list:
								sga_results_json[mode][method_name][scenario_name][cf][
									percentage_num] = self.fetch_empty_metrics_json()
		
		for sga_result in db_results:
			mode = sga_result.mode
			method_name = sga_result.method_name
			scenario_name = sga_result.scenario_name
			if sga_result.context_fraction is None:
				continue
			cf = sga_result.context_fraction
			if scenario_name == "full":
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sga_results_json[mode][method_name][scenario_name][cf] = completed_metrics_json
				continue
			else:
				percentage_num = sga_result.partial_percentage if scenario_name == "partial" else sga_result.label_noise_percentage
				with_constraint_metrics = sga_result.result_details.with_constraint_metrics
				no_constraint_metrics = sga_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sga_result.result_details.semi_constraint_metrics
				completed_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sga_results_json[mode][method_name][scenario_name][cf][percentage_num] = completed_metrics_json
		
		return sga_results_json
	
	# ---------------------------------------------------------------------------------------------------
	# ------------------------------------ CSV GENERATION METHODS ---------------------------------------
	# ---------------------------------------------------------------------------------------------------
	
	def generate_sga_results_csvs_method_wise(self, sga_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sga_combined_method_wise_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "mode_results_csvs",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario Name", "Severity Level", "Context Fraction",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100"
				])
				for method_name in self.method_list:
					for scenario_name in self.scenario_list:
						if scenario_name == "full":
							for cf in self.context_fraction_list:
								writer.writerow([
									method_name,
									scenario_name,
									"-",
									cf,
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@10"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@20"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@50"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@100"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@10"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@20"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@50"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@100"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@10"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@20"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@50"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@100"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@10"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@20"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@50"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@100"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@10"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@20"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@50"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@100"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@10"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@20"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@50"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@100"]
								])
						else:
							percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
							for percentage_num in percentage_list:
								for cf in self.context_fraction_list:
									writer.writerow([
										method_name,
										scenario_name,
										percentage_num,
										cf,
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@100"]
									])
	
	def generate_sga_results_csvs_context_wise(self, sga_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sga_combined_context_wise_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "mode_results_csvs",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Context Fraction", "Method Name", "Scenario Name", "Severity Level",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100",
					"R@10", "R@20", "R@50", "R@100",
					"mR@10", "mR@20", "mR@50", "mR@100"
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
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@10"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@20"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@50"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["R@100"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@10"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@20"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@50"],
									sga_results_json[mode][method_name][scenario_name][cf][0]["mR@100"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@10"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@20"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@50"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["R@100"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@10"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@20"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@50"],
									sga_results_json[mode][method_name][scenario_name][cf][1]["mR@100"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@10"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@20"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@50"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["R@100"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@10"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@20"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@50"],
									sga_results_json[mode][method_name][scenario_name][cf][2]["mR@100"]
								])
							else:
								percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
								for percentage_num in percentage_list:
									writer.writerow([
										cf,
										method_name,
										scenario_name,
										percentage_num,
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"R@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][0][
											"mR@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"R@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][1][
											"mR@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"R@100"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@10"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@20"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@50"],
										sga_results_json[mode][method_name][scenario_name][cf][percentage_num][2][
											"mR@100"]
									])
	
	def compile_sga_results(self):
		sga_results_json = self.fetch_sga_results_json()
		self.generate_sga_results_csvs_method_wise(sga_results_json)
		self.generate_sga_results_csvs_context_wise(sga_results_json)
	
	# ---------------------------------------------------------------------------------------------------
	# ------------------------------------ LATEX GENERATION METHODS -------------------------------------
	# ---------------------------------------------------------------------------------------------------
	
	@staticmethod
	def generate_sga_paper_latex_header():
		latex_header = "\\begin{table}[!h]\n"
		latex_header += "    \\centering\n"
		latex_header += "    \\captionsetup{font=small}\n"
		latex_header += "    \\caption{Results for SGG.}\n"
		latex_header += "    \\label{tab:sgg_mean_recall_results}\n"
		latex_header += "    \\renewcommand{\\arraystretch}{1.2} \n"
		latex_header += "    \\resizebox{\\textwidth}{!}{\n"
		latex_header += "    \\begin{tabular}{ll|ccccccccc|ccccccccc|ccccccccc}\n"
		latex_header += "    \\hline\n"
		latex_header += "       &  & \\multicolumn{9}{c}{\\textbf{AGS}} & \\multicolumn{9}{c}{\\textbf{PGAGS}} & \\multicolumn{9}{c}{\\textbf{GAGS}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1-2}\\cmidrule(lr){3-11} \\cmidrule(lr){12-20} \\cmidrule(lr){21-29} \n "
		latex_header += "       &  & \\multicolumn{3}{c}{\\textbf{With}} & \\multicolumn{3}{c}{\\textbf{No}} & \\multicolumn{3}{c}{\\textbf{Semi}} & \\multicolumn{3}{c}{\\textbf{With}} & \\multicolumn{3}{c}{\\textbf{No}} & \\multicolumn{3}{c}{\\textbf{Semi}} & \\multicolumn{3}{c}{\\textbf{With}} & \\multicolumn{3}{c}{\\textbf{No}} & \\multicolumn{3}{c}{\\textbf{Semi}} \\\\ \n"
		latex_header += "        \\cmidrule(lr){1-2} \\cmidrule(lr){3-5} \\cmidrule(lr){6-8} \\cmidrule(lr){9-11} \\cmidrule(lr){12-14} \\cmidrule(lr){15-17} \\cmidrule(lr){18-20} \\cmidrule(lr){21-23} \\cmidrule(lr){24-26} \\cmidrule(lr){27-29} \n "
		
		latex_header += (
				"    $\\mathcal{F}$ &  \\textbf{Method} & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50} & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & "
				"\\textbf{mR@10} & \\textbf{mR@20} & \\textbf{mR@50}  & " + " \\\\ \\hline\n")
		return latex_header


def main():
	prepare_results_sga = PrepareResultsSGA()
	prepare_results_sga.compile_sga_results()


def combine_results():
	prepare_results_sga = PrepareResultsSGA()
	prepare_results_sga.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_csvs",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_csvs\sga_combined_results.xlsx"
	)


if __name__ == "__main__":
	# main()
	combine_results()
