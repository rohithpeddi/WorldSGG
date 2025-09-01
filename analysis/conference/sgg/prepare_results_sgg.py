import csv
import os

from analysis.conference.prepare_results_base import PrepareResultsBase


class PrepareResultsSGG(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsSGG, self).__init__()
		self.scenario_list = ["full", "partial", "labelnoise"]
		self.mode_list = ["sgcls", "sgdet", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		self.partial_percentages = [10]
		self.label_noise_percentages = [20]
		
		self.task_name = "sgg"
	
	def fetch_sgg_results_json(self):
		db_results = self.fetch_db_sgg_results()
		sgg_results_json = {}
		for mode in self.mode_list:
			sgg_results_json[mode] = {}
			for method_name in self.method_list:
				sgg_results_json[mode][method_name] = {}
				for scenario_name in self.scenario_list:
					if scenario_name == "full":
						sgg_results_json[mode][method_name][scenario_name] = self.fetch_empty_metrics_json()
					else:
						sgg_results_json[mode][method_name][scenario_name] = {}
						percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
						for percentage_num in percentage_list:
							sgg_results_json[mode][method_name][scenario_name][
								percentage_num] = self.fetch_empty_metrics_json()
		
		for sgg_result in db_results:
			mode = sgg_result.mode
			method_name = sgg_result.method_name
			scenario_name = sgg_result.scenario_name
			if scenario_name == "full":
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				
				completed_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				sgg_results_json[mode][method_name][scenario_name] = completed_metrics_json
				
				continue
			else:
				percentage_num = sgg_result.partial_percentage if scenario_name == "partial" else sgg_result.label_noise_percentage
				
				with_constraint_metrics = sgg_result.result_details.with_constraint_metrics
				no_constraint_metrics = sgg_result.result_details.no_constraint_metrics
				semi_constraint_metrics = sgg_result.result_details.semi_constraint_metrics
				
				completed_metrics_json = self.fetch_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
					semi_constraint_metrics
				)
				
				sgg_results_json[mode][method_name][scenario_name][percentage_num] = completed_metrics_json
		
		return sgg_results_json
	
	def generate_sgg_recall_results_csvs_method_wise(self, sgg_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sgg_recall_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "mode_results_csvs",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario Name", "Severity Level", "R@10", "R@20", "R@50", "R@100",
					"R@10", "R@20", "R@50", "R@100", "R@10", "R@20", "R@50", "R@100"
				])
				for method_name in self.method_list:
					for scenario_name in self.scenario_list:
						if scenario_name == "full":
							writer.writerow([
								method_name,
								scenario_name,
								"-",
								sgg_results_json[mode][method_name][scenario_name][0]["R@10"],
								sgg_results_json[mode][method_name][scenario_name][0]["R@20"],
								sgg_results_json[mode][method_name][scenario_name][0]["R@50"],
								sgg_results_json[mode][method_name][scenario_name][0]["R@100"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@10"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@20"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@50"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@100"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@10"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@20"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@50"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@100"]
							])
							continue
						else:
							percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
							for percentage_num in percentage_list:
								writer.writerow([
									method_name,
									scenario_name,
									percentage_num,
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@100"]
								])
	
	def generate_sgg_mean_recall_results_csvs_method_wise(self, sgg_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sgg_mean_recall_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "mode_results_csvs",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario Name", "Severity Level", "mR@10", "mR@20", "mR@50", "mR@100",
					"mR@10", "mR@20", "mR@50", "mR@100", "mR@10", "mR@20", "mR@50", "mR@100"
				])
				for method_name in self.method_list:
					for scenario_name in self.scenario_list:
						if scenario_name == "full":
							writer.writerow([
								method_name,
								scenario_name,
								"-",
								sgg_results_json[mode][method_name][scenario_name][0]["mR@10"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@20"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@50"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@100"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@10"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@20"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@50"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@100"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@10"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@20"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@50"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@100"]
							])
							continue
						else:
							percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
							for percentage_num in percentage_list:
								writer.writerow([
									method_name,
									scenario_name,
									percentage_num,
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@100"]
								])
	
	def generate_sgg_combined_results_csvs_method_wise(self, sgg_results_json):
		for mode in self.mode_list:
			csv_file_name = f"sgg_combined_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "mode_results_csvs",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario Name", "Severity Level",
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
							writer.writerow([
								method_name,
								scenario_name,
								"-",
								sgg_results_json[mode][method_name][scenario_name][0]["R@10"],
								sgg_results_json[mode][method_name][scenario_name][0]["R@20"],
								sgg_results_json[mode][method_name][scenario_name][0]["R@50"],
								sgg_results_json[mode][method_name][scenario_name][0]["R@100"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@10"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@20"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@50"],
								sgg_results_json[mode][method_name][scenario_name][0]["mR@100"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@10"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@20"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@50"],
								sgg_results_json[mode][method_name][scenario_name][1]["R@100"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@10"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@20"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@50"],
								sgg_results_json[mode][method_name][scenario_name][1]["mR@100"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@10"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@20"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@50"],
								sgg_results_json[mode][method_name][scenario_name][2]["R@100"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@10"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@20"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@50"],
								sgg_results_json[mode][method_name][scenario_name][2]["mR@100"]
							])
							continue
						else:
							percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
							for percentage_num in percentage_list:
								writer.writerow([
									method_name,
									scenario_name,
									percentage_num,
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["R@100"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@10"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@20"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@50"],
									sgg_results_json[mode][method_name][scenario_name][percentage_num][2]["mR@100"]
								])
	
	def compile_sgg_method_wise_results(self):
		sgg_results_json = self.fetch_sgg_results_json()
		self.generate_sgg_recall_results_csvs_method_wise(sgg_results_json)
		self.generate_sgg_mean_recall_results_csvs_method_wise(sgg_results_json)
		self.generate_sgg_combined_results_csvs_method_wise(sgg_results_json)


def main():
	prepare_results_sgg = PrepareResultsSGG()
	prepare_results_sgg.compile_sgg_method_wise_results()


def combine_results():
	prepare_results_sgg = PrepareResultsSGG()
	prepare_results_sgg.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_sgg",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_sgg\sgg_modes_combined_results.xlsx"
	)


if __name__ == '__main__':
	main()
