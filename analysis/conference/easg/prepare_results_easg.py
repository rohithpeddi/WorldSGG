import csv
import os

from analysis.conference.prepare_results_base import PrepareResultsBase


class PrepareResultsEASG(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareResultsEASG, self).__init__()
		self.scenario_list = ["full", "partial", "labelnoise"]
		self.mode_list = ["sgcls", "easgcls", "predcls"]
		self.method_list = ["easg"]
		self.partial_percentages = [10, 40, 70]
		self.label_noise_percentages = [10, 20, 30]
		
		self.task_name = "easg"
	
	def fetch_easg_results_json(self):
		db_results = self.fetch_db_easg_results()
		easg_results_json = {}
		for mode in self.mode_list:
			easg_results_json[mode] = {}
			for method_name in self.method_list:
				easg_results_json[mode][method_name] = {}
				for scenario_name in self.scenario_list:
					if scenario_name == "full":
						easg_results_json[mode][method_name][scenario_name] = self.fetch_easg_empty_metrics_json()
					else:
						easg_results_json[mode][method_name][scenario_name] = {}
						percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
						for percentage_num in percentage_list:
							easg_results_json[mode][method_name][scenario_name][
								percentage_num] = self.fetch_easg_empty_metrics_json()
		
		for easg_result in db_results:
			mode = easg_result.mode
			method_name = easg_result.method_name
			scenario_name = easg_result.scenario_name
			if scenario_name == "full":
				with_constraint_metrics = easg_result.result_details.with_constraint_metrics
				no_constraint_metrics = easg_result.result_details.no_constraint_metrics
				
				completed_metrics_json = self.fetch_easg_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics
				)
				easg_results_json[mode][method_name][scenario_name] = completed_metrics_json
				
				continue
			else:
				percentage_num = easg_result.partial_percentage if scenario_name == "partial" else easg_result.label_noise_percentage
				
				with_constraint_metrics = easg_result.result_details.with_constraint_metrics
				no_constraint_metrics = easg_result.result_details.no_constraint_metrics
				
				completed_metrics_json = self.fetch_easg_completed_metrics_json(
					with_constraint_metrics,
					no_constraint_metrics,
				)
				
				easg_results_json[mode][method_name][scenario_name][percentage_num] = completed_metrics_json
		
		return easg_results_json
	
	def generate_easg_combined_results_csvs_method_wise(self, easg_results_json):
		for mode in self.mode_list:
			csv_file_name = f"easg_combined_{mode}.csv"
			csv_file_path = os.path.join(os.path.dirname(__file__), "../../results_docs", "mode_results_easg",
			                             csv_file_name)
			os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
			with open(csv_file_path, "a", newline='') as csv_file:
				writer = csv.writer(csv_file, quoting=csv.QUOTE_NONNUMERIC)
				writer.writerow([
					"Method Name", "Scenario Name", "Severity Level",
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
								easg_results_json[mode][method_name][scenario_name][0]["R@10"],
								easg_results_json[mode][method_name][scenario_name][0]["R@20"],
								easg_results_json[mode][method_name][scenario_name][0]["R@50"],
								easg_results_json[mode][method_name][scenario_name][0]["R@100"],
								easg_results_json[mode][method_name][scenario_name][0]["mR@10"],
								easg_results_json[mode][method_name][scenario_name][0]["mR@20"],
								easg_results_json[mode][method_name][scenario_name][0]["mR@50"],
								easg_results_json[mode][method_name][scenario_name][0]["mR@100"],
								easg_results_json[mode][method_name][scenario_name][1]["R@10"],
								easg_results_json[mode][method_name][scenario_name][1]["R@20"],
								easg_results_json[mode][method_name][scenario_name][1]["R@50"],
								easg_results_json[mode][method_name][scenario_name][1]["R@100"],
								easg_results_json[mode][method_name][scenario_name][1]["mR@10"],
								easg_results_json[mode][method_name][scenario_name][1]["mR@20"],
								easg_results_json[mode][method_name][scenario_name][1]["mR@50"],
								easg_results_json[mode][method_name][scenario_name][1]["mR@100"]
							])
							continue
						else:
							percentage_list = self.partial_percentages if scenario_name == "partial" else self.label_noise_percentages
							for percentage_num in percentage_list:
								writer.writerow([
									method_name,
									scenario_name,
									percentage_num,
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@10"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@20"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@50"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["R@100"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@10"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@20"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@50"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][0]["mR@100"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@10"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@20"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@50"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["R@100"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@10"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@20"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@50"],
									easg_results_json[mode][method_name][scenario_name][percentage_num][1]["mR@100"]
								])
	
	def compile_easg_method_wise_results(self):
		sgg_results_json = self.fetch_easg_results_json()
		self.generate_easg_combined_results_csvs_method_wise(sgg_results_json)


def main():
	prepare_results_easg = PrepareResultsEASG()
	prepare_results_easg.compile_easg_method_wise_results()


def combine_results():
	prepare_results_easg = PrepareResultsEASG()
	prepare_results_easg.combine_csv_to_excel(
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_easg",
		r"C:\Users\rohit\PycharmProjects\stl_stsg\analysis\results_docs\mode_results_easg\easg_modes_combined_results.xlsx"
	)


if __name__ == '__main__':
	main()
	combine_results()