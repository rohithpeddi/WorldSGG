import csv
import os

from analysis.conference.transfer_base import process_result_details_from_csv_row, \
	process_result_details_from_csv_row_no_method, get_mode_name, get_partial_percentage
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


# -----------------------------------------------------------------------------------
# ----------------------------- TRANSFER FULL PARTIAL FILES -------------------
# -----------------------------------------------------------------------------------


# Methods: sttran, dsgdetr
# Modes: sgcls, sgdet, predcls
# Partial Percentages: 10, 40, 70
# Scenario: partial, full
def transfer_sgg(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGG
):
	base_file_name = os.path.basename(result_file_path)
	
	mode_name = get_mode_name(base_file_name)
	assert mode_name == mode
	assert mode in [const.SGCLS, const.SGDET, const.PREDCLS]
	
	with open(result_file_path, 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		for row in csv_reader:
			result_details, method_name = process_result_details_from_csv_row(row)
			result = Result(
				task_name=task_name,
				scenario_name=scenario_name,
				method_name=method_name,
				mode=mode,
			)
			
			if scenario_name == const.PARTIAL:
				partial_percentage = get_partial_percentage(base_file_name)
				result.partial_percentage = partial_percentage
			elif scenario_name == const.LABELNOISE:
				# label_noise_percentage = 20
				# result.label_noise_percentage = label_noise_percentage
				raise Exception("Label Noise percentage not implemented")
			
			result.add_result_details(result_details)
			print(f"[{task_name}][{scenario_name}][{mode_name}][{base_file_name}] Saving result: {result.result_id}")
			db_service.update_result_to_db("results_21_11_sgg", result.result_id, result.to_dict())
			print(f"[{task_name}][{scenario_name}][{mode_name}][{base_file_name}] Saved result: {result.result_id}")
			print("-----------------------------------------------------------------------------------")


def transfer_results_from_directories_sgg():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	task_directory_path = r"E:\results\legal\sgg"
	task_name = const.SGG
	for scenario_name in os.listdir(task_directory_path):
		# Convert the scenario name to lowercase
		scenario_name = scenario_name.lower()
		print(f"[{task_name}][{scenario_name}] Processing files for scenario: ", scenario_name)
		scenario_name_path = os.path.join(task_directory_path, scenario_name)
		for method_name_csv_file in os.listdir(scenario_name_path):
			# Convert the method name to lowercase
			method_name_csv_file = method_name_csv_file.lower()
			method_name_csv_path = os.path.join(scenario_name_path, method_name_csv_file)
			
			mode_name = get_mode_name(method_name_csv_file.split('.')[0])
			assert mode_name in [const.SGCLS, const.SGDET, const.PREDCLS]
			if task_name == const.SGG and scenario_name in [const.PARTIAL, const.FULL]:
				print(
					f"[{task_name}][{scenario_name}][{mode_name}][{method_name_csv_file[:-4]}] Processing file: ",
					method_name_csv_path)
				transfer_sgg(
					mode=mode_name,
					result_file_path=method_name_csv_path,
					scenario_name=scenario_name
				)


# -----------------------------------------------------------------------------------
# ----------------------------- TRANSFER CORRUPTION RELATED FILES -------------------
# -----------------------------------------------------------------------------------


def transfer_sgg_corruptions(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGG
):
	base_file_name = os.path.basename(result_file_path)
	details = (base_file_name.split('.')[0]).split('_')
	
	# self._corruption_name = (f"{self._conf.dataset_corruption_mode}_{self._conf.video_corruption_mode}_"
	# f"{self._conf.dataset_corruption_type}_{self._conf.corruption_severity_level}")
	
	index_of_mode = details.index(mode)
	
	dataset_corruption_mode = details[index_of_mode + 1]
	video_corruption_mode = details[index_of_mode + 2]
	corruption_severity_level = details[-1]
	
	partial_percentage = None
	if scenario_name == const.PARTIAL:
		partial_percentage = details[index_of_mode - 1]
	
	assert mode in [const.SGCLS, const.SGDET, const.PREDCLS]
	
	with open(result_file_path, 'r') as read_obj:
		# pass the file object to reader() to get the reader object
		csv_reader = csv.reader(read_obj)
		for row in csv_reader:
			
			if corruption_severity_level == "1":
				result_details = process_result_details_from_csv_row_no_method(row)
				if method_name is None:
					method_name = details[0]
			else:
				result_details, method_name = process_result_details_from_csv_row(row)
			
			if "sttran" in method_name:
				method_name = "sttran"
			elif "dsgdetr" in method_name:
				method_name = "dsgdetr"
			
			result = Result(
				task_name=task_name,
				scenario_name=scenario_name,
				method_name=method_name,
				mode=mode,
			)
			
			result.partial_percentage = partial_percentage
			result.dataset_corruption_mode = dataset_corruption_mode
			result.video_corruption_mode = video_corruption_mode
			result.corruption_severity_level = corruption_severity_level
			if dataset_corruption_mode == const.FIXED:
				dataset_corruption_type = "_".join(details[index_of_mode + 3:-1])
				result.dataset_corruption_type = dataset_corruption_type
			else:
				dataset_corruption_type = details[index_of_mode + 3]
				result.dataset_corruption_type = dataset_corruption_type
			
			result.add_result_details(result_details)
			print("-----------------------------------------------------------------------------------")
			print("Saving result: ", result.result_id)
			db_service.update_result_to_db("results_14_11_sgg_corruptions", result.result_id, result.to_dict())
			print("Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_corruption_results_from_directories_sgg():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	task_directory_path = r"E:\results\legal\sgg"
	task_name = const.SGG
	for scenario_name in os.listdir(task_directory_path):
		scenario_name_path = os.path.join(task_directory_path, scenario_name)
		# Convert the scenario name to lowercase
		scenario_name = scenario_name.lower()
		
		if scenario_name != const.CORRUPTION:
			continue
		
		print(f"[{task_name}][{scenario_name}] Processing files for scenario: ", scenario_name)
		for method_name_csv_file in os.listdir(scenario_name_path):
			method_name_csv_path = os.path.join(scenario_name_path, method_name_csv_file)
			# Convert the method name to lowercase
			method_name_csv_file = method_name_csv_file.lower()
			
			train_scenario_name = const.FULL
			if "_partial_" in method_name_csv_file:
				train_scenario_name = const.PARTIAL
			
			mode_name = None
			if "_sgcls_" in method_name_csv_file:
				mode_name = const.SGCLS
			elif "_sgdet_" in method_name_csv_file:
				mode_name = const.SGDET
			elif "_predcls_" in method_name_csv_file:
				mode_name = const.PREDCLS
			
			assert mode_name in [const.SGCLS, const.SGDET, const.PREDCLS]
			if task_name == const.SGG and scenario_name == const.CORRUPTION:
				print(
					f"[{task_name}][{scenario_name}][{mode_name}][{method_name_csv_file[:-4]}] Processing file: ",
					method_name_csv_path)
				transfer_sgg_corruptions(
					mode=mode_name,
					result_file_path=method_name_csv_path,
					scenario_name=train_scenario_name
				)


if __name__ == '__main__':
	db_service = FirebaseService()
	transfer_corruption_results_from_directories_sgg()
	# transfer_results_from_directories_sgg()
