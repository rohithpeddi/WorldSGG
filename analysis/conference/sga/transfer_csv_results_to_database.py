import csv
import os

from analysis.conference.transfer_base import process_result_details_from_csv_row, \
	process_result_details_from_csv_row_no_method, get_mode_name, get_partial_percentage, get_sga_method_name
from analysis.results.FirebaseService import FirebaseService
from analysis.results.Result import Result
from constants import ResultConstants as const


# Methods: sttran_ant, sttran_gen_ant, dsgdetr_ant, dsgdetr_gen_ant, ode, sde
# Modes: sgcls, sgdet, predcls
# Partial Percentages: 10, 40, 70
# Label Noise Percentages: 10, 20, 30
# Scenario: partial, labelnoise, full
def transfer_sga(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGA
):
	base_file_name = os.path.basename(result_file_path)
	# As context fraction has "." in it so we cannot split by "."
	eval_csv_file_name = base_file_name[:-4]
	mode_name = get_mode_name(eval_csv_file_name)
	assert mode_name == mode
	
	csv_attributes = eval_csv_file_name.split("_")
	
	# Test future frames
	index_of_test = csv_attributes.index("test")
	train_future_frames = csv_attributes[index_of_test - 1]
	test_num = csv_attributes[index_of_test + 1]
	
	# If "." is present in the test future frames then it is context fraction else it is test future frames
	if "." in test_num:
		test_context_fraction = test_num
		test_future_frames = None
	else:
		test_context_fraction = None
		test_future_frames = test_num
	
	if not os.path.exists(result_file_path):
		print("File does not exist: ", result_file_path)
		return
	
	if test_context_fraction is None:
		print("Skipping test future frame results!")
		return
	
	with open(result_file_path, 'r') as read_obj:
		csv_reader = csv.reader(read_obj)
		num_rows = len(list(csv_reader))
	
	with open(result_file_path, 'r') as read_obj:
		# pass the file object to reader() to get the reader object
		csv_reader = csv.reader(read_obj)
		for row_id, row in enumerate(csv_reader):
			if len(row) == 0:
				print(f"[{task_name}][{scenario_name}][{mode}] Skipping empty row")
				continue
			
			# As the first row here corresponds to the full annotations and the second corresponds to the partial annotations
			if scenario_name == const.PARTIAL and row_id == 0 and num_rows > 1:
				print(f"[{task_name}][{scenario_name}][{mode}] Skipping header row")
				continue
			
			result_details, method_name = process_result_details_from_csv_row(row)
			method_name = get_sga_method_name(method_name)
			
			result = Result(
				task_name=task_name,
				scenario_name=scenario_name,
				method_name=method_name,
				mode=mode,
			)
			if scenario_name == const.PARTIAL:
				try:
					partial_percentage = get_partial_percentage(base_file_name)
				except Exception as e:
					print("ERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERROR")
					print(f"[{task_name}][{scenario_name}][{mode}] Error in getting partial percentage: ", e)
					print("ERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERRORERROR")
					partial_percentage = 40
				result.partial_percentage = partial_percentage
			elif scenario_name == const.LABELNOISE:
				raise Exception("Label Noise scenario not supported for SGA")
			
			result.train_num_future_frames = train_future_frames
			result.test_num_future_frames = test_future_frames
			result.context_fraction = test_context_fraction
			
			result.add_result_details(result_details)
			print(f"[{task_name}][{scenario_name}][{mode}] Saving result: ", result.result_id)
			db_service.update_result_to_db("results_21_11_sga", result.result_id, result.to_dict())
			print(f"[{task_name}][{scenario_name}][{mode}] Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_results_from_directories_sga():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	task_directory_path = r"E:\results\legal\sga"
	task_name = const.SGA
	for scenario_name in os.listdir(task_directory_path):
		scenario_name_path = os.path.join(task_directory_path, scenario_name)
		# Convert the scenario name to lowercase
		scenario_name = scenario_name.lower()
		
		if scenario_name not in [const.PARTIAL, const.FULL]:
			continue
		
		print(f"[{task_name}][{scenario_name}] Processing files for scenario: ", scenario_name)
		for mode_name in os.listdir(scenario_name_path):
			# Convert the mode name to lowercase
			mode_name_path = os.path.join(scenario_name_path, mode_name)
			mode_name = mode_name.lower()
			for method_name_csv_file in os.listdir(mode_name_path):
				# Convert the method name to lowercase
				method_name_csv_path = os.path.join(mode_name_path, method_name_csv_file)
				if task_name == const.SGA and scenario_name in [const.PARTIAL, const.FULL]:
					print(f"[{task_name}][{scenario_name}][{mode_name}][{method_name_csv_file[:-4]}] Processing file: ",
					      method_name_csv_path)
					transfer_sga(
						mode=mode_name,
						result_file_path=method_name_csv_path,
						scenario_name=scenario_name
					)


# -----------------------------------------------------------------------------------
# ----------------------------- TRANSFER CORRUPTION RELATED FILES -------------------
# -----------------------------------------------------------------------------------


def transfer_sga_corruptions(
		mode,
		result_file_path,
		scenario_name,
		task_name=const.SGG
):
	base_file_name = os.path.basename(result_file_path)
	details = (base_file_name[:-4]).split('_')
	
	# self._corruption_name = (f"{self._conf.dataset_corruption_mode}_{self._conf.video_corruption_mode}_"
	# f"{self._conf.dataset_corruption_type}_{self._conf.corruption_severity_level}")
	# dsgdetr_ant_partial_10_sgcls_future_3_test_0.5_fixed_fixed_dust_3
	
	index_of_mode = details.index(mode)
	
	dataset_corruption_mode = details[index_of_mode + 5]
	video_corruption_mode = details[index_of_mode + 6]
	corruption_severity_level = details[-1]
	
	train_future_frame = details[index_of_mode + 2]
	test_num = details[index_of_mode + 4]
	
	if "." in test_num:
		test_context_fraction = test_num
		test_future_frames = None
	else:
		test_context_fraction = None
		test_future_frames = test_num
	
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
			else:
				result_details, method_name = process_result_details_from_csv_row(row)
			
			if "sttran_ant" in method_name:
				method_name = "sttran_ant"
			elif "dsgdetr_ant" in method_name:
				method_name = "dsgdetr_ant"
			elif "sttran_gen_ant" in method_name:
				method_name = "sttran_gen_ant"
			elif "dsgdetr_gen_ant" in method_name:
				method_name = "dsgdetr_gen_ant"
			elif "ode" in method_name:
				method_name = "ode"
			elif "sde" in method_name:
				method_name = "sde"
			
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
			result.train_num_future_frames = train_future_frame
			result.test_num_future_frames = test_future_frames
			result.context_fraction = test_context_fraction
			
			if dataset_corruption_mode == const.FIXED:
				dataset_corruption_type = "_".join(details[index_of_mode + 7:-1])
				result.dataset_corruption_type = dataset_corruption_type
			else:
				dataset_corruption_type = details[index_of_mode + 7]
				result.dataset_corruption_type = dataset_corruption_type
			
			result.add_result_details(result_details)
			print("-----------------------------------------------------------------------------------")
			print("Saving result: ", result.result_id)
			db_service.update_result_to_db("results_14_11_sga_corruptions", result.result_id, result.to_dict())
			print("Saved result: ", result.result_id)
			print("-----------------------------------------------------------------------------------")


def transfer_corruption_results_from_directories_sga():
	# results_parent_directory = os.path.join(os.path.dirname(__file__), "..", "docs", "csvs", "new")
	task_directory_path = r"E:\results\legal\sga"
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
				transfer_sga_corruptions(
					mode=mode_name,
					result_file_path=method_name_csv_path,
					scenario_name=train_scenario_name
				)


if __name__ == '__main__':
	db_service = FirebaseService()
	# transfer_corruption_results_from_directories_sga()
	transfer_results_from_directories_sga()
