from analysis.results.Result import Metrics, ResultDetails
from constants import ResultConstants as const

# -----------------------------------------------------------------------------------
# ---------------------------- METHOD DETAILS EXTRACTION ---------------------------
# -----------------------------------------------------------------------------------

def get_mode_name(result_file_name):
	if "_sgdet" in result_file_name:
		return const.SGDET
	elif "_sgcls" in result_file_name:
		return const.SGCLS
	elif "_predcls" in result_file_name:
		return const.PREDCLS
	
	raise Exception(f"Mode not found for method name: {result_file_name}")


def get_partial_percentage(method_name):
	if "_partial" in method_name:
		method_vars = method_name.split('_')
		index_of_partial = method_vars.index("partial")
		return int(method_vars[index_of_partial + 1])
	
	raise Exception(f"Partial percentage not found for method name: {method_name}")


def get_sga_method_name(method_name):
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
	else:
		raise Exception(f"Method name not found for method name: {method_name}")
	return method_name


# -----------------------------------------------------------------------------------
# ---------------------------- PROCESS CSV ROWS EXTRACTION ---------------------------
# -----------------------------------------------------------------------------------


def process_result_details_from_csv_row(row):
	method_name = row[0]
	print("Processing method: ", method_name)
	with_constraint_metrics = Metrics(
		row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12]
	)
	no_constraint_metrics = Metrics(
		row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23],
		row[24]
	)
	semi_constraint_metrics = Metrics(
		row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34], row[35],
		row[36]
	)
	result_details = ResultDetails()
	result_details.add_with_constraint_metrics(with_constraint_metrics)
	result_details.add_no_constraint_metrics(no_constraint_metrics)
	result_details.add_semi_constraint_metrics(semi_constraint_metrics)
	return result_details, method_name


def process_result_details_from_csv_row_no_method(row):
	with_constraint_metrics = Metrics(
		row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]
	)
	no_constraint_metrics = Metrics(
		row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22],
		row[23]
	)
	semi_constraint_metrics = Metrics(
		row[24], row[25], row[26], row[27], row[28], row[29], row[30], row[31], row[32], row[33], row[34],
		row[35]
	)
	result_details = ResultDetails()
	result_details.add_with_constraint_metrics(with_constraint_metrics)
	result_details.add_no_constraint_metrics(no_constraint_metrics)
	result_details.add_semi_constraint_metrics(semi_constraint_metrics)
	return result_details
