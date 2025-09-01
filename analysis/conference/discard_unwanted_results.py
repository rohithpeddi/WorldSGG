from analysis.aaai.prepare_results_base import fetch_rob_sgg_results
from analysis.results.FirebaseService import FirebaseService


def discard_method_results():
	rob_sgg_results = fetch_rob_sgg_results()
	for result in rob_sgg_results:
		if result.method_name == "sttran" and result.mode in ["sgcls", "predcls"]:
			print(f"Removing result with name: {result.method_name} and mode: {result.mode}")
			db_service.remove_result(result.result_id)


if __name__ == "__main__":
	db_service = FirebaseService()
	discard_method_results()
