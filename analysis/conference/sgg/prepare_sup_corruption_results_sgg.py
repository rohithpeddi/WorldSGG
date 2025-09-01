from analysis.conference.prepare_results_base import PrepareResultsBase, fetch_value, fetch_rounded_value
from constants import CorruptionConstants as const


class PrepareSupSGGCorruptions(PrepareResultsBase):
	
	def __init__(self):
		super(PrepareSupSGGCorruptions, self).__init__()
		self.mode_list = ["sgcls", "predcls"]
		self.method_list = ["sttran", "dsgdetr"]
		self.scenario_list = ["full", "partial"]
		self.partial_percentages = [70, 40, 10]
		
		self.corruption_types = [
			const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
			const.GAUSSIAN_BLUR, const.DEFOCUS_BLUR, const.FOG, const.FROST,
			const.SPATTER, const.CONTRAST, const.BRIGHTNESS, const.PIXELATE,
			const.JPEG_COMPRESSION, const.SUN_GLARE, const.DUST, const.SATURATE
		]
		self.dataset_corruption_modes = [const.FIXED, const.MIXED]
		self.video_corruption_modes = [const.FIXED, const.MIXED]
		self.severity_levels = ["3"]
		
		self.latex_mode_list = ["sgcls", "predcls"]
		self.latex_method_list = ["sttran", "dsgdetr"]
		self.latex_method_list = [
			"sttran_full", "sttran_partial",
			"dsgdetr_full", "dsgdetr_partial"
		]
		
		self.latex_corruption_types = [
			const.GAUSSIAN_NOISE, const.SHOT_NOISE, const.IMPULSE_NOISE, const.SPECKLE_NOISE,
			const.FOG, const.FROST, const.BRIGHTNESS, const.SUN_GLARE
		]
		
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
