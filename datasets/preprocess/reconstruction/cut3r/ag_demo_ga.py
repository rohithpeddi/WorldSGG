import argparse
import os
import pickle
from src.dust3r.model import ARCroco3DStereo
import random
import torch

random.seed(42)


def get_video_belongs_to_split(video_id):
	"""
	Get the split that the video belongs to based on its ID.
	"""
	first_letter = video_id[0]
	if first_letter.isdigit() and int(first_letter) < 5:
		return "04"
	elif first_letter.isdigit() and int(first_letter) >= 5:
		return "59"
	elif first_letter in "ABCD":
		return "AD"
	elif first_letter in "EFGH":
		return "EH"
	elif first_letter in "IJKL":
		return "IL"
	elif first_letter in "MNOP":
		return "MP"
	elif first_letter in "QRST":
		return "QT"
	elif first_letter in "UVWXYZ":
		return "UZ"


class AgCut3r:
	
	def __init__(
			self,
			args,
			datapath,
			ag_root_dir
	):
		self.datapath = datapath
		self.ag_root_dir = ag_root_dir
		
		self.frames_path = os.path.join(self.datapath, "frames")
		self.annotations_path = os.path.join(self.datapath, "annotations")
		self.video_list = sorted(os.listdir(self.frames_path))
		self.gt_annotations = sorted(os.listdir(self.annotations_path))
		print("Total number of ground truth annotations: ", len(self.gt_annotations))
		
		video_id_frame_id_list_pkl_file_path = os.path.join(self.datapath, "4d_video_frame_id_list.pkl")
		if os.path.exists(video_id_frame_id_list_pkl_file_path):
			with open(video_id_frame_id_list_pkl_file_path, "rb") as f:
				self.video_id_frame_id_list = pickle.load(f)
		else:
			assert False, f"Please generate {video_id_frame_id_list_pkl_file_path} first"
		self.ag_root_dir = ag_root_dir
		self.ag_cut3r_root = os.path.join(ag_root_dir, "ag4D", "cut3r_ga")
		os.makedirs(self.ag_cut3r_root, exist_ok=True)
		
		# -------------------------------- CUT3R PARAMS --------------------------------
		self.weights_path = args.model_path
		self.device = args.device
		self.croco_model = ARCroco3DStereo.from_pretrained(args.model_path).to(self.device)
		self.croco_model.eval()
	
	@staticmethod
	def prepare_input(
			img_paths, img_mask, size, raymaps=None, raymap_mask=None, revisit=1, update=True
	):
		"""
		Prepare input views for inference from a list of image paths.

		Args:
			img_paths (list): List of image file paths.
			img_mask (list of bool): Flags indicating valid images.
			size (int): Target image size.
			raymaps (list, optional): List of ray maps.
			raymap_mask (list, optional): Flags indicating valid ray maps.
			revisit (int): How many times to revisit each view.
			update (bool): Whether to update the state on revisits.

		Returns:
			list: A list of view dictionaries.
		"""
		# Import image loader (delayed import needed after adding ckpt path).
		from src.dust3r.utils.image import load_images
		
		images = load_images(img_paths, size=size)
		views = []
		num_views = len(images)
		all_permutations = forward_backward_permutations(num_views, interval=2)
		for permute in all_permutations:
			_views = []
			for idx, i in enumerate(permute):
				view = {
					"img": images[i]["img"],
					"ray_map": torch.full(
						(
							images[i]["img"].shape[0],
							6,
							images[i]["img"].shape[-2],
							images[i]["img"].shape[-1],
						),
						torch.nan,
					),
					"true_shape": torch.from_numpy(images[i]["true_shape"]),
					"idx": i,
					"instance": str(i),
					"camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(
						0
					),
					"img_mask": torch.tensor(True).unsqueeze(0),
					"ray_mask": torch.tensor(False).unsqueeze(0),
					"update": torch.tensor(True).unsqueeze(0),
					"reset": torch.tensor(False).unsqueeze(0),
				}
				_views.append(view)
			views.append(_views)
		return views


def parse_args():
	"""Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description="Run 3D point cloud inference and visualization using ARCroco3DStereo."
	)
	parser.add_argument(
		"--model_path",
		type=str,
		default="src/cut3r_512_dpt_4_64.pth",
		help="Path to the pretrained model checkpoint.",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda",
		help="Device to run inference on (e.g., 'cuda' or 'cpu').",
	)
	parser.add_argument(
		"--vis_threshold",
		type=float,
		default=1.5,
		help="Visualization threshold for the point cloud viewer. Ranging from 1 to INF",
	)
	parser.add_argument(
		"--split",
		type=str,
		choices=["04", "59", "AD", "EH", "IL", "MP", "QT", "UZ"],
		help="Split for the data"
	)
	parser.add_argument(
		"--datapath",
		type=str,
		default="/data/rohith/ag/",
		help="Path to the data"
	)
	
	parser.add_argument("--ag_root_dir", type=str, default="/data/rohith/ag/")
	parser.add_argument("--niter", type=int, default=200, help="number of iterations")
	parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
	
	parser.add_argument("--split", type=str, help="path to the model weights")
	parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
	
	parser.add_argument("--silent", action="store_true", default=False, help="silence logs")
	parser.add_argument(
		"--window_wise",
		action="store_true",
		default=False,
		help="Use window wise mode for optimization"
	)
	parser.add_argument("--window_size", type=int, default=100, help="Window size")
	parser.add_argument("--window_overlap_ratio", type=float, default=0.5, help="Window overlap ratio")
	parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
	
	parser.add_argument("--fps", type=int, default=0, help="FPS for video processing")
	parser.add_argument("--num_frames", type=int, default=200, help="Maximum number of frames for video processing")
	parser.add_argument("--share", action="store_true", default=False, help="Share the demo")
	
	return parser.parse_args()
