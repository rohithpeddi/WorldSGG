import os

import torch

from uni4d.engine import Engine


class AgEngine:

    def __init__(
            self,
            args
    ):
        self.ag_root_dir = "/data/rohith/ag"
        self.opt = args
        self.ag_frames_dir = os.path.join(self.ag_root_dir, "frames")
        self.ag_4D_dir = os.path.join(self.ag_root_dir, "ag4D")
        self.uni4D_dir = os.path.join(self.ag_4D_dir, "uni4D")
        self.unidepth_dir = os.path.join(self.uni4D_dir, "unidepth")
        os.makedirs(self.unidepth_dir, exist_ok=True)

        self.frames_path = os.path.join(self.ag_root_dir, "frames")
        self.annotations_path = os.path.join(self.ag_root_dir, "annotations")
        self.video_list = sorted(os.listdir(self.frames_path))
        self.gt_annotations = sorted(os.listdir(self.annotations_path))

        self.cut3r_dir = os.path.join(self.ag_4D_dir, "cut3r")

    def process_ag_video(self, video_name):

        cut3r_video_output_path = os.path.join(self.cut3r_dir, f"{video_name[:-4]}.pkl")
        if not os.path.exists(cut3r_video_output_path):
            print(f"Cut3R output for {video_name} not found. Skipping video.")
            return

        init_cam_dict = torch.load(cut3r_video_output_path)["cam_dict"]
        # init_cam_dict = None

        engine = Engine(self.opt, init_cam_dict=init_cam_dict)
        self.opt.video_name = video_name
        engine.video_name = video_name
        print("Initializing engine for video:", video_name)
        engine.initialize()

        print("Initializing optimization for video:", video_name)
        engine.optimize_init()
        engine.log_timer("init")

        print("Optimizing bundle adjustment for video:", video_name)
        engine.optimize_BA()

        print("Reinitializing static points for video:", video_name)
        engine.reinitialize_static()  # add more static points

        engine.log_timer("BA")
        if engine.num_points_dyn > 0:
            print("Initializing dynamic control points for video:", video_name)
            engine.init_dyn_cp()

            print("Optimizing dynamic points for video:", video_name)
            engine.optimize_dyn()

            print("Filtering dynamic points for video:", video_name)
            engine.filter_dyn()
            engine.log_timer("dyn")

        print("Saving results for video:", video_name)
        engine.save_results(save_fused_points=self.opt.vis_4d)
        del engine
