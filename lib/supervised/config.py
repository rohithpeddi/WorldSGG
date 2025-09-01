from argparse import ArgumentParser

BATCHNORM_MOMENTUM = 0.01


class Config(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """
        Defaults
        """
        self.mode = None
        self.save_path = None
        self.data_path = None
        self.datasize = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = True
        self.lr = 1e-5
        self.enc_layer = 1
        self.dec_layer = 3
        self.nepoch = 10
        self.results_path = None
        self.method_name = None
        self.task_name = "sga"

        # ---------------- SGA ----------------
        self.max_window = 5
        self.brownian_size = 1
        self.ode_ratio = 1.0
        self.sde_ratio = 1.0
        self.bbox_ratio = 0.1
        self.baseline_context = 3
        self.max_future = 5
        self.hp_recon_loss = 1.0

        # ---------------- Corruptions ----------------
        self.use_input_corruptions = False
        self.dataset_corruption_mode = "fixed"
        self.corruption_severity_level = 1

        self.video_corruption_mode = "fixed"
        self.dataset_corruption_type = None

        self.is_video_based_corruption = False
        self.corruption_frames_directory = None

        # ---------------- Partial Annotations --------------------
        self.use_partial_annotations = False
        self.partial_percentage = 100
        self.maintain_distribution = False

        # ---------------- Label Noise ----------------
        self.use_label_noise = False
        self.label_noise_percentage = 20

        # ---------------- Use wandb ----------------
        self.use_wandb = False
        
        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        """
        Sets up an argument parser
        :return:
        """
        parser = ArgumentParser(description='training code')
        parser.add_argument('--mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
        parser.add_argument('--save_path', default='/data/rohith/ag/checkpoints', type=str)
        parser.add_argument('--method_name', default='sttran', type=str)
        parser.add_argument('--results_path', default='results', type=str)

        parser.add_argument('--baseline_context', default=3, type=int)
        parser.add_argument('--max_future', default=3, type=int)
        parser.add_argument('--data_path', default='/data/rohith/ag', type=str)
        parser.add_argument('--datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
        parser.add_argument('--ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
        parser.add_argument('--optimizer', help='adamw/adam/sgd', default='adamw', type=str)
        parser.add_argument('--lr', dest='lr', help='learning rate', default=1e-5, type=float)
        parser.add_argument('--nepoch', help='epoch number', default=4, type=int)
        parser.add_argument('--enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
        parser.add_argument('--dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)
        parser.add_argument('--bce_loss', action='store_true')
        parser.add_argument('--modified_gt', action='store_true')
        parser.add_argument("--task_name", default="sga", type=str)

        # ---------------- SGA ----------------
        parser.add_argument('--max_window', default=3, type=int)
        parser.add_argument('--brownian_size', default=1, type=int)
        parser.add_argument('--ode_ratio', default=1.0, type=float)
        parser.add_argument('--sde_ratio', default=1.0, type=float)
        parser.add_argument('--bbox_ratio', default=0.1, type=float)

        # =============================================================================
        # Settings where the input is modified by introducing corruptions like noise, blur, etc.
        # =============================================================================

        # ---------------- Corruptions ----------------
        parser.add_argument("--use_input_corruptions", action="store_true")

        # ---------------- Corruption Related Parameters ----------------
        parser.add_argument('--dataset_corruption_mode', default='fixed', type=str)
        parser.add_argument('--corruption_severity_level', default=1, type=int)
        parser.add_argument('--video_corruption_mode', default='fixed', type=str)
        parser.add_argument('--dataset_corruption_type', default=None, type=str)

        parser.add_argument('--is_video_based_corruption', action='store_true')
        parser.add_argument('--corruption_frames_directory', default=None, type=str)

        # =============================================================================
        # Settings where the ground truth is modified either in the form of missing labels or noisy labels
        # =============================================================================

        # ---------------- Partial Annotations ----------------
        parser.add_argument("--use_partial_annotations", action="store_true")
        parser.add_argument("--partial_percentage", default=100, type=int)

        parser.add_argument("--maintain_distribution", action="store_true")
        # ---------------- Label Noise ----------------
        parser.add_argument("--use_label_noise", action="store_true")
        parser.add_argument("--label_noise_percentage", default=20, type=int)

        # ---------------- Use wandb ----------------
        parser.add_argument("--use_wandb", action="store_true")

        return parser
