from argparse import ArgumentParser

import torch


class EgoConfig(object):

    def __init__(self):
        self.save_path = None
        self.data_path = None
        self.ckpt = None
        self.optimizer = None
        self.bce_loss = True
        self.lr = 1e-5
        self.num_epoch = 10
        self.results_path = None
        self.method_name = "easg"
        self.task_name = "easg"
        self.split = "train"

        self.path_to_annotations = '/data/rohith/easg/annotations'
        self.path_to_data = '/data/rohith/easg/features'
        self.path_to_output = '/data/rohith/easg/checkpoints'

        # ---------------- Corruptions ----------------
        self.use_input_corruptions = False

        # ---------------- Partial Annotations --------------------
        self.use_partial_annotations = False
        self.partial_percentage = 100
        self.maintain_distribution = False

        # ---------------- Label Noise ----------------
        self.use_label_noise = False
        self.label_noise_percentage = 20

        # ---------------- Use wandb ----------------
        self.use_wandb = False

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.parser = self.setup_parser()
        self.args = vars(self.parser.parse_args())
        self.__dict__.update(self.args)

    def setup_parser(self):
        parser = ArgumentParser(description='training code')

        parser.add_argument('--method_name', type=str, default='easg', help='method name')
        parser.add_argument('--save_path', type=str, default='/data/rohith/easg/checkpoints',
                            help='path to save the model')
        parser.add_argument('--data_path', type=str, default='/data/rohith/easg/features', help='path to data')
        parser.add_argument('--ckpt', type=str, default=None, help='path to load the model')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer')
        parser.add_argument('--task_name', type=str, default='easg', help='task name')

        # ----------------------- EGO CENTRIC DATA CONFIGURATION -----------------------

        parser.add_argument('--path_to_annotations', default='/data/rohith/easg/annotations', type=str)
        parser.add_argument('--path_to_data', default='/data/rohith/easg/features', type=str)
        parser.add_argument('--path_to_output', default='/data/rohith/easg/checkpoints', type=str)
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--sch_param', type=int, default=10, help='parameter for lr scheduler')
        parser.add_argument('--num_epoch', type=int, default=10, help='total number of epochs')
        parser.add_argument('--random_guess', action='store_true', help='for random guessing')
        parser.add_argument('--split', type=str, default='train', help='train or test')

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
