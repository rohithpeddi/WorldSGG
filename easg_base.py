import os

import torch
import wandb

from constants import EgoConstants as const
from lib.supervised.ego_evaluation_recall import BasicEgoActionSceneGraphEvaluator


class EASGBase:

    def __init__(self, conf):
        self._conf = conf

        self._train_dataset = None
        self._val_dataset = None
        self._dataloader_train = None
        self._dataloader_val = None
        self._evaluator = None
        self._model = None
        self._device = None

        # Load checkpoint name
        self._checkpoint_name = None
        self._checkpoint_save_dir_path = None
        self._corruption_name = None

        # Init Wandb
        self._enable_wandb = self._conf.use_wandb

    def _init_evaluators(self):
        self._evaluator = BasicEgoActionSceneGraphEvaluator(self._conf)

    def _load_checkpoint(self):
        if self._model is None:
            raise ValueError("Model is not initialized")

        if self._conf.ckpt:
            if os.path.exists(self._conf.ckpt) is False:
                raise ValueError(f"Checkpoint file {self._conf.ckpt} does not exist")

            try:
                # Load checkpoint to the specified device
                ckpt = torch.load(self._conf.ckpt, map_location=self._device)

                # Determine the key for the state_dict based on availability
                state_dict_key = 'state_dict' if 'state_dict' in ckpt else f'{self._conf.method_name}_state_dict'

                # Load the state dictionary
                self._model.load_state_dict(ckpt[state_dict_key], strict=False)
                print(f"Loaded model from checkpoint {self._conf.ckpt}")

            except FileNotFoundError:
                print(f"Error: Checkpoint file {self._conf.ckpt} not found.")
            except KeyError:
                print(f"Error: Appropriate state_dict not found in the checkpoint.")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
        else:
            print("--------------------------------------------------------")
            print("Checkpoint not provided. Training from scratch")
            print("--------------------------------------------------------")

    @staticmethod
    def _save_model(model, epoch, checkpoint_save_file_path, checkpoint_name, method_name):
        print("*" * 40)
        os.makedirs(checkpoint_save_file_path, exist_ok=True)
        torch.save({f"{method_name}_state_dict": model.state_dict()},
                   os.path.join(checkpoint_save_file_path, f"{checkpoint_name}_epoch_{epoch}.tar"))
        print(f"Saved {method_name} checkpoint after {epoch} epochs")
        print("*" * 40)

    def _init_config(self, is_train=True):
        print('The CKPT saved here:', self._conf.save_path)
        os.makedirs(self._conf.save_path, exist_ok=True)

        # Set the preferred device
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self._conf.ckpt is not None:
            if self._conf.task_name == const.EASG:
                # Checkpoint name format for model trained with full annotations: easg_epoch_1.tar
                # Checkpoint name format for model trained with partial annotations: easg_partial_10_epoch_1.tar
                # Checkpoint name format for model trained with label noise: easg_label_noise_10_epoch_1.tar
                self._checkpoint_name_with_epoch = os.path.basename(self._conf.ckpt).split('.')[0]
                self._checkpoint_name = "_".join(self._checkpoint_name_with_epoch.split('_')[:-2])
                print("--------------------------------------------------------")
                print(f"Loading checkpoint with name: {self._checkpoint_name}")
                print("--------------------------------------------------------")
        else:
            # Set the checkpoint name and save path details
            if self._conf.task_name == const.EASG:
                if self._conf.use_partial_annotations:
                    self._checkpoint_name = f"{self._conf.method_name}_partial_{self._conf.partial_percentage}"
                elif self._conf.use_label_noise:
                    self._checkpoint_name = f"{self._conf.method_name}_labelnoise_{self._conf.label_noise_percentage}"
                else:
                    self._checkpoint_name = f"{self._conf.method_name}"
                print("--------------------------------------------------------")
                print(f"Training model with name: {self._checkpoint_name}")
                print("--------------------------------------------------------")

        self._checkpoint_save_dir_path = os.path.join(self._conf.save_path, self._conf.task_name, self._conf.method_name)
        os.makedirs(self._checkpoint_save_dir_path, exist_ok=True)

        # --------------------------- W&B CONFIGURATION ---------------------------
        if self._enable_wandb:
            wandb.init(project=self._checkpoint_name, config=self._conf)

        print("-------------------- CONFIGURATION DETAILS ------------------------")
        for i in self._conf.args:
            print(i, ':', self._conf.args[i])
        print("-------------------------------------------------------------------")