from easydict import EasyDict
from pathlib import Path
import json


class Config:
    def __init__(self):
        # NOTE: Edit this configutation if you want
        ############################################
        # General
        ############################################
        # Options about experiment
        self.cfg = EasyDict()
        self.cfg.EXP = EasyDict()
        self.cfg.EXP.path = '/app/NAS2_sejong/SRPruning/vdsr'
        self.cfg.EXP.version = 'v43'
        self.cfg.EXP.description = "Rewinding L2NormPruning 0.8 with PatchMSELoss"
        self.cfg.EXP.seed = 903
        # Options for save path
        self.cfg.SAVE = EasyDict()
        self.cfg.SAVE.cfg_dir = f"{self.cfg.EXP.path}/config/"
        self.cfg.SAVE.cfg_file_path = f"{self.cfg.EXP.path}/config/{self.cfg.EXP.version}.cfg"
        self.cfg.SAVE.save_dir = f"{self.cfg.EXP.path}/samples/{self.cfg.EXP.version}"
        self.cfg.SAVE.checkpoint_dir = f"{self.cfg.EXP.path}/checkpoint/{self.cfg.EXP.version}"
        self.cfg.SAVE.summary_dir = f"{self.cfg.EXP.path}/summary/{self.cfg.EXP.version}"
        # Options for dataset
        self.cfg.DATA = EasyDict()
        self.cfg.DATA.div2k_dir = '../../dataset/DIV2K/'
        self.cfg.DATA.set5_dir = '../../dataset/Set5/'
        self.cfg.DATA.hr_size = 128
        self.cfg.DATA.lr_size = 32
        ############################################
        # Training
        ############################################
        # Options for training
        self.cfg.TRAIN = EasyDict()
        self.cfg.TRAIN.batch_size = 64
        self.cfg.TRAIN.dataloader_num_worker = 16
        self.cfg.TRAIN.end_epoch = 10000
        self.cfg.TRAIN.period_log = 5  # epoch
        self.cfg.TRAIN.period_save = 500  # epoch
        self.cfg.TRAIN.learning_rate = 1e-4
        self.cfg.TRAIN.lr_step_milestones = [10000 * (x + 1) for x in range(int(10000 / 10000))]
        self.cfg.TRAIN.lr_step_gamma = 0.1
        self.cfg.TRAIN.resume = True
        self.cfg.TRAIN.load_checkpoint_path = f"{self.cfg.EXP.path}/checkpoint/v22/SRPruning_epoch_0.pth"
        self.cfg.TRAIN.loss = 'PatchMSELoss'
        self.cfg.TRAIN.pruning = True 
        self.cfg.TRAIN.pruning_version = 'p34'
        self.cfg.TRAIN.pruning_dir = f"{self.cfg.EXP.path}/pruning/{self.cfg.TRAIN.pruning_version}"
        ############################################
        # Pruning
        ############################################
        self.cfg.PRUNE = EasyDict()
        self.cfg.PRUNE.description = "Test pruning_map"
        self.cfg.PRUNE.exp_ver = 'p46'
        self.cfg.PRUNE.trained_checkpoint_path = f"{self.cfg.EXP.path}/checkpoint/v22/SRPruning_epoch_10000.pth"
        self.cfg.PRUNE.method = 'MagnitudeFilterPruning'  # 'RandomPruning', 'MagnitudePruning', 'ActivationPreservingPruning', 'MagnitudeFilterPruning'
        self.cfg.PRUNE.pruning_rate = 0.1
        self.cfg.PRUNE.random_prune_try_cnt = 1
        self.cfg.SAVE.pruning_dir = f"{self.cfg.EXP.path}/pruning/{self.cfg.PRUNE.exp_ver}"
        ############################################
        # Train pruning model
        ############################################
        self.cfg.TRAIN_PRUNE = EasyDict()
        self.cfg.TRAIN_PRUNE.model_parameters = f"{self.cfg.EXP.path}/checkpoint/v22/SRPruning_epoch_10000.pth"

    def prepare_training(self):
        if Path(self.cfg.SAVE.cfg_file_path).exists():
            print(
                f"[ERROR] Configuration {self.cfg.SAVE.cfg_file_path} already exists")
            print(f"[ERROR] Stop Experiment {self.cfg.EXP.version}")
            exit(-1)
        else:
            self._make_directory(self.cfg.SAVE.cfg_dir)
            self._make_directory(self.cfg.SAVE.save_dir)
            self._make_directory(self.cfg.SAVE.checkpoint_dir)
            self._make_directory(self.cfg.SAVE.summary_dir)
            self._save_config_file(self.cfg.SAVE.cfg_file_path)
            print(f"[INFO] Experiment {self.cfg.EXP.version} set up")
    
    def prepare_pruning(self):
        if Path(self.cfg.SAVE.pruning_dir).exists():
            print(
                f"[ERROR] Pruning directory {self.cfg.SAVE.pruning_dir} already exists")
            print(f"[ERROR] Stop pruning {self.cfg.PRUNE.exp_ver}")
            exit(-1)
        else:
            self._make_directory(self.cfg.SAVE.pruning_dir)
            print(f"[INFO] Experiment {self.cfg.PRUNE.exp_ver} set up")

    def _make_directory(self, path):
        print(f"[INFO] Make directory {path}")
        if not Path(path).exists():
            Path(path).mkdir(parents=True)

    def _save_config_file(self, path):
        print(f"[INFO] Save config to {path}")
        with open(path, 'w') as f:
            f.write(json.dumps(self.cfg, indent=4))
