from easydict import EasyDict
from pathlib import Path
import json


class Config:
    def __init__(self):
        # NOTE: Edit this configutation if you want
        self.cfg = EasyDict()
        self.cfg.EXP = EasyDict()
        self.cfg.EXP.path = '../../REWIND-vdsr-scratch'
        self.cfg.EXP.version = 'v1'
        self.cfg.EXP.subversion = '0'  # Inc this val for training from checkpoint
        self.cfg.EXP.description = "From scratch"

        self.cfg.SAVE = EasyDict()
        self.cfg.SAVE.cfg_dir = f"{self.cfg.EXP.path}/config/"
        self.cfg.SAVE.cfg_file_path = f"{self.cfg.EXP.path}/config/{self.cfg.EXP.version}.{self.cfg.EXP.subversion}.cfg"
        self.cfg.SAVE.save_dir = f"{self.cfg.EXP.path}/samples/{self.cfg.EXP.version}"
        self.cfg.SAVE.checkpoint_dir = f"{self.cfg.EXP.path}/checkpoint/{self.cfg.EXP.version}"
        self.cfg.SAVE.summary_dir = f"{self.cfg.EXP.path}/summary/{self.cfg.EXP.version}"

        self.cfg.DATA = EasyDict()
        self.cfg.DATA.div2k_dir = '../../dataset/DIV2K/'
        self.cfg.DATA.set5_dir = '../../dataset/Set5'
        self.cfg.DATA.hr_size = 128
        self.cfg.DATA.lr_size = 32

        self.cfg.TRAIN = EasyDict()
        self.cfg.TRAIN.seed = 903
        # dataloader setting
        self.cfg.TRAIN.batch_size = 64
        self.cfg.TRAIN.dataloader_num_worker = 4
        # training setting
        self.cfg.TRAIN.end_epoch = 100
        self.cfg.TRAIN.log_per_epoch = 4
        self.cfg.TRAIN.learning_rate = 1e-4
        # resume setting
        self.cfg.TRAIN.resume = False
        self.cfg.TRAIN.load_checkpoint_path = None
        self.cfg.TRAIN.start_epoch = 0

    def prepare_experiment(self):
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
            print("[INFO] Experiment set up")

    def _make_directory(self, path):
        print(f"[INFO] Make directory {path}")
        if not Path(path).exists():
            Path(path).mkdir(parents=True)

    def _save_config_file(self, path):
        print(f"[INFO] Save config to {path}")
        with open(path, 'w') as f:
            f.write(json.dumps(self.cfg, indent=4))
