from easydict import EasyDict as edict
from pathlib import Path
import json

config = edict()

# save configuration
config.SAVE = edict()
config.SAVE.exp_path = '../../REWIND-vdsr'
config.SAVE.exp_version = 'v23'
config.SAVE.description = "Add clamp from v22"
config.SAVE.cfg_dir = f'{config.SAVE.exp_path}/configs/'
config.SAVE.cfg_file_path = f'{config.SAVE.exp_path}/configs/{config.SAVE.exp_version}.cfg'
config.SAVE.save_dir = f"{config.SAVE.exp_path}/samples/{config.SAVE.exp_version}"
config.SAVE.checkpoint_dir = f"{config.SAVE.exp_path}/checkpoint/{config.SAVE.exp_version}"
config.SAVE.summary_dir = f"{config.SAVE.exp_path}/summary/{config.SAVE.exp_version}"

config.DATA = edict()
# config.DATA.train_db_path = '../../dataset/291/train.h5'
config.DATA.train_hr_path = '../../dataset/DIV2K/DIV2K_train_HR'
config.DATA.train_lr_path = '../../dataset/DIV2K/DIV2K_train_LR_bicubic/X4'
config.DATA.valid_hr_path = '../../dataset/DIV2K/DIV2K_valid_HR'
config.DATA.valid_lr_path = '../../dataset/DIV2K/DIV2K_valid_LR_bicubic/X4'
config.DATA.sr_size = 64

config.TRAIN = edict()
config.TRAIN.batch_size = 128
config.TRAIN.learning_rate = 1e-4

# other options
def load_config(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f.read())
        print(cfg)
        return cfg

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write(json.dumps(cfg, indent=4))

if Path(config.SAVE.cfg_file_path).exists():
    print(f"Configuration {config.SAVE.cfg_file_path} already exists")
    print(f"Stop Experiment {config.SAVE.exp_version}")
    exit(-1)
else:
    if not Path(config.SAVE.cfg_dir).exists():
        Path(config.SAVE.cfg_dir).mkdir(parents=True)
    if not Path(config.SAVE.save_dir).exists():
        Path(config.SAVE.save_dir).mkdir(parents=True)
    if not Path(config.SAVE.checkpoint_dir).exists():
        Path(config.SAVE.checkpoint_dir).mkdir(parents=True)
    if not Path(config.SAVE.summary_dir).exists():
        Path(config.SAVE.summary_dir).mkdir(parents=True)
    log_config(config.SAVE.cfg_file_path, config)