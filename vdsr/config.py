from easydict import EasyDict as EasyDict
import json

config = edict()

# save configuration
config.SAVE = edict()
config.SAVE.exp_path = '../../REWIND-srgan'
config.SAVE.exp_version = 'v9'
config.SAVE.cfg_dir = f'{config.SAVE.exp_path}/configs/'
config.SAVE.cfg_file_path = f'{config.SAVE.exp_path}/configs/{config.SAVE.exp_version}.cfg'
config.SAVE.save_dir = f"{config.SAVE.exp_path}/samples/{config.SAVE.exp_version}"
config.SAVE.checkpoint_dir = f"{config.SAVE.exp_path}/checkpoint/{config.SAVE.exp_version}"
config.SAVE.summary_dir = f"{config.SAVE.exp_path}/summary/{config.SAVE.exp_version}"

config.TRAIN = edict()

def load_config(filename):
    with open(filename, 'r') as f:
        cfg = json.load(f.read())
        print(cfg)
        return cfg

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write(json.dumps(cfg, indent=4))
