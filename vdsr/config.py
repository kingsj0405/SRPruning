from easydict import easyDict as EasyDict
from pathlib import Path
import json

config = edict()

# save configuration
config.SAVE = edict()
config.SAVE.exp_path = '../../REWIND-vdsr'
config.SAVE.exp_version = 'v1'
config.SAVE.cfg_dir = f'{config.SAVE.exp_path}/configs/'
config.SAVE.cfg_file_path = f'{config.SAVE.exp_path}/configs/{config.SAVE.exp_version}.cfg'
config.SAVE.save_dir = f"{config.SAVE.exp_path}/samples/{config.SAVE.exp_version}"
config.SAVE.checkpoint_dir = f"{config.SAVE.exp_path}/checkpoint/{config.SAVE.exp_version}"
config.SAVE.summary_dir = f"{config.SAVE.exp_path}/summary/{config.SAVE.exp_version}"

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
        Path(config.SAVE.save_dir).mkdir(parents=True)
    if not Path(config.SAVE.summary_dir).exists():
        Path(config.SAVE.save_dir).mkdir(parents=True)
    log_config(config.SAVE.cfg_file_path)

# Add options