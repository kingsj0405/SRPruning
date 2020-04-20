from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 16 # [16] use 8 if your GPU memory is small, and use [2, 4] in tl.vis.save_images / use 16 for faster training
config.TRAIN.grid = [4, 4]
config.TRAIN.lr_init = 1e-6
config.TRAIN.beta1 = 0.9

## initialize G
config.TRAIN.n_epoch_init = 100
config.TRAIN.lr_decay_init = 0.1
config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

## adversarial learning (SRGAN)
config.TRAIN.n_epoch = 2000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

## train set location
config.TRAIN.hr_img_path = '../../dataset/DIV2K/DIV2K_train_HR/'
config.TRAIN.lr_img_path = '../../dataset/DIV2K/DIV2K_train_LR_bicubic/X4/'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '../../dataset/DIV2K/DIV2K_valid_HR/'
config.VALID.lr_img_path = '../../dataset/DIV2K/DIV2K_valid_LR_bicubic/X4/'

## save location
config.SAVE = edict()
config.SAVE.exp_name = '../../REWIND-srgan'
config.SAVE.exp_version = 'v8'
config.SAVE.cfg_dir = f'{config.SAVE.exp_name}/configs/'
config.SAVE.cfg_file_path = f'{config.SAVE.exp_name}/configs/{config.SAVE.exp_version}.cfg'
config.SAVE.save_dir = f"{config.SAVE.exp_name}/samples/{config.SAVE.exp_version}"
config.SAVE.checkpoint_dir = f"{config.SAVE.exp_name}/checkpoint/{config.SAVE.exp_version}"
config.SAVE.summary_dir = f"{config.SAVE.exp_name}/summary/{config.SAVE.exp_version}"

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
