import random
import time
import numpy as np

from torch.utils import data
from pathlib import Path
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SRDatasetFromDIV2K(data.Dataset):
    """
    Summary:

        Return only high resolution image and apply matlab downsample
        and upsample on batch in train loop

    Parameters:

        dir_path should be as following

    $ tree -d dataset/DIV2K/
    dataset/DIV2K/
    ├── DIV2K_train_HR
    ├── DIV2K_train_LR_bicubic
    │   └── X4
    ├── DIV2K_valid_HR
    └── DIV2K_valid_LR_bicubic
        └── X4

    6 directories
    """

    def __init__(self, dir_path="dataset/DIV2K", transform=None, transform_lr=None):
        super(SRDatasetFromDIV2K, self).__init__()
        self.hr_list = sorted(
            list(Path(f"{dir_path}/DIV2K_train_HR").glob('*.png')))
        # self.lr_list = sorted(
        #     list(Path(f"{dir_path}/DIV2K_train_LR_bicubic/X4").glob('*.png')))
        # assert(len(self.hr_list) == len(self.lr_list))
        self.len = len(self.hr_list)
        self.transform = transform
        self.transform_lr = transform_lr

    def __getitem__(self, index):
        hr_img = Image.open(self.hr_list[index])
        # lr_img = Image.open(self.lr_list[index])
        seed = np.random.randint(time.time())
        if self.transform is not None:
            random.seed(seed)
            hr_img = self.transform(hr_img)
        # if self.transform_lr is not None:
        #     random.seed(seed)
        #     lr_img = self.transform_lr(lr_img)
        return hr_img#, lr_img

    def __len__(self):
        return self.len
