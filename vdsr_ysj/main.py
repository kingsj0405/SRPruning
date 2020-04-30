import fire
import numpy
import torch

from torchvision import transforms
from torchvision.utils import save_image

from config import Config
from dataset import SRDatasetFromDIV2K
from util import DownSample2DMatlab, UpSample2DMatlab


class Main:
    def test(self):
        raise NotImplementedError

    def train(self):
        print("[INFO] Set configuration")
        config = Config()
        config.prepare_experiment()
        # FIXME: After this issue resolved
        # https://github.com/makinacorpus/easydict/issues/20
        config = config.cfg
        print("[INFO] Set random seed")
        numpy.random.seed(config.TRAIN.seed)
        torch.manual_seed(config.TRAIN.seed)
        torch.cuda.manual_seed(config.TRAIN.seed)
        print("[INFO] Get training dataset and data_loader")
        train_set = SRDatasetFromDIV2K(dir_path=config.DATA.div2k_dir, transform=transforms.Compose(
            [transforms.RandomCrop([config.DATA.hr_size, config.DATA.hr_size]), transforms.ToTensor()]))
        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_set,
            num_workers=4,
            batch_size=config.TRAIN.batch_size,
            shuffle=True)
        # FIXME: make as real train loop
        for index, hr_image in enumerate(train_dataloader):
            # Make low resolution input from high resolution image
            lr_image = DownSample2DMatlab(hr_image, 1/4, cuda=False)
            # Suuuper resolution
            out = UpSample2DMatlab(lr_image, 4, cuda=False)
            # Save images
            save_image(lr_image, f"{config.SAVE.save_dir}/lr_sample_{index}.png")
            save_image(out, f"{config.SAVE.save_dir}/out_sample_{index}.png")
            save_image(hr_image, f"{config.SAVE.save_dir}/hr_sample_{index}.png")


if __name__ == '__main__':
    fire.Fire(Main)
