import fire
import numpy
import torch

from torchvision import transforms
from torchvision.utils import save_image

from config import Config
from dataset import SRDatasetFromDIV2K


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
        for index, batch in enumerate(train_dataloader):
            print(f"[{index}] {batch.shape}")
            save_image(batch, f"{config.SAVE.save_dir}/sample_{index}.png")


if __name__ == '__main__':
    fire.Fire(Main)
