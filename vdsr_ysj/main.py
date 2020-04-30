import fire
import numpy
import torch

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from dataset import SRDatasetFromDIV2K
from model import VDSR
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
        print("[INFO] Prepare net, optimizer, loss for training")
        net = VDSR().cuda()
        net.train()
        optimizer = torch.optim.Adam(
            net.parameters(), lr=config.TRAIN.learning_rate)
        criterion = torch.nn.MSELoss().cuda()
        print("[INFO] Start training loop")
        writer = SummaryWriter(config.SAVE.summary_dir)
        global_step = 0
        log_timing = (len(train_set) // config.TRAIN.batch_size) / \
            config.TRAIN.log_per_epoch
        for epoch in tqdm(
            range(
                config.TRAIN.start_epoch +
                1,
                config.TRAIN.end_epoch +
                1)):
            for index, hr_image in enumerate(tqdm(train_dataloader)):
                # Make low resolution input from high resolution image
                hr_image = hr_image.cuda()
                lr_image = DownSample2DMatlab(hr_image, 1 / 4, cuda=True)
                # Forward
                out = UpSample2DMatlab(lr_image, 4, cuda=True)
                out = net(out)
                loss = criterion(out, hr_image)
                # Back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Save images
                if index % log_timing:
                    # Summary to tensorboard
                    writer.add_scalar(
                        'MSE', loss.item(), global_step=global_step)
                    writer.flush()
                    # Save sample images
                    save_image(
                        lr_image, f"{config.SAVE.save_dir}/lr_epoch_{epoch}.png")
                    save_image(
                        out, f"{config.SAVE.save_dir}/out_epoch_{epoch}.png")
                    save_image(
                        hr_image, f"{config.SAVE.save_dir}/hr_epoch_{epoch}.png")
                # Add count
                global_step += config.TRAIN.batch_size


if __name__ == '__main__':
    fire.Fire(Main)
