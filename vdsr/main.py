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
from layer import DownSample2DMatlab, UpSample2DMatlab
from util import psnr_set5


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
        train_set = SRDatasetFromDIV2K(dir_path=config.DATA.div2k_dir,
                                       transform=transforms.Compose([
                                           transforms.RandomCrop(
                                               [config.DATA.hr_size, config.DATA.hr_size]),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor()]),
                                       transform_lr=transforms.Compose([
                                           transforms.RandomCrop(
                                               [config.DATA.lr_size, config.DATA.lr_size]),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.RandomVerticalFlip(),
                                           transforms.ToTensor()
                                       ]))
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.TRAIN.lr_step_size)
        criterion = torch.nn.MSELoss().cuda()
        print("[INFO] Start training loop")
        writer = SummaryWriter(config.SAVE.summary_dir)
        global_step = 0
        log_timing = (len(train_set) // config.TRAIN.batch_size) / \
            config.TRAIN.log_per_epoch
        for epoch in tqdm(range(config.TRAIN.start_epoch + 1,
                                config.TRAIN.end_epoch + 1)):
            for index, hr_image in enumerate(
                    tqdm(train_dataloader)):
                # Make low resolution input from high resolution image
                hr_image = hr_image.cuda()
                lr_image = DownSample2DMatlab(hr_image, 1 / 4, cuda=True)
                # Forward
                bicubic_image = UpSample2DMatlab(lr_image, 4, cuda=True)
                out = net(bicubic_image)
                loss = criterion(out, hr_image)
                # Back-propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Check training status
                if index % log_timing == 0:
                    # Add images to tensorboard
                    writer.add_images('1 hr', hr_image.clamp(0, 1))
                    writer.add_images('2 out', out.clamp(0, 1))
                    writer.add_images('3 bicubic', bicubic_image.clamp(0, 1))
                    # writer.add_images('4 model_output', model_output)# Memory
                    writer.add_images('5 lr', lr_image.clamp(0, 1))
                    # Add values to tensorboard
                    writer.add_scalar(
                        '1 MSE', loss.item(), global_step=global_step)
                    app, apb = psnr_set5(net,
                                         set5_dir=config.DATA.set5_dir,
                                         save_dir=config.SAVE.save_dir)
                    writer.add_scalar(
                        '2 Set5 PSNR VDSR', app, global_step=global_step)
                    writer.add_scalar(
                        '3 Set5 PSNR bicubic', apb, global_step=global_step)
                    writer.add_scalar(
                        '4 learning rate', optimizer.param_groups[0]['lr'],
                        global_step=global_step)
                    writer.flush()
                    # Save sample images
                    save_image(lr_image,
                               f"{config.SAVE.save_dir}/epoch_{epoch}_lr.png")
                    save_image(out,
                               f"{config.SAVE.save_dir}/epoch_{epoch}_out.png")
                    # save_image(model_output,
                    #            f"{config.SAVE.save_dir}/epoch_{epoch}_model_output.png")
                    save_image(hr_image,
                               f"{config.SAVE.save_dir}/epoch_{epoch}_hr.png")
                    # Save checkpoint
                    torch.save({
                        'config': config,
                        'epoch': epoch,
                        'global_step': global_step,
                        'net': net.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_{epoch}.pth")
                # Add count
                global_step += config.TRAIN.batch_size
            scheduler.step()


if __name__ == '__main__':
    fire.Fire(Main)
