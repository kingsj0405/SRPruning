import fire
import numpy
import json
import torch

from easydict import EasyDict
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from dataset import SRDatasetFromDIV2K
from model import VDSR
from layer import DownSample2DMatlab, UpSample2DMatlab
from util import psnr_set5
from pruning import RandomPruning, MagnitudePruning
from visualization import _filter


def train():
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
    if torch.cuda.device_count() > 1:
        print(
            f"[INFO] Use multiple gpus with count {torch.cuda.device_count()}")
        net = torch.nn.DataParallel(net)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.TRAIN.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, config.TRAIN.lr_step_size)
    if config.TRAIN.resume:
        print(
            f"[INFO] Load checkpoint from {config.TRAIN.load_checkpoint_path}")
        checkpoint = torch.load(config.TRAIN.load_checkpoint_path)
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        start_epoch = 0
        global_step = 0
    criterion = torch.nn.MSELoss().cuda()
    print("[INFO] Start training loop")
    net.train()
    writer = SummaryWriter(config.SAVE.summary_dir)
    log_timing = (len(train_set) // config.TRAIN.batch_size) / \
        config.TRAIN.period_log
    for epoch in tqdm(range(start_epoch + 1,
                            config.TRAIN.end_epoch + 1)):
        if epoch == (start_epoch + 1) or epoch % config.TRAIN.period_save == 0:
            # Save checkpoint
            torch.save({
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_{epoch}.pth")
        for index, hr_image in enumerate(tqdm(train_dataloader)):
            # Make low resolution input from high resolution image
            hr_image = hr_image.cuda()
            lr_image = DownSample2DMatlab(hr_image, 1 / 4, cuda=True)
            # Forward
            bicubic_image = UpSample2DMatlab(lr_image, 4, cuda=True)
            out = net(bicubic_image)
            loss = criterion(out, hr_image)
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
            # Back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Add count
            global_step += config.TRAIN.batch_size
        scheduler.step()


def random_pruning(checkpoint_path, save_dir, pruning_rate, try_cnt):
    print("[INFO] Set random seed")
    numpy.random.seed(903)
    torch.manual_seed(903)
    torch.cuda.manual_seed(903)
    print(f"[INFO] Load from checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = VDSR().cuda()
    net.load_state_dict(checkpoint['net'])
    print(f"[INFO] Get psnr5 from randomly pruned network")
    data = []
    psnrs = []
    for i in tqdm(range(1, try_cnt + 1)):
        # Prune
        pruning = RandomPruning(net.parameters(), pruning_rate)
        pruning.step()
        pruning.zero()
        # # Calculate psnr5
        # psnr, _ = psnr_set5(net, set5_dir, save_dir, False)
        # # Append to lists
        # data.append({
        #     'psnr': psnr,
        #     'masks': pruning.masks
        # })
        # psnrs.append(psnr)
    # print(f"[INFO] psnr statistics")
    # psnrs = numpy.array(psnrs)
    # statistics = {
    #     'min': psnrs.min(),
    #     'max': psnrs.max(),
    #     'median': psnrs.median(),
    #     'mean': psnrs.mean()
    # }
    # print(f"[INFO] Save masks and psnr value to {save_dir}/random-pruning.json")
    # save_data = EasyDict()
    # save_data.statistics = statistics
    # save_data.data = data
    # with open(f"{save_dir}/random-pruning.json", 'w') as f:
    #     f.write(json.dumps(save_data, index=4))


def test():
    raise NotImplementedError


def filter(checkpoint_path, save_dir, target_conv_index, filter_index):
    """
    summary:
        visualize filter by get activation of random input

    parameters:
        checkpoint_path: path to checkpoint
        save_dir: path to save visualization
        target_conv_index: conv index, start from 1
        filter_index: filter index, start from 1
    """
    print(f"[INFO] Load checkpoitn from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = VDSR()
    net.load_state_dict(checkpoint['net'])
    _filter(net, save_dir, target_conv_index, filter_index)


if __name__ == '__main__':
    fire.Fire({
        'train': train,
        'test': test,
        'filter': filter,
        'random_pruning': random_pruning
    })
