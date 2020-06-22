import fire
import numpy
import json
import torch
import pickle

from easydict import EasyDict
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from config import Config
from dataset import SRDatasetFromDIV2K
from model import VDSR
from layer import DownSample2DMatlab, UpSample2DMatlab
from util import psnr_set5
from pruning import pruning_map, Pruning, RandomPruning, MagnitudePruning, MagnitudeFilterPruning, AttentionPruning
from visualization import _filter


def train():
    print("[INFO] Set configuration")
    config = Config()
    config.prepare_training()
    # FIXME: After this issue resolved
    # https://github.com/makinacorpus/easydict/issues/20
    config = config.cfg
    print("[INFO] Set random seed")
    numpy.random.seed(config.EXP.seed)
    torch.manual_seed(config.EXP.seed)
    torch.cuda.manual_seed(config.EXP.seed)
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
    # Use multiple gpus if possible
    if torch.cuda.device_count() > 1:
        print(
            f"[INFO] Use multiple gpus with count {torch.cuda.device_count()}")
    net = torch.nn.DataParallel(net)
    optimizer = torch.optim.Adam(
        net.parameters(), lr=config.TRAIN.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.TRAIN.lr_step_milestones,
        gamma=config.TRAIN.lr_step_gamma)
    criterion = torch.nn.MSELoss().cuda()
    # Re-load from checkpoint, this can be rewinding
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
    # Set pruning mask
    if config.TRAIN.pruning:
        json_path = f"{config.TRAIN.pruning_dir}/pruning-report.json"
        print(f"[INFO] Load pruning report from {json_path}")
        with open(json_path, 'r') as f:
            pruning_report = json.load(f)
            pruned_index = int(pruning_report['statistics']['argmax']) + 1
        channel_mask_path = f"{config.TRAIN.pruning_dir}/channel_mask_{pruned_index}.pickle"
        print(f"[INFO] Load mask index from {channel_mask_path}")
        with open(channel_mask_path, 'rb') as f:
            channel_mask = pickle.load(f)
        pruning_method = pruning_report['meta']['config']['method']
        pruning_rate = pruning_report['meta']['config']['pruning_rate']
        if pruning_method in pruning_map.keys():
            pruning_method = pruning_map[pruning_method]
            pruning = pruning_method(net.parameters(), pruning_rate)
            pruning.update(channel_mask)
        else:
            raise Exception(
                f"Not proper config.PRUNE.method, cur var is: {config.PRUNE.method}")
    print("[INFO] Start training loop")
    net.train()
    writer = SummaryWriter(config.SAVE.summary_dir)
    print("[INFO] Save checkpoint before training")
    torch.save({
        'config': config,
        'epoch': 0,
        'global_step': global_step,
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_0.pth")
    for epoch in tqdm(range(start_epoch + 1,
                            config.TRAIN.end_epoch + 1), position=0, leave=True):
        for index, hr_image in enumerate(tqdm(train_dataloader, position=1, leave=False)):
            # Make low resolution input from high resolution image
            hr_image = hr_image.cuda()
            lr_image = DownSample2DMatlab(hr_image, 1 / 4, cuda=True)
            # Zero masked value with pruning
            if config.TRAIN.pruning:
                pruning.zero()
            # Forward
            bicubic_image = UpSample2DMatlab(lr_image, 4, cuda=True)
            out = net(bicubic_image)
            loss = criterion(out, hr_image)
            # Back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Add count
            global_step += config.TRAIN.batch_size
        if epoch == 1 or epoch % config.TRAIN.period_log == 0:
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
        if epoch % config.TRAIN.period_save == 0:
            # Save checkpoint
            torch.save({
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_{epoch}.pth")
        scheduler.step()


def pruning():
    print("[INFO] Set configuration")
    config = Config()
    config.prepare_pruning()
    # FIXME: After this issue resolved
    # https://github.com/makinacorpus/easydict/issues/20
    config = config.cfg
    print("[INFO] Set random seed")
    numpy.random.seed(config.EXP.seed)
    torch.manual_seed(config.EXP.seed)
    torch.cuda.manual_seed(config.EXP.seed)
    print(
        f"[INFO] Load from checkpoint {config.PRUNE.trained_checkpoint_path}")
    checkpoint = torch.load(config.PRUNE.trained_checkpoint_path)
    print(f"[INFO] Get psnr set5 from randomly pruned network")
    result = EasyDict()
    psnrs = []
    for i in tqdm(range(1, config.PRUNE.random_prune_try_cnt + 1)):
        # Load net
        net = VDSR().cuda()
        net = torch.nn.DataParallel(net)
        net.load_state_dict(checkpoint['net'])
        # Prune
        if config.PRUNE.method in pruning_map.keys():
            pruning_method = pruning_map[config.PRUNE.method]
            pruning = pruning_method(
                net.parameters(), config.PRUNE.pruning_rate)
        else:
            raise Exception(
                f"Not proper config.PRUNE.method, cur var is: {config.PRUNE.method}")
        pruning.update()
        pruning.zero()
        # Calculate psnr5
        psnr, _ = psnr_set5(net,
                            set5_dir=config.DATA.set5_dir,
                            save_dir=config.SAVE.save_dir,
                            save=False)
        # Save results
        with open(f"{config.SAVE.pruning_dir}/channel_mask_{i}.pickle", 'wb') as f:
            pickle.dump(pruning.channel_mask, f)
        psnrs.append(psnr)
    result.psnrs = psnrs
    print(f"[INFO] Get meta and statistics of experiment")
    result.meta = EasyDict()
    result.meta.config = config.PRUNE
    result.statistics = EasyDict()
    psnrs = numpy.array(psnrs)
    result.statistics.min = psnrs.min()
    result.statistics.max = psnrs.max()
    result.statistics.mean = psnrs.mean()
    result.statistics.argmax = int(psnrs.argmax())
    json_path = f"{config.SAVE.pruning_dir}/pruning-report.json"
    print(f"[INFO] Save masks and psnr value to {json_path}")
    with open(json_path, 'w') as f:
        json_txt = json.dumps(result, indent=4)
        f.write(json_txt)


def train_pruning_net():
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Use multiple gpus with count {torch.cuda.device_count()}")
    print("[INFO] Set configuration")
    config = Config()
    config.prepare_training()
    # FIXME: After this issue resolved
    # https://github.com/makinacorpus/easydict/issues/20
    config = config.cfg
    print("[INFO] Set random seed")
    numpy.random.seed(config.EXP.seed)
    torch.manual_seed(config.EXP.seed)
    torch.cuda.manual_seed(config.EXP.seed)
    print("[INFO] Load dataset for activation")
    train_set = SRDatasetFromDIV2K(dir_path=config.DATA.div2k_dir,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(
                                           [config.DATA.hr_size, config.DATA.hr_size]),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor()]),
                                   transform_lr=transforms.Compose([
                                       transforms.RandomCrop([config.DATA.lr_size, config.DATA.lr_size]),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.RandomVerticalFlip(),
                                       transforms.ToTensor()
                                   ]),
                                   mode='valid')
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_set,
        num_workers=4,
        batch_size=config.TRAIN.batch_size,
        shuffle=True)
    print("[INFO] Prepare net, optimizer, loss for training")
    net = VDSR().cuda()
    net = torch.nn.DataParallel(net)
    pruning = AttentionPruning(net.parameters(), None)
    pruning.net = pruning.net.cuda()
    pruning.net = torch.nn.DataParallel(pruning.net)
    optimizer = torch.optim.Adam(
        pruning.net.parameters(), lr=config.TRAIN.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=config.TRAIN.lr_step_milestones,
        gamma=config.TRAIN.lr_step_gamma)
    criterion = torch.nn.MSELoss().cuda()
    print(f"[INFO] Load checkpoint from {config.TRAIN_PRUNE.model_parameters}")
    checkpoint = torch.load(config.PRUNE.trained_checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    print("[INFO] Start training")
    net.eval()
    pruning.net.train()
    writer = SummaryWriter(config.SAVE.summary_dir)
    start_epoch = 0
    global_step = 0
    print("[INFO] Save checkpoint before training")
    torch.save({
        'config': config,
        'epoch': 0,
        'global_step': global_step,
        'net': net.state_dict(),
        'pruning.net': pruning.net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_0.pth")
    for epoch in tqdm(range(start_epoch + 1,
                            config.TRAIN.end_epoch + 1), position=0, leave=True):
        for index, image in enumerate(tqdm(train_dataloader, position=1, leave=False)):
            # input and model parameters(from pruning)
            image = image.cuda()
            image = DownSample2DMatlab(image, 1 / 4, cuda=True)
            # forward through pruning.net
            w_0 = pruning.clone_params()
            pruning.update(image)
            # get pruned activation
            with torch.no_grad():
                pruning.zero()
                output = net(image)
            # get original activation
            with torch.no_grad():
                pruning.rewind(w_0)
                origin = net(image)
            # Get reconstruction loss
            loss = criterion(origin, output)
            # Back-propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Add count
            global_step += config.TRAIN.batch_size
        if epoch % config.TRAIN.period_log == 0:
            writer.add_scalar(
                '1 MSE', loss.item(), global_step=global_step
            )
        if epoch % config.TRAIN.period_save == 0:
            # Save checkpoint
            torch.save({
                'config': config,
                'epoch': epoch,
                'global_step': global_step,
                'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, f"{config.SAVE.checkpoint_dir}/SRPruning_epoch_{epoch}.pth")
        scheduler.step()


def visualize_filter(checkpoint_path, save_dir, target_conv_index, filter_index):
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
        'pruning': pruning,
        'train_pruning_net': train_pruning_net,
        'visualize': visualize_filter
    })
