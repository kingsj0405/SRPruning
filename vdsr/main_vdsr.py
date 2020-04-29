import argparse
import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from PIL.Image import BICUBIC
from tqdm import tqdm
from vdsr import Net
from dataset import DatasetFromHdf5, DatasetFromDIV2K
from config import config


def psnr(img1, img2, max=255):
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""
    if max == 1:
        img1 = torch.clamp(img1 * 255.0, 0, 255.0)
        img2 = torch.clamp(img2 * 255.0, 0, 255.0)
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse))


def train(training_data_loader, optimizer, model, criterion, epoch, writer):
    global step
    model.train()
    for iteration, batch in enumerate(tqdm(training_data_loader, position=1, dynamic_ncols=True), 1):
        # Get the image
        train_img, label_img = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        train_img = train_img.cuda()
        label_img = label_img.cuda()
        # Forward
        output = model(train_img)
        output = torch.clamp(output, 0.0, 1.0)
        loss = criterion(output, label_img)
        # Back-propagation
        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()
        if ((iteration - 1) % config.TRAIN.log_period == 0):
            writer.add_scalar('MSE', loss.item(), global_step=step)
            writer.add_scalar('PSNR', psnr(output, label_img, 1), global_step=step)
            writer.add_scalar('PSNR bicubic', psnr(train_img, label_img, 1), global_step=step)
            writer.add_images('output', output, global_step=step)
            writer.add_images('output_model', model.output_model, global_step=step)
            writer.add_images('label_img', label_img, global_step=step)
            writer.add_images('train_img', train_img, global_step=step)
            writer.flush()
            save_image(output, f"{config.SAVE.save_dir}/train_{step}.png")
        step += 1


def save_checkpoint(model, epoch):
    model_out_path = f"{config.SAVE.checkpoint_dir}/model_epoch_{epoch}.pth"
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch VDSR")
    parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")

    # Variables
    global opt, model, step, writer
    opt = parser.parse_args()
    print(opt)
    step = 0

    print("Random Seed: ", config.TRAIN.seed)
    np.random.seed(config.TRAIN.seed)
    torch.manual_seed(config.TRAIN.seed)
    torch.cuda.manual_seed(config.TRAIN.seed)

    print("===> Loading datasets")
    # train_set = DatasetFromHdf5(config.DATA.train_db_path)
    sr_size = config.DATA.sr_size
    train_set = DatasetFromDIV2K(train_dirpath=config.DATA.train_lr_path,
                                 label_dirpath=config.DATA.train_hr_path,
                                 train_transform=transforms.Compose([
                                     transforms.Resize([int(sr_size / 4), int(sr_size / 4)], BICUBIC),
                                     transforms.Resize([sr_size, sr_size], BICUBIC),
                                 ]),
                                 label_transform=transforms.Compose([
                                     transforms.RandomCrop([sr_size, sr_size]),
                                 ]),
                                 all_transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=opt.threads,
                                      batch_size=config.TRAIN.batch_size,
                                      shuffle=True)

    print("===> Building model")
    model = Net().cuda()
    criterion = nn.MSELoss().cuda()

    # # optionally resume from a checkpoint
    # if opt.resume:
    #     if os.path.isfile(opt.resume):
    #         print("=> loading checkpoint '{}'".format(opt.resume))
    #         checkpoint = torch.load(opt.resume)
    #         opt.start_epoch = checkpoint["epoch"] + 1
    #         model.load_state_dict(checkpoint["model"].state_dict())
    #     else:
    #         print("=> no checkpoint found at '{}'".format(opt.resume))

    # # optionally copy weights from a checkpoint
    # if opt.pretrained:
    #     if os.path.isfile(opt.pretrained):

    #         weights = torch.load(opt.pretrained)
    #         model.load_state_dict(weights['model'].state_dict())
    #     else:
    #         print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.learning_rate)

    print("===> Training")
    writer = SummaryWriter(config.SAVE.summary_dir)
    for epoch in tqdm(range(config.TRAIN.start_epoch, config.TRAIN.end_epoch + 1), position=0, dynamic_ncols=True):
        train(training_data_loader, optimizer, model, criterion, epoch, writer)
        save_checkpoint(model, epoch)


if __name__ == "__main__":
    main()