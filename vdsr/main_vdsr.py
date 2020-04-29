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


class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()


# Training settings
parser = argparse.ArgumentParser(description="PyTorch VDSR")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=50, help="Number of epochs to train for")
parser.add_argument("--lr", type=float, default=0.1, help="Learning Rate. Default=0.1")
parser.add_argument("--step", type=int, default=10, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--cuda", action="store_true", help="Use cuda?")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--start-epoch", default=1, type=int, help="Manual epoch number (useful on restarts)")
parser.add_argument("--clip", type=float, default=0.4, help="Clipping Gradients. Default=0.4")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")
parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="Weight decay, Default: 1e-4")
parser.add_argument('--pretrained', default='', type=str, help='path to pretrained model (default: none)')
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")

def main():
    global opt, model, step, writer
    opt = parser.parse_args()
    print(opt)
    step = 0

    cuda = opt.cuda
    if cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = 903
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

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
    model = Net()
    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # optionally copy weights from a checkpoint
    if opt.pretrained:
        if os.path.isfile(opt.pretrained):

            weights = torch.load(opt.pretrained)
            model.load_state_dict(weights['model'].state_dict())
        else:
            print("=> no model found at '{}'".format(opt.pretrained))  

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.learning_rate)

    print("===> Training")
    writer = SummaryWriter(config.SAVE.summary_dir)
    for epoch in tqdm(range(opt.start_epoch, opt.nEpochs + 1), position=0, dynamic_ncols=True):
        train(training_data_loader, optimizer, model, criterion, epoch, writer)
        save_checkpoint(model, epoch)

def train(training_data_loader, optimizer, model, criterion, epoch, writer):
    global step
    model.train()
    for iteration, batch in enumerate(tqdm(training_data_loader, position=1, dynamic_ncols=True), 1):
        # Get the image
        train_img, label_img = Variable(batch[0]), Variable(batch[1], requires_grad=False)
        if opt.cuda:
            train_img = train_img.cuda()
            label_img = label_img.cuda()
        # Forward
        output = model(train_img)
        output = torch.clamp(output, 0.0, 1.0)
        loss = criterion(output, label_img)
        # Back-propagation
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), opt.clip) 
        optimizer.step()
        if ((iteration - 1) % 50 == 0):
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

if __name__ == "__main__":
    main()