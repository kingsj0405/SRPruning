import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torchvision import transforms
from tqdm import tqdm
from vdsr import Net
from dataset import DatasetFromHdf5, DatasetFromDIV2K
from config import config

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

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    # train_set = DatasetFromHdf5(config.DATA.train_db_path)
    sr_size = config.DATA.sr_size
    train_set = DatasetFromDIV2K(train_dirpath=config.DATA.train_lr_path, label_dirpath=config.DATA.train_hr_path,
                                 train_transform=transforms.Compose([
                                     transforms.RandomCrop([sr_size / 4, sr_size / 4]),
                                     transforms.Resize([sr_size, sr_size]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
                                 ]),
                                 label_transform=transforms.Compose([
                                     transforms.RandomCrop([sr_size, sr_size]),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
                                 ]))
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=config.TRAIN.batch_size, shuffle=True)

    print("===> Building model")
    model = Net()
    criterion = nn.MSELoss(size_average=False)

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
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)

    print("===> Training")
    writer = SummaryWriter(config.SAVE.summary_dir)
    for epoch in tqdm(range(opt.start_epoch, opt.nEpochs + 1), position=0, dynamic_ncols=True):
        train(training_data_loader, optimizer, model, criterion, epoch, writer)
        save_checkpoint(model, epoch)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (0.1 ** (epoch // opt.step))
    return lr

def train(training_data_loader, optimizer, model, criterion, epoch, writer):
    global step
    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # print("Epoch = {}, lr = {}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()

    for iteration, batch in enumerate(tqdm(training_data_loader, position=1, dynamic_ncols=True), 1):
        train_img, label_img = Variable(batch[0]), Variable(batch[1], requires_grad=False)

        if opt.cuda:
            train_img = input.cuda()
            label_img = target.cuda()

        output = model(train_img)
        loss = criterion(output, label_img)
        optimizer.zero_grad()
        loss.backward() 
        nn.utils.clip_grad_norm(model.parameters(),opt.clip) 
        optimizer.step()

        if ((iteration - 1) % 100 == 0):
            # print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))
            writer.add_scalar('MSE', loss, global_step=step)
            writer.add_scalar('learning_rate', lr, global_step=step)
            writer.add_images('output', output * 127.5 + 127.5, global_step=step)
            writer.add_images('label_img', target * 127.5 + 127.5, global_step=step)
            writer.add_images('train_img', input * 127.5 + 127.5, global_step=step)
            save_image(output * 127.5 + 127.5, f"{config.SAVE.save_dir}/train_{step}.png")
        step += 1

def save_checkpoint(model, epoch):
    model_out_path = f"{config.SAVE.checkpoint_dir}/model_epoch_{epoch}.pth"
    state = {"epoch": epoch ,"model": model}
    torch.save(state, model_out_path)
    # print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()