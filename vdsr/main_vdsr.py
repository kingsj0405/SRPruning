import argparse
import os
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TVF
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from vdsr import Net
from dataset import DatasetFromHdf5, DatasetFromDIV2K
from config import config, start_experiment


def train(training_data_loader, epoch, writer):
    def psnr(img1, img2, max=255):
        """Peak Signal to Noise Ratio
        img1 and img2 have range [0, 255]"""
        if max == 1:
            img1 = torch.clamp(img1 * 255.0, 0, 255.0)
            img2 = torch.clamp(img2 * 255.0, 0, 255.0)
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
    # train
    global step, model, optimizer, criterion
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
            writer.add_images('model.output_model', model.output_model, global_step=step)
            writer.add_images('label_img', label_img, global_step=step)
            writer.add_images('train_img', train_img, global_step=step)
            writer.flush()
            save_image(output, f"{config.SAVE.save_dir}/train_{step}.png")
            psnr_set5(model, config.DATA.set5_dir, config.SAVE.save_dir, writer)
        step += config.TRAIN.batch_size

def psnr_set5(model, set5_dir, save_dir, writer):
    def psnr(y_true,y_pred, shave_border=4):
        '''
            Input must be 0-255, 2D
        '''
        # rgb to YCbCr
        def _rgb2ycbcr(img, maxVal=255):
            # Same as MATLAB's rgb2ycbcr
            # Updated at 03/14/2017
            # Not tested for cb and cr
            O = np.array([[16],
                        [128],
                        [128]])
            T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                        [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                        [0.439215686274510, -0.367788235294118, -0.071427450980392]])
            if maxVal == 1:
                O = O / 255.0
            t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
            t = np.dot(t, np.transpose(T))
            t[:, 0] += O[0]
            t[:, 1] += O[1]
            t[:, 2] += O[2]
            ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
            return ycbcr
        # psnr
        y_true = _rgb2ycbcr(y_true)[:,:,0]
        y_pred = _rgb2ycbcr(y_pred)[:,:,0]
        target_data = np.array(y_true, dtype=np.float32)
        ref_data = np.array(y_pred, dtype=np.float32)
        diff = ref_data - target_data
        if shave_border > 0:
            diff = diff[shave_border:-shave_border, shave_border:-shave_border]
        rmse = np.sqrt(np.mean(np.power(diff, 2)))
        return 20 * np.log10(255./rmse)
    # psnr_set5
    global step
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    avg_elapsed_time = 0.0
    count = 0.0
    scale = 4
    image_list = list(Path(set5_dir).glob('*.bmp'))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for image_path in image_list:
        count += 1
        # Make bicubic image from label image
        label_img = Image.open(image_path)
        test_img = TVF.resize(label_img, (int(label_img.size[1] / 4), int(label_img.size[0] / 4)))
        bicubic_img = TVF.resize(test_img, (label_img.size[1], label_img.size[0]))
        # Numpy
        label_img = np.asarray(label_img)
        bicubic_img = np.asarray(bicubic_img)
        # Get psnr of bicubic
        psnr_bicubic = psnr(label_img, bicubic_img, scale)
        avg_psnr_bicubic += psnr_bicubic
        # Normalize
        bicubic_img = bicubic_img - 127.5
        bicubic_img = bicubic_img / 127.5
        bicubic_img = Variable(torch.from_numpy(bicubic_img).float()).view(1, -1, bicubic_img.shape[0], bicubic_img.shape[1])
        bicubic_img = bicubic_img.cuda()
        # Super-Resolution
        start_time = time.time()
        predicted_img = model(bicubic_img)
        elapsed_time = time.time() - start_time
        avg_elapsed_time += elapsed_time
        # Denomalize
        predicted_img = predicted_img.cpu()
        predicted_img = predicted_img.data[0].view(bicubic_img.shape[2], bicubic_img.shape[3], -1).numpy()
        predicted_img = predicted_img * 127.5 + 127.5
        predicted_img[predicted_img < 0] = 0
        predicted_img[predicted_img > 255.] = 255.
        predicted_img = predicted_img.astype(np.uint8)
        # Get psnr of generated
        psnr_predicted = psnr(label_img, predicted_img, scale)
        avg_psnr_predicted += psnr_predicted
        # Save if
        Image.fromarray(predicted_img).save(f"{save_dir}/{image_path.stem}_step_{step}.png")
    # writer to tensorboard
    writer.add_scalar('Set5 PSNR VDSR', avg_psnr_predicted/count, global_step=step)
    writer.add_scalar('Set5 PSNR bicubic', avg_psnr_bicubic/count, global_step=step)
    writer.add_scalar('Set5 average time', avg_elapsed_time/count, global_step=step)
    writer.flush()


def save_checkpoint(model, epoch, step):
    model_out_path = f"{config.SAVE.checkpoint_dir}/model_epoch_{epoch}.pth"
    state = {"epoch": epoch, "step": step ,"model": model}
    torch.save(state, model_out_path)


def main():
    global step, model, optimizer, criterion
    print("Random Seed: ", config.TRAIN.seed)
    np.random.seed(config.TRAIN.seed)
    torch.manual_seed(config.TRAIN.seed)
    torch.cuda.manual_seed(config.TRAIN.seed)
    print("===> Loading datasets")
    sr_size = config.DATA.sr_size
    train_set = DatasetFromDIV2K(train_dirpath=config.DATA.train_lr_path,
                                 label_dirpath=config.DATA.train_hr_path,
                                 train_transform=transforms.Compose([
                                     transforms.Resize([int(sr_size / 4), int(sr_size / 4)], Image.BICUBIC),
                                     transforms.Resize([sr_size, sr_size], Image.BICUBIC),
                                 ]),
                                 label_transform=transforms.Compose([
                                     transforms.RandomCrop([sr_size, sr_size]),
                                 ]),
                                 all_transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))
    training_data_loader = DataLoader(dataset=train_set,
                                      num_workers=4,
                                      batch_size=config.TRAIN.batch_size,
                                      shuffle=True)
    print("===> Building model")
    model = Net().cuda()
    criterion = nn.MSELoss().cuda()
    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.learning_rate, momentum=0.9, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)  # step_size is epoch_size explicit by scheduler.step()
    print("===> Prepare experiment")
    if config.TRAIN.start_epoch == 0:
        start_epoch = 0
        step = 0
        start_experiment(config)
    else:
        checkpoint = torch.load(f"{config.SAVE.checkpoint_dir}/model_epoch_{config.TRAIN.start_epoch}.pth")
        start_epoch = checkpoint["epoch"] + 1
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"].state_dict())
    print("===> Training")
    writer = SummaryWriter(config.SAVE.summary_dir)
    for epoch in tqdm(range(start_epoch + 1, config.TRAIN.end_epoch + 1), position=0, dynamic_ncols=True):
        train(training_data_loader, epoch, writer)
        save_checkpoint(model, epoch, step)
        # scheduler.step()


if __name__ == "__main__":
    main()