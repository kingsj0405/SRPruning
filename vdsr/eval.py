import argparse
import os
import torch
import torchvision.transforms.functional as TVF
import numpy as np
import time
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from PIL import Image

scale = 4
version = 'v23'
image_list = list(Path(f'../../dataset/Set5/').glob('*.bmp'))
writer = SummaryWriter(f'../../REWIND-vdsr/summary/{version}/')

def PSNR(y_true,y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''
    # Same as MATLAB's rgb2ycbcr
    # Updated at 03/14/2017
    # Not tested for cb and cr
    def _rgb2ycbcr(img, maxVal=255):
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
    y_true = _rgb2ycbcr(y_true)[:,:,0]
    y_pred = _rgb2ycbcr(y_pred)[:,:,0]
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)
    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 20 * np.log10(255./rmse)

def main():
    for epoch in range(1, 51):
        model = torch.load(f'../../REWIND-vdsr/checkpoint/{version}/model_epoch_{epoch}.pth', map_location=lambda storage, loc: storage)["model"]
        avg_psnr_predicted = 0.0
        avg_psnr_bicubic = 0.0
        avg_elapsed_time = 0.0
        count = 0.0
        for image_path in image_list:
            count += 1
            print(f"Processing {image_path}")
            label_img = Image.open(image_path)
            test_img = TVF.resize(label_img, (int(label_img.size[1] / 4), int(label_img.size[0] / 4)))
            bicubic_img = TVF.resize(test_img, (label_img.size[1], label_img.size[0]))

            label_img = np.asarray(label_img)
            test_img = np.asarray(test_img)
            bicubic_img = np.asarray(bicubic_img)

            psnr_bicubic = PSNR(label_img, bicubic_img, scale)
            avg_psnr_bicubic += psnr_bicubic

            bicubic_img = bicubic_img - 127.5
            bicubic_img = bicubic_img / 127.5

            bicubic_img = Variable(torch.from_numpy(bicubic_img).float()).view(1, -1, bicubic_img.shape[0], bicubic_img.shape[1])

            model = model.cuda()
            bicubic_img = bicubic_img.cuda()

            start_time = time.time()
            predicted_img = model(bicubic_img)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time

            predicted_img = predicted_img.cpu()
            predicted_img = predicted_img.data[0].view(bicubic_img.shape[2], bicubic_img.shape[3], -1).numpy()
            predicted_img = predicted_img * 127.5 + 127.5
            predicted_img[predicted_img < 0] = 0
            predicted_img[predicted_img > 255.] = 255.
            predicted_img = predicted_img.astype(np.uint8)

            psnr_predicted = PSNR(label_img, predicted_img, scale)
            avg_psnr_predicted += psnr_predicted
            if epoch == 50:
                Path(f"../../REWIND-vdsr/test/{version}").mkdir(parents=True, exist_ok=True)
                Image.fromarray(predicted_img).save(f"../../REWIND-vdsr/test/{version}/{image_path.stem}.png")

        writer.add_scalar('Set5 PSNR VDSR', avg_psnr_predicted/count, global_step=epoch*8)
        writer.add_scalar('Set5 PSNR bicubic', avg_psnr_bicubic/count, global_step=epoch*8)
        writer.add_scalar('Set5 average time', avg_elapsed_time/count, global_step=epoch*8)
        writer.flush()

if __name__ == '__main__':
    main()