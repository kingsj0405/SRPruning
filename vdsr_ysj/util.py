# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms.functional as TVF

from torch.autograd import Variable
from pathlib import Path
from PIL import Image


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
    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])
    return ycbcr


def psnr(y_true, y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''
    y_true = _rgb2ycbcr(y_true)[:, :, 0]
    y_pred = _rgb2ycbcr(y_pred)[:, :, 0]
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)
    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 20 * np.log10(255. / rmse)


def psnr_set5(model, set5_dir, save_dir, writer, epoch, global_step):
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    count = 0.0
    scale = 4
    image_list = list(Path(set5_dir).glob('*.bmp'))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for image_path in image_list:
        count += 1
        # Make bicubic image from label image
        label_img = Image.open(image_path)
        test_img = TVF.resize(label_img,
                              (int(label_img.size[1] / 4),
                               int(label_img.size[0] / 4)))
        bicubic_img = TVF.resize(
            test_img, (label_img.size[1], label_img.size[0]))
        # Numpy
        label_img = np.asarray(label_img)
        bicubic_img = np.asarray(bicubic_img)
        # Get psnr of bicubic
        psnr_bicubic = psnr(label_img, bicubic_img, scale)
        avg_psnr_bicubic += psnr_bicubic
        # Normalize
        bicubic_img = bicubic_img - 127.5
        bicubic_img = bicubic_img / 127.5
        bicubic_img = Variable(
            torch.from_numpy(bicubic_img).float()).view(
            1, -1, bicubic_img.shape[0], bicubic_img.shape[1])
        bicubic_img = bicubic_img.cuda()
        # Super-Resolution
        predicted_img = model(bicubic_img)
        # Denomalize
        predicted_img = predicted_img.cpu()
        predicted_img = predicted_img.data[0].view(
            bicubic_img.shape[2], bicubic_img.shape[3], -1).numpy()
        predicted_img = predicted_img * 127.5 + 127.5
        predicted_img[predicted_img < 0] = 0
        predicted_img[predicted_img > 255.] = 255.
        predicted_img = predicted_img.astype(np.uint8)
        # Get psnr of generated
        psnr_predicted = psnr(label_img, predicted_img, scale)
        avg_psnr_predicted += psnr_predicted
        Image.fromarray(predicted_img).save(
            f"{save_dir}/epoch_{epoch}_{image_path.stem}.png")
    # writer to tensorboard
    writer.add_scalar(
        'Set5 PSNR VDSR',
        avg_psnr_predicted / count,
        global_step=global_step)
    writer.add_scalar(
        'Set5 PSNR bicubic',
        avg_psnr_bicubic / count,
        global_step=global_step)
    writer.flush()
