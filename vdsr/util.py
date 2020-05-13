# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.autograd import Variable
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image

from layer import DownSample2DMatlab, UpSample2DMatlab


def _load_img_array(path, color_mode='RGB',
                    channel_mean=None, modcrop=[0, 0, 0, 0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    # Load image
    from PIL import Image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')
    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:, :, 0:1]
    # Normalize To 0-1
    x *= 1.0 / 255.0
    if channel_mean:
        x[:, :, 0] -= channel_mean[0]
        x[:, :, 1] -= channel_mean[1]
        x[:, :, 2] -= channel_mean[2]
    if modcrop[0] * modcrop[1] * modcrop[2] * modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]
    return x


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
    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)
    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    return 20 * np.log10(255. / rmse)


def psnr_set5(model, set5_dir, save_dir, save=True):
    avg_psnr_predicted = 0.0
    avg_psnr_bicubic = 0.0
    count = 0.0
    scale = 4
    image_list = list(Path(set5_dir).glob('*.bmp'))
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for image_path in image_list:
        count += 1
        # Load label image
        label_img = _load_img_array(image_path)
        label_img = torch.from_numpy(label_img)
        label_img = label_img.unsqueeze(0)
        label_img = Variable(label_img).cuda()
        # Make bicubic image and net-through image
        # NOTE: permute for tensor
        label_img = label_img.permute(0, 3, 1, 2)
        bicubic_img = DownSample2DMatlab(label_img, 1 / scale)
        bicubic_img = bicubic_img.clamp(0, 1)
        bicubic_img = torch.round(bicubic_img * 255) / 255
        bicubic_img = UpSample2DMatlab(bicubic_img, scale)
        bicubic_img = bicubic_img.clamp(0, 1)
        bicubic_img = torch.round(bicubic_img * 255) / 255
        predicted_img = model(bicubic_img)
        label_img = label_img.permute(0, 2, 3, 1)
        bicubic_img = bicubic_img.permute(0, 2, 3, 1)
        predicted_img = predicted_img.permute(0, 2, 3, 1)
        # Denormalize
        label_img = label_img[0].cpu().data.numpy()
        label_img = label_img.clip(0, 1)
        label_img = (label_img * 255).astype(np.uint8)
        bicubic_img = bicubic_img[0].cpu().data.numpy()
        bicubic_img = bicubic_img.clip(0, 1)
        bicubic_img = (bicubic_img * 255).astype(np.uint8)
        predicted_img = predicted_img[0].cpu().data.numpy()
        predicted_img = predicted_img.clip(0, 1)
        predicted_img = (predicted_img * 255).astype(np.uint8)
        # Get psnr of bicubic, predicted
        psnr_bicubic = psnr(_rgb2ycbcr(label_img)[:, :, 0],
                            _rgb2ycbcr(bicubic_img)[:, :, 0],
                            scale)
        psnr_predicted = psnr(_rgb2ycbcr(label_img)[:, :, 0],
                              _rgb2ycbcr(predicted_img)[:, :, 0],
                              scale)
        avg_psnr_bicubic += psnr_bicubic
        avg_psnr_predicted += psnr_predicted
        # Save image
        if save:
            Image.fromarray(predicted_img).save(
                f"{save_dir}/Set5_{image_path.stem}.png")
    return (avg_psnr_predicted / count), (avg_psnr_bicubic / count)
