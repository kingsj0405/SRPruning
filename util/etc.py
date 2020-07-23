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

from model import DownSample2DMatlab, UpSample2DMatlab


def count_num_of_parameters(model):
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    return number_parameters


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
        with torch.no_grad():
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
            psnr_predicted = psnr(_rgb2ycbcr(label_img)[:, :, 0],
                                  _rgb2ycbcr(predicted_img)[:, :, 0],
                                  scale)
            avg_psnr_predicted += psnr_predicted
            # Save image
            if save:
                Image.fromarray(predicted_img).save(
                    f"{save_dir}/Set5_{image_path.stem}.png")
    return (avg_psnr_predicted / count)


def calculate_psnr(testsets_H, testset_O):
    count = 0
    avg_psnr_value = 0.0
    h_list = sorted(list(Path(testsets_H).glob('*.png')))
    o_list = sorted(list(Path(testset_O).glob('*.png')))
    for h_path, o_path in tqdm(zip(h_list, o_list)):
        with torch.no_grad():
            # Load label image
            h_img = _load_img_array(h_path)
            h_img = (h_img * 255).astype(np.uint8)
            # Load output image
            o_img = _load_img_array(o_path)
            o_img = (o_img * 255).astype(np.uint8)
            # Get psnr of bicubic, predicted
            psnr_value = psnr(h_img,#_rgb2ycbcr(h_img)[:, :, 0],
                              o_img,#_rgb2ycbcr(o_img)[:, :, 0],
                              4)
            count += 1
            avg_psnr_value += psnr_value
    return (avg_psnr_value / count)


def test(model, L_folder, out_folder, logger, save, ensemble=False):
    # --------------------------------
    # read image
    # --------------------------------
    util.mkdir(out_folder)

    # record PSNR, runtime
    test_results = OrderedDict()
    test_results['runtime'] = []

    logger.info(L_folder)
    logger.info(out_folder)
    idx = 0

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_folder):

        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info('{:->4d}--> {:>10s}'.format(idx, img_name+ext))

        img_L = util.imread_uint(img, n_channels=3)
        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)

        if ensemble:
            start.record()
            # Original
            img_E = model(img_L)
            # Rotation 90
            img_E += model(img_L.rot90(1, (-1, -2))).rot90(3, (-1, -2))
            # Rotation 180
            img_E += model(img_L.rot90(2, (-1, -2))).rot90(2, (-1, -2))
            # Rotation 270
            img_E += model(img_L.rot90(3, (-1, -2))).rot90(1, (-1, -2))
            # H flip
            img_L_hflip = img_L.flip(-1)
            img_E += model(img_L_hflip).flip(-1)
            # H flip + rot90
            img_E += model(img_L_hflip.rot90(1, (-1, -2))).rot90(3, (-1, -2)).flip(-1)
            # H flip + rot180
            img_E += model(img_L_hflip.rot90(2, (-1, -2))).rot90(2, (-1, -2)).flip(-1)
            # H flip + rot270
            img_E += model(img_L_hflip.rot90(3, (-1, -2))).rot90(1, (-1, -2)).flip(-1)
            img_E /= 8
            end.record()
        else:
            start.record()
            img_E = model(img_L)
            end.record()
        torch.cuda.synchronize()
        test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        if save:
            new_name = '{:3d}'.format(int(img_name.split('x')[0]))
            path = os.path.join(out_folder, new_name+ext)
            print(f'[INFO] Save {idx:4d} to {path:10s}')
            util.imsave(img_E, path)
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
    print(f'[INFO] Average runtime of ({L_folder}) is : {ave_runtime:.6f} seconds')