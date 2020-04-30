import numpy as np
import torch
import torch.nn as nn

# MATLAB imresize function
# Key difference from other resize funtions is antialiasing when downsampling
# This function only for downsampling


def DownSample2DMatlab(tensor, scale, method='cubic',
                       antialiasing=True, cuda=True):
    '''
    This gives same result as MATLAB downsampling
    tensor: 4D tensor [Batch, Channel, Height, Width],
            height and width must be divided by the denominator of scale factor
    scale: Even integer denominator scale factor only (e.g. 1/2,1/4,1/8,...)
           Or list [1/2, 1/4] : [V scale, H scale]
    method: 'cubic' as default, currently cubic supported
    antialiasing: True as default
    '''
    # For cubic interpolation,
    # Cubic Convolution Interpolation for Digital Image Processing, ASSP, 1981
    def cubic(x):
        absx = np.abs(x)
        absx2 = np.multiply(absx, absx)
        absx3 = np.multiply(absx2, absx)

        f = np.multiply((1.5 * absx3 - 2.5 * absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2),
                        np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f
    # Generate resize kernel (resize weight computation)

    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
            kernel_width = kernel_width / scale

        x = np.ones((1, 1))

        u = x / scale + 0.5 * (1 - 1 / scale)

        left = np.floor(u - kernel_width / 2)

        P = int(np.ceil(kernel_width) + 2)

        indices = np.tile(left, (1, P)) + np.expand_dims(np.arange(0, P), 0)

        if scale < 1 and antialiasing:
            weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
        else:
            weights = kernel(np.tile(u, (1, P)) - indices)

        weights = weights / np.expand_dims(np.sum(weights, 1), 1)

        save = np.where(np.any(weights, 0))
        weights = weights[:, save[0]]

        return weights

    # Resize along a specified dimension
    def resizeAlongDim(tensor, scale_v, scale_h,
                       kernel_width, weights):  # , indices):
        if scale_v < 1 and antialiasing:
            kernel_width_v = kernel_width / scale_v
        else:
            kernel_width_v = kernel_width
        if scale_h < 1 and antialiasing:
            kernel_width_h = kernel_width / scale_h
        else:
            kernel_width_h = kernel_width

        # Generate filter
        f_height = np.transpose(weights[0][0:1, :])
        f_width = weights[1][0:1, :]
        f = np.dot(f_height, f_width)
        f = f[np.newaxis, np.newaxis, :, :]
        F = torch.from_numpy(f.astype('float32'))

        # Reflect padding
        i_scale_v = int(1 / scale_v)
        i_scale_h = int(1 / scale_h)
        pad_top = int((kernel_width_v - i_scale_v) / 2)
        if i_scale_v == 1:
            pad_top = 0
        pad_bottom = int((kernel_width_h - i_scale_h) / 2)
        if i_scale_h == 1:
            pad_bottom = 0
        pad_array = ([pad_bottom, pad_bottom, pad_top, pad_top])
        kernel_width_v = int(kernel_width_v)
        kernel_width_h = int(kernel_width_h)

        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(1, 1, (kernel_width_v, kernel_width_h),
                       (i_scale_v, i_scale_h), bias=False)
        FT.weight.data = F
        if cuda:
            FT.cuda()
        FT.requires_grad = False

        # actually, we want 'symmetric' padding, not 'reflect'
        outs = []
        for c in range(num_channel):
            padded = nn.functional.pad(
                tensor[:, c:c + 1, :, :], pad_array, 'reflect')
            outs.append(FT(padded))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4

    if isinstance(scale, list):
        scale_v = float(scale[0])
        scale_h = float(scale[1])

        weights = []
        for i in range(2):
            W = contributions(
                float(
                    scale[i]),
                kernel,
                kernel_width,
                antialiasing)
            weights.append(W)
    else:
        scale = float(scale)

        scale_v = scale
        scale_h = scale

        weights = []
        for i in range(2):
            W = contributions(scale, kernel, kernel_width, antialiasing)
            weights.append(W)

    # np.save('bic_x4_downsample_h.npy', weights[0])

    tensor = resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights)

    return tensor


def UpSample2DMatlab(tensor, scale, method='cubic',
                     antialiasing=True, cuda=True):
    '''
    This gives same result as MATLAB upsampling
    tensor: 4D tensor [Batch, Channel, Height, Width],
            height and width must be divided by the denominator of scale factor
    scale: integer scale factor only (e.g. 2,3,4,...)
    method: 'cubic' as default, currently cubic supported
    antialiasing: True as default
    '''

    # For cubic interpolation,
    # Cubic Convolution Interpolation for Digital Image Processing, ASSP, 1981
    def cubic(x):
        absx = np.abs(x)
        absx2 = np.multiply(absx, absx)
        absx3 = np.multiply(absx2, absx)

        f = np.multiply((1.5 * absx3 - 2.5 * absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2),
                        np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
            kernel_width = kernel_width / scale

        x = np.arange(1, kernel_width + 1)
        x = np.expand_dims(x, 1)

        u = x / scale + 0.5 * (1 - 1 / scale)

        left = np.floor(u - kernel_width / 2)

        P = int(np.ceil(kernel_width) + 2)

        indices = np.tile(left, (1, P)) + np.expand_dims(np.arange(0, P), 0)

        if scale < 1 and antialiasing:
            weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
        else:
            weights = kernel(np.tile(u, (1, P)) - indices)

        weights = weights / np.expand_dims(np.sum(weights, 1), 1)

        save = np.where(np.any(weights, 0))
        weights = weights[:, save[0]]

        return weights

    # Resize along a specified dimension
    def resizeAlongDim(tensor, scale, kernel_width, weights):  # , indices):
        # if scale < 1 and antialiasing:
        #   kernel_width = kernel_width / scale

        scale = int(scale)
        # Generate filter
        new_kernel_width = kernel_width
        if kernel_width % 2 == 0:
            new_kernel_width += 1
        F = np.zeros((scale * scale, 1, new_kernel_width, new_kernel_width))

        b = scale // 2
        for y in range(scale):
            for x in range(scale):
                f_height = np.transpose(weights[0][y:y + 1, :])
                f_width = weights[1][x:x + 1, :]
                f = np.dot(f_height, f_width)

                sy = 0
                if y >= b:
                    sy += 1
                sx = 0
                if x >= b:
                    sx += 1

                F[y * scale + x, 0, sy:sy + kernel_width, sx:sx + kernel_width] = f

        # import scipy.io
        # a={}
        # a['f'] = F
        # scipy.io.savemat('upsample_x2_f.mat', a)

        F = torch.from_numpy(F.astype('float32'))

        # Reflect padding
        pad_array = ([2, 2, 2, 2])

        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(
            1, 1, (new_kernel_width, new_kernel_width), (1, 1), bias=False)
        FT.weight.data = F
        if cuda:
            FT.cuda()
        FT.requires_grad = False

        outs = []
        for i in range(num_channel):
            padded = nn.functional.pad(
                tensor[:, i:i + 1, :, :], pad_array, 'reflect')
            outs.append(FT(padded))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4
    scale = float(scale)
    weights = []

    for i in range(2):
        W = contributions(scale, kernel, kernel_width, antialiasing)
        weights.append(W)

    tensor = resizeAlongDim(tensor, scale, kernel_width, weights)
    tensor = nn.PixelShuffle(int(scale))(tensor)

    return tensor
