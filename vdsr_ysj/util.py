# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def DepthToSpace(input, upscale_factor):
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels // (upscale_factor ** 2)

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor

    input_view = input.contiguous().view(
        batch_size,
        out_channels,
        upscale_factor,
        upscale_factor,
        in_height,
        in_width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


def SpaceToDepth(input, downscale_factor):
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * downscale_factor ** 2

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size,
        in_channels,
        out_height,
        downscale_factor,
        out_width,
        downscale_factor)

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


def SpaceToDepth3D(input, downscale_factor):
    outs = []
    for i in range(input.size(2)):
        outs.append(SpaceToDepth(input[:, :, i], downscale_factor))
    return torch.stack(outs, 2)

# MATLAB imresize function
# Key difference from other resize funtions is antialiasing when downsampling
# This function only for downsampling


def DownSample2DMatlab(
        tensor,
        scale,
        method='cubic',
        antialiasing=True,
        cuda=True):
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

        f = np.multiply(
            (1.5 * absx3 - 2.5 * absx2 + 1),
            np.less_equal(
                absx,
                1)) + np.multiply(
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2),
            np.logical_and(
                np.less(
                    1,
                    absx),
                np.less_equal(
                    absx,
                    2)))

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
    def resizeAlongDim(
            tensor,
            scale_v,
            scale_h,
            kernel_width,
            weights):  # , indices):
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


def Blur2DMatlab(tensor, scale, method='cubic', antialiasing=True, cuda=True):
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

        f = np.multiply(
            (1.5 * absx3 - 2.5 * absx2 + 1),
            np.less_equal(
                absx,
                1)) + np.multiply(
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2),
            np.logical_and(
                np.less(
                    1,
                    absx),
                np.less_equal(
                    absx,
                    2)))

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
    def resizeAlongDim(
            tensor,
            scale_v,
            scale_h,
            kernel_width,
            weights):  # , indices):
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

        # # Reflect padding
        # i_scale_v = int(1/scale_v)
        # i_scale_h = int(1/scale_h)
        # pad_top = int((kernel_width_v - i_scale_v) / 2)
        # pad_bottom = int((kernel_width_h - i_scale_h) / 2)
        # pad_array = ([pad_top+1, pad_bottom+1, pad_top+1, pad_bottom+1])
        # kernel_width_v = int(kernel_width_v)
        # kernel_width_h = int(kernel_width_h)

        # Reflect padding
        i_scale_v = int(1 / scale_v)
        i_scale_h = int(1 / scale_h)
        pad_top = int((kernel_width_v) // 2)
        if i_scale_v == 1:
            pad_top = 0
        pad_bottom = int((kernel_width_h) // 2)
        if i_scale_h == 1:
            pad_bottom = 0
        pad_array = ([pad_bottom - 1, pad_bottom, pad_top - 1, pad_top])
        kernel_width_v = int(kernel_width_v)
        kernel_width_h = int(kernel_width_h)

        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(
            1, 1, (kernel_width_v, kernel_width_h), (1, 1), bias=False)
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
            # outs.append(FT(padded[:,:,1:,1:]))
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

    tensor = resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights)

    return tensor


def DownSample2DGaussian(x, scale=4, gpu_no=0, cuda=True):
    '''
    Downsampling images with Gaussian smoothing
    x: [B,C,T,H,W], 5D
    scale: 2,3,4 are available, e.g. 4 means 1/4 downsampling
    '''

    def _gkern(kernlen=13, nsig=1.6):
        import scipy.ndimage.filters as fi
        # create nxn zeros
        inp = np.zeros((kernlen, kernlen))
        # set element at the middle to one, a dirac delta
        inp[kernlen // 2, kernlen // 2] = 1
        # gaussian-smooth the dirac, resulting in a gaussian filter mask
        return fi.gaussian_filter(inp, nsig)

    if scale == 2:
        h = _gkern(13, 0.8)
    elif scale == 3:
        h = _gkern(13, 1.2)
    elif scale == 4:
        h = _gkern()
    else:
        raise ValueError

    h = h[np.newaxis, np.newaxis, :, :].astype(np.float32)

    xs = x.size()
    x = x.contiguous().view(xs[0], xs[1] * xs[2], xs[3], xs[4])  # B,C*T,H,W

    # Reflect padding
    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = ([pad_top, pad_bottom, pad_left, pad_right])

    #
    W = torch.from_numpy(h.astype('float32'))
    FT = nn.Conv2d(1, 1, (filter_height, filter_width),
                   (scale, scale), bias=False)
    FT.weight.data = W
    if cuda:
        FT.cuda(gpu_no)
    FT.requires_grad = False

    # actually, we want 'symmetric' padding, not 'reflect'
    outs = []
    for c in range(xs[1] * xs[2]):
        padded = nn.functional.pad(x[:, c:c + 1, :, :], pad_array, 'reflect')
        outs.append(FT(padded))
    out = torch.cat(outs, 1)

    os = out.size()
    out = out.view(xs[0], xs[1], xs[2], os[2], os[3])  # B,C,T,H,W

    return out


def UpSample2DMatlab(
        tensor,
        scale,
        method='cubic',
        antialiasing=True,
        cuda=True):
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

        f = np.multiply(
            (1.5 * absx3 - 2.5 * absx2 + 1),
            np.less_equal(
                absx,
                1)) + np.multiply(
            (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2),
            np.logical_and(
                np.less(
                    1,
                    absx),
                np.less_equal(
                    absx,
                    2)))

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


def BilinearStn(input, flow, name='STN'):
    '''
    input: [B, C, H, W]
    flow: [B, 2, H, W], 2 for dx dy, relative displacement in px
    output: [B, C, H, W]
    WARNING: only applicable when r=4 currently
    '''

    # flow = tf.zeros_like(flow)
    fs = flow.size()

    # pad to accurate upsample
    xs = input.size()

    # base grid
    # x_t, y_t = tf.meshgrid(tf.linspace(0., tf.cast(xs[2]-1, tf.float32), fs[2]),
    # tf.linspace(0., tf.cast(xs[1]-1, tf.float32), fs[1]))

    # Due to when transform
    base_x = torch.arange(fs[3]).view(1, 1, fs[3])
    base_x = base_x.expand(fs[0], fs[2], fs[3])

    base_y = torch.arange(fs[2]).view(1, fs[2], 1)
    base_y = base_y.expand(fs[0], fs[2], fs[3])

    base_grid = torch.stack([base_x, base_y], 1)
    base_grid = Variable(base_grid, requires_grad=False).cuda()

    new_grid = base_grid + flow
    x = new_grid[:, 0, :, :].contiguous().view(-1)
    y = new_grid[:, 1, :, :].contiguous().view(-1)

    # base index
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, xs[3] - 1)
    x1 = torch.clamp(x1, 0, xs[3] - 1)
    y0 = torch.clamp(y0, 0, xs[2] - 1)
    y1 = torch.clamp(y1, 0, xs[2] - 1)

    def _repeat(x, n_repeats):
        rep = torch.ones([1, n_repeats])
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)

    base = _repeat(torch.arange(xs[0]) * xs[2] * xs[3], fs[2] * fs[3]).cuda()
    base_y0 = base + y0 * xs[3]
    base_y1 = base + y1 * xs[3]
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # sampling
    # use indices to lookup pixels in the flat image and restore
    im_flat = input.permute(0, 2, 3, 1).contiguous().view(-1, xs[1])
    Ia = torch.gather(im_flat,
                      0,
                      idx_a.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
    Ib = torch.gather(im_flat,
                      0,
                      idx_b.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
    Ic = torch.gather(im_flat,
                      0,
                      idx_c.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
    Id = torch.gather(im_flat,
                      0,
                      idx_d.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)

    # and finally calculate bilinearly interpolated values
    wa = ((x1 - x) * (y1 - y)).view(-1, 1).expand(-1,
                                                  xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
    wb = ((x1 - x) * (y - y0)).view(-1, 1).expand(-1,
                                                  xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
    wc = ((x - x0) * (y1 - y)).view(-1, 1).expand(-1,
                                                  xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
    wd = ((x - x0) * (y - y0)).view(-1, 1).expand(-1,
                                                  xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)

    return [Ia, Ib, Ic, Id], [wa, wb, wc, wd]


def BilinearStnWithW(input, flow, weight, init=False, name='STN'):
    '''
    input: [B, C, H, W]
    flow: [B, 2, H, W], 2 for dx dy, relative displacement in px
    weight: [B, 4, H, W]
    output: [B, C, H, W]
    WARNING: only applicable when r=4 currently
    '''

    # flow = tf.zeros_like(flow)
    fs = flow.size()

    # pad to accurate upsample
    xs = input.size()

    # base grid
    # x_t, y_t = tf.meshgrid(tf.linspace(0., tf.cast(xs[2]-1, tf.float32), fs[2]),
    # tf.linspace(0., tf.cast(xs[1]-1, tf.float32), fs[1]))

    # Due to when transform
    base_x = torch.arange(fs[3]).view(1, 1, fs[3])
    base_x = base_x.expand(fs[0], fs[2], fs[3])

    base_y = torch.arange(fs[2]).view(1, fs[2], 1)
    base_y = base_y.expand(fs[0], fs[2], fs[3])

    base_grid = torch.stack([base_x, base_y], 1)
    base_grid = Variable(base_grid, requires_grad=False).cuda()

    new_grid = base_grid + flow
    x = new_grid[:, 0, :, :].contiguous().view(-1)
    y = new_grid[:, 1, :, :].contiguous().view(-1)

    # base index
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, xs[3] - 1)
    x1 = torch.clamp(x1, 0, xs[3] - 1)
    y0 = torch.clamp(y0, 0, xs[2] - 1)
    y1 = torch.clamp(y1, 0, xs[2] - 1)

    def _repeat(x, n_repeats):
        rep = torch.ones([1, n_repeats])
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)

    base = _repeat(torch.arange(xs[0]) * xs[2] * xs[3], fs[2] * fs[3]).cuda()
    base_y0 = base + y0 * xs[3]
    base_y1 = base + y1 * xs[3]
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # sampling
    # use indices to lookup pixels in the flat image and restore
    im_flat = input.permute(0, 2, 3, 1).contiguous().view(-1, xs[1])
    Ia = torch.gather(im_flat,
                      0,
                      idx_a.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
    Ib = torch.gather(im_flat,
                      0,
                      idx_b.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
    Ic = torch.gather(im_flat,
                      0,
                      idx_c.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)
    Id = torch.gather(im_flat,
                      0,
                      idx_d.view(-1,
                                 1).expand(-1,
                                           xs[1]).long()).view(xs[0],
                                                               xs[2],
                                                               xs[3],
                                                               xs[1]).permute(0,
                                                                              3,
                                                                              1,
                                                                              2)

    if init:
        # and finally calculate bilinearly interpolated values
        wa = ((x1 - x) * (y1 - y)).view(-1, 1).expand(-1,
                                                      xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
        wb = ((x1 - x) * (y - y0)).view(-1, 1).expand(-1,
                                                      xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
        wc = ((x - x0) * (y1 - y)).view(-1, 1).expand(-1,
                                                      xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
        wd = ((x - x0) * (y - y0)).view(-1, 1).expand(-1,
                                                      xs[1]).view(xs[0], xs[2], xs[3], xs[1]).permute(0, 3, 1, 2)
        return Ia * wa + Ib * wb + Ic * wc + Id * wd

    else:
        return Ia * weight[:, 0:1] + Ib * weight[:, 1:2] + \
            Ic * weight[:, 2:3] + Id * weight[:, 3:4]
