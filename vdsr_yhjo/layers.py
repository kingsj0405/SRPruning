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
        batch_size, out_channels, upscale_factor, upscale_factor, in_height, in_width)

    shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


def SpaceToDepth(input, downscale_factor):
    batch_size, in_channels, in_height, in_width = input.size()
    out_channels = in_channels * downscale_factor ** 2

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = input.contiguous().view(
        batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor)

    shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()
    return shuffle_out.view(batch_size, out_channels, out_height, out_width)

def SpaceToDepth3D(input, downscale_factor):
    outs = []
    for i in range(input.size(2)):
        outs.append(SpaceToDepth(input[:,:,i], downscale_factor))
    return torch.stack(outs, 2)


# MATLAB imresize function
# Key difference from other resize funtions is antialiasing when downsampling
# This function only for downsampling
def DownSample2DMatlab(tensor, scale, method='cubic', antialiasing=True, cuda=True):
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

        f = np.multiply((1.5*absx3 - 2.5*absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5*absx3 + 2.5*absx2 - 4*absx + 2), \
            np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
          kernel_width = kernel_width / scale

        x = np.ones((1, 1))

        u = x/scale + 0.5 * (1 - 1/scale)

        left = np.floor(u - kernel_width/2)

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
    def resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights):#, indices):
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
        i_scale_v = int(1/scale_v)
        i_scale_h = int(1/scale_h)
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
        FT = nn.Conv2d(1, 1, (kernel_width_v, kernel_width_h), (i_scale_v, i_scale_h), bias=False)
        FT.weight.data = F
        if cuda:
           FT.cuda()
        FT.requires_grad = False

        # actually, we want 'symmetric' padding, not 'reflect'
        outs = []
        for c in range(num_channel):
            padded = nn.functional.pad(tensor[:,c:c+1,:,:], pad_array, 'reflect')
            outs.append(FT(padded))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4

    if type(scale) is list:
        scale_v = float(scale[0])
        scale_h = float(scale[1])

        weights = []
        for i in range(2):
            W = contributions(float(scale[i]), kernel, kernel_width, antialiasing)
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

        f = np.multiply((1.5*absx3 - 2.5*absx2 + 1), np.less_equal(absx, 1)) + \
            np.multiply((-0.5*absx3 + 2.5*absx2 - 4*absx + 2), \
            np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

        return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
        if scale < 1 and antialiasing:
          kernel_width = kernel_width / scale

        x = np.ones((1, 1))

        u = x/scale + 0.5 * (1 - 1/scale)

        left = np.floor(u - kernel_width/2)

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
    def resizeAlongDim(tensor, scale_v, scale_h, kernel_width, weights):#, indices):
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
        i_scale_v = int(1/scale_v)
        i_scale_h = int(1/scale_h)
        pad_top = int((kernel_width_v) // 2)
        if i_scale_v == 1:
            pad_top = 0
        pad_bottom = int((kernel_width_h) // 2)
        if i_scale_h == 1:
            pad_bottom = 0
        pad_array = ([pad_bottom-1, pad_bottom, pad_top-1, pad_top])
        kernel_width_v = int(kernel_width_v)
        kernel_width_h = int(kernel_width_h)


        #
        tensor_shape = tensor.size()
        num_channel = tensor_shape[1]
        FT = nn.Conv2d(1, 1, (kernel_width_v, kernel_width_h), (1, 1), bias=False)
        FT.weight.data = F
        if cuda:
           FT.cuda()
        FT.requires_grad = False

        # actually, we want 'symmetric' padding, not 'reflect'
        outs = []
        for c in range(num_channel):
            padded = nn.functional.pad(tensor[:,c:c+1,:,:], pad_array, 'reflect')
            outs.append(FT(padded))
            # outs.append(FT(padded[:,:,1:,1:]))
        out = torch.cat(outs, 1)

        return out

    if method == 'cubic':
        kernel = cubic

    kernel_width = 4

    if type(scale) is list:
        scale_v = float(scale[0])
        scale_h = float(scale[1])

        weights = []
        for i in range(2):
            W = contributions(float(scale[i]), kernel, kernel_width, antialiasing)
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
        inp[kernlen//2, kernlen//2] = 1
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

    h = h[np.newaxis,np.newaxis,:,:].astype(np.float32)

    xs = x.size()
    x = x.contiguous().view(xs[0], xs[1]*xs[2], xs[3], xs[4])  # B,C*T,H,W

    ## Reflect padding
    filter_height, filter_width = 13, 13
    pad_height = filter_height - 1
    pad_width = filter_width - 1

    # When pad_height (pad_width) is odd, we pad more to bottom (right),
    # following the same convention as conv2d().
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    pad_array = ([pad_top, pad_bottom,pad_left, pad_right])

    #
    W = torch.from_numpy(h.astype('float32'))
    FT = nn.Conv2d(1, 1, (filter_height, filter_width), (scale, scale), bias=False)
    FT.weight.data = W
    if cuda:
        FT.cuda(gpu_no)
    FT.requires_grad = False

    # actually, we want 'symmetric' padding, not 'reflect'
    outs = []
    for c in range(xs[1]*xs[2]):
        padded = nn.functional.pad(x[:,c:c+1,:,:], pad_array, 'reflect')
        outs.append(FT(padded))
    out = torch.cat(outs, 1)

    os = out.size()
    out = out.view(xs[0], xs[1], xs[2], os[2], os[3])  # B,C,T,H,W

    return out



def UpSample2DMatlab(tensor, scale, method='cubic', antialiasing=True, cuda=True):
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

      f = np.multiply((1.5*absx3 - 2.5*absx2 + 1), np.less_equal(absx, 1)) + \
          np.multiply((-0.5*absx3 + 2.5*absx2 - 4*absx + 2), \
                      np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))

      return f

    # Generate resize kernel (resize weight computation)
    def contributions(scale, kernel, kernel_width, antialiasing):
      if scale < 1 and antialiasing:
          kernel_width = kernel_width / scale

      x = np.arange(1, kernel_width+1)
      x = np.expand_dims(x, 1)

      u = x/scale + 0.5 * (1 - 1/scale)

      left = np.floor(u - kernel_width/2)

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
    def resizeAlongDim(tensor, scale, kernel_width, weights):#, indices):
        # if scale < 1 and antialiasing:
        #   kernel_width = kernel_width / scale

        scale = int(scale)
        # Generate filter
        new_kernel_width = kernel_width
        if kernel_width % 2 == 0:
           new_kernel_width += 1
        F = np.zeros((scale*scale, 1, new_kernel_width, new_kernel_width))

        b = scale // 2
        for y in range(scale):
            for x in range(scale):
                f_height = np.transpose(weights[0][y:y+1, :])
                f_width = weights[1][x:x+1, :]
                f = np.dot(f_height, f_width)

                sy = 0
                if y >= b:
                    sy += 1
                sx = 0
                if x >= b:
                    sx += 1

                F[y*scale+x, 0, sy:sy+kernel_width, sx:sx+kernel_width] = f

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
        FT = nn.Conv2d(1, 1, (new_kernel_width, new_kernel_width), (1, 1), bias=False)
        FT.weight.data = F
        if cuda:
           FT.cuda()
        FT.requires_grad = False

        outs = []
        for i in range(num_channel):
            padded = nn.functional.pad(tensor[:,i:i+1,:,:], pad_array, 'reflect')
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


#
# # 3D Kernel for X, Y, and T-axis
# def DownSample2DMatlab_3DKernel(tensor, scale, kernel_t=3, out_t=3, method='cubic', antialiasing=True):
#     '''
#     tensor: 4D tensor [Batch, Height, Width, Channel],
#             height and width must be divided by the denominator of scale factor
#     scale: Even integer denominator scale factor only (e.g. 1/2,1/4,1/8,...)
#     method: 'cubic' as default, currently cubic supported
#     antialiasing: True as default
#     '''
#
#     # For cubic interpolation,
#     # Cubic Convolution Interpolation for Digital Image Processing, ASSP, 1981
#     def cubic(x):
#         absx = np.abs(x)
#         absx2 = np.multiply(absx, absx)
#         absx3 = np.multiply(absx2, absx)
#
#         f = np.multiply((1.5*absx3 - 2.5*absx2 + 1), np.less_equal(absx, 1)) + \
#             np.multiply((-0.5*absx3 + 2.5*absx2 - 4*absx + 2), \
#                         np.logical_and(np.less(1, absx), np.less_equal(absx, 2)))
#
#         return f
#
# #    # Generate resize kernel (resize weight computation)
# #    def contributions(in_length, out_length, scale, kernel, kernel_width, antialiasing):
# #        if scale < 1 and antialiasing:
# #            kernel_width = kernel_width / scale;
# #
# #        x = np.arange(1, out_length+1)
# #        x = np.expand_dims(x, 1)
# #
# #        u = x/scale + 0.5 * (1 - 1/scale)
# #
# #        left = np.floor(u - kernel_width/2)
# #
# #        P = int(np.ceil(kernel_width) + 2)
# #
# #        indices = np.tile(left, (1, P)) + np.tile(np.expand_dims(np.arange(0, P), 0), (out_length, 1))
#
# #        if scale < 1 and antialiasing:
# #            weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
# #        else:
# #            weights = kernel(np.tile(u, (1, P)) - indices)
# #
# #        weights = weights / np.expand_dims(np.sum(weights, 1), 1)
# #
# #        aux = np.concatenate((np.arange(1, in_length+1), np.arange(in_length, 0, -1)))
# #        indices = aux[np.mod(indices-1, aux.shape[0]).astype(np.int)]
# #
# #        save = np.where(np.any(weights, 0))
# #        weights = weights[:, save[0]]
# #        indices = indices[:, save[0]]
# #
# #        return weights, indices
#
#     # Generate resize kernel (resize weight computation)
#     def contributions(scale, kernel, kernel_width, antialiasing):
#         if scale < 1 and antialiasing:
#             kernel_width = kernel_width / scale
#
#         x = np.ones((1, 1))
#
#         u = x/scale + 0.5 * (1 - 1/scale)
#
#         left = np.floor(u - kernel_width/2)
#
#         P = int(np.ceil(kernel_width) + 2)
#
#         indices = np.tile(left, (1, P)) + np.expand_dims(np.arange(0, P), 0)
#
#         if scale < 1 and antialiasing:
#             weights = scale * kernel(scale * (np.tile(u, (1, P)) - indices))
#         else:
#             weights = kernel(np.tile(u, (1, P)) - indices)
#
#         weights = weights / np.expand_dims(np.sum(weights, 1), 1)
#
#         save = np.where(np.any(weights, 0))
#         weights = weights[:, save[0]]
#
#         return weights
#
#     # Resize along a specified dimension
#     def resizeAlongDim(tensor, scale, kernel_width, weights, kernel_t, out_t):#, indices):
#         if scale < 1 and antialiasing:
#             kernel_width = kernel_width / scale
#
#         # Generate filter
#         # Spatial
#         f_height = np.transpose(weights[0][0:1, :])
#         f_width = weights[1][0:1, :]
#         f = np.dot(f_height, f_width)
#         # Temporal
#         mu = 0.0
#         sig = 1.0
#         numer = np.zeros((1, kernel_t))
#         denom = 0.0
#         for i in range(kernel_t):
#             t = np.exp(-np.power(((i-kernel_t/2) - mu)/sig, 2.) / 2.0)
#             numer[0, i] = t
#             denom += t
#         numer /= denom
# #        numer = np.zeros((1, kernel_t), dtype=np.float32)
# #        numer[0,:] = [0.274068619061197, 0.451862761877606, 0.274068619061197]
#
#         # Total
#         f = np.matmul(f[:,:,np.newaxis], np.tile(numer, (f.shape[0],1,1)))
#         f = np.transpose(f, [2, 1, 0])
#         f = f[:, :, :, np.newaxis, np.newaxis]
#         F = tf.Variable(f.astype('float32'), trainable=False)
#
#         # Reflect padding
#         tensor_shape = tensor.get_shape().as_list()
#         num_channel = tensor_shape[4]
#         num_temporal = tensor_shape[1]
#         i_scale = int(1/scale)
#
#         pad_top = int((kernel_width - i_scale) / 2)
#         pad_bottom = int((kernel_width - i_scale) / 2)
#         pad_before = int(((kernel_t + (out_t-1)) - num_temporal) / 2)
#         pad_after = ((kernel_t + (out_t-1)) - num_temporal) - pad_before
# #        pad_before = 1
# #        pad_after = 1
#         pad_array = [[0,0], [pad_before, pad_after], [pad_top, pad_bottom], [pad_top, pad_bottom], [0,0]]
#
#         #
#         outs = []
#         tensor_padded = tf.pad(tensor, pad_array, mode='SYMMETRIC')
#         for i in range(num_channel):
#             t = tf.nn.conv3d(tensor_padded[:,:,:,:,i:i+1], F, [1,1,i_scale,i_scale,1], 'VALID')
# #            outs.append(t[:,:,:,:,0])
#             outs.append(t)
#         out = tf.concat(outs, 4)
# #        out = tf.stack(outs, 4)
#
#         return out
#
#
#     if method == 'cubic':
#         kernel = cubic
#
#     kernel_width = 4
#
# #    tensor_shape = tensor.get_shape().as_list()
#
#     scale = float(scale)
# #    output_size = np.round(np.asarray(tensor_shape[1:3]) * scale).astype(np.int)
#
#     weights = []
# #    indices = []
#     for i in range(2):
#         W = contributions(scale, kernel, kernel_width, antialiasing)
# #        W, I = contributions(tensor_shape[i+1], output_size[i], scale, kernel, kernel_width, antialiasing)
#         weights.append(W)
# #        indices.append(I)
#
#     tensor = resizeAlongDim(tensor, scale, kernel_width, weights, kernel_t, out_t)
# #    tensor = resizeAlongDim(tensor, int(1/scale), i, weights, indices)
#
#     return tensor
#
#


#
#
#
# # Using pywt
# def DWT2D(x, wname='haar'):
#     import pywt
#     assert wname in {'haar', 'sym4'}, 'wname must be in {haar, sym4}'
#
#     if x.get_shape().dims[-1].value != 1:
#         assert 'DWT2D: Channel of input tensor must be 1'
#
#     w = pywt.Wavelet(wname)
#
#     w_filter = np.asarray(w.filter_bank)
#     Lo_D = w_filter[0:1, :]
#     Hi_D = w_filter[1:2, :]
# #        Lo_R = w_filter[2:3, :]
# #        Hi_R = w_filter[3:4, :]
#
#     filter_LL = np.dot(np.transpose(Lo_D), Lo_D)
#     filter_LH = np.dot(np.transpose(Hi_D), Lo_D)
#     filter_HL = np.dot(np.transpose(Lo_D), Hi_D)
#     filter_HH = np.dot(np.transpose(Hi_D), Hi_D)
#
#     pad = 1 # True for haar and sym4
#     filters = np.zeros((filter_LL.shape[0]+pad, filter_LL.shape[1]+pad, 1, 4))
#     filters[:-pad, :-pad, 0, 0] = np.fliplr(np.flipud(filter_LL))
#     filters[:-pad, :-pad, 0, 1] = np.fliplr(np.flipud(filter_LH))
#     filters[:-pad, :-pad, 0, 2] = np.fliplr(np.flipud(filter_HL))
#     filters[:-pad, :-pad, 0, 3] = np.fliplr(np.flipud(filter_HH))
#
#     filters = tf.Variable(filters.astype('float32'), trainable=False, name='DWT2_Filter')
#
#     # Periodization padding
#     b_x, h_x, w_x, c_x = x.get_shape().as_list()
#     h_f, w_f, _, _ = filters.get_shape().as_list()
#
#     x = tf.tile(x, [1, 3, 3, 1])
#     x = x[:, h_x-(h_f/2):h_x*2+(h_f/2), w_x-(w_f/2):w_x*2+(w_f/2), :]
#     x = tf.nn.conv2d(x, filters, [1, 1, 1, 1], 'VALID')
#
#     x = x[:, 1::2, 1::2, :]
#
#     return tf.split(3, 4, x)
#
# def IDWT2D(x, wname='haar'):
#     import pywt
#     assert wname in {'haar', 'sym4'}, 'wname must be in {haar, sym4}'
#     assert len(x) == 4, 'IDWT2D: Four components are needed'
#
#     # Zero padding, extended copy of x, inserting 0s
#     b, h, w, c = x[0].get_shape().as_list()
#     assert isinstance(b, int), 'IDWT2D: Batch size must be assigned'
#     dummy = np.zeros((b, h, w, 3))
#     dummy = tf.Variable(dummy.astype('float32'), trainable=False, name='IDWT2_Dummy')
#
#     LL = PeriodicShuffle(tf.concat(3, [x[0], dummy]), 2)
#     LH = PeriodicShuffle(tf.concat(3, [x[1], dummy]), 2)
#     HL = PeriodicShuffle(tf.concat(3, [x[2], dummy]), 2)
#     HH = PeriodicShuffle(tf.concat(3, [x[3], dummy]), 2)
#
#     x = tf.concat(3, [LL, LH, HL, HH])
#     assert x.get_shape().dims[-1].value == 4, 'IDWT2D: Channel of each component must be 1'
#
#     #
#     w = pywt.Wavelet(wname)
#
#     w_filter = np.asarray(w.filter_bank)
# #    Lo_D = w_filter[0:1, :]
# #    Hi_D = w_filter[1:2, :]
#     Lo_R = w_filter[2:3, :]
#     Hi_R = w_filter[3:4, :]
#
#     filter_LL = np.dot(np.transpose(Lo_R), Lo_R)
#     filter_LH = np.dot(np.transpose(Hi_R), Lo_R)
#     filter_HL = np.dot(np.transpose(Lo_R), Hi_R)
#     filter_HH = np.dot(np.transpose(Hi_R), Hi_R)
#
#     pad = 1 # True for haar and sym4
#     filters = np.zeros((filter_LL.shape[0]+pad, filter_LL.shape[1]+pad, 4, 1))
#     filters[:-pad, :-pad, 0, 0] = np.fliplr(np.flipud(filter_LL))
#     filters[:-pad, :-pad, 1, 0] = np.fliplr(np.flipud(filter_LH))
#     filters[:-pad, :-pad, 2, 0] = np.fliplr(np.flipud(filter_HL))
#     filters[:-pad, :-pad, 3, 0] = np.fliplr(np.flipud(filter_HH))
#
#     filters = tf.Variable(filters.astype('float32'), trainable=False, name='IDWT2_Filter')
#
#     # Periodization padding
#     b_x, h_x, w_x, c_x = x.get_shape().as_list()
#     h_f, w_f, _, _ = filters.get_shape().as_list()
#
#     x = tf.tile(x, [1, 3, 3, 1])
#     x = x[:, h_x-(h_f/2):h_x*2+(h_f/2), w_x-(w_f/2):w_x*2+(w_f/2), :]
#     x = tf.nn.conv2d(x, filters, [1, 1, 1, 1], 'VALID')
#
#     return x



# # check
# import cv2
# from utils import _load_img_array
# import scipy.io
# #
# # ins = cv2.imread('download.jpg')
# # ins = np.transpose(ins, (2, 0, 1))
# # ins = ins[np.newaxis, ...]
# # ins = ins[:,:,:800,:800]
# # ins = torch.from_numpy((ins/255.0).astype('float32'))
# # out = DownSample2DMatlab(ins, 1/4.0)
# #
# # out = out.data.numpy()
# # out = out[0]
# # out = np.transpose(out, (1, 2, 0))
# # out = np.clip(np.around(out*255), 0, 255)
# # cv2.imwrite('download_p.png', out.astype(np.uint8))
#
#
# from PIL import Image
# # ins = _load_img_array('download.jpg')
# ins = scipy.io.loadmat('download.mat')
# ins = ins['download']
# ins = ins / 255.0
# ins = np.transpose(ins, (2, 0, 1))
# ins = ins[np.newaxis, ...]
# ins = ins[:,:,:800,:800]
# ins = torch.from_numpy(ins.astype('float32')).cuda()
# out = UpSample2DMatlab(ins, 3)
# out = out.cpu().data.numpy()
# out = out[0]
# out = np.transpose(out, (1, 2, 0))
#
# a={}
# a['p'] = out
# scipy.io.savemat('p.mat', a)
#
# out = np.clip(np.around(out*255), 0, 255).astype(np.uint8)
# # out[:,:,[0,1,2]] = out[:,:,[2,1,0]]
# out = Image.fromarray(out)
# out.save('download_p.png')




def DynamicFiltering(x, f, filter_size, cuda=True):
    '''
    Dynamic filtering: https://arxiv.org/abs/1605.09673
    x: (B, T, H, W)
    f: (B, in_depth, output_depth, H, W)
    filter_shape (ft, fh, fw)
    '''
      
    # make tower
    filter_localexpand_np = np.reshape(np.eye(np.prod(filter_size), np.prod(filter_size), dtype=np.float32), (np.prod(filter_size), filter_size[0], filter_size[1], filter_size[2]))
    filter_localexpand = torch.from_numpy(filter_localexpand_np)

    conv_localexpand = nn.Conv2d(1, int(np.prod(filter_size)), (filter_size[1],filter_size[2]), (1,1), (filter_size[1]//2,filter_size[2]//2), bias=False) # Bx25xHxW
    conv_localexpand.weight.data = filter_localexpand
    conv_localexpand.requires_grad = False

    if cuda:
       conv_localexpand.cuda()
    x_localexpand = conv_localexpand(x) # Bx25xHxW

    x_localexpand = torch.transpose(x_localexpand, 1, 2) # BxHx25xW
    x_localexpand = torch.transpose(x_localexpand, 2, 3) # BxHxWx25
    s = x_localexpand.size()
    x_localexpand = x_localexpand.contiguous().view(s[0], s[1], s[2], 1, s[3]) # BxHxWx1x25

    f = torch.transpose(f, 1, 3) # BxHx16x25xW
    f = torch.transpose(f, 2, 4) # BxHxWx25x16

    x = torch.matmul(x_localexpand, f) # BxHxWx1x16
    x = torch.squeeze(x, 3) # BxHxWx16
    x = torch.transpose(x, 1, 3) # Bx16xWxH
    x = torch.transpose(x, 2, 3) # Bx16xHxW

    return x




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
    #                        tf.linspace(0., tf.cast(xs[1]-1, tf.float32), fs[1]))    

    # Due to when transform
    base_x = torch.arange(fs[3]).view(1, 1, fs[3])
    base_x = base_x.expand(fs[0], fs[2], fs[3])

    base_y = torch.arange(fs[2]).view(1, fs[2], 1)
    base_y = base_y.expand(fs[0], fs[2], fs[3])

    base_grid = torch.stack([base_x, base_y], 1)
    base_grid = Variable(base_grid, requires_grad=False).cuda()
    
    new_grid = base_grid + flow
    x = new_grid[:,0,:,:].contiguous().view(-1)
    y = new_grid[:,1,:,:].contiguous().view(-1)

    # base index
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, xs[3]-1)
    x1 = torch.clamp(x1, 0, xs[3]-1)
    y0 = torch.clamp(y0, 0, xs[2]-1)
    y1 = torch.clamp(y1, 0, xs[2]-1)

    def _repeat(x, n_repeats):
        rep = torch.ones([1, n_repeats])
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)
            
    base = _repeat(torch.arange(xs[0])*xs[2]*xs[3], fs[2]*fs[3]).cuda()
    base_y0 = base + y0*xs[3]
    base_y1 = base + y1*xs[3]
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # sampling
    # use indices to lookup pixels in the flat image and restore
    im_flat = input.permute(0,2,3,1).contiguous().view(-1, xs[1])
    Ia = torch.gather(im_flat, 0, idx_a.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    Ib = torch.gather(im_flat, 0, idx_b.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    Ic = torch.gather(im_flat, 0, idx_c.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    Id = torch.gather(im_flat, 0, idx_d.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)

    # and finally calculate bilinearly interpolated values
    wa = ((x1-x) * (y1-y)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    wb = ((x1-x) * (y-y0)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    wc = ((x-x0) * (y1-y)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    wd = ((x-x0) * (y-y0)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)

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
    #                        tf.linspace(0., tf.cast(xs[1]-1, tf.float32), fs[1]))    

    # Due to when transform
    base_x = torch.arange(fs[3]).view(1, 1, fs[3])
    base_x = base_x.expand(fs[0], fs[2], fs[3])

    base_y = torch.arange(fs[2]).view(1, fs[2], 1)
    base_y = base_y.expand(fs[0], fs[2], fs[3])

    base_grid = torch.stack([base_x, base_y], 1)
    base_grid = Variable(base_grid, requires_grad=False).cuda()
    
    new_grid = base_grid + flow
    x = new_grid[:,0,:,:].contiguous().view(-1)
    y = new_grid[:,1,:,:].contiguous().view(-1)

    # base index
    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, xs[3]-1)
    x1 = torch.clamp(x1, 0, xs[3]-1)
    y0 = torch.clamp(y0, 0, xs[2]-1)
    y1 = torch.clamp(y1, 0, xs[2]-1)

    def _repeat(x, n_repeats):
        rep = torch.ones([1, n_repeats])
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)
            
    base = _repeat(torch.arange(xs[0])*xs[2]*xs[3], fs[2]*fs[3]).cuda()
    base_y0 = base + y0*xs[3]
    base_y1 = base + y1*xs[3]
    idx_a = base_y0 + x0
    idx_b = base_y1 + x0
    idx_c = base_y0 + x1
    idx_d = base_y1 + x1

    # sampling
    # use indices to lookup pixels in the flat image and restore
    im_flat = input.permute(0,2,3,1).contiguous().view(-1, xs[1])
    Ia = torch.gather(im_flat, 0, idx_a.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    Ib = torch.gather(im_flat, 0, idx_b.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    Ic = torch.gather(im_flat, 0, idx_c.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
    Id = torch.gather(im_flat, 0, idx_d.view(-1,1).expand(-1,xs[1]).long()).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)

    if init:
        # and finally calculate bilinearly interpolated values
        wa = ((x1-x) * (y1-y)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
        wb = ((x1-x) * (y-y0)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
        wc = ((x-x0) * (y1-y)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
        wd = ((x-x0) * (y-y0)).view(-1,1).expand(-1,xs[1]).view(xs[0],xs[2],xs[3],xs[1]).permute(0,3,1,2)
        return Ia*wa + Ib*wb + Ic*wc + Id*wd

    else:
        return Ia*weight[:,0:1] + Ib*weight[:,1:2] + Ic*weight[:,2:3] + Id*weight[:,3:4]