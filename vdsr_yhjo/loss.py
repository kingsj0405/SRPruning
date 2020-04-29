# -*- coding: utf-8 -*-
# import tensorflow as tf
import torch

# def MSE(y_true, y_pred, axis=None):
#     return tf.reduce_mean(tf.square(y_true - y_pred), axis=axis)

# def MAE(y_true, y_pred, axis=None):
#     return tf.reduce_mean(tf.abs(y_pred - y_true), axis=axis)
    
def Huber(input, target, delta=0.01, reduce=True):
    abs_error = torch.abs(input - target)
    quadratic = torch.clamp(abs_error, max=delta)

    # The following expression is the same in value as
    # tf.maximum(abs_error - delta, 0), but importantly the gradient for the
    # expression when abs_error == delta is 0 (for tf.maximum it would be 1).
    # This is necessary to avoid doubling the gradient, since there is already a
    # nonzero contribution to the gradient from the quadratic term.
    linear = (abs_error - quadratic)
    losses = 0.5 * torch.pow(quadratic, 2) + delta * linear
    
    if reduce:
        return torch.mean(losses)
    else:
        return losses
    
'''
Input tensor must be 0-255 integer type, 3D (batch, h, w)
'''
def PSNR_Y(y_true, y_pred, crop_factor):
    import numpy as np

    psnrs = []
    shape = y_true.get_shape().as_list()
    for i in range(0, shape[0]):
        imdff = y_true.astype(np.float64) - y_pred.astype(np.float64)
        imdff = imdff[crop_factor:-crop_factor, crop_factor:-crop_factor]
        imdff = imdff.flatten()
        rmse = np.sqrt(np.mean(np.power(imdff, 2)))
        psnr = 20 * np.log10(255.0 / rmse)
        
        psnrs.append(psnr)

    return np.mean(psnr)
    

# '''
# GDL from https://arxiv.org/pdf/1511.05440v6.pdf

# '''
# def GDL(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert len(y_pred.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'

#     if len(y_true.get_shape().as_list()) == 4:
#         t1 = tf.abs(y_true[:,1:,1:,:] - y_true[:,:-1,1:,:]) - tf.abs(y_pred[:,1:,1:,:] - y_pred[:,:-1,1:,:])
#         t2 = tf.abs(y_true[:,1:,1:,:] - y_true[:,1:,:-1,:]) - tf.abs(y_pred[:,1:,1:,:] - y_pred[:,1:,:-1,:])
#     elif len(y_true.get_shape().as_list()) == 5:
#         t1 = tf.abs(y_true[:,:,1:,1:,:] - y_true[:,:,:-1,1:,:]) - tf.abs(y_pred[:,:,1:,1:,:] - y_pred[:,:,:-1,1:,:])
#         t2 = tf.abs(y_true[:,:,1:,1:,:] - y_true[:,:,1:,:-1,:]) - tf.abs(y_pred[:,:,1:,1:,:] - y_pred[:,:,1:,:-1,:])
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1) + tf.abs(t2))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1) + tf.multiply(t2,t2))
        
# def GDLv2(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert len(y_pred.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'

#     if len(y_true.get_shape().as_list()) == 4:
#         t1 = (y_true[:,1:,1:,:] - y_true[:,:-1,1:,:]) - (y_pred[:,1:,1:,:] - y_pred[:,:-1,1:,:])
#         t2 = (y_true[:,1:,1:,:] - y_true[:,1:,:-1,:]) - (y_pred[:,1:,1:,:] - y_pred[:,1:,:-1,:])
#     elif len(y_true.get_shape().as_list()) == 5:
#         t1 = (y_true[:,:,1:,1:,:] - y_true[:,:,:-1,1:,:]) - (y_pred[:,:,1:,1:,:] - y_pred[:,:,:-1,1:,:])
#         t2 = (y_true[:,:,1:,1:,:] - y_true[:,:,1:,:-1,:]) - (y_pred[:,:,1:,1:,:] - y_pred[:,:,1:,:-1,:])
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1) + tf.abs(t2))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1) + tf.multiply(t2,t2))
    
# def GDLonC(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert len(y_pred.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'

#     if len(y_true.get_shape().as_list()) == 4:
#         t1 = tf.abs(y_true[:,:,:,1:] - y_true[:,:,:,:-1]) - tf.abs(y_pred[:,:,:,1:] - y_pred[:,:,:,:-1])
#     elif len(y_true.get_shape().as_list()) == 5:
#         t1 = tf.abs(y_true[:,1:,:,:,:] - y_true[:,:-1,:,:,:]) - tf.abs(y_pred[:,1:,:,:,:] - y_pred[:,:-1,:,:,:])
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1))
    
# def GDLonCv2(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert len(y_pred.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'
    
#     shape = y_true.get_shape().as_list()
    
#     if len(y_true.get_shape().as_list()) == 4:
#         t = shape[3]
#         if t == None:
#             t_c = 1
#             t = 3
#         else:
#             t_c = t // 2
#         Ts = []
#         for i in range(t):
#             Ts.append((y_true[:,:,:,i:i+1] - y_true[:,:,:,t_c:t_c+1]) - (y_pred[:,:,:,i:i+1] - y_pred[:,:,:,t_c:t_c+1]))
#         t1 = tf.concat(Ts, 3)
#     elif len(y_true.get_shape().as_list()) == 5:
#         t = shape[1]
#         if t == None:
#             t_c = 1
#             t = 3
#         else:
#             t_c = t // 2
#         Ts = []
#         for i in range(t):
#             Ts.append((y_true[:,i:i+1,:,:,:] - y_true[:,t_c:t_c+1,:,:,:]) - (y_pred[:,i:i+1,:,:,:] - y_pred[:,t_c:t_c+1,:,:,:]))
#         t1 = tf.concat(Ts, 1)
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1))
        
# def GDLonCv2_(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert len(y_pred.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'

#     if len(y_true.get_shape().as_list()) == 4:
#         t1 = (y_true[:,:,:,1:] - y_true[:,:,:,:-1]) - (y_pred[:,:,:,1:] - y_pred[:,:,:,:-1])
#         t1 = tf.concat([t1, (y_true[:,:,:,2:3] - y_true[:,:,:,0:1]) - (y_pred[:,:,:,2:3] - y_pred[:,:,:,0:1])], 3)
#     elif len(y_true.get_shape().as_list()) == 5:
#         t1 = (y_true[:,1:,:,:,:] - y_true[:,:-1,:,:,:]) - (y_pred[:,1:,:,:,:] - y_pred[:,:-1,:,:,:])
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1))
    
# def GDLonCv2SharpDiff(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'

#     if len(y_true.get_shape().as_list()) == 4:
#         dy_true = y_true[:,1:,1:,:] - y_true[:,:-1,1:,:]
#         dy_pred = y_pred[:,1:,1:,:] - y_pred[:,:-1,1:,:]
        
#         dx_true = y_true[:,1:,1:,:] - y_true[:,1:,:-1,:]
#         dx_pred = y_pred[:,1:,1:,:] - y_pred[:,1:,:-1,:]
        
#         dt_dy_true = dy_true[:,:,:,1:] - dy_true[:,:,:,:-1]
#         dt_dy_pred = dy_pred[:,:,:,1:] - dy_pred[:,:,:,:-1]
        
#         dt_dx_true = dx_true[:,:,:,1:] - dx_true[:,:,:,:-1]
#         dt_dx_pred = dx_pred[:,:,:,1:] - dx_pred[:,:,:,:-1]
        
#         t1 = dt_dy_true - dt_dy_pred
#         t2 = dt_dx_true - dt_dx_pred
        
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1)) + tf.reduce_mean(tf.abs(t2))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1)) + tf.reduce_mean(tf.multiply(t2,t2))
    
# def GDLonCv3(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert len(y_pred.get_shape().as_list()) in [4, 5], 'GDL: Dimension of input tensor must be in [4, 5]'
#     assert alpha in [1, 2], 'GDL: Alpha must be in [1, 2]'

#     if len(y_true.get_shape().as_list()) == 4:
#         t1 = (y_true[:,:,:,1:] - y_true[:,:,:,:-1]) - (y_pred[:,:,:,1:] - y_pred[:,:,:,:-1])
#     elif len(y_true.get_shape().as_list()) == 5:
#         t1 = (y_true[:,1:,:,:,:] - y_true[:,:-1,:,:,:]) - (y_pred[:,1:,:,:,:] - y_pred[:,:-1,:,:,:])
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.exp(tf.abs(t1))-1)
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.exp(tf.multiply(t1,t1))-1)
    
# def GDL3DonT(y_true, y_pred, alpha=1):
#     assert len(y_true.get_shape().as_list()) in [5], 'GDL3DonT: Dimension of input tensor must be 5'
#     assert len(y_pred.get_shape().as_list()) in [5], 'GDL3DonT: Dimension of input tensor must be 5'
#     assert alpha in [1, 2], 'GDL3DonT: Alpha must be in [1, 2]'

#     t1 = tf.abs(y_true[:,1:,1:,:] - y_true[:,:-1,1:,:]) - tf.abs(y_pred[:,1:,1:,:] - y_pred[:,:-1,1:,:])
#     t2 = tf.abs(y_true[:,1:,1:,:] - y_true[:,1:,:-1,:]) - tf.abs(y_pred[:,1:,1:,:] - y_pred[:,1:,:-1,:])
    
#     if alpha == 1:
#         return tf.reduce_mean(tf.abs(t1) + tf.abs(t2))
#     elif alpha == 2:        
#         return tf.reduce_mean(tf.multiply(t1,t1) + tf.multiply(t2,t2))
    
#    
#def mse(y_true, y_pred):
#    return K.mean(K.square(y_pred - y_true))
#
#def mae(y_true, y_pred):
#    return K.mean(K.abs(y_pred - y_true)) 
#
#def D_KL(y_true, y_pred):
#    y_true = K.batch_flatten(y_true)
#    y_pred = K.batch_flatten(y_pred)
#    y_true = K.clip(y_true, K.epsilon(), 1)
#    y_pred = K.clip(y_pred, K.epsilon(), 1)
#    return K.sum(y_true * K.log(y_true / y_pred), axis=-1)   