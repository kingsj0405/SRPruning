import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, Function
from torch.utils.data import DataLoader

import PIL
from PIL import Image

import numpy as np
# import matplotlib.pyplot as plt
import time
from os import listdir, mkdir
from os.path import isfile, join, isdir
import math
from tqdm import tqdm
import glob


import sys
sys.path.insert(1, '../local/experiment/VSR/')
from sr_utils_pytorch.layers import DownSample2DMatlab, UpSample2DMatlab, SpaceToDepth
from sr_utils_pytorch.utils import DirectoryIterator_DIV2K, DirectoryIterator_VSR_TOFlow, GeneratorEnqueuer, _load_img_array, _rgb2ycbcr, _ycbcr2rgb, AVG_PSNR, PSNR
from sr_utils_pytorch.loss import Huber




# USER PARAMS
EXP_NAME = "SISRusingVideo"
VERSION = "499"

UPSCALE = 4     # upscaling factor
NB_BATCH = 16    # 1*4
NB_FRAME = 1
NB_ITER = 200000     # Total number of iterations

I_DISPLAY = 100
I_VALIDATION = 100
I_SAVE = 1000

best_avg_psnr = 26    # DUF-16: 26.8, FRVSR-10-128: 26.9, RLSP-7-128: 27.46

from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='../pt_log/{}/v{}'.format(EXP_NAME, str(VERSION)))

class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(torch.nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()

        self.residual_layer = self.make_layer(Block, 18)

        self.input = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.output = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.kaiming_normal(m.weight)
                nn.init.constant(m.bias, 0)

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)

        return out + x

model_VDSR = VDSR().cuda()



        


# Iteration
print('===> Training start')
l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
l_accum_n = 0.
dT = 0.
rT = 0.





if not isdir('{}'.format(EXP_NAME)):
    mkdir('{}'.format(EXP_NAME))
if not isdir('{}/checkpoint'.format(EXP_NAME)):
    mkdir('{}/checkpoint'.format(EXP_NAME))
if not isdir('{}/result'.format(EXP_NAME)):
    mkdir('{}/result'.format(EXP_NAME))
if not isdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/checkpoint/v{}'.format(EXP_NAME, str(VERSION)))
if not isdir('{}/result/v{}'.format(EXP_NAME, str(VERSION))):
    mkdir('{}/result/v{}'.format(EXP_NAME, str(VERSION)))




params_G = list(filter(lambda p: p.requires_grad, model_VDSR.parameters()))
opt_G = optim.Adam(params_G, lr=0.0001)




# # Training dataset
# Iter_H = GeneratorEnqueuer(DirectoryIterator_VSR_TOFlow(  #GeneratorEnqueuer
#                                  listfile = '/host/media/ssd1/users/yhjo/dataset/vimeo_septuplet/sep_trainlist.txt',
#                                  datadir = '/host/media/ssd1/users/yhjo/dataset/vimeo_septuplet/sequences/',
#                                  nframe = NB_FRAME,
#                                  target_size = 128+16,
#                                  out_batch_size=NB_BATCH, 
#                                  shuffle=True))
# Iter_H.start(max_q_size=16, workers=2)

# # Training dataset
Iter_H = GeneratorEnqueuer(DirectoryIterator_DIV2K(  #GeneratorEnqueuer
                           datadir='/host/media/ssd1/users/yhjo/dataset/DIV2K/',
                           crop_size=32,
                           crop_per_image=4,
                           out_batch_size=NB_BATCH, 
                           shuffle=True))
Iter_H.start(max_q_size=16, workers=4)




def SaveCheckpoint():
    torch.save(model_VDSR, '{}/checkpoint/v{}/model_VDSR_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_ESRGAN2, '{}/checkpoint/v{}/model_ESRGAN2_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_ESRGAN_x2, '{}/checkpoint/v{}/model_ESRGAN_x2_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))

    # torch.save(model_FH, '{}/checkpoint/v{}/model_FH_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_D, '{}/checkpoint/v{}/model_D_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(model_NL, '{}/checkpoint/v{}/model_NL_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))

    torch.save(opt_G, '{}/checkpoint/v{}/opt_G_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    # torch.save(opt_D, '{}/checkpoint/v{}/opt_D_i{:06d}.pth'.format(EXP_NAME, str(VERSION), i))
    print("Checkpoint saved")





START_ITER = 0
accum_samples = 0


# TRAINING
for i in tqdm(range(START_ITER+1, NB_ITER+1)):


    model_VDSR.train()


    # Data preparing
    # EDVR (vimeo): 7 frames, Matlab downsampling
    st = time.time()
    batch_L_Matlab, batch_H = Iter_H.dequeue()  # BxCxTxHxW
    batch_H = Variable(torch.from_numpy(batch_H)).cuda()  # Matlab downsampled
    batch_L_Matlab = Variable(torch.from_numpy(batch_L_Matlab)).cuda()  # Matlab downsampled
    batch_L_Matlab = UpSample2DMatlab(batch_L_Matlab, UPSCALE)
    dT += time.time() - st



    st = time.time()

    opt_G.zero_grad()

    batch_S = model_VDSR(batch_L_Matlab)

    # Pixel loss
    loss_Pixel = nn.MSELoss()(batch_S, batch_H)


    loss_G = loss_Pixel
    loss_G.backward()
    torch.nn.utils.clip_grad_norm_(params_G, 0.1)
    opt_G.step()
    rT += time.time() - st

    accum_samples += NB_BATCH

    l_accum[0] += loss_Pixel.item()


    if i % I_DISPLAY == 0:
        writer.add_scalar('loss_Pixel', l_accum[0]/I_DISPLAY, i)

        print("{} {}| Iter:{:6d}, Sample:{:6d}, Pixel:{:.2e}, dT:{:.4f}, rT:{:.4f}".format(
            EXP_NAME, VERSION, i, accum_samples, l_accum[0]/I_DISPLAY, dT/I_DISPLAY, rT/I_DISPLAY))
        l_accum = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        l_accum_n = 0.
        n_mix=0
        dT = 0.
        rT = 0.


    if i % I_SAVE == 0:
        SaveCheckpoint()


    if i % I_VALIDATION == 0:
        with torch.no_grad():
            model_VDSR.eval()

            # Test for Set5
            vid4_dir = '../local/dataset/Set5/'
            files = glob.glob(vid4_dir + '/*.bmp')
            files.sort()

            psnrs = []
            lpips = []
            if not isdir('{}/result/v{}/{}'.format(EXP_NAME, str(VERSION), 'Set5')):
                mkdir('{}/result/v{}/{}'.format(EXP_NAME, str(VERSION), 'Set5'))

            for fn in files:
                tmp = _load_img_array(fn)

                val_H = np.asarray(tmp).astype(np.float32) # HxWxC
                val_H_ = val_H
                val_H_ = np.transpose(val_H_, [2, 0, 1]) # CxHxW
                batch_H = val_H_[np.newaxis, ...]


                h_least_multiple = UPSCALE
                w_least_multiple = UPSCALE

                h_pad = [0, 0]
                w_pad = [0, 0]
                if batch_H.shape[2] % h_least_multiple > 0:
                    t = h_least_multiple - (batch_H.shape[2] % h_least_multiple )
                    h_pad = [t//2, t-t//2]
                    batch_H = np.lib.pad(batch_H, pad_width=((0,0),(0,0),(h_pad[0],h_pad[1]),(0,0)), mode = 'reflect')
                if batch_H.shape[3] % w_least_multiple > 0:
                    t = w_least_multiple - (batch_H.shape[3] % w_least_multiple )
                    w_pad = [t//2, t-t//2]
                    batch_H = np.lib.pad(batch_H, pad_width=((0,0),(0,0),(0,0),(w_pad[0],w_pad[1])), mode = 'reflect')



                # val_H_ = batch_H[np.newaxis, ...]  # BxCxTxHxW
                batch_H = Variable(torch.from_numpy(batch_H), volatile=True).cuda()
                        
                # Down sampling
                xh = DownSample2DMatlab(batch_H, 1/UPSCALE)
                xh = torch.clamp(xh, 0, 1)
                xh = torch.round(xh*255)/255
                xh = UpSample2DMatlab(xh, UPSCALE)
                xh = torch.clamp(xh, 0, 1)
                xh = torch.round(xh*255)/255


                batch_output = model_VDSR(xh)

                #
                out_FI2 = (batch_output).cpu().data.numpy()
                out_FI2 = np.clip(out_FI2[0], 0. , 1.) # CxHxW
                out_FI2 = np.transpose(out_FI2, [1, 2, 0])

                if h_pad[0] > 0:
                    out_FI2 = out_FI2[h_pad[0]:,:,:]
                if h_pad[1] > 0:
                    out_FI2 = out_FI2[:-h_pad[1],:,:]
                if w_pad[0] > 0:
                    out_FI2 = out_FI2[:,w_pad[0]:,:]
                if w_pad[1] > 0:
                    out_FI2 = out_FI2[:,:-w_pad[1],:]
                    
                # Save to file
                img_gt = (val_H*255).astype(np.uint8)
                img_target = ((out_FI2)*255).astype(np.uint8)

                Image.fromarray(np.around(out_FI2*255).astype(np.uint8)).save('{}/result/v{}/{}/{}.png'.format(EXP_NAME, str(VERSION), 'Set5', fn.split('/')[-1]))
                psnrs.append(PSNR(_rgb2ycbcr(img_gt)[4:-4,4:-4,0], _rgb2ycbcr(img_target)[4:-4,4:-4,0], 0))


        print('AVG PSNR: Set5: {}'.format(np.mean(np.asarray(psnrs))))

        writer.add_scalar('set5', np.mean(np.asarray(psnrs)), i)

        if np.mean(np.asarray(psnrs)) < best_avg_psnr:
            best_avg_psnr = np.mean(np.asarray(psnrs))
            SaveCheckpoint()