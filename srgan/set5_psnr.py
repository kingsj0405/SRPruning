import tensorflow as tf    
import tensorlayer as tl
import numpy as np

from model import get_G
from pathlib import Path

from matlab_imresize import Resize


test_hr_img_list = sorted(tl.files.load_file_list(path='../../dataset/Set5', regx='.*.bmp', printable=False))
test_hr_imgs = tl.vis.read_images(test_hr_img_list, path='../../dataset/Set5', n_threads=32)

G = get_G([1, None, None, 3])
G.load_weights('../../REWIND-srgan/checkpoint/v10/g_init_50.h5')
G.eval()

resize = Resize(4)

tl.files.exists_or_mkdir('../../dataset/Set5_lr')
tl.files.exists_or_mkdir('../../REWIND-srgan/test/v10/')

# Same as MATLAB's rgb2ycbcr
# Updated at 03/14/2017
# Not tested for cb and cr
def _rgb2ycbcr(img, maxVal=255):
#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

#    ycbcr = np.empty([img.shape[0], img.shape[1], img.shape[2]])

    if maxVal == 1:
        O = O / 255.0

#    ycbcr[:,:,0] = ((T[0,0] * r) + (T[0,1] * g) + (T[0,2] * b) + O[0])
#    ycbcr[:,:,1] = ((T[1,0] * r) + (T[1,1] * g) + (T[1,2] * b) + O[1])
#    ycbcr[:,:,2] = ((T[2,0] * r) + (T[2,1] * g) + (T[2,2] * b) + O[2])

    t = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

#    print(np.all((ycbcr - ycbcr_) < 1/255.0/2.0))

    return ycbcr

def PSNR(y_true,y_pred, shave_border=4):
    '''
        Input must be 0-255, 2D
    '''

    target_data = np.array(y_true, dtype=np.float32)
    ref_data = np.array(y_pred, dtype=np.float32)

    diff = ref_data - target_data
    if shave_border > 0:
        diff = diff[shave_border:-shave_border, shave_border:-shave_border]
    rmse = np.sqrt(np.mean(np.power(diff, 2)))

    return 20 * np.log10(255./rmse)

# save
psnrs = []
for i, img in enumerate(test_hr_imgs):
    # Generate low resolution
    img = (img / 127.5) - 1
    img_shape = img.shape
    img_lr = resize(img)
    img_lr = np.asarray(img_lr, dtype=np.float32)
    img_lr = img_lr[np.newaxis,:,:,:]
    out = G(img_lr).numpy()
    # Save lr image
    img_lr = (img_lr + 1) * 127.5
    img_lr = np.clip(img_lr, 0, 255)
    img_lr = img_lr.astype(np.uint8)
    tl.vis.save_image(img_lr[0], f'../../dataset/Set5_lr/{test_hr_img_list[i]}')
    # Calculate psnr
    out = (out + 1) * 127.5
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)
    img = np.asarray((img + 1) * 127.5, dtype=np.uint8)
    psnr = PSNR(_rgb2ycbcr(img)[:,:,0], _rgb2ycbcr(out[0])[:,:,0], 4)
    psnrs.append(psnr)
    print(f"PSNR of {test_hr_img_list[i]}: {psnr}")
    # Save out image
    tl.vis.save_image(out[0], f'../../REWIND-srgan/test/v10/{test_hr_img_list[i]}')
print(f"Average of set5 psnr is {np.asarray(psnrs).mean()}")