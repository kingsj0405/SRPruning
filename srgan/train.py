#! /usr/bin/python
# -*- coding: utf8 -*-

import os
import time
import random
import numpy as np
import scipy, multiprocessing
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from model import get_G, get_D
from config import config, log_config

# NOTE: https://github.com/python-pillow/Pillow/issues/1510#issuecomment-151458026
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size  # use 8 if your GPU memory is small, and change [4, 4] in tl.vis.save_images to [2, 4]
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
lr_decay_init = config.TRAIN.lr_decay_init
decay_every_init = config.TRAIN.decay_every_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
shuffle_buffer_size = 128
step_size = int((800 - 1) / batch_size) + 1

# ni = int(np.sqrt(batch_size))

log_config(config.SAVE.cfg_file_path, config)
save_dir = config.SAVE.save_dir
checkpoint_dir = config.SAVE.checkpoint_dir
summary_dir = config.SAVE.summary_dir

def get_train_data():
    # load dataset
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))#[0:20]
        # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
        # valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
        # valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the entire train set.
    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
        # for im in train_hr_imgs:
        #     print(im.shape)
        # valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
        # for im in valid_lr_imgs:
        #     print(im.shape)
        # valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
        # for im in valid_hr_imgs:
        #     print(im.shape)
        
    # dataset API and augmentation
    def generator_train():
        for img in train_hr_imgs:
            yield img
    def _map_fn_train(img):
        hr_patch = tf.image.random_crop(img, [384, 384, 3])
        hr_patch = hr_patch / (255. / 2.)
        hr_patch = hr_patch - 1.
        hr_patch = tf.image.random_flip_left_right(hr_patch)
        lr_patch = tf.image.resize(hr_patch, size=[96, 96])
        return lr_patch, hr_patch
    train_ds = tf.data.Dataset.from_generator(generator_train, output_types=(tf.float32))
    train_ds = train_ds.map(_map_fn_train, num_parallel_calls=multiprocessing.cpu_count())
        # train_ds = train_ds.repeat(n_epoch_init + n_epoch)
    train_ds = train_ds.shuffle(shuffle_buffer_size)
    train_ds = train_ds.prefetch(buffer_size=2)
    train_ds = train_ds.batch(batch_size)
        # value = train_ds.make_one_shot_iterator().get_next()
    return train_ds

def train():
    G = get_G((batch_size, 96, 96, 3))
    D = get_D((batch_size, 384, 384, 3))
    VGG = tl.models.vgg19(pretrained=True, end_with='pool4', mode='static')
    writer = tf.summary.create_file_writer(summary_dir)

    lr_v = tf.Variable(lr_init)
    g_optimizer_init = tf.optimizers.Adam(lr_v, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)
    d_optimizer = tf.optimizers.Adam(lr_v, beta_1=beta1)

    G.train()
    D.train()
    VGG.train()

    train_ds = get_train_data()

    step_base = 0
    ## initialize learning (G)
    if config.TRAIN.load_init:
        G.load_weights(config.LOAD.load_init_path)
    else:
        init_g()
    def init_g():
        print("Start init G")
        n_step_epoch = round(n_epoch_init // batch_size)
        for epoch in tqdm(range(n_epoch_init), desc='epoch init learn', dynamic_ncols=True, position=0):
            for step, (lr_patchs, hr_patchs) in enumerate(tqdm(train_ds, desc='step', dynamic_ncols=True, total=step_size, position=1)):
                if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                    break
                step_time = time.time()
                with tf.GradientTape() as tape:
                    fake_hr_patchs = G(lr_patchs)
                    mse_loss = tl.cost.mean_squared_error(fake_hr_patchs, hr_patchs, is_mean=True)
                grad = tape.gradient(mse_loss, G.trainable_weights)
                g_optimizer_init.apply_gradients(zip(grad, G.trainable_weights))
                # print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, mse: {:.3f} ".format(
                #     epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss))
                with writer.as_default():
                    tf.summary.scalar("mse_loss", mse_loss, step=step_base+step)
                    tf.summary.image("training example", lr_patchs, max_outputs=3, step=step_base+step)
                    tf.summary.image("generated output", fake_hr_patchs, max_outputs=3, step=step_base+step)
                    writer.flush()
            step_base += step
            # update the learning rate
            if epoch != 0 and (epoch % decay_every_init == 0):
                new_lr_decay = lr_decay_init**(epoch // decay_every_init)
                lr_v.assign(lr_init * new_lr_decay)
                log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
                print(log)
            if epoch % 10 == 0:
                G.save_weights(os.path.join(checkpoint_dir, 'g_init_{}.h5'.format(epoch)))
                tl.vis.save_images(fake_hr_patchs.numpy(), config.TRAIN.grid, os.path.join(save_dir, 'train_g_init_{}.png'.format(epoch)))
        print("Finish init G")
        G.save_weights(os.path.join(checkpoint_dir, 'g_init.h5'))
        print("Initialized G is saved")

    ## adversarial learning (G, D)
    print("Start adv learning G and D")
    n_step_epoch = round(n_epoch // batch_size)
    for epoch in tqdm(range(n_epoch), desc='epoch adv learn', dynamic_ncols=True, position=0):
        for step, (lr_patchs, hr_patchs) in enumerate(tqdm(train_ds, desc='step', dynamic_ncols=True, total=step_size, position=1)):
            if lr_patchs.shape[0] != batch_size: # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                logits_fake = D(fake_patchs)
                feature_fake = VGG((fake_patchs+1)/2.) # the pre-trained VGG uses the input range of [0, 1]
                feature_real = VGG((hr_patchs+1)/2.)
                g_gan_loss = 1e-3 * tl.cost.sigmoid_cross_entropy(logits_fake, tf.ones_like(logits_fake))
                mse_loss = tl.cost.mean_squared_error(fake_patchs, hr_patchs, is_mean=True)
                vgg_loss = 2e-6 * tl.cost.mean_squared_error(feature_fake, feature_real, is_mean=True)
                g_loss = mse_loss + vgg_loss + g_gan_loss
            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            with tf.GradientTape(persistent=True) as tape:
                fake_patchs = G(lr_patchs)
                fake_patchs.detach()
                logits_fake = D(fake_patchs)
                logits_real = D(hr_patchs)
                d_loss1 = tl.cost.sigmoid_cross_entropy(logits_real, tf.ones_like(logits_real))
                d_loss2 = tl.cost.sigmoid_cross_entropy(logits_fake, tf.zeros_like(logits_fake))
                d_loss = d_loss1 + d_loss2
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            # print("Epoch: [{}/{}] step: [{}/{}] time: {:.3f}s, g_loss(mse:{:.3f}, vgg:{:.3f}, adv:{:.3f}) d_loss: {:.3f}".format(
            #     epoch, n_epoch_init, step, n_step_epoch, time.time() - step_time, mse_loss, vgg_loss, g_gan_loss, d_loss))
            with writer.as_default():
                tf.summary.scalar("mse_loss", mse_loss, step=step_base + step)
                tf.summary.scalar("vgg_loss", vgg_loss, step=step_base + step)
                tf.summary.scalar("g_gan_loss", g_gan_loss, step=step_base + step)
                tf.summary.scalar("d1_loss(real)", d_loss1, step=step_base + step)
                tf.summary.scalar("d2_loss(fake)", d_loss2, step=step_base + step)
                tf.summary.scalar("d_loss", d_loss, step=step_base + step)
                tf.summary.scalar("g_loss(mse_loss + vgg_loss + g_gan_loss)", g_loss, step=step_base + step)
                tf.summary.image("training example", lr_patchs, max_outputs=3, step=step_base+step)
                tf.summary.image("generated output", fake_patchs, max_outputs=3, step=step_base+step)
                tf.summary.image("ground truth", hr_patchs, max_outputs=3, step=step_base+step)
                writer.flush()
        step_base += step

        # update the learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            lr_v.assign(lr_init * new_lr_decay)
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        if epoch % 100 == 0:
            tl.vis.save_images(fake_patchs.numpy(), config.TRAIN.grid, os.path.join(save_dir, 'train_g_{}.png'.format(epoch)))
            G.save_weights(os.path.join(checkpoint_dir, 'g_{}.h5'.format(epoch)))
            D.save_weights(os.path.join(checkpoint_dir, 'd_{}.h5'.format(epoch)))
    print("Finish adv learning G and D")

def evaluate():
    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## if your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    G = get_G([1, None, None, 3])
    G.load_weights(os.path.join(checkpoint_dir, 'g.h5'))
    G.eval()

    valid_lr_img = np.asarray(valid_lr_img, dtype=np.float32)
    valid_lr_img = valid_lr_img[np.newaxis,:,:,:]
    size = [valid_lr_img.shape[1], valid_lr_img.shape[2]]

    out = G(valid_lr_img).numpy()

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], os.path.join(save_dir, 'valid_gen.png'))
    tl.vis.save_image(valid_lr_img[0], os.path.join(save_dir, 'valid_lr.png'))
    tl.vis.save_image(valid_hr_img, os.path.join(save_dir, 'valid_hr.png'))

    out_bicu = scipy.misc.imresize(valid_lr_img[0], [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, os.path.join(save_dir, 'valid_bicubic.png'))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='srgan', help='srgan, evaluate')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'srgan':
        if os.path.exists(config.SAVE.cfg_file_path):
            print(f"There is {config.SAVE.cfg_file_path} already")
            print(f"Close experiment")
            exit(0)
        tl.files.exists_or_mkdir(save_dir)
        tl.files.exists_or_mkdir(checkpoint_dir)
        tl.files.exists_or_mkdir(summary_dir)
        tl.files.exists_or_mkdir(config.SAVE.cfg_dir)
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    elif tl.global_flag['mode'] == 'test':
        test()
    else:
        raise Exception("Unknow --mode")