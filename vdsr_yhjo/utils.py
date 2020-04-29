# -*- coding: utf-8 -*-
from __future__ import absolute_import, division

# import torch
import numpy as np
# from matplotlib import pyplot as plt
import h5py
import os
import threading
import math
import glob
#import matlab.engine
import time



import sys
sys.path.insert(1, '../local/experiment/VSR/')
from sr_utils_pytorch.layers import DownSample2DMatlab

    
## History buffer
class Buffer(object):
  '''
  modified from https://github.com/carpedm20/simulated-unsupervised-tensorflow/blob/master/buffer.py
  '''
  
  def __init__(self, buffer_size, H_dims, L_dims):
    self.rng = np.random.RandomState(None)
    self.buffer_size = buffer_size
    self.H_buffer = np.zeros([buffer_size]+H_dims).astype(np.float32)
    self.G_buffer = np.zeros([buffer_size]+[H_dims[0]]+[H_dims[1]-2]+H_dims[2:]).astype(np.float32)
    # self.G_buffer = np.zeros([buffer_size]+[H_dims[0]]+[H_dims[1]-2-1]+H_dims[2:]).astype(np.float32)
    self.L_buffer = np.zeros([buffer_size]+L_dims).astype(np.float32)
#    self.batch_size = config.batch_size

    self.idx = 0

  def push(self, batch_H, batch_G, batch_L, quantize=False):
    if quantize:
        batch_G = self._quantize(batch_G)
        
    push_size = len(batch_H)
    # overflow
    if self.idx + push_size > self.buffer_size:
      random_idx = self.rng.choice(self.idx, push_size, replace=False)
      self.H_buffer[random_idx] = batch_H
      self.G_buffer[random_idx] = batch_G
      self.L_buffer[random_idx] = batch_L
    #underflow
    else:
      self.H_buffer[self.idx:self.idx+push_size] = batch_H
      self.G_buffer[self.idx:self.idx+push_size] = batch_G
      self.L_buffer[self.idx:self.idx+push_size] = batch_L
      self.idx += push_size

  def sample(self, n):
    random_idx = self.rng.choice(self.idx, n, replace=False)
    return self.H_buffer[random_idx], self.G_buffer[random_idx], self.L_buffer[random_idx]
    
  def push_and_sample(self, batch_H, batch_G, batch_L, ratio=0.5):
    push_size = len(batch_H)
    idx_push = self.rng.choice(push_size, int(push_size*ratio), replace=False)
    self.push(batch_H[idx_push], batch_G[idx_push], batch_L[idx_push])
      
    sample_H, sample_GL, sample_L = batch_H, batch_G, batch_L
    sample_H[idx_push], sample_GL[idx_push], sample_L[idx_push] = self.sample(int(push_size*ratio))
    return sample_H, sample_GL, sample_L

  def _quantize(self, x):
      x = (x+1.)/2. * 255.
      x = np.rint(x)
      x = (x/255.)*2.-1. 
      return x
      
  def quantize_and_push_and_sample(self, batch_H, batch_GL, batch_L, ratio=0.5):      
    return self.push_and_sample(batch_H, self._quantize(batch_GL), batch_L, ratio=ratio)




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


def AVG_PSNR(vid_true, vid_pred, vmin=0, vmax=255, t_border=2, sp_border=8, is_T_Y=False, is_P_Y=False):
    '''
        This include RGB2ycbcr and VPSNR computed in Y

    '''
    input_shape = vid_pred.shape
    if is_T_Y:
      Y_true = to_uint8(vid_true, vmin, vmax)
    else:
      Y_true = np.empty(input_shape[:-1])
      for t in range(input_shape[0]):
        Y_true[t] = _rgb2ycbcr(to_uint8(vid_true[t], vmin, vmax), 255)[:,:,0]

    if is_P_Y:
      Y_pred = to_uint8(vid_pred, vmin, vmax)
    else:
      Y_pred = np.empty(input_shape[:-1])
      for t in range(input_shape[0]):
        Y_pred[t] = _rgb2ycbcr(to_uint8(vid_pred[t], vmin, vmax), 255)[:,:,0]

    diff =  Y_true - Y_pred
    diff = diff[t_border: input_shape[0]- t_border, sp_border: input_shape[1]- sp_border, sp_border: input_shape[2]- sp_border]

    psnrs = []
    for t in range(diff.shape[0]):
      rmse = np.sqrt(np.mean(np.power(diff[t],2)))
      psnrs.append(20*np.log10(255./rmse))

    return np.mean(np.asarray(psnrs))


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

def _ycbcr2rgb(img, maxVal =255):
    r=img[:,:,0]
    g=img[:,:,1]
    b=img[:,:,2]

    if(maxVal == 255):
        ycbcr = np.empty( [img.shape[0], img.shape[1], img.shape[2]])
        ycbcr[:,:,0] = ((1.1644* r) + (0.000* g) + (1.5960 * b)-222.9216)
        ##Cb Equation
        ycbcr[:,:,1] = ((1.1644 * r) + (-0.3918 * g) + (-0.8130 * b) +135.5754)
        ## Cr Equantion
        ycbcr[:,:,2] = ((1.1644 * r) + (2.0172 * g) + (0.0000 * b) - 276.8363)
    else :
        ycbcr = np.empty( [img.shape[0], img.shape[1], img.shape[2]])
        ycbcr[:,:,0] = ((1.1644* r) + (0.000* g) + (1.5960 * b)- 222.9216/255)
        ##Cb Equation
        ycbcr[:,:,1] = ((1.1644 * r) + (-0.3918 * g) + (-0.8130 * b) +135.5754/255)
        ## Cr Equantion
        ycbcr[:,:,2] = ((1.1644 * r) + (2.0172 * g) + (0.0000 * b) - 276.8363/255)

    return ycbcr

def to_uint8(x, vmin, vmax):
    x = x.astype('float32')
    x = (x-vmin)/(vmax-vmin)*255 # 0~255
    return np.clip(np.round(x), 0, 255)


def VPSNR(vid_true, vid_pred, vmin=0, vmax=255, t_border=1, sp_border=16, is_Y=False):
    '''
        This include RGB2ycbcr and VPSNR computed in Y

    '''
    input_shape= vid_pred.shape
    if is_Y:
        Y_true = to_uint8(vid_true, vmin, vmax)
        Y_pred = to_uint8(vid_pred, vmin, vmax)
    else:
        Y_true = np.empty(input_shape[:-1])
        Y_pred = np.empty(input_shape[:-1])
        for t in range(input_shape[0]):
            Y_true[t] = _rgb2ycbcr(to_uint8(vid_true[t], vmin, vmax), 255)[:,:,0]
            Y_pred[t] = _rgb2ycbcr(to_uint8(vid_pred[t], vmin, vmax), 255)[:,:,0]

    diff =  Y_true - Y_pred
    diff = diff[t_border: input_shape[0]- t_border, sp_border: input_shape[1]- sp_border, sp_border: input_shape[2]- sp_border]
    rmse = np.sqrt(np.mean(np.power(diff,2)))

    return 20*np.log10(255./rmse)



def SaveParams(sess, params, out_file='parmas.hdf5'):
    f = h5py.File(out_file, 'w')
    g = f.create_group('params')

    # Flatten list
    params = [item for sublist in params for item in sublist]

    for param in params:
        # To make easy to save hdf5, replace '/' to '_'
#        print(param.name.replace('_','__').replace('/','_'))
        g.create_dataset(param.name.replace('_','__').replace('/','_'), data=sess.run(param))

    print('Parameters are saved to: %s' % out_file)


def LoadParams(sess, params, in_file='parmas.hdf5'):
    f = h5py.File(in_file, 'r')
    g = f['params']
    assign_ops = []
    # Flatten list
    params = [item for sublist in params for item in sublist]

    for param in params:
        flag = False
        for idx, name in enumerate(g):
            #
            parsed_name = list(name)
            for i in range(0+1, len(parsed_name)-1):
                if parsed_name[i] == '_' and (parsed_name[i-1] != '_' and parsed_name[i+1] != '_'):
                    parsed_name[i] = '/'
            parsed_name = ''.join(parsed_name)
            parsed_name = parsed_name.replace('__','_')

            if param.name == parsed_name:
                flag = True
#                print(param.name)
                assign_ops += [param.assign(g[name][()])]

        if not flag:
             print('Warning::Cant find param: {}, ignore if intended.'.format(param.name))

    sess.run(assign_ops)

    print('Parameters are loaded')


#################################################################
### Batch Iterators #############################################
#################################################################
class Iterator(object):
    '''
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''
    def __init__(self, N, batch_size, shuffle, seed, infinite):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed, infinite)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None, infinite=True):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    index_array = np.random.permutation(N)

            if infinite == True:
                current_index = (self.batch_index * batch_size) % N
                if N >= current_index + batch_size:
                    current_batch_size = batch_size
                    self.batch_index += 1
                else:
                    current_batch_size = N - current_index
                    self.batch_index = 0
            else:
                current_index = (self.batch_index * batch_size)
                if current_index >= N:
                    self.batch_index = 0
                    raise StopIteration()
                elif N >= current_index + batch_size:
                    current_batch_size = batch_size
                else:
                    current_batch_size = N - current_index
                self.batch_index += 1
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

def _load_img_array(path, color_mode='RGB', channel_mean=None, modcrop=[0,0,0,0]):
    '''Load an image using PIL and convert it into specified color space,
    and return it as an numpy array.

    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    The code is modified from Keras.preprocessing.image.load_img, img_to_array.
    '''
    ## Load image
    from PIL import Image
    img = Image.open(path)
    if color_mode == 'RGB':
        cimg = img.convert('RGB')
        x = np.asarray(cimg, dtype='float32')

    elif color_mode == 'YCbCr' or color_mode == 'Y':
        cimg = img.convert('YCbCr')
        x = np.asarray(cimg, dtype='float32')
        if color_mode == 'Y':
            x = x[:,:,0:1]

    ## To 0-1
    x *= 1.0/255.0

    if channel_mean:
        x[:,:,0] -= channel_mean[0]
        x[:,:,1] -= channel_mean[1]
        x[:,:,2] -= channel_mean[2]

    if modcrop[0]*modcrop[1]*modcrop[2]*modcrop[3]:
        x = x[modcrop[0]:-modcrop[1], modcrop[2]:-modcrop[3], :]

    return x

def _flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


import time

class GeneratorEnqueuer(object):
    """Builds a queue out of a data generator.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.
    # Arguments
        generator: a generator function which endlessly yields data
        pickle_safe: use multiprocessing if True, otherwise threading

    **copied from https://github.com/fchollet/keras/blob/master/keras/engine/training.py

    Usage:
    enqueuer = GeneratorEnqueuer(generator, pickle_safe=pickle_safe)
    enqueuer.start(max_q_size=max_q_size, workers=workers)

    while enqueuer.is_running():
        if not enqueuer.queue.empty():
            generator_output = enqueuer.queue.get()
            break
        else:
            time.sleep(wait_time)
    """

    def __init__(self, generator, use_multiprocessing=True, wait_time=0.00001, random_seed=int(time.time())):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed



    def start(self, workers=1, max_q_size=10):
        """Kicks off threads which add data from the generator into the queue.
        # Arguments
            workers: number of worker threads
            max_q_size: queue size (when full, threads could block on put())
            wait_time: time to sleep in-between calls to put()
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_q_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            import multiprocessing
            try:
                import queue
            except ImportError:
                import Queue as queue

            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_q_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stop running threads and wait for them to exit, if necessary.
        Should be called by the same thread which called start().
        # Arguments
            timeout: maximum time to wait on thread.join()
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None


    def dequeue(self):
        while self.is_running():
            if not self.queue.empty():
                return self.queue.get()
                break
            else:
                time.sleep(self.wait_time)






######################################################################
######################################################################
######################################################################
##########          Video batch iterator                    ##########
##########          For VSR                                 ##########
######################################################################
######################################################################
######################################################################



import subprocess as sp
import re
import logging
logging.captureWarnings(True)

DEVNULL = open(os.devnull, 'wb')

def cvsecs(time):
    """
    Will convert any time into seconds.
    Here are the accepted formats:
    >>> cvsecs(15.4) -> 15.4 # seconds
    >>> cvsecs( (1,21.5) ) -> 81.5 # (min,sec)
    >>> cvsecs( (1,1,2) ) -> 3662 # (hr, min, sec)
    >>> cvsecs('01:01:33.5') -> 3693.5  #(hr,min,sec)
    >>> cvsecs('01:01:33.045') -> 3693.045
    >>> cvsecs('01:01:33,5') #coma works too

    Copied from
    https://github.com/Zulko/moviepy/blob/master/moviepy/tools.py
    """
    def is_string(obj):
        """ Returns true if s is string or string-like object,
        compatible with Python 2 and Python 3."""
        try:
            return isinstance(obj, basestring)
        except NameError:
            return isinstance(obj, str)


    if is_string(time):
        if (',' not in time) and ('.' not in time):
            time = time + '.0'
        expr = r"(\d+):(\d+):(\d+)[,|.](\d+)"
        finds = re.findall(expr, time)[0]
        nums = list( map(float, finds) )
        return ( 3600*int(finds[0])
                + 60*int(finds[1])
                + int(finds[2])
                + nums[3]/(10**len(finds[3])))

    elif isinstance(time, tuple):
        if len(time)== 3:
            hr, mn, sec = time
        elif len(time)== 2:
            hr, mn, sec = 0, time[0], time[1]
        return 3600*hr + 60*mn + sec

    else:
        return time



class FFMPEG_VideoReader:
    """
    This module implements all the functions to read a video or a picture
    using ffmpeg. It is quite ugly, as there are many pitfalls to avoid

    Modified from
    https://github.com/Zulko/moviepy/blob/master/moviepy/video/io/ffmpeg_reader.py

    """

    def __init__(self, filename, print_infos=False, bufsize = None, check_duration=True,
                 fps_source='tbr'):

#        tt = time.time()
        self.filename = filename
#        t = time.time()
        infos = ffmpeg_parse_infos(filename, print_infos, check_duration,
                                   fps_source)
#        print('inner-infos:', time.time() - t)

        self.fps = infos['video_fps']
        self.size = infos['video_size']
        self.rotation = infos['video_rotation']



        self.duration = infos['video_duration']
        self.ffmpeg_duration = infos['duration']
        self.nframes = infos['video_nframes']

        self.infos = infos
        self.depth = 3

        if bufsize is None:
            w, h = self.size
            bufsize = self.depth * w * h + 1000000

        self.bufsize = bufsize
#        print('inner-total:', time.time() - tt)

    def get_duration(self):
        return self.duration

    def initialize(self, starttime=0):
        """Opens the file, creates the pipe. """

        self.close() # if any

        if starttime != 0 :
            i_arg = [
                     #'-hwaccel', 'cuvid',
                     '-ss', "%.06f" % starttime,
                     '-i', self.filename]

        else:
            i_arg = [ '-i', self.filename]

        cmd = (["ffmpeg"] + i_arg +
               ['-f', 'image2pipe',
                '-pix_fmt', 'rgb24',
                '-vcodec', 'rawvideo', '-'
                ]
                )

        popen_params = {"bufsize": self.bufsize,
                        "stdout": sp.PIPE,
                        "stderr": sp.PIPE,
                        "stdin": DEVNULL}

        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        self.proc = sp.Popen(cmd, **popen_params)



    def read_frame(self):
        w, h = self.size
        nbytes = self.depth*w*h
#        s = ''
#
#        while(len(s)//(w*h) != 3):
        s = self.proc.stdout.read(nbytes)
        # print(nbytes)
        # print(len(s))

        result = np.fromstring(s, dtype='uint8')
        result.shape = (h, w, len(s)//(w*h)) # reshape((h, w, len(s)//(w*h)))

#            if len(s)//(w*h) != 3:
#                print(self.filename + str(len(s)//(w*h)))

        return result


    def get_frames(self, p, N):
        """ Read a file video N frame at relative position p (0-1).
        Note for coders: getting an arbitrary frame in the video with
        ffmpeg can be painfully slow if some decoding has to be done.
        This function tries to avoid fectching arbitrary frames
        whenever possible, by moving between adjacent frames.
        """

        # these definitely need to be rechecked sometime. Seems to work.

        # I use that horrible '+0.00001' hack because sometimes due to numerical
        # imprecisions a 3.0 can become a 2.99999999... which makes the int()
        # go to the previous integer. This makes the fetching more robust in the
        # case where you get the nth frame by writing get_frame(n/fps).

        t = (self.duration-1) * p
        tt = time.time()
        self.initialize(t)
        tt_init = time.time() - tt
        tt = time.time()
        self.read_frame() # dump first frame
        w, h = self.size
        frms = np.empty((N, h, w, self.depth), dtype=np.uint8)
        t = self.read_frame()
        for n in range(N):
            frms[n] = self.read_frame()
        tt_read = time.time()-tt
#        print(tt_init, tt_read)
        return frms


    def close(self):
        if hasattr(self,'proc'):
            self.proc.kill()
            self.proc.terminate()
            self.proc.stdout.close()
            self.proc.stderr.close()
            self.proc.wait()
            del self.proc

    def __del__(self):
        self.close()
        if hasattr(self,'lastread'):
            del self.lastread




def ffmpeg_parse_infos(filename, print_infos=False, check_duration=True,
                       fps_source='tbr'):
    """Get file infos using ffmpeg.
    Returns a dictionary with the fields:
    "video_found", "video_fps", "duration", "video_nframes",
    "video_duration", "audio_found", "audio_fps"
    "video_duration" is slightly smaller than "duration" to avoid
    fetching the uncomplete frames at the end, which raises an error.
    """


    # open the file in a pipe, provoke an error, read output
    is_GIF = filename.endswith('.gif')
    cmd = ["ffmpeg", "-i", filename]
    if is_GIF:
        cmd += ["-f", "null", "/dev/null"]

    popen_params = {"bufsize": 10**5,
                    "stdout": sp.PIPE,
                    "stderr": sp.PIPE,
                    "stdin": DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = sp.Popen(cmd, **popen_params)

    proc.stdout.readline()
    proc.terminate()
    infos = proc.stderr.read().decode('utf8')
    del proc

    if print_infos:
        # print the whole info text returned by FFMPEG
        print( infos )


    lines = infos.splitlines()
    if "No such file or directory" in lines[-1]:
        raise IOError(("MoviePy error: the file %s could not be found !\n"
                      "Please check that you entered the correct "
                      "path.")%filename)

    result = dict()


    # get duration (in seconds)
    result['duration'] = None

    if check_duration:
        try:
            keyword = ('frame=' if is_GIF else 'Duration: ')
            # for large GIFS the "full" duration is presented as the last element in the list.
            index = -1 if is_GIF else 0
            line = [l for l in lines if keyword in l][index]
            match = re.findall("([0-9][0-9]:[0-9][0-9]:[0-9][0-9].[0-9][0-9])", line)[0]
            result['duration'] = cvsecs(match)
        except:
            raise IOError(("MoviePy error: failed to read the duration of file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s")%(
                              filename, infos))

    # get the output line that speaks about video
    lines_video = [l for l in lines if ' Video: ' in l and re.search('\d+x\d+', l)]

    result['video_found'] = ( lines_video != [] )

    if result['video_found']:
        try:
            line = lines_video[0]

            # get the size, of the form 460x320 (w x h)
            match = re.search(" [0-9]*x[0-9]*(,| )", line)
            s = list(map(int, line[match.start():match.end()-1].split('x')))
            result['video_size'] = s
        except:
            raise IOError(("MoviePy error: failed to read video dimensions in file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s")%(
                              filename, infos))

        # Get the frame rate. Sometimes it's 'tbr', sometimes 'fps', sometimes
        # tbc, and sometimes tbc/2...
        # Current policy: Trust tbr first, then fps unless fps_source is
        # specified as 'fps' in which case try fps then tbr

        # If result is near from x*1000/1001 where x is 23,24,25,50,
        # replace by x*1000/1001 (very common case for the fps).

        def get_tbr():
            match = re.search("( [0-9]*.| )[0-9]* tbr", line)

            # Sometimes comes as e.g. 12k. We need to replace that with 12000.
            s_tbr = line[match.start():match.end()].split(' ')[1]
            if "k" in s_tbr:
                tbr = float(s_tbr.replace("k", "")) * 1000
            else:
                tbr = float(s_tbr)
            return tbr

        def get_fps():
            match = re.search("( [0-9]*.| )[0-9]* fps", line)
            fps = float(line[match.start():match.end()].split(' ')[1])
            return fps

        if fps_source == 'tbr':
            try:
                result['video_fps'] = get_tbr()
            except:
                result['video_fps'] = get_fps()

        elif fps_source == 'fps':
            try:
                result['video_fps'] = get_fps()
            except:
                result['video_fps'] = get_tbr()

        # It is known that a fps of 24 is often written as 24000/1001
        # but then ffmpeg nicely rounds it to 23.98, which we hate.
        coef = 1000.0/1001.0
        fps = result['video_fps']
        for x in [23,24,25,30,50]:
            if (fps!=x) and abs(fps - x*coef) < .01:
                result['video_fps'] = x*coef

        if check_duration:
            result['video_nframes'] = int(result['duration']*result['video_fps'])+1
            result['video_duration'] = result['duration']
        else:
            result['video_nframes'] = 1
            result['video_duration'] = None
        # We could have also recomputed the duration from the number
        # of frames, as follows:
        # >>> result['video_duration'] = result['video_nframes'] / result['video_fps']

        # get the video rotation info.
        try:
            rotation_lines = [l for l in lines if 'rotate          :' in l and re.search('\d+$', l)]
            if len(rotation_lines):
                rotation_line = rotation_lines[0]
                match = re.search('\d+$', rotation_line)
                result['video_rotation'] = int(rotation_line[match.start() : match.end()])
            else:
                result['video_rotation'] = 0
        except:
            raise IOError(("MoviePy error: failed to read video rotation in file %s.\n"
                           "Here are the file infos returned by ffmpeg:\n\n%s")%(
                              filename, infos))


    lines_audio = [l for l in lines if ' Audio: ' in l]

    result['audio_found'] = lines_audio != []

    if result['audio_found']:
        line = lines_audio[0]
        try:
            match = re.search(" [0-9]* Hz", line)
            result['audio_fps'] = int(line[match.start()+1:match.end()])
        except:
            result['audio_fps'] = 'unknown'

    return result






#%%
class DirectoryIterator_VSR(Iterator):
    '''
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
    '''

    def __init__(self,
                 H_directory,
                 wildcard='train360p.*/*.mp4',
                 target_size=128,
                 nframe = 3,
                 maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
                 crop_per_image=1,
                 out_batch_size=32,
                 shuffle=True,
                 seed=None,
                 infinite=True):

        self.H_directory = H_directory
        self.wildcard = wildcard
        self.target_size = target_size
        self.nframe = nframe
        self.maxbframe = maxbframe
        self.crop_per_image = crop_per_image

        self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)


        self.out_batch_size = out_batch_size
        self.H_fnames = glob.glob(self.H_directory + self.wildcard)

        import pickle
        with open("vsr_traindata_filelist.pickle","wb") as f:
            pickle.dump(self.H_fnames, f)

        self.nb_sample = len(self.H_fnames)

        print('Found %d H videos' % self.nb_sample)

        super(DirectoryIterator_VSR, self).__init__(self.nb_sample, 1, shuffle, seed, infinite)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        batch_H = np.zeros((self.out_batch_size,) + self.H_shape, dtype='float32')
        batch_I = []

        fname_idx = index_array[0]
        H_name = self.H_fnames[fname_idx]
        VReader = FFMPEG_VideoReader(H_name)

        idx = 0
        retry = 0
        while (idx < self.out_batch_size):
            retry += 1

            p = np.random.uniform(high=0.98)
            # Temporal aug.
            bframe = np.random.randint(1, self.maxbframe+1)
            x = VReader.get_frames(p, self.nframe+(self.nframe-1)*(bframe-1)) #x:  T,H,W,C
            x = x[::bframe,:,:,:]

            #
            if x.shape[3] != 3:
                continue

#            # Scale augmentation
#            s = 1/float(np.random.randint(1, 4))
#            if s != 1:
#                x = _DownSample2DMatlab(x, s)

            # Read another file when video spatial size is smaller than patch size
            if retry > 10 or x.shape[1] - self.target_size < self.crop_per_image or x.shape[2] - self.target_size < self.crop_per_image:
                fname_idx += 1
                if fname_idx >= len(self.H_fnames):
                    fname_idx = 0
                H_name = self.H_fnames[fname_idx]
                VReader = FFMPEG_VideoReader(H_name)

                retry = 0

                continue

            cr_y = np.random.randint(0, x.shape[1] - self.target_size, self.crop_per_image)
            cr_x = np.random.randint(0, x.shape[2] - self.target_size, self.crop_per_image)

            for c in range(self.crop_per_image):
                _H = x[:,cr_y[c]:cr_y[c]+self.target_size, cr_x[c]:cr_x[c]+self.target_size,:]
                _H = _H.astype('float32') / 255.

#                # check enough high frequencies
##                from scipy import signal
##                h = np.array([[0., -1., 0.],
##                              [-1., 4., -1.],
##                                [0., -1., 0.]])
#                HF = np.mean(np.abs(4*_H[0,1:-1,1:-1,:] - _H[0,0:-2,1:-1,:] - _H[0,2:,1:-1,:] - _H[0,1:-1,0:-2,:] - _H[0,1:-1,2:,:]))
##                HF = np.mean(np.abs(signal.convolve2d(_H[0,:,:,1], h)))
#                if not HF > 0.1:
##                    print(HF)
#                    continue

                # check enought motion on consecutive frames
                dc = False
                for f in range(self.nframe-1):
    #                print(np.mean((x[f]-x[f+1])**2))
                    if np.mean((_H[f]-_H[f+1])**2) > 0.03 or np.mean((_H[f]-_H[f+1])**2) < 0.0003:   # psnr btw 15-35
                        dc = True
                        break
    #            print(' ')
                if dc:
                    continue

                # Random Aug
                # Rot
                ri = np.random.randint(0,4)
                _H = np.transpose(_H, [1,2,3,0])
                _H = np.rot90(_H, ri)
                _H = np.transpose(_H, [3,0,1,2])
                # LR flip
                lrf = 0
                if np.random.random() < 0.5:
                    lrf = 1
                    _H = _flip_axis(_H, 2)
                # Temporal flip
                tf = 0
                if np.random.random() < 0.5:
                    tf = 1
                    _H = _flip_axis(_H, 0)

                batch_H[idx] = _H
                idx += 1


                t = {'fname_idx':fname_idx, 'p':p, 'cr_y':cr_y, 'cr_x':cr_x, 'rot90':ri, 'lr_flip':lrf, 't_flip':tf}
                batch_I.append(t)


                if idx >= self.out_batch_size:
                  break

        return batch_H, batch_I
#        return batch_I




#%%
from PIL import Image
class DirectoryIterator_VSR_FromList(Iterator):
  def __init__(self,
               H_directory,
               listfile = 'vsr_traindata_filelist.pickle',
               datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
               total_samples = 160000,
               target_size = 144,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               crop_per_image = 2,
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    self.H_directory = H_directory
    self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe
    self.crop_per_image = crop_per_image

    self.to_Y = to_Y
    if to_Y:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (1,)
    else:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

    self.out_batch_size = out_batch_size

    import pickle
    with open(listfile, 'rb') as f:
      self.H_fnames = pickle.load(f, encoding='utf8')
    for i in range(len(self.H_fnames)):
      self.H_fnames[i] = H_directory + '/' + self.H_fnames[i].split('/')[-2] +'/'+ self.H_fnames[i].split('/')[-1]

    print('Found %d H videos' % len(self.H_fnames))

    self.nb_sample = total_samples
    with open(datafile, 'rb') as f:
      self.list = pickle.load(f, encoding='latin1')

    super(DirectoryIterator_VSR_FromList, self).__init__(self.nb_sample-int(out_batch_size/crop_per_image)+1, int(out_batch_size/crop_per_image), shuffle, seed, infinite)

  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = np.zeros((self.out_batch_size,) + self.H_shape, dtype='float32')

    fname_idx_prev = -1

    # idx = 0
    for i in range(int(self.out_batch_size/self.crop_per_image)):
      fname_idx = self.list[current_index+i]['fname_idx']

      if fname_idx_prev != fname_idx:
        fname_idx_prev = fname_idx
        H_name = self.H_fnames[fname_idx] #.split('/')[-1]
        VReader = FFMPEG_VideoReader(H_name)

      p = self.list[current_index+i]['p']
      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)
      x = VReader.get_frames(p, self.nframe+(self.nframe-1)*(bframe-1)) #x:  T,H,W,C
      x = x[::bframe,:,:,:]
    #   print(x.shape)

      cr_y = self.list[current_index+i]['cr_y']
      cr_x = self.list[current_index+i]['cr_x']

      for c in range(self.crop_per_image):
        _H = x[:,cr_y[c]:cr_y[c]+self.target_size, cr_x[c]:cr_x[c]+self.target_size,:]
        if _H.shape[2] != self.H_shape[2] and _H.shape[1] != self.H_shape[1]:
          _H = x[:,(x.shape[1]-self.target_size)//2:(x.shape[1]-self.target_size)//2+self.target_size, (x.shape[2]-self.target_size)//2:(x.shape[2]-self.target_size)//2+self.target_size,:]
        if _H.shape[1] != self.H_shape[1]:
          _H = x[:,(x.shape[1]-self.target_size)//2:(x.shape[1]-self.target_size)//2+self.target_size, cr_x[c]:cr_x[c]+self.target_size,:]
        if _H.shape[2] != self.H_shape[2]:
          _H = x[:,cr_y[c]:cr_y[c]+self.target_size, (x.shape[2]-self.target_size)//2:(x.shape[2]-self.target_size)//2+self.target_size,:]

        if self.to_Y:
          for n in range(self.nframe):
            _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
          _H = _H[:,:,:,0:1]

        _H = _H.astype('float32') / 255.
        # Random Aug
        # Rot
        ri = np.random.randint(0,4)
        _H = np.transpose(_H, [1,2,3,0])
        _H = np.rot90(_H, ri)
        _H = np.transpose(_H, [3,0,1,2])
        # LR flip
        if np.random.random() < 0.5:
          _H = _flip_axis(_H, 2)
        # Temporal flip
        if np.random.random() < 0.5:
          _H = _flip_axis(_H, 0)  # TxHxWxC

        batch_H[i*self.crop_per_image + c] = _H
    #   idx += 1

    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # CxTxHxW
    return batch_H



#%%
from PIL import Image
class DirectoryIterator_VSR_FromList_ToPngs(Iterator):
  def __init__(self,
               H_directory,
               listfile = 'vsr_traindata_filelist.pickle',
               datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
               total_samples = 160000,
               target_size = 144,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               crop_per_image = 2,
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    self.H_directory = H_directory
    self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe
    self.crop_per_image = crop_per_image

    self.to_Y = to_Y
    if to_Y:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (1,)
    else:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

    self.out_batch_size = out_batch_size

    import pickle
    with open(listfile, 'rb') as f:
      self.H_fnames = pickle.load(f, encoding='utf8')

    print('Found %d H videos' % len(self.H_fnames))

    self.nb_sample = total_samples
    with open(datafile, 'rb') as f:
      self.list = pickle.load(f, encoding='latin1')

    self.global_f_idx = 181
    self.global_idx = 46

    super(DirectoryIterator_VSR_FromList_ToPngs, self).__init__(self.nb_sample-int(out_batch_size/crop_per_image)+1, int(out_batch_size/crop_per_image), shuffle, seed, infinite)

  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = np.zeros((self.out_batch_size,) + self.H_shape, dtype='float32')

    fname_idx_prev = -1

    for i in range(int(self.out_batch_size/self.crop_per_image)):
    #   fname_idx = self.list[current_index+i]['fname_idx']
      fname_idx = self.list[current_index+i+80000]['fname_idx']

      if fname_idx_prev != fname_idx:
        fname_idx_prev = fname_idx
        H_name = self.H_fnames[fname_idx].split('/')[-1]
        VReader = FFMPEG_VideoReader(self.H_directory+'/'+self.H_fnames[fname_idx].split('/')[-2] +'/'+H_name)

      if self.global_f_idx != fname_idx:
        self.global_f_idx = fname_idx
        self.global_idx = 0
        if not os.path.isdir('/hdd2/datasets/VSR/train360p_ext/{:s}'.format(H_name)):
          os.mkdir('/hdd2/datasets/VSR/train360p_ext/{:s}'.format(H_name))

      p = self.list[current_index+i]['p']
      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)
      x = VReader.get_frames(p, self.nframe+(self.nframe-1)*(bframe-1)) #x:  T,H,W,C
      x = x[::bframe,:,:,:]

      cr_y = self.list[current_index+i]['cr_y']
      cr_x = self.list[current_index+i]['cr_x']

      for c in range(self.crop_per_image):
        _H = x[:,cr_y[c]:cr_y[c]+self.target_size, cr_x[c]:cr_x[c]+self.target_size,:]
        if _H.shape[2] != self.H_shape[2] and _H.shape[1] != self.H_shape[1]:
          _H = x[:,(x.shape[1]-self.target_size)//2:(x.shape[1]-self.target_size)//2+self.target_size, (x.shape[2]-self.target_size)//2:(x.shape[2]-self.target_size)//2+self.target_size,:]
        if _H.shape[1] != self.H_shape[1]:
          _H = x[:,(x.shape[1]-self.target_size)//2:(x.shape[1]-self.target_size)//2+self.target_size, cr_x[c]:cr_x[c]+self.target_size,:]
        if _H.shape[2] != self.H_shape[2]:
          _H = x[:,cr_y[c]:cr_y[c]+self.target_size, (x.shape[2]-self.target_size)//2:(x.shape[2]-self.target_size)//2+self.target_size,:]

        if self.to_Y:
          for n in range(self.nframe):
            _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
          _H = _H[:,:,:,0:1]

        for t in range(_H.shape[0]):
          result = Image.fromarray(_H[t,:,:,:])
          result.save('/hdd2/datasets/VSR/train360p_ext/{:s}/sample{:06d}_c{:01d}_f{:02d}.png'.format(H_name, self.global_idx, c, t))


      self.global_idx += 1

    return batch_H



#%%
from PIL import Image
class DirectoryIterator_VSR_FromList_FromPngs(Iterator):
  def __init__(self,
               H_directory,
               listfile = 'vsr_traindata_filelist.pickle',
               datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
               datadir = '/hdd2/datasets/VSR/train360p_ext/',
               total_samples = 160000,
               target_size = 144,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               crop_per_image = 2,
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    self.H_directory = H_directory
    self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe
    self.crop_per_image = crop_per_image

    self.to_Y = to_Y
    if to_Y:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (1,)
    else:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

    self.out_batch_size = out_batch_size

    import pickle
    import sys
    pv = sys.version
    
    with open(listfile, 'rb') as f:
      if pv[0] == '2':
        self.H_fnames = pickle.load(f)
      elif pv[0] == '3':
        self.H_fnames = pickle.load(f, encoding='utf8')
    for i in range(len(self.H_fnames)):
      self.H_fnames[i] = self.H_fnames[i].split('/')[-1]

    self.datadir = datadir

    print('Found %d H videos' % len(self.H_fnames))

    self.nb_sample = total_samples
    with open(datafile, 'rb') as f:
      if pv[0] == '2':
        self.list = pickle.load(f)
      elif pv[0] == '3':
        self.list = pickle.load(f, encoding='latin1')

    self.global_f_idx = -1
    self.global_idx = -1

    # super(DirectoryIterator_VSR_FromList_FromPngs, self).__init__(self.nb_sample, int(out_batch_size/crop_per_image), shuffle, seed, infinite)
    super(DirectoryIterator_VSR_FromList_FromPngs, self).__init__(self.nb_sample, out_batch_size, shuffle, seed, infinite)
  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    H_names = []
    H_idxs = []
    
    i = 0
    while i < int(self.out_batch_size):
    # for i in range(int(self.out_batch_size)):
      fname_idx = self.list[current_index+i]['fname_idx']
    #   if fname_idx < 33:
    #     fname_idx += 33

      if self.global_f_idx != fname_idx:
        self.global_f_idx = fname_idx
        self.global_idx = 0

      H_name = self.H_fnames[fname_idx]

      if H_name == "Omarion Ft. Chris Brown & Jhene Aiko - Post To Be (Official Video)-aPxVSCfoYnU.mp4" and (self.global_idx == "39" or self.global_idx == "63" or self.global_idx == "71" or self.global_idx == "112"):
        self.global_idx += 1
        fname_idx_prev = fname_idx
        continue
      else:
        H_names.append(H_name)
        H_idxs.append(self.global_idx)
    #   H_names.append(H_name)
    #   H_idxs.append(self.global_idx)

      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)

    #   for c in range(self.crop_per_image):
      c = 1
      _H = []
    #   print('{} {} {} {}'.format(H_name, fname_idx, self.global_idx, c))
      for f in range(0, self.nframe+(self.nframe-1)*(bframe-1), bframe):
        # if i == 0 and f == 0:
        #     print(H_name + '/sample{:06d}_c{:1d}_f{:02d}.png'.format(self.global_idx, c, f))
        _H.append(_load_img_array(self.datadir + '/' + H_name + '/sample{:06d}_c{:1d}_f{:02d}.png'.format(self.global_idx, c, f)))
      _H = np.asarray(_H)

      if self.to_Y:
        for n in range(self.nframe):
          _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
        _H = _H[:,:,:,0:1]

      # _H = _H.astype('float32') / 255.
      # Random Aug
      # Rot
      ri = np.random.randint(0,4)
      _H = np.transpose(_H, [1,2,3,0])
      _H = np.rot90(_H, ri)
      _H = np.transpose(_H, [3,0,1,2])
      # LR flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 2)
      # Temporal flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 0)  # TxHxWxC

      batch_H.append(_H)

      self.global_idx += 1
      fname_idx_prev = fname_idx
      i += 1

    batch_H = np.asarray(batch_H)
    if batch_H.shape[0] != self.out_batch_size:
      print('err: batchsize '+str(batch_H.shape[0]))
      return None
    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW
    return batch_H, H_names, H_idxs



#%%
from PIL import Image
class DirectoryIterator_VSR_FromList_FromPngs_(Iterator):
  def __init__(self,
               H_directory = '/hdd2/datasets/VSR/train360p_ext/',
               listfile = 'vsr_traindata_filelist.pickle',
               datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
               total_samples = 160000,
               target_size = 144,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               crop_per_image = 2,
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe
    self.crop_per_image = crop_per_image

    self.to_Y = to_Y
    if to_Y:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (1,)
    else:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

    self.out_batch_size = out_batch_size

    import pickle
    with open(listfile, 'rb') as f:
      self.H_fnames = pickle.load(f, encoding='utf8')
    for i in range(len(self.H_fnames)):
      self.H_fnames[i] = self.H_fnames[i].split('/')[-1]

    self.datadir = H_directory

    print('Found %d H videos' % len(self.H_fnames))

    self.nb_sample = total_samples
    with open(datafile, 'rb') as f:
        self.list = pickle.load(f, encoding='latin1')
    #   self.list = pickle.load(f, encoding='utf8')

    self.global_f_idx = -1
    self.global_idx = -1

    super(DirectoryIterator_VSR_FromList_FromPngs_, self).__init__(self.nb_sample-int(out_batch_size/crop_per_image)+1, int(out_batch_size/crop_per_image), shuffle, seed, infinite)

  def next(self):
    # import imageio

    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    
    for i in range(int(self.out_batch_size/self.crop_per_image)):
      fname_idx = self.list[current_index+i]['fname_idx']

      if self.global_f_idx != fname_idx:
        self.global_f_idx = fname_idx
        self.global_idx = 0

      H_name = self.H_fnames[fname_idx]

      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)

      for c in range(self.crop_per_image):
        # print('{}_{}_{}'.format(current_index, i, c))
        #   c = 1
        _H = []
        #   print('{} {} {} {}'.format(H_name, fname_idx, self.global_idx, c))
        for f in range(0, self.nframe+(self.nframe-1)*(bframe-1), bframe):
            t = _load_img_array(self.datadir + '/' + H_name + '/sample{:06d}_c{:1d}_f{:02d}.png'.format(self.global_idx, c, f))
            # imageio.imwrite(self.datadir + '/' + H_name + '/sample{:06d}_c{:1d}_f{:02d}.ppm'.format(self.global_idx, c, f), t, format='PPM-FI', flags=1)
            _H.append(t)
        _H = np.asarray(_H)

        if self.to_Y:
            for n in range(self.nframe):
                _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
            _H = _H[:,:,:,0:1]

        # _H = _H.astype('float32') / 255.
        # Random Aug
        # Rot
        ri = np.random.randint(0,4)
        _H = np.transpose(_H, [1,2,3,0])
        _H = np.rot90(_H, ri)
        _H = np.transpose(_H, [3,0,1,2])
        # LR flip
        if np.random.random() < 0.5:
            _H = _flip_axis(_H, 2)
        # Temporal flip
        if np.random.random() < 0.5:
            _H = _flip_axis(_H, 0)  # TxHxWxC

        batch_H.append(_H)

      self.global_idx += 1
      fname_idx_prev = fname_idx

    batch_H = np.asarray(batch_H)
    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW
    
    return batch_H




class DirectoryIterator_VSR_FromList_FromPngs_CDVL(Iterator):
  def __init__(self,
               listfile = 'vsr_traindata_filelist.pickle',
               datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
               datadir = '/hdd2/datasets/VSR/train360p_ext/',
               target_size = 144,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               crop_per_image = 2,
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe
    self.crop_per_image = crop_per_image

    self.to_Y = to_Y
    if to_Y:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (1,)
    else:
      self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

    self.out_batch_size = out_batch_size

    import pickle
    import sys
    pv = sys.version

    with open(listfile, 'rb') as f:
      if pv[0] == '2':
        self.H_fnames = pickle.load(f)
      elif pv[0] == '3':
        self.H_fnames = pickle.load(f, encoding='utf8')

    for i in range(len(self.H_fnames)):
      self.H_fnames[i] = self.H_fnames[i].split('/')[-1]

    self.datadir = datadir

    datafiles = glob.glob(datafile)
    datafiles.sort()
    self.list = []
    for d in datafiles:
        with open(d, 'rb') as f:
            if pv[0] == '2':
                self.list += pickle.load(f)
            elif pv[0] == '3':
                self.list += pickle.load(f, encoding='latin1')
    self.nb_sample = len(self.list)

    # self.global_f_idx = -1
    # self.global_idx = 0

    print('Found %d H videos, %d H samples' % (len(self.H_fnames), self.nb_sample))

    super(DirectoryIterator_VSR_FromList_FromPngs_CDVL, self).__init__(self.nb_sample, int(out_batch_size/crop_per_image), shuffle, seed, infinite)
    # super(DirectoryIterator_VSR_FromList_FromPngs_CDVL, self).__init__(self.nb_sample, out_batch_size, shuffle, seed, infinite)
  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    
    for i in range(int(self.out_batch_size)):
      fname_idx = self.list[index_array[i]]['fname_idx']
      global_idx = self.list[index_array[i]]['global_sample_idx']
      
    #   if self.global_f_idx != fname_idx:
    #     self.global_f_idx = fname_idx
    #     self.global_idx = 0

      H_name = self.H_fnames[fname_idx]
      
      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)

    #   for c in range(self.crop_per_image):
      _H = []
      for f in range(0, self.nframe+(self.nframe-1)*(bframe-1), bframe):
        _H.append(_load_img_array(self.datadir + '/' + H_name + '/globalsample{:06d}_f{:02d}.png'.format(global_idx, f)))
      _H = np.asarray(_H)
      
      if self.to_Y:
        for n in range(self.nframe):
          _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
        _H = _H[:,:,:,0:1]

      # _H = _H.astype('float32') / 255.
      # Random Aug
      # Rot
      ri = np.random.randint(0,4)
      _H = np.transpose(_H, [1,2,3,0])
      _H = np.rot90(_H, ri)
      _H = np.transpose(_H, [3,0,1,2])
      # LR flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 2)
      # Temporal flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 0)  # TxHxWxC

      batch_H.append(_H)

    #   self.global_idx += 1
    #   fname_idx_prev = fname_idx

    batch_H = np.asarray(batch_H)
    # if batch_H.shape[0] != self.out_batch_size:
    #   return None

    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW
    return batch_H




class DirectoryIterator_VSR_PoseTrack(Iterator):
  import glob
  def __init__(self,
            #    H_directory,
               datadir1 = '/mnt/hdd1/posetrack_data/images/bonn_mpii_train_5sec/',
               datadir2 = '/mnt/hdd1/posetrack_data/images/bonn_mpii_train_v2_5sec/',
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    # self.H_directory = H_directory
    # self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe

    self.to_Y = to_Y

    self.out_batch_size = out_batch_size

    a = glob.glob(datadir1 + '/*')
    a.sort()
    b = glob.glob(datadir2 + '/*')
    b.sort()
    self.datadir = a + b

    self.total_samples = len(self.datadir)
    print('Found %d H sequences' % self.total_samples)

    # super(DirectoryIterator_VSR_PoseTrack, self).__init__(self.total_samples, out_batch_size, shuffle, seed, infinite)
    super(DirectoryIterator_VSR_PoseTrack, self).__init__(self.total_samples, 1, shuffle, seed, infinite)


  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []

    i_dir = self.datadir[current_index]
    dir_files = glob.glob(i_dir + '/*.jpg')
    dir_files.sort()
    
    for i in range(int(self.out_batch_size)):
      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)

      s = np.random.randint(0, len(dir_files) - (self.nframe+(self.nframe-1)*(bframe-1)) - 1)

      _H = []
      for f in range(s, s+self.nframe+(self.nframe-1)*(bframe-1), bframe):
        _H.append(_load_img_array(dir_files[f]))
      _H = np.asarray(_H)

      if self.to_Y:
        for n in range(self.nframe):
          _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
        _H = _H[:,:,:,0:1]

      # Random Aug
      # LR flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 2)
      # Temporal flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 0)  # TxHxWxC

      batch_H.append(_H)

    batch_H = np.stack(batch_H, 0)

    # Random Aug
    # Rot
    ri = np.random.randint(0,4) # may be problem here
    batch_H = np.transpose(batch_H, [2,3,4,0,1])
    batch_H = np.rot90(batch_H, ri)
    batch_H = np.transpose(batch_H, [3,4,0,1,2])

    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW
    return batch_H


    


class DirectoryIterator_VSR_TOFlow(Iterator):
  def __init__(self,
            #    H_directory,
               listfile = '/mnt/hdd1/vimeo_septuplet/vimeo_septuplet/sep_trainlist.txt',
               datadir = '/mnt/hdd1/vimeo_septuplet/vimeo_septuplet/sequences/',
                total_samples = None,
                target_size = 144,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    # self.H_directory = H_directory
    # self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe
    self.target_size = target_size

    self.to_Y = to_Y

    self.out_batch_size = out_batch_size

    f = open(listfile, 'r')
    self.H_list = f.readlines()
    f.close()

    self.datadir = datadir
    if total_samples is None:
        self.total_samples = len(self.H_list)
    else:
        self.total_samples = total_samples
    print('Found %d H sequences' % self.total_samples)

    super(DirectoryIterator_VSR_TOFlow, self).__init__(self.total_samples, out_batch_size, shuffle, seed, infinite)


  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    
    for i in range(int(self.out_batch_size)):
      i_dir = self.H_list[(current_index+i) % self.total_samples].rstrip()

      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)

      _H = []
      for f in range(0, self.nframe+(self.nframe-1)*(bframe-1), bframe):
          
        try:
          _H.append(_load_img_array(self.datadir + '/' + i_dir + '/im{:1d}.png'.format(f+1)))
        except:
          print("File open error: {}".format(self.datadir + '/' + i_dir + '/im{:1d}.png'.format(f+1)))
          continue

        
      _H = np.asarray(_H)

      if self.to_Y:
        for n in range(self.nframe):
          _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
        _H = _H[:,:,:,0:1]

      # Random Aug
      # LR flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 2)
      # Temporal flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 0)  # TxHxWxC

      batch_H.append(_H)

      

    batch_H = np.stack(batch_H, 0)



    # # Random Aug
    # # Rot
    # ri = np.random.randint(0,4) # may be problem here
    # batch_H = np.transpose(batch_H, [2,3,4,0,1])
    # batch_H = np.rot90(batch_H, ri)
    # batch_H = np.transpose(batch_H, [3,4,0,1,2])

    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW

    if self.target_size is not None:
        ch = batch_H.shape[3] - self.target_size
        cw = batch_H.shape[4] - self.target_size

        if ch > 0:
            sh = np.random.randint(ch)
            batch_H = batch_H[:,:,:,sh:sh+self.target_size]

        if cw > 0:
            sw = np.random.randint(cw)
            batch_H = batch_H[:,:,:,:,sw:sw+self.target_size]

    return batch_H



    

class DirectoryIterator_VSR_FlyingChairs(Iterator):
  def __init__(self,
            #    H_directory,
               datadir = '/host/media/VSR/FlyingChairs_release/data/',
               otal_samples = None,
               nframe = 3,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               out_batch_size = 16,
               to_Y = False,
               shuffle = True,
               seed = None,
               infinite = True):

    # self.H_directory = H_directory
    # self.target_size = target_size
    self.nframe = nframe
    self.maxbframe = maxbframe

    self.to_Y = to_Y

    self.out_batch_size = out_batch_size

    f = open(listfile, 'r')
    self.H_list = f.readlines()
    f.close()

    self.datadir = datadir
    if total_samples is None:
        self.total_samples = len(self.H_list)
    else:
        self.total_samples = total_samples
    print('Found %d H sequences' % self.total_samples)

    super(DirectoryIterator_VSR_TOFlow, self).__init__(self.total_samples, out_batch_size, shuffle, seed, infinite)


  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    
    for i in range(int(self.out_batch_size)):
      i_dir = self.H_list[current_index+i].rstrip()

      # Temporal aug.
      bframe = np.random.randint(1, self.maxbframe+1)

      _H = []
      for f in range(0, self.nframe+(self.nframe-1)*(bframe-1), bframe):
        _H.append(_load_img_array(self.datadir + '/' + i_dir + '/im{:1d}.png'.format(f+1)))
      _H = np.asarray(_H)

      if self.to_Y:
        for n in range(self.nframe):
          _H[n,:,:,0] = _rgb2ycbcr(_H[n,:,:,:], maxVal=1)[:,:,0]
        _H = _H[:,:,:,0:1]

      # Random Aug
      # LR flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 2)
      # Temporal flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 0)  # TxHxWxC

      batch_H.append(_H)

    batch_H = np.stack(batch_H, 0)

    # Random Aug
    # Rot
    ri = np.random.randint(0,4) # may be problem here
    batch_H = np.transpose(batch_H, [2,3,4,0,1])
    batch_H = np.rot90(batch_H, ri)
    batch_H = np.transpose(batch_H, [3,4,0,1,2])

    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW
    return batch_H





class DirectoryIterator_VSR_MPISintel(Iterator):
    def __init__(self,
               datadir = '',
               target_size = 144,
               nframe = 3,
               crop_per_image = 2,
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

        self.datadir = datadir
        self.target_size = target_size
        self.nframe = nframe
        self.crop_per_image = crop_per_image
        self.out_batch_size = out_batch_size

        self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

        scenes = glob.glob(datadir + '/clean/*')
        self.scenes = [s.split('/')[-1] for s in scenes]

        self.frameCount = []
        self.cleanFrames = []
        self.flows = []
        self.occlusions = []
        for s in self.scenes:
            frames = glob.glob(datadir + '/clean/' + s + '/*.png')
            flows = glob.glob(datadir + '/clean/' + s + '/*.png')
            occs = glob.glob(datadir + '/clean/' + s + '/*.png')
            self.cleanFrames.append(frames)
            self.flows.append(flows)
            self.occlusions.append(occs)
            self.frameCount.append(len(frames))

        print('Found %d scenes, total %d frames' % (len(self.scenes), np.sum(np.asarray(self.frameCount))))

        super(DirectoryIterator_VSR_MPISintel, self).__init__(len(self.scenes), 1, shuffle, seed, infinite)

    def next(self):
        import sys
        sys.path.insert(0,'../flowlib/')
        sys.path.insert(0,'../STN/flowlib/')
        from lib import flowlib as fl



        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        batch_frame = []
        batch_flow = []
        batch_occlusion = []

        scene = self.scenes[current_index]
        frameCount = self.frameCount[current_index]
        
        i = 0
        while i < self.out_batch_size:
            maxCount = frameCount - self.nframe - 1
            startCount = np.random.randint(0, maxCount)
            _H = []
            _F = []
            _O = []
            for f in range(startCount, startCount+self.nframe):
                _H.append(_load_img_array(self.datadir + '/clean/' + scene + '/frame_{:04d}.png'.format(f+1)))
                _F.append(fl.read_flow(self.datadir + '/flow/' + scene + '/frame_{:04d}.flo'.format(f+1), verbose=False))
                _O.append(_load_img_array(self.datadir + '/occlusions/' + scene + '/frame_{:04d}.png'.format(f+1)))
            # _H.append(_load_img_array(self.datadir + '/clean/' + scene + '/frame_{:04d}.png'.format(f+2)))
            _H = np.asarray(_H) # T, H, W, 3, range: [0-1]
            _F = np.asarray(_F) # T, H, W, 2
            _O = np.asarray(_O) # T, H, W, 3

            # Random Aug
            # Rot
            # r = np.random.randint(0,4)
            # _H = np.transpose(_H, [1,2,3,0])
            # _H = np.rot90(_H, r)
            # _H = np.transpose(_H, [3,0,1,2])

            # _F = np.transpose(_F, [1,2,3,0])
            # _F = np.rot90(_F, r)
            # _F = np.transpose(_F, [3,0,1,2])
            # dx = np.copy(_F[:,:,:,0])
            # dy = np.copy(_F[:,:,:,1])
            # if r == 1:
            #     _F[:,:,:,0] = -dy
            #     _F[:,:,:,1] = dx
            # elif r == 2:
            #     _F[:,:,:,0] = -dx
            #     _F[:,:,:,1] = -dy
            # elif r == 3:
            #     _F[:,:,:,0] = dy
            #     _F[:,:,:,1] = -dx

            # _O = np.transpose(_O, [1,2,3,0])
            # _O = np.rot90(_O, r)
            # _O = np.transpose(_O, [3,0,1,2])
            # LR flip
            if np.random.random() < 0.5:
                _H = _flip_axis(_H, 2)
                _F = _flip_axis(_F, 2)
                _F[:,:,:,0] = -_F[:,:,:,0]
                _O = _flip_axis(_O, 2)
            # TD flip
            if np.random.random() < 0.5:
                _H = _flip_axis(_H, 1)
                _F = _flip_axis(_F, 1)
                _F[:,:,:,1] = -_F[:,:,:,1]
                _O = _flip_axis(_O, 1)
            # # Temporal flip
            # if np.random.random() < 0.5:
            #     _H = _flip_axis(_H, 0)  
            #     _F = _flip_axis(_F, 0)  
            #     _F = -_F
            #     _O = _flip_axis(_O, 0)  

            if self.target_size is not None:
                sh = np.random.randint(0, _H.shape[1]-self.target_size, self.crop_per_image)
                sw = np.random.randint(0, _H.shape[2]-self.target_size, self.crop_per_image)
                for c in range(0, self.crop_per_image):
                    batch_frame.append(_H[:, sh[c]:sh[c]+self.target_size, sw[c]:sw[c]+self.target_size, :])
                    batch_flow.append(_F[:, sh[c]:sh[c]+self.target_size, sw[c]:sw[c]+self.target_size, :])
                    batch_occlusion.append(_O[:, sh[c]:sh[c]+self.target_size, sw[c]:sw[c]+self.target_size, :])
                    # batch_frame.append(_H)
                    # batch_flow.append(_F)
                    # batch_occlusion.append(_O)
                    i += 1
            else:
                batch_frame.append(_H)
                batch_flow.append(_F)
                batch_occlusion.append(_O)
                i += 1
                    

        batch_frame = np.asarray(batch_frame)
        batch_flow = np.asarray(batch_flow)
        batch_occlusion = np.asarray(batch_occlusion)
        batch_frame = np.transpose(batch_frame, [0,4,1,2,3])  # B, C, T, H, W
        batch_flow = np.transpose(batch_flow, [0,4,1,2,3])  # B, C, T, H, W
        batch_occlusion = np.transpose(batch_occlusion, [0,4,1,2,3])  # B, C, T, H, W

        return batch_frame, batch_flow, batch_occlusion


    



class DirectoryIterator_FlyingChairs(Iterator):
    def __init__(self,
                H_directory,
                out_batch_size = 16,
                # mask_mode = 0,  # 0: random, 1: box
                shuffle = True,
                seed = None,
                infinite = True):

        self.out_batch_size = out_batch_size
        # self.mask_mode = mask_mode

        self.flows = glob.glob(H_directory + '/*-flow_01.flo')
        if shuffle:
            import random
            random.shuffle(self.flows)
        self.nb_sample = len(self.flows)
        print('Found %d FlyingChairs' % self.nb_sample)

        super(DirectoryIterator_FlyingChairs, self).__init__(self.nb_sample-out_batch_size, out_batch_size, shuffle, seed, infinite)

    def load_flow(self, path):
        with open(path, 'rb') as f:
            magic = float(np.fromfile(f, np.float32, count = 1)[0])
            if magic == 202021.25:
                w, h = np.fromfile(f, np.int32, count = 1)[0], np.fromfile(f, np.int32, count = 1)[0]
                data = np.fromfile(f, np.float32, count = h*w*2)
                data.resize((h, w, 2))
                return data
            return None
            
    def make_holes_static(self, rnd, img_size, min_ratio, max_ratio):
        def where_mask(mask):
            H,W = mask.shape
            ys, xs = np.where(mask > 0.5)
            if len(ys) == 0:
                return mask, 0, H, 0, W
            else:
                ymin = np.min(ys)
                ymax = np.max(ys)
                xmin = np.min(xs)
                xmax = np.max(xs)
                height = ymax - ymin + 1
                width = xmax - xmin + 1

            return ymin, ymax, xmin, xmax, height, width

        def crop_minimal(mask):
            H,W = mask.shape
            ymin, ymax, xmin, xmax, height, width = where_mask(mask)
            pad_y = 1 / 2. * height
            pad_x = 1 / 2. * width
            ymin = int(max(0, ymin-pad_y))
            ymax = int(min(H-1, ymax+pad_y))
            xmin = int(max(0, xmin-pad_x))
            xmax = int(min(W-1, xmax+pad_x))
            return mask[ymin:ymax+1, xmin:xmax+1], ymin, ymax+1, xmin, xmax+1

        shorther_edge = min(img_size[0], img_size[1])
        hmin = int(img_size[0]*min_ratio)
        hmax = int(img_size[0]*max_ratio)
        wmin = int(img_size[1]*min_ratio)
        wmax = int(img_size[1]*max_ratio)

        h1 = np.zeros(img_size, dtype=np.uint8) 
        hole_H = rnd.randint(hmin, hmax)
        hole_W = rnd.randint(wmin, wmax)
        st_H = rnd.randint(0, img_size[0]-hole_H-1)
        st_W = rnd.randint(0, img_size[1]-hole_W-1)
        h1[st_H:st_H+hole_H,st_W:st_W+hole_W] = 1

        crop_h1, st_y, end_y, st_x, end_x = crop_minimal(h1)
        # h1[st_y:end_y, st_x:end_x] = random_transform([crop_h1], rnd, rt=45, sh=45)[0]

        h2 = np.zeros(img_size, dtype=np.uint8) 
        for t in range(1000):
            h2[:] = 0
            hole_H = rnd.randint(hmin, hmax)
            hole_W = rnd.randint(wmin, wmax)
            st_H = rnd.randint(0, img_size[0]-hole_H-1)
            st_W = rnd.randint(0, img_size[1]-hole_W-1)
            h2[st_H:st_H+hole_H,st_W:st_W+hole_W] = 1

            # check 
            _, _, _, _, height, width = where_mask(h1+h2)
            if height > 0.9*shorther_edge or width > 0.9*shorther_edge:
                continue

            crop_h2, st_y, end_y, st_x, end_x = crop_minimal(h2)
            # h2[st_y:end_y, st_x:end_x] = random_transform([crop_h2], rnd, rt=45, sh=45)[0]

            # check 
            ymin, ymax, xmin, xmax, height, width = where_mask(h1+h2)
            if height > 0.9*shorther_edge or width > 0.9*shorther_edge:
                continue

            if np.sum(h1*h2) == 0:
                break
        
        return h1, h2, (height, width), t

    def next(self):
        from PIL import Image

        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        flos = []
        flo_masks = []
        img1s = []
        img2s = []
    
        for i in range(int(self.out_batch_size)):
            # 0000000-flow_01.flo
            # 0000000-img_0.png
            # 0000000-img_1.png
            # 0000000-occ_01.png

            flo_file = self.flows[current_index+i]
            flo = self.load_flow(flo_file)

            flo_mask = (np.asarray(Image.open(flo_file[:-11]+'occ_01.png')) / 255.).astype(np.float32)[:,:,np.newaxis]
            # flo_mask = (np.asarray(Image.open(flo_file[:-4]+'mask.png')) / 255.).astype(np.float32)[:,:,np.newaxis]
            img1 = (np.asarray(Image.open(flo_file[:-11]+'img_0.png')) / 255.).astype(np.float32)
            img2 = (np.asarray(Image.open(flo_file[:-11]+'img_1.png')) / 255.).astype(np.float32)

            # Random crop for square shape
            st = np.random.randint(flo.shape[1] - flo.shape[0])
            flo = flo[:,st:st+flo.shape[0]]
            flo_mask = flo_mask[:,st:st+flo.shape[0]]
            img1 = img1[:,st:st+flo.shape[0]]
            img2 = img2[:,st:st+flo.shape[0]]

            # # Uniform noise or random box
            # if self.mask_mode == 0:
            #     uniform_noise = np.random.uniform(0, 1, [flo.shape[0], flo.shape[1], 1])
            #     uniform_noise[uniform_noise > 0.2] = 1
            #     uniform_noise[uniform_noise <= 0.2] = 0
            #     uniform_noise = np.tile(uniform_noise, [1,1,2])
            #     flo_corrupted *= uniform_noise
            #     flo_corrupted[:,:,0] += np.mean(flo[:,:,0])*(1-uniform_noise[:,:,0])
            #     flo_corrupted[:,:,1] += np.mean(flo[:,:,1])*(1-uniform_noise[:,:,0])
            #     masks.append(uniform_noise[:,:,0:1])
            # elif self.mask_mode == 1:
            #     hole1, hole2, reg_size, tried = self.make_holes_static(np.random, flo.shape[:-1], min_ratio=0.1, max_ratio=0.3)
            #     hole1 = np.tile(hole1[...,np.newaxis], [1,1,2])
            #     hole2 = np.tile(hole2[...,np.newaxis], [1,1,2])
            #     hole = (1 - hole1) * (1 - hole2)
            #     flo_corrupted *= hole
            #     flo_corrupted[:,:,0] += np.mean(flo[:,:,0])*(1-hole[:,:,0])
            #     flo_corrupted[:,:,1] += np.mean(flo[:,:,1])*(1-hole[:,:,0])
            #     masks.append(hole[:,:,0:1])
                
            # For check
            # plt.subplot(1,2,1)
            # plt.imshow(flo[:,:,0])
            # plt.subplot(1,2,2)
            # plt.imshow(flo_corrupted[:,:,0])
            # plt.show(block=True)

            # # Random Rot
            # ri = np.random.randint(0,4)
            # flo = np.rot90(flo, ri)
            # flo_corrupted = np.rot90(flo_corrupted, ri)
            # img1 = np.rot90(img1, ri)
            # img2 = np.rot90(img2, ri)
            # # Random LR flip
            # if np.random.random() < 0.5:
            #     flo = _flip_axis(flo, 1)
            #     flo_corrupted = _flip_axis(flo_corrupted, 1)
            #     img1 = _flip_axis(img1, 1)
            #     img2 = _flip_axis(img2, 1)

            flos.append(flo)
            flo_masks.append(flo_mask)
            img1s.append(img1)
            img2s.append(img2)

        flos = np.stack(flos, 0).transpose((0,3,1,2))
        flo_masks = np.stack(flo_masks, 0).astype(np.float32).transpose((0,3,1,2))
        img1s = np.stack(img1s, 0).transpose((0,3,1,2))
        img2s = np.stack(img2s, 0).transpose((0,3,1,2))
        
        return flos, flo_masks, img1s, img2s




from PIL import Image
import random
import re
class DirectoryIterator_FlyingThings3D(Iterator):
    def __init__(self,
                H_directory,
                out_batch_size = 16,
                # mask_mode = 0,  # 0: random, 1: box
                shuffle = True,
                crop_size = [128,128],
                rescale_range = [2.0, 3.0],
                phase = 'Train',
                seed = None,
                infinite = True):

        self.base_directory = H_directory
        self.out_batch_size = out_batch_size
        self.phase = phase

        # optical_flow/TRAIN~TEST/A~B~C/xxxx/into_past/left~right/OpticalFlowIntoPast_0006~0015_L~R.pfm
        # optical_flow/TRAIN~TEST/A~B~C/xxxx/into_future/left~right/OpticalFlowIntoFuture_0006~0015_L~R.pfm

        # frames_cleanpass/TRAIN~TEST/A~B~C/xxxx/left~right/0006~0015.png

        # all_unused_files.txt

        f = open(H_directory + "/all_unused_files.txt", "r")
        unused_files = []
        for line in f:
            unused_files.append(line.strip())

        import os.path
        import time
        train_frames_d1 = []    # TRAIN/TEST
        train_frames_d2 = []    # A/B/C
        train_frames_d3 = []    # xxxx
        train_frames_d4 = []    # left/right
        train_frames_d5 = []    # 00xx.png
        train_d0 = sorted(glob.glob(H_directory + "/frames_cleanpass/*"))

        if(self.phase == 'Train'):
            d0 = train_d0[1]
        else: # test
            d0 = train_d0[0]

        train_d1 = glob.glob(d0+"/*")
        for d1 in train_d1:
            if os.path.isdir(d1):   # A / B / C
                train_d2 = glob.glob(d1+"/*")
                for d2 in train_d2:
                    if os.path.isdir(d2):   # xxxx                    
                        train_d3 = glob.glob(d2+"/*")
                        for d3 in train_d3:
                            if os.path.isdir(d3):   # left / right
                                train_d4 = glob.glob(d3+"/*.png")
                                for d4 in train_d4:
                                    t_d1 = d4.split('/')[-5]
                                    t_d2 = d4.split('/')[-4]
                                    t_d3 = d4.split('/')[-3]
                                    t_d4 = d4.split('/')[-2]
                                    t_d5 = d4.split('/')[-1]
                                    inner_path = t_d1+'/'+t_d2+'/'+t_d3+'/'+t_d4+'/'+t_d5
                                    if inner_path in unused_files:
                                        continue
                                    if '0006' in t_d5 or '0015' in t_d5:
                                        continue
                                    train_frames_d1.append(t_d1)
                                    train_frames_d2.append(t_d2)
                                    train_frames_d3.append(t_d3)
                                    train_frames_d4.append(t_d4)
                                    train_frames_d5.append(t_d5[:-4])

        self.train_frames_d1 = train_frames_d1
        self.train_frames_d2 = train_frames_d2
        self.train_frames_d3 = train_frames_d3
        self.train_frames_d4 = train_frames_d4
        self.train_frames_d5 = train_frames_d5

        self.crop_size = crop_size
        self.rescale_range = rescale_range

        self.nb_sample = len(train_frames_d1)
        print('Found %d FlyingThings3D frames (except %d unused frames)' % (self.nb_sample, len(unused_files)))

        import random
        if shuffle:
            combined = list(zip(self.train_frames_d1,self.train_frames_d2,self.train_frames_d3,self.train_frames_d4,self.train_frames_d5))
            random.shuffle(combined)
            self.train_frames_d1,self.train_frames_d2,self.train_frames_d3,self.train_frames_d4,self.train_frames_d5 = zip(*combined)

        super(DirectoryIterator_FlyingThings3D, self).__init__(self.nb_sample-out_batch_size, out_batch_size, False, seed, infinite)

    # PFM
    def readPFM(self, file):
        file = open(file, 'rb')

        color = None
        width = None
        height = None
        scale = None
        endian = None

        header = file.readline().rstrip()
        if header == b'PF':
            color = True
        elif header == b'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(br'^(\d+)\s(\d+)\s$', file.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        scale = float(file.readline().rstrip())
        if scale < 0: # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>' # big-endian

        data = np.fromfile(file, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data, scale

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        imgs = []
        flos_past = []
        flos_future = []
    
        for i in range(int(self.out_batch_size)):
            t_d1 = self.train_frames_d1[current_index+i]
            t_d2 = self.train_frames_d2[current_index+i]
            t_d3 = self.train_frames_d3[current_index+i]
            t_d4 = self.train_frames_d4[current_index+i]
            t_d5 = self.train_frames_d5[current_index+i]

            left_no = (int(t_d5)-1)
            if left_no < 10:
                left_no = '000'+str(left_no)
            elif left_no < 100:
                left_no = '00'+str(left_no)
            frame_left_file = self.base_directory+'/frames_cleanpass/'+t_d1+'/'+t_d2+'/'+t_d3+'/'+t_d4+'/'+left_no+'.png'
            frame_center_file = self.base_directory+'/frames_cleanpass/'+t_d1+'/'+t_d2+'/'+t_d3+'/'+t_d4+'/'+t_d5+'.png'
            right_no = (int(t_d5)+1)
            if right_no < 10:
                right_no = '000'+str(right_no)
            elif right_no < 100:
                right_no = '00'+str(right_no)
            frame_right_file = self.base_directory+'/frames_cleanpass/'+t_d1+'/'+t_d2+'/'+t_d3+'/'+t_d4+'/'+right_no+'.png'

            LR_indicator = 'L'
            if 'right' in t_d4:
                LR_indicator = 'R'
            flow_past_left_file = self.base_directory+'/optical_flow/'+t_d1+'/'+t_d2+'/'+t_d3+'/into_past/'+t_d4+'/OpticalFlowIntoPast_'+left_no+'_'+LR_indicator+'.pfm'
            flow_past_center_file = self.base_directory+'/optical_flow/'+t_d1+'/'+t_d2+'/'+t_d3+'/into_past/'+t_d4+'/OpticalFlowIntoPast_'+t_d5+'_'+LR_indicator+'.pfm'
            flow_past_right_file = self.base_directory+'/optical_flow/'+t_d1+'/'+t_d2+'/'+t_d3+'/into_past/'+t_d4+'/OpticalFlowIntoPast_'+right_no+'_'+LR_indicator+'.pfm'
            flow_future_left_file = self.base_directory+'/optical_flow/'+t_d1+'/'+t_d2+'/'+t_d3+'/into_future/'+t_d4+'/OpticalFlowIntoFuture_'+left_no+'_'+LR_indicator+'.pfm'
            flow_future_center_file = self.base_directory+'/optical_flow/'+t_d1+'/'+t_d2+'/'+t_d3+'/into_future/'+t_d4+'/OpticalFlowIntoFuture_'+t_d5+'_'+LR_indicator+'.pfm'
            flow_future_right_file = self.base_directory+'/optical_flow/'+t_d1+'/'+t_d2+'/'+t_d3+'/into_future/'+t_d4+'/OpticalFlowIntoFuture_'+right_no+'_'+LR_indicator+'.pfm'


            frame_left = (np.asarray(Image.open(frame_left_file)) / 255.).astype(np.float32)
            frame_center = (np.asarray(Image.open(frame_center_file)) / 255.).astype(np.float32)
            frame_right = (np.asarray(Image.open(frame_right_file)) / 255.).astype(np.float32)

            flow_past_left, _ = self.readPFM(flow_past_left_file)
            flow_past_center, _ = self.readPFM(flow_past_center_file)
            flow_past_right, _ = self.readPFM(flow_past_right_file)
            flow_future_left, _ = self.readPFM(flow_future_left_file)
            flow_future_center, _ = self.readPFM(flow_future_center_file)
            flow_future_right, _ = self.readPFM(flow_future_right_file)


            # Random rescale
            if self.rescale_range is not None:
                from scipy.misc import imresize
                l, u = self.rescale_range
                h,w,_ = frame_center.shape
                r  = 1/(random.uniform(l,u))

                if(self.crop_size is not None):
                    if(int(h/u) <self.crop_size[0] or int(w/u) < self.crop_size[1]):
                        eline = ('Rescaled image is smaller than crop size. Set the rescaeld value < [{:02f}]'.format( min(h/self.crop_size[0], w/self.crop_size[1])))
                        raise Exception(eline)

                frame_left = imresize(frame_left, r, interp='bilinear')
                frame_center = imresize(frame_center, r, interp='bilinear')
                frame_right = imresize(frame_right, r, interp='bilinear')

                flow_past_left = imresize(flow_past_left, r, interp='nearest') * r
                flow_past_center = imresize(flow_past_center, r, interp='nearest') * r
                flow_past_right = imresize(flow_past_right, r, interp='nearest') * r
                flow_future_left = imresize(flow_future_left, r, interp='nearest') * r
                flow_future_center = imresize(flow_future_center, r, interp='nearest') * r
                flow_future_right = imresize(flow_future_right, r, interp='nearest') * r

            # Random crop
            if self.crop_size is not None:
                th, tw = self.crop_size
                h,w,_ = frame_center.shape

                if(h-th <0 or w-tw<0):
                    raise Exception('Rescaled image is smaller than crop size.')

                h1 = np.random.randint(0, h - th)
                w1 = np.random.randint(0, w - tw)

                frame_left = frame_left[h1:(h1+th), w1:(w1+tw), :]
                frame_center = frame_center[h1:(h1+th), w1:(w1+tw), :]
                frame_right = frame_right[h1:(h1+th), w1:(w1+tw), :]
                
                flow_past_left = flow_past_left[h1:(h1+th), w1:(w1+tw), :]
                flow_past_center = flow_past_center[h1:(h1+th), w1:(w1+tw), :]
                flow_past_right = flow_past_right[h1:(h1+th), w1:(w1+tw), :]
                flow_future_left = flow_future_left[h1:(h1+th), w1:(w1+tw), :]
                flow_future_center = flow_future_center[h1:(h1+th), w1:(w1+tw), :]
                flow_future_right = flow_future_right[h1:(h1+th), w1:(w1+tw), :]
            

            imgs.append( np.stack([frame_left, frame_center, frame_right], 0) )
            flos_past.append( np.stack([flow_past_left, flow_past_center, flow_past_right], 0) )
            flos_future.append( np.stack([flow_future_left, flow_future_center, flow_future_right], 0) )

        imgs = np.stack(imgs, 0).transpose((0,4,1,2,3))
        flos_past = np.stack(flos_past, 0).transpose((0,4,1,2,3))[:,0:2]
        flos_future = np.stack(flos_future, 0).transpose((0,4,1,2,3))[:,0:2]
        
        return imgs, flos_past, flos_future




class DirectoryIterator_MSCOCO(Iterator):
    def __init__(self,
                H_directory,
                out_batch_size = 16,
                shuffle = True,
                crop_size = [128,128],
                rescale_range = [2.0, 3.0],
                seed = None,
                infinite = True):

        # self.base_directory = H_directory
        self.out_batch_size = out_batch_size

        self.files = glob.glob(H_directory+'/*.jpg')

        self.crop_size = crop_size
        self.rescale_range = rescale_range

        self.nb_sample = len(self.files)
        print('Found %d MSCOCO images' % (self.nb_sample))

        import random
        if shuffle:
            random.shuffle(self.files)

        super(DirectoryIterator_MSCOCO, self).__init__(self.nb_sample-out_batch_size, out_batch_size, False, seed, infinite)


    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)

        imgs = []
    
        def load_img(i):
            img = (np.asarray(Image.open(self.files[i])) / 255.).astype(np.float32)
            if len(img.shape) < 3:
                img = np.stack([img,img,img], 2)
            elif len(img.shape) == 3 and img.shape[2] == 1:
                img = np.concatenate([img,img,img], 2)
            return img


        for i in range(int(self.out_batch_size)):
            img = load_img(current_index+i)

            # Random rescale
            if self.rescale_range is not None:
                l, u = self.rescale_range
                h,w,_ = img.shape
                r  = 1/(random.uniform(l,u))

                if(self.crop_size is not None):
                    if(int(h/u) <self.crop_size[0] or int(w/u) < self.crop_size[1]):
                        eline = ('Rescaled image is smaller than crop size. Set the rescaeld value < [{:02f}]'.format( min(h/self.crop_size[0], w/self.crop_size[1])))
                        raise Exception(eline)
                

                from scipy.misc import imresize
                img = imresize(img, r, interp='bilinear')

            # Random crop
            if self.crop_size is not None:
                th, tw = self.crop_size
                h,w,_ = img.shape
                c = 1

                while (h-th <0 or w-tw<0):
                    img = load_img(current_index+i+c)
                    h,w,_ = img.shape
                    c += 1
                    
                if c > 1:
                    print('Image size smaller than {:d}x{:d} {:s}'.format(th,tw,self.files[current_index+i]))

                h1 = np.random.randint(0, h - th + 1)
                w1 = np.random.randint(0, w - tw + 1)

                img = img[h1:(h1+th), w1:(w1+tw), :]

            imgs.append( img )

        imgs = np.stack(imgs, 0).transpose((0,3,1,2))
        
        return imgs




#%%
class DirectoryIterator_VSR_FromHdf5(Iterator):
  def __init__(self,
               h5dir = '/mnt/hdd2/dataset/train360p_ext_h5/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               minbframe = 1,
               maxbframe = 1,  # maxbframe: temporal boundary for temporal aug.
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

    self.h5dir = h5dir
    self.nframe = nframe
    self.minbframe = minbframe
    self.maxbframe = maxbframe
    self.out_batch_size = out_batch_size

    import pickle
    import sys
    from PIL import Image
    import glob
    import os
    import h5py
    from random import shuffle

    # self.list = []
    # h5files = glob.glob(os.path.join(h5dir, "*.h5"))
    # for fname in h5files:
    #     print(fname)
    #     f = h5py.File(fname, 'r')
    #     for s in range(len(f.keys())):
    #         for c in range(2):
    #             self.list += [[fname.split('/')[-1], "{:06d}".format(s), "{:1d}".format(c)]]
    #             # print([fname.split('/')[-1], s, c])

    # with open(os.path.join(h5dir, 'index.pickle'), 'wb') as handle:
    #     pickle.dump(self.list, handle)
    # input('wait')


    # # comparison with old pickle
    # with open('/mnt/hdd2/experiments/VSR/STN/vsr_traindata_filelist_py2.pickle', 'rb') as handle:
    #     flist = pickle.load(handle, encoding='utf-8')

    # with open('/mnt/hdd2/experiments/VSR/STN/vsr_traindata_144_nframe31_cpi2_batch16_i10000_.pickle', 'rb') as handle:
    #     olist = pickle.load(handle, encoding='latin1')

    # cnt = {}
    # for l in olist:
    #     fname = flist[l['fname_idx']][-15:-4]
    #     # print(fname)
    #     if fname not in cnt:
    #         cnt[fname] = 1
    #     else:
    #         cnt[fname] += 1

    # new_cnt = {}
    # with open(os.path.join(h5dir, 'index.pickle'), 'rb') as handle:
    #     self.list = pickle.load(handle)

    # for l in self.list:
    #     fname = l[0][-18:-7]
    #     if fname not in new_cnt:
    #         new_cnt[fname] = 1
    #     else:
    #         new_cnt[fname] += 1

    # cnt_sum = 0
    # new_cnt_sum = 0
    # for key, val in cnt.items():
    #     cnt_sum += val
    #     new_cnt_sum += new_cnt[key]
    #     if val*2 != new_cnt[key]:
    #         print(key, val, new_cnt[key])
    # input('{} {}'.format(cnt_sum, new_cnt_sum))

    with open(os.path.join(h5dir, 'index.pickle'), 'rb') as handle:
        self.list = pickle.load(handle)
        
    if shuffle:
        shuffle(self.list)

    print('Found %d H videos' % len(self.list))

    self.info = None
    self.h5file = None

    super(DirectoryIterator_VSR_FromHdf5, self).__init__(len(self.list)-out_batch_size, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    
    for i in range(int(self.out_batch_size)):
      info = self.list[current_index+i]

      if self.info != info[0]:
        self.info = info[0]
        self.h5file = h5py.File(os.path.join(self.h5dir, info[0]), 'r')

      # Temporal aug.
      bframe = np.random.randint(self.minbframe, self.maxbframe+1)
      _H = []
      for f in range(0, self.nframe+(self.nframe-1)*(bframe-1), bframe):
        _H.append( self.h5file[info[1]][info[2]]["{:02d}".format(f)][()] )
      _H = np.asarray(_H)   # T, H, W, C

      # Random Aug
      # Rot
      ri = np.random.randint(0,4)
      _H = np.transpose(_H, [1,2,3,0])  # h, w, c, t
      _H = np.rot90(_H, ri)
      _H = np.transpose(_H, [3,0,1,2])  # T, H, W, C
      # LR flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 2)
      # Temporal flip
      if np.random.random() < 0.5:
        _H = _flip_axis(_H, 0)  # TxHxWxC

      batch_H.append(_H)

    batch_H = (np.asarray(batch_H) / 255.0).astype(np.float32)   # b, t, h, w, c

    return batch_H.transpose((0,4,1,2,3))





class DirectoryIterator_VSR_MPISintel_(Iterator):
    def __init__(self,
               datadir = '',
            #    target_size = 144,
               nframe = 3,
            #    crop_per_image = 2,
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

        self.datadir = datadir
        # self.target_size = target_size
        self.nframe = nframe
        # self.crop_per_image = crop_per_image
        self.out_batch_size = out_batch_size

        # self.H_shape = (self.nframe, self.target_size, self.target_size) + (3,)

        scenes = glob.glob(datadir + '/*')
        self.scenes = [s.split('/')[-1] for s in scenes]
        # self.scenes = [s for s in self.scenes if 'NT_' in s]
        self.scenes = [s for s in self.scenes if 'NT_' not in s]

        self.frameCount = []
        self.cleanFrames = []
        # self.flows = []
        # self.occlusions = []
        for s in self.scenes:
            frames = glob.glob(datadir + '/' + s + '/*.png')
            # flows = glob.glob(datadir + '/clean/' + s + '/*.png')
            # occs = glob.glob(datadir + '/clean/' + s + '/*.png')
            self.cleanFrames.append(frames)
            # self.flows.append(flows)
            # self.occlusions.append(occs)
            self.frameCount.append(len(frames))

        print('Found %d scenes, total %d frames' % (len(self.scenes), np.sum(np.asarray(self.frameCount))))

        super(DirectoryIterator_VSR_MPISintel_, self).__init__(len(self.scenes), 1, shuffle, seed, infinite)

    def next(self):
        import sys
        sys.path.insert(0,'../flowlib/')
        sys.path.insert(0,'../STN/flowlib/')
        from lib import flowlib as fl


        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel

        batch_frame = []
        # batch_flow = []
        # batch_occlusion = []

        scene = self.scenes[current_index]
        frameCount = self.frameCount[current_index]
        
        i = 0
        while i < self.out_batch_size:
            maxCount = frameCount - self.nframe + 1
            startCount = np.random.randint(0, maxCount)
            _H = []
            # _F = []
            # _O = []
            for f in range(startCount, startCount+self.nframe):
                # print(scene, startCount)
                if frameCount > 8:
                    _H.append(_load_img_array(self.datadir + '/' + scene + '/frame_{:04d}.png'.format(f+1)))
                else:
                    _H.append(_load_img_array(self.datadir + '/' + scene + '/frame{:02d}.png'.format(f+7)))
                # _F.append(fl.read_flow(self.datadir + '/flow/' + scene + '/frame_{:04d}.flo'.format(f+1), verbose=False))
                # _O.append(_load_img_array(self.datadir + '/occlusions/' + scene + '/frame_{:04d}.png'.format(f+1)))
            # _H.append(_load_img_array(self.datadir + '/clean/' + scene + '/frame_{:04d}.png'.format(f+2)))
            _H = np.asarray(_H) # T, H, W, 3, range: [0-1]
            # _F = np.asarray(_F) # T, H, W, 2
            # _O = np.asarray(_O) # T, H, W, 3

            # Random Aug
            # Rot
            # r = np.random.randint(0,4)
            # _H = np.transpose(_H, [1,2,3,0])
            # _H = np.rot90(_H, r)
            # _H = np.transpose(_H, [3,0,1,2])

            # _F = np.transpose(_F, [1,2,3,0])
            # _F = np.rot90(_F, r)
            # _F = np.transpose(_F, [3,0,1,2])
            # dx = np.copy(_F[:,:,:,0])
            # dy = np.copy(_F[:,:,:,1])
            # if r == 1:
            #     _F[:,:,:,0] = -dy
            #     _F[:,:,:,1] = dx
            # elif r == 2:
            #     _F[:,:,:,0] = -dx
            #     _F[:,:,:,1] = -dy
            # elif r == 3:
            #     _F[:,:,:,0] = dy
            #     _F[:,:,:,1] = -dx

            # _O = np.transpose(_O, [1,2,3,0])
            # _O = np.rot90(_O, r)
            # _O = np.transpose(_O, [3,0,1,2])
            # LR flip
            if np.random.random() < 0.5:
                _H = _flip_axis(_H, 2)
                # _F = _flip_axis(_F, 2)
                # _F[:,:,:,0] = -_F[:,:,:,0]
                # _O = _flip_axis(_O, 2)
            # TD flip
            if np.random.random() < 0.5:
                _H = _flip_axis(_H, 1)
                # _F = _flip_axis(_F, 1)
                # _F[:,:,:,1] = -_F[:,:,:,1]
                # _O = _flip_axis(_O, 1)
            # Temporal flip
            if np.random.random() < 0.5:
                _H = _flip_axis(_H, 0)  
            #     _F = _flip_axis(_F, 0)  
            #     _F = -_F
            #     _O = _flip_axis(_O, 0)  

            # if self.target_size is not None:
            #     sh = np.random.randint(0, _H.shape[1]-self.target_size, self.crop_per_image)
            #     sw = np.random.randint(0, _H.shape[2]-self.target_size, self.crop_per_image)
            #     for c in range(0, self.crop_per_image):
            #         batch_frame.append(_H[:, sh[c]:sh[c]+self.target_size, sw[c]:sw[c]+self.target_size, :])
            #         batch_flow.append(_F[:, sh[c]:sh[c]+self.target_size, sw[c]:sw[c]+self.target_size, :])
            #         batch_occlusion.append(_O[:, sh[c]:sh[c]+self.target_size, sw[c]:sw[c]+self.target_size, :])
            #         # batch_frame.append(_H)
            #         # batch_flow.append(_F)
            #         # batch_occlusion.append(_O)
            #         i += 1
            # else:
            batch_frame.append(_H)
            # batch_flow.append(_F)
            # batch_occlusion.append(_O)
            i += 1
                    

        batch_frame = np.asarray(batch_frame)
        # batch_flow = np.asarray(batch_flow)
        # batch_occlusion = np.asarray(batch_occlusion)
        batch_frame = np.transpose(batch_frame, [0,4,1,2,3])  # B, C, T, H, W
        # batch_flow = np.transpose(batch_flow, [0,4,1,2,3])  # B, C, T, H, W
        # batch_occlusion = np.transpose(batch_occlusion, [0,4,1,2,3])  # B, C, T, H, W

        return batch_frame #, batch_flow, batch_occlusion




from PIL import Image
class DirectoryIterator_VSR_NTIRE2019(Iterator):
  def __init__(self,
               datadir = '../../dataset/NTIRE2019_VSR/',
               target_size = 256,
               nframe = 3,
               crop_per_image = 4,
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

    self.datadir = datadir
    self.target_size = target_size
    self.nframe = nframe
    self.crop_per_image = crop_per_image
    self.out_batch_size = out_batch_size


    self.H_files = []
    self.L_files = []
    self.nb_sample = 0

    import glob
    for i in range(0, 240):
        self.H_files.append( [] )
        pngs = glob.glob(datadir + '/{:03d}/*.png'.format(i))
        pngs.sort()
        self.H_files[i] = pngs

        self.L_files.append( [] )
        pngs = glob.glob(datadir + '/X4/{:03d}/*.png'.format(i))
        pngs.sort()
        self.L_files[i] = pngs

        self.nb_sample += len(self.H_files[i])
    print('Found %d scenes, %d frames' % (240, self.nb_sample))

    super(DirectoryIterator_VSR_NTIRE2019, self).__init__(240, out_batch_size//crop_per_image, False, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    Hs = []
    Ls = []
    i = 0
    
    while len(Hs) < self.out_batch_size:
        scene = self.H_files[(current_index+i) % 240]

        total_frame = len(scene)
        sn = np.random.randint(total_frame-self.nframe)

        _Hs = []
        _Ls = []
        for f in range(sn, sn+self.nframe):
            _Hs.append( _load_img_array(self.datadir + '/{:03d}/{:08d}.png'.format(current_index, f)) )
            _Ls.append( _load_img_array(self.datadir + '/X4/{:03d}/{:08d}.png'.format(current_index, f)) )
        _H = np.asarray(_Hs)    # T H W C
        _L = np.asarray(_Ls)

        _, Lh, Lw, _ = _L.shape
        sh = np.random.randint(Lh-self.target_size//4, size=self.crop_per_image)
        sw = np.random.randint(Lw-self.target_size//4, size=self.crop_per_image)

        for c in range(self.crop_per_image):
            cH = _H[:, sh[c]*4:sh[c]*4+self.target_size, sw[c]*4:sw[c]*4+self.target_size]
            cL = _L[:, sh[c]:sh[c]+self.target_size//4, sw[c]:sw[c]+self.target_size//4]
                    
            # Motion and texture check
            dc = False
            for f in range(self.nframe-1):
                # Pass only psnr btw 15-35
                if np.mean((cH[f]-cH[f+1])**2) > 0.03 or np.mean((cH[f]-cH[f+1])**2) < 0.0003:
                    dc = True
                    break
            if dc:
                continue

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            cH = np.transpose(cH, [1,2,3,0])
            cH = np.rot90(cH, ri)
            cH = np.transpose(cH, [3,0,1,2])
            cL = np.transpose(cL, [1,2,3,0])
            cL = np.rot90(cL, ri)
            cL = np.transpose(cL, [3,0,1,2])
            # LR flip
            if np.random.random() < 0.5:
                cH = _flip_axis(cH, 2)
                cL = _flip_axis(cL, 2)
            # Temporal flip
            if np.random.random() < 0.5:
                cH = _flip_axis(cH, 0)
                cL = _flip_axis(cL, 0)

            Hs.append(cH)
            Ls.append(cL)

        i += 1

    batch_H = np.asarray(Hs)    # B T H W C
    batch_L = np.asarray(Ls)

    batch_H = np.transpose(batch_H, [0,4,1,2,3])  # BxCxTxHxW
    batch_L = np.transpose(batch_L, [0,4,1,2,3])

    return batch_H, batch_L





#%%
class DirectoryIterator_Deblur_GOPRO_Large(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/GOPRO_Large/train/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 256,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size

    import glob
    import random


    blur_pngs = []
    sharp_pngs = []

    subdirs = glob.glob(train_dir+'/*')
    subdirs.sort()
    for sd in subdirs:
        blurs = glob.glob(sd+'/blur/*.png')
        blurs.sort()
        sharps = glob.glob(sd+'/sharp/*.png')
        sharps.sort()

        for i in range(len(blurs)-nframe+1):
            bs = []
            ss = []
            for j in range(nframe):
                bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            blur_pngs.append( bs )
            sharp_pngs.append( ss )
            # blur_pngs.append( [blurs[i], blurs[i+1], blurs[i+2]] )
            # sharp_pngs.append( [sharps[i], sharps[i+1], sharps[i+2]] )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        
    self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(blur_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_Deblur_GOPRO_Large, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_blur = []
    batch_sharp = []
    
    for i in range(int(self.out_batch_size)):
        blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        B = []
        S = []
        for j in range(len(blurs)):
            B.append( _load_img_array(blurs[j]) )
            S.append( _load_img_array(sharps[j]) )
        B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C

        for j in range(4):
        # for j in range(2):    # for SDNET4
            bs = B_.shape
            sh = np.random.randint(0, bs[1]-self.crop_size+1)
            sw = np.random.randint(0, bs[2]-self.crop_size+1)
            B = B_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            B = np.rot90(B, ri)
            B = np.transpose(B, [3,0,1,2])  # T, H, W, C
            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            batch_blur.append(B)
            batch_sharp.append(S)

    batch_blur = np.stack(batch_blur, 0).astype(np.float32)
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_blur.transpose((0,4,1,2,3)), batch_sharp.transpose((0,4,1,2,3))



#%%
class DirectoryIterator_Deblur_DVD(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/DeepVideoDeblurring_Dataset/quantitative_datasets',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 256,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size

    import glob
    import random


    blur_pngs = []
    sharp_pngs = []
    TEST_SETS = [
        'IMG_0030',
        'IMG_0049',
        'IMG_0021',
        '720p_240fps_2',
        'IMG_0032',
        'IMG_0033',
        'IMG_0031',
        'IMG_0003',
        'IMG_0039',
        'IMG_0037',
    ]

    subdirs = glob.glob(train_dir+'/*')
    subdirs.sort()
    for sd in subdirs:
        if sd.split('/')[-1] in TEST_SETS:
            continue

        blurs = glob.glob(sd+'/input/*.jpg')
        blurs.sort()
        sharps = glob.glob(sd+'/GT/*.jpg')
        sharps.sort()

        for i in range(len(blurs)-nframe+1):
            bs = []
            ss = []
            for j in range(nframe):
                bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            blur_pngs.append( bs )
            sharp_pngs.append( ss )
            # blur_pngs.append( [blurs[i], blurs[i+1], blurs[i+2]] )
            # sharp_pngs.append( [sharps[i], sharps[i+1], sharps[i+2]] )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        
    self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(blur_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_Deblur_DVD, self).__init__(self.total_count, out_batch_size//4, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_blur = []
    batch_sharp = []
    
    for i in range(int(self.out_batch_size)):
        blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        B = []
        S = []
        for j in range(len(blurs)):
            B.append( _load_img_array(blurs[j]) )
            S.append( _load_img_array(sharps[j]) )
        B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C

        for j in range(4):
        # for j in range(2):    # for SDNET4
            bs = B_.shape
            sh = np.random.randint(0, bs[1]-self.crop_size+1)
            sw = np.random.randint(0, bs[2]-self.crop_size+1)
            B = B_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            B = np.rot90(B, ri)
            B = np.transpose(B, [3,0,1,2])  # T, H, W, C
            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            batch_blur.append(B)
            batch_sharp.append(S)

    batch_blur = np.stack(batch_blur, 0).astype(np.float32)
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_blur.transpose((0,4,1,2,3)), batch_sharp.transpose((0,4,1,2,3))



#%%
class DirectoryIterator_Deblur_REDS(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/REDS/train/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 256,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size

    import glob
    import random


    blur_pngs = []
    sharp_pngs = []

    subdirs = glob.glob(train_dir+'/train_blur/*')
    subdirs.sort()
    for sd in subdirs:
        # exclude REDS4
        if sd.split("/")[-1] in ['000','011','015','020']:
            continue

        blurs = glob.glob(sd+'/*.png')
        blurs.sort()
        sharps = glob.glob(sd.replace("train_blur","train_sharp",1)+'/*.png')
        sharps.sort()

        for i in range(len(blurs)-nframe+1):
            bs = []
            ss = []
            for j in range(nframe):
                if blurs[i+j].split("/")[-1] != sharps[i+j].split("/")[-1]:
                    print("DATASET ERROR {} {}".format(blurs[i+j].split("/")[-1], sharps[i+j].split("/")[-1]))
                bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            blur_pngs.append( bs )
            sharp_pngs.append( ss )
            # blur_pngs.append( [blurs[i], blurs[i+1], blurs[i+2]] )
            # sharp_pngs.append( [sharps[i], sharps[i+1], sharps[i+2]] )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        
    self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(blur_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_Deblur_REDS, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_blur = []
    batch_sharp = []
    
    for i in range(int(self.out_batch_size)):
        blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        B = []
        S = []

        # try:
        for j in range(len(blurs)):
            B.append( _load_img_array(blurs[j]) )
            S.append( _load_img_array(sharps[j]) )
        B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C
        # except:



        for j in range(4):
            bs = B_.shape
            sh = np.random.randint(0, bs[1]-self.crop_size+1)
            sw = np.random.randint(0, bs[2]-self.crop_size+1)
            B = B_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            B = np.rot90(B, ri)
            B = np.transpose(B, [3,0,1,2])  # T, H, W, C
            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            batch_blur.append(B)
            batch_sharp.append(S)

    batch_blur = np.stack(batch_blur, 0).astype(np.float32)
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_blur.transpose((0,4,1,2,3)), batch_sharp.transpose((0,4,1,2,3))




#%%
class DirectoryIterator_VSR_REDS(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/REDS/train/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 64,
               scale_factor=4,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size
    self.r = scale_factor

    import glob
    import random


    blur_pngs = []
    sharp_pngs = []

    subdirs = glob.glob(train_dir+'/X4/*')
    subdirs.sort()
    for sd in subdirs:
        # exclude REDS4
        if sd.split("/")[-1] in ['000','011','015','020']:
            continue

        blurs = glob.glob(sd+'/*.png')
        blurs.sort()
        sharps = glob.glob(sd.replace("X4","train_sharp",1)+'/*.png')
        sharps.sort()

        for i in range(len(blurs)-nframe+1):
            bs = []
            ss = []
            for j in range(nframe):
                if blurs[i+j].split("/")[-1] != sharps[i+j].split("/")[-1]:
                    print("DATASET ERROR {} {}".format(blurs[i+j].split("/")[-1], sharps[i+j].split("/")[-1]))
                bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            blur_pngs.append( bs )
            sharp_pngs.append( ss )
            # blur_pngs.append( [blurs[i], blurs[i+1], blurs[i+2]] )
            # sharp_pngs.append( [sharps[i], sharps[i+1], sharps[i+2]] )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        
    self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(blur_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_VSR_REDS, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_blur = []
    batch_sharp = []
    
    i = 0
    while len(batch_blur) < int(self.out_batch_size)*4:
    # for i in range(int(self.out_batch_size)):
        blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]
        i += 1

        B = []
        S = []
        try:
            for j in range(len(blurs)):
                B.append( _load_img_array(blurs[j]) )
                S.append( _load_img_array(sharps[j]) )
        except:
            print("File open error: {} {}".format(blurs[j], sharps[j]))
            continue
            # raise
        B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C

        for j in range(4):
            bs = B_.shape
            sh = np.random.randint(0, bs[1]-self.crop_size+1)
            sw = np.random.randint(0, bs[2]-self.crop_size+1)
            B = B_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[:, sh*self.r:(sh+self.crop_size)*self.r, sw*self.r:(sw+self.crop_size)*self.r]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            B = np.rot90(B, ri)
            B = np.transpose(B, [3,0,1,2])  # T, H, W, C
            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            batch_blur.append(B)
            batch_sharp.append(S)

    batch_blur = np.stack(batch_blur, 0).astype(np.float32)
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_blur.transpose((0,4,1,2,3)), batch_sharp.transpose((0,4,1,2,3))







class DirectoryIterator_DIV2K(Iterator):
  def __init__(self,
            #    H_directory,
               datadir = '/mnt/hdd1/DIV2K/',
                crop_size = 32,
                crop_per_image = 4,
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

    self.crop_size = crop_size
    self.out_batch_size = out_batch_size
    self.crop_per_image = crop_per_image
    self.datadir = datadir
    self.r = 4

    import glob
    import random

    lrs = glob.glob(datadir+'/DIV2K_train_LR_bicubic/X4/*.png')
    lrs.sort()

    sharps = glob.glob(datadir+'/DIV2K_train_HR/*.png')
    sharps.sort()

    if len(lrs) != len(sharps):
        print("file count mismatch")
        raise

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        lrs, sharps = shuffle_list(lrs, sharps)
        
    self.lr_pngs = lrs
    self.sharp_pngs = sharps
    self.total_count = len(lrs)

    print('Found %d images' % self.total_count)

    super(DirectoryIterator_DIV2K, self).__init__(self.total_count, out_batch_size//crop_per_image, shuffle, seed, infinite)


  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_blur = []
    batch_sharp = []
    
    i = 0
    while (len(batch_blur) < self.out_batch_size):
        blurs = self.lr_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        try:
            B_ = _load_img_array(blurs) 
            S_ = _load_img_array(sharps) 
        except:
            print("File open error: {} {}".format(blurs, sharps))
            raise

        for j in range(self.crop_per_image):
            if (len(batch_blur) > self.out_batch_size):
                break

            bs = B_.shape   # h, w, c
            sh = np.random.randint(0, bs[0]-self.crop_size+1)
            sw = np.random.randint(0, bs[1]-self.crop_size+1)
            B = B_[sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[sh*self.r:(sh+self.crop_size)*self.r, sw*self.r:(sw+self.crop_size)*self.r]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            B = np.rot90(B, ri)
            S = np.rot90(S, ri)

            # LR flip
            if np.random.random() < 0.5:
                B = _flip_axis(B, 1)
                S = _flip_axis(S, 1)

            batch_blur.append(B)
            batch_sharp.append(S)

        i += 1
        
    batch_blur = np.stack(batch_blur, 0).astype(np.float32) # BxHxWxC
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_blur.transpose((0,3,1,2)), batch_sharp.transpose((0,3,1,2))





import torchvision

class DirectoryIterator_NTIRE2020(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/NTIRE2020/train/',
               out_batch_size = 16,
               crop_size= 64,
               scale_factor=4,
               crop_per_frame=4,
               aug_gt=False,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.crop_size = crop_size
    self.crop_per_frame = crop_per_frame
    self.out_batch_size = out_batch_size
    self.r = scale_factor
    self.aug_gt = aug_gt
    seed = time.time()

    import glob
    import random


    gt_files = glob.glob(train_dir+'/*.png')
    gt_files.sort()

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        # blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        random.shuffle(gt_files)
        
    self.total_count = len(gt_files)
    self.gt_files = gt_files

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_NTIRE2020, self).__init__(self.total_count, int(out_batch_size/crop_per_frame), shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_sharp = []
    
    i = 0
    while len(batch_sharp) < int(self.out_batch_size):
        sharps = self.gt_files[(current_index+i) % self.total_count]
        i += 1

        try:
            S_ = _load_img_array(sharps) # H,W,C
        except:
            print("File open error: {}".format(sharps))
            continue

        ss = S_.shape
        sh = np.random.randint(0, ss[0]-self.crop_size+1, self.crop_per_frame)
        sw = np.random.randint(0, ss[1]-self.crop_size+1, self.crop_per_frame)

        for j in range(self.crop_per_frame):
            S = S_[sh[j]:(sh[j]+self.crop_size), sw[j]:(sw[j]+self.crop_size)]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            S = np.rot90(S, ri)

            # LR flip
            if np.random.random() < 0.5:
                S = _flip_axis(S, 1)

            augs = []
            augs += [ S ]

            if self.aug_gt > 1:
                aug_cnt = 0
                S_pil = PIL.Image.fromarray((S*255).astype(np.uint8))

                while aug_cnt < self.aug_gt:
                    for ang in [0, -0.1, 0.1]:
                        if aug_cnt >= self.aug_gt:
                            break
                        for she in [0, -0.1, 0.1]:
                            if aug_cnt >= self.aug_gt:
                                break
                            for s in [100, 99, 101]:
                                if aug_cnt >= self.aug_gt:
                                    break
                                for x in [-0.5, 0.5, -1, 1]:
                                    if aug_cnt >= self.aug_gt:
                                        break
                                    for y in [-0.5, 0.5, -1, 1]:
                                        if aug_cnt >= self.aug_gt:
                                            break

                                        if np.random.rand() < 0.5:
                                            continue

                                        # angle (sin, cos), translate, scale, shear (tan),
                                        imgt = torchvision.transforms.functional.affine(
                                            S_pil, ang, [x,y], s/100, she, resample=PIL.Image.BICUBIC, fillcolor=None
                                        )

                                        augs += [ np.array(imgt) ]
                                        aug_cnt += 1

            batch_sharp.append( np.stack(augs, 0).astype(np.float32)/255.0 ) 

    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32).transpose((0,4,1,2,3))   # B, C, Aug, H, W

    return  batch_sharp




#%%
class DirectoryIterator_VSR_REDS_upx8(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/REDS/train/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 64,
               scale_factor=4,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size
    self.r = scale_factor

    import glob
    import random


    # blur_pngs = []
    sharp_pngs = []

    subdirs = glob.glob(train_dir+'/X4/*')
    subdirs.sort()
    for sd in subdirs:
        # exclude REDS4
        if sd.split("/")[-1] in ['000','011','015','020']:
            continue

        # blurs = glob.glob(sd+'/*.png')
        # blurs.sort()
        sharps = glob.glob(sd.replace("X4","train_sharp",1)+'/*.png')
        sharps.sort()

        for i in range(len(sharps)-nframe+1):
            # bs = []
            ss = []
            for j in range(nframe):
                # if blurs[i+j].split("/")[-1] != sharps[i+j].split("/")[-1]:
                #     print("DATASET ERROR {} {}".format(blurs[i+j].split("/")[-1], sharps[i+j].split("/")[-1]))
                # bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            # blur_pngs.append( bs )
            sharp_pngs.append( ss )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        # sharp_pngs = shuffle_list(sharp_pngs)
        random.shuffle(sharp_pngs)
        
    # self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(sharp_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_VSR_REDS_upx8, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    # batch_blur = []
    batch_sharp = []
    
    for i in range(int(self.out_batch_size)):
        # blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        # B = []
        S = []
        for j in range(len(sharps)):
            # B.append( _load_img_array(blurs[j]) )
            S.append( _load_img_array(sharps[j]) )
        # B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C

        for j in range(4):
            bs = S_.shape
            sh = np.random.randint(0, bs[1]-self.crop_size*self.r+1)
            sw = np.random.randint(0, bs[2]-self.crop_size*self.r+1)
            # B = B_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[:, sh:sh+self.crop_size*self.r, sw:sw+self.crop_size*self.r]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            # B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            # B = np.rot90(B, ri)
            # B = np.transpose(B, [3,0,1,2])  # T, H, W, C
            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                # B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                # B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            # batch_blur.append(B)
            batch_sharp.append(S)

    # batch_blur = np.stack(batch_blur, 0).astype(np.float32)
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_sharp.transpose((0,4,1,2,3))



import PIL
from PIL import Image
from PIL import ImageFilter
#%%
class DirectoryIterator_VSR_REDS_For_Inpainting(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/REDS/train/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 256,
            #    scale_factor=4,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size
    # self.r = scale_factor

    import glob
    import random


    # blur_pngs = []
    sharp_pngs = []

    subdirs = glob.glob(train_dir+'/train_sharp/*')
    subdirs.sort()
    for sd in subdirs:
        # exclude REDS4
        if sd.split("/")[-1] in ['000','011','015','020']:
            continue

        sharps = glob.glob(sd+'/*.png')
        sharps.sort()
        # sharps = glob.glob(sd.replace("X4","train_sharp",1)+'/*.png')
        # sharps.sort()

        for i in range(len(sharps)-nframe+1):
            # bs = []
            ss = []
            for j in range(nframe):
                # if blurs[i+j].split("/")[-1] != sharps[i+j].split("/")[-1]:
                #     print("DATASET ERROR {} {}".format(blurs[i+j].split("/")[-1], sharps[i+j].split("/")[-1]))
                # bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            # blur_pngs.append( bs )
            sharp_pngs.append( ss )
            # blur_pngs.append( [blurs[i], blurs[i+1], blurs[i+2]] )
            # sharp_pngs.append( [sharps[i], sharps[i+1], sharps[i+2]] )

    if shuffle:
        # def shuffle_list(*ls):
        #     l = list(zip(*ls))
        #     random.shuffle(l)
        #     return zip(*l)

        # blur_pngs, sharp_pngs = shuffle_list(blur_pngs, sharp_pngs)
        random.shuffle(sharp_pngs)

    # self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(sharp_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_VSR_REDS_For_Inpainting, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_mask = []
    batch_sharp = []
    
    for i in range(int(self.out_batch_size)):
        # blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        # M = []
        S = []
        for j in range(len(sharps)):
            try:
                # B.append( _load_img_array(blurs[j]) )
                S.append( _load_img_array(sharps[j]) )
            except:
                print("File open error: {}".format(sharps[j]))
                raise
        # B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C

        for j in range(4):
            bs = S_.shape
            sh = np.random.randint(0, bs[1]-self.crop_size+1)
            sw = np.random.randint(0, bs[2]-self.crop_size+1)
            # B = B_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            S = S_[:, sh:(sh+self.crop_size), sw:(sw+self.crop_size)]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            # B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            # B = np.rot90(B, ri)
            # B = np.transpose(B, [3,0,1,2])  # T, H, W, C
            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                # B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                # B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            # random MASK
            M = np.zeros((self.crop_size, self.crop_size)).astype(np.uint8)
            sh, sw = random.randint(0, self.crop_size//2), random.randint(0, self.crop_size//2)
            M[sh:sh+self.crop_size//2, sw:sw+self.crop_size//2] = 255

            # mask aug
            M = Image.fromarray(M).rotate(np.random.randint(0,45), resample=Image.NEAREST, expand=True)
            M = M.filter(ImageFilter.MaxFilter(3))
            M = M.resize((self.crop_size, self.crop_size), resample=Image.NEAREST)

            batch_mask.append(np.asarray(M))
            batch_sharp.append(S)

    batch_mask = np.stack(batch_mask, 0).astype(np.float32)[:, np.newaxis, ...] # B, C, H, W
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_mask/255, batch_sharp.transpose((0,4,1,2,3))*2-1.





#%%
class DirectoryIterator_Prediction_CalPed(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/calped/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 2,
               out_batch_size = 16,
               crop_size= 256,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size

    import glob
    import random

    input_pngs = []
    output_pngs = []

    for i in range(6):
        subdirs = glob.glob(train_dir+'/set{:02d}/*'.format(i))
        subdirs.sort()

        for sd in subdirs:
            jpgs = glob.glob(sd+'/*.jpg')
            jpgs.sort()

            for j in range(len(jpgs)-nframe):
                input_pngs.append( [jpgs[j], jpgs[j+1]] )
                output_pngs.append( [jpgs[j+2]] )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        input_pngs, output_pngs = shuffle_list(input_pngs, output_pngs)
        
    self.input_pngs = input_pngs
    self.output_pngs = output_pngs
    self.total_count = len(input_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_Prediction_CalPed, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_input = []
    batch_output = []
    
    for i in range(int(self.out_batch_size)):
        inputs = self.input_pngs[(current_index+i) % self.total_count]
        outputs = self.output_pngs[(current_index+i) % self.total_count]

        I = []
        O = []
        for j in range(len(inputs)):
            I.append( _load_img_array(inputs[j]) )
        for j in range(len(outputs)):
            O.append( _load_img_array(outputs[j]) )
        I_ = np.stack(I, 0)
        O_ = np.stack(O, 0)  # B H W C

        # Crop boundary (black area)
        I_ = I_[:, 10:-10, 10:-10, :]
        O_ = O_[:, 10:-10, 10:-10, :]

        for j in range(4):
            Is = I_.shape
            sh = np.random.randint(0, Is[1]-self.crop_size+1)
            sw = np.random.randint(0, Is[2]-self.crop_size+1)
            I = I_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            O = O_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            I = np.transpose(I, [1,2,3,0])  # h, w, c, t
            I = np.rot90(I, ri)
            I = np.transpose(I, [3,0,1,2])  # T, H, W, C
            O = np.transpose(O, [1,2,3,0])  # h, w, c, t
            O = np.rot90(O, ri)
            O = np.transpose(O, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                I = _flip_axis(I, 2)
                O = _flip_axis(O, 2)

            batch_input.append(I)
            batch_output.append(O)

    batch_input = np.stack(batch_input, 0).astype(np.float32)
    batch_output = np.stack(batch_output, 0).astype(np.float32)

    return batch_input.transpose((0,4,1,2,3)), batch_output.transpose((0,4,1,2,3))



#%%
class DirectoryIterator_Prediction_CalPed_(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/calped/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 2,
               out_batch_size = 16,
               crop_size= 256,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size

    import glob
    import random

    input_pngs = []
    # output_pngs = []

    for i in range(6):
        subdirs = glob.glob(train_dir+'/set{:02d}/*'.format(i))
        subdirs.sort()

        for sd in subdirs:
            jpgs = glob.glob(sd+'/*.jpg')
            jpgs.sort()

            for j in range(len(jpgs)-nframe):
                inp = []

                for k in range(nframe):
                    inp.append( jpgs[k] )

                input_pngs.append( inp )
                # output_pngs.append( [jpgs[j+2]] )

    if shuffle:
        # def shuffle_list(*ls):
        #     l = list(zip(*ls))
        #     random.shuffle(l)
        #     return zip(*l)

        # input_pngs, output_pngs = shuffle_list(input_pngs, output_pngs)
        random.shuffle(input_pngs)
        
    self.input_pngs = input_pngs
    # self.output_pngs = output_pngs
    self.total_count = len(input_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_Prediction_CalPed_, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_input = []
    # batch_output = []
    
    for i in range(int(self.out_batch_size)):
        inputs = self.input_pngs[(current_index+i) % self.total_count]
        # outputs = self.output_pngs[(current_index+i) % self.total_count]

        I = []
        # O = []
        for j in range(len(inputs)):
            I.append( _load_img_array(inputs[j]) )
        # for j in range(len(outputs)):
        #     O.append( _load_img_array(outputs[j]) )
        I_ = np.stack(I, 0)
        # O_ = np.stack(O, 0)  # B H W C

        # Crop boundary (black area)
        I_ = I_[:, 10:-10, 10:-10, :]
        # O_ = O_[:, 10:-10, 10:-10, :]

        for j in range(4):
            Is = I_.shape
            sh = np.random.randint(0, Is[1]-self.crop_size+1)
            sw = np.random.randint(0, Is[2]-self.crop_size+1)
            I = I_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            # O = O_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            I = np.transpose(I, [1,2,3,0])  # h, w, c, t
            I = np.rot90(I, ri)
            I = np.transpose(I, [3,0,1,2])  # T, H, W, C
            # O = np.transpose(O, [1,2,3,0])  # h, w, c, t
            # O = np.rot90(O, ri)
            # O = np.transpose(O, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                I = _flip_axis(I, 2)
                # O = _flip_axis(O, 2)

            batch_input.append(I)
            # batch_output.append(O)

    batch_input = np.stack(batch_input, 0).astype(np.float32)
    # batch_output = np.stack(batch_output, 0).astype(np.float32)

    return batch_input.transpose((0,4,1,2,3))




#%%
class DirectoryIterator_Inpainting_CalPed(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/calped/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 2,
               out_batch_size = 16,
               crop_size= 256,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size

    import glob
    import random

    input_pngs = []

    for i in range(6):
        subdirs = glob.glob(train_dir+'/set{:02d}/*'.format(i))
        subdirs.sort()

        for sd in subdirs:
            jpgs = glob.glob(sd+'/*.jpg')
            jpgs.sort()

            for i in range(len(jpgs)-nframe):
                t = []
                for j in range(nframe):
                    t.append( jpgs[i+j] )
                input_pngs.append( t )

    if shuffle:
        # def shuffle_list(*ls):
        #     l = list(zip(*ls))
        #     random.shuffle(l)
        #     return zip(*l)

        random.shuffle(input_pngs)
        
    self.input_pngs = input_pngs
    self.total_count = len(input_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_Inpainting_CalPed, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_input = []
    batch_mask = []
    batch_output = []
    
    for i in range(int(self.out_batch_size)):
        inputs = self.input_pngs[(current_index+i) % self.total_count]

        I = []
        for j in range(len(inputs)):
            I.append( _load_img_array(inputs[j]) )
        I_ = np.stack(I, 0)

        # Crop boundary (black area)
        I_ = I_[:, 10:-10, 10:-10, :]

        for j in range(4):
            Is = I_.shape
            sh = np.random.randint(0, Is[1]-self.crop_size+1)
            sw = np.random.randint(0, Is[2]-self.crop_size+1)
            I = np.copy(I_[:, sh:sh+self.crop_size, sw:sw+self.crop_size])
            O = np.copy(I)

            # Mask
            M = np.zeros_like(I)    # T, H, W, C
            sh = np.random.randint(0, self.crop_size-self.crop_size//4)
            sw = np.random.randint(0, self.crop_size-self.crop_size//4)
            M[:, sh:sh+self.crop_size//4, sw:sw+self.crop_size//4] = 1

            I[M==1] = 0.5

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            I = np.transpose(I, [1,2,3,0])  # h, w, c, t
            I = np.rot90(I, ri)
            I = np.transpose(I, [3,0,1,2])  # T, H, W, C
            O = np.transpose(O, [1,2,3,0])  # h, w, c, t
            O = np.rot90(O, ri)
            O = np.transpose(O, [3,0,1,2])  # T, H, W, C
            M = np.transpose(M, [1,2,3,0])  # h, w, c, t
            M = np.rot90(M, ri)
            M = np.transpose(M, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                I = _flip_axis(I, 2)
                O = _flip_axis(O, 2)
                M = _flip_axis(M, 2)

            batch_input.append(I)
            batch_mask.append(M)
            batch_output.append(O)

    batch_input = np.stack(batch_input, 0).astype(np.float32)
    batch_mask = np.stack(batch_mask, 0).astype(np.float32)
    batch_output = np.stack(batch_output, 0).astype(np.float32)

    return batch_input.transpose((0,4,1,2,3)), batch_mask.transpose((0,4,1,2,3)), batch_output.transpose((0,4,1,2,3))






#%%
class DirectoryIterator_PennAction_FromHdf5(Iterator):
  def __init__(self,
               h5dir = '/host/media/ssd1/users/yhjo/PennAction/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

    self.h5dir = h5dir
    self.nframe = nframe
    self.out_batch_size = out_batch_size

    # import pickle
    # import sys
    # from PIL import Image
    import glob
    # import os
    # import h5py
    from random import shuffle

    sequences = glob.glob(h5dir+"/*.h5")
    sequences.sort()
    self.list = sequences
        
    if shuffle:
        shuffle(self.list)

    print('Found %d H videos' % len(self.list))

    super(DirectoryIterator_PennAction_FromHdf5, self).__init__(len(self.list)-out_batch_size, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    
    for i in range(int(self.out_batch_size)):
        filename = self.list[current_index+i]
        frames = h5py.File(filename, 'r')['frames']
        total_frames = frames.shape[0]

        start_frame = np.random.randint(total_frames - self.nframe)
        _H = []
        for f in range(start_frame, start_frame+self.nframe):
            _H.append( frames[f] )
        _H = np.asarray(_H)   # T, H, W, C

        # Random Aug
        # # Rot
        # ri = np.random.randint(0,4)
        # _H = np.transpose(_H, [1,2,3,0])  # h, w, c, t
        # _H = np.rot90(_H, ri)
        # _H = np.transpose(_H, [3,0,1,2])  # T, H, W, C
        # LR flip
        if np.random.random() < 0.5:
            _H = _flip_axis(_H, 2)
        # Temporal flip
        if np.random.random() < 0.5:
            _H = _flip_axis(_H, 0)  # TxHxWxC

        batch_H.append(_H)

    batch_H = (np.array(batch_H) / 255.0).astype(np.float32)   # b, t, h, w, c

    return batch_H.transpose((0,4,1,2,3))   # b, c, t, h, w



#%%
class DirectoryIterator_UCF101_FromHdf5(Iterator):
  def __init__(self,
               h5dir = '/host/media/ssd1/users/yhjo/UCF101/',
               c = None,
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               shuffle = True,
               seed = None,
               infinite = True):

    self.h5dir = h5dir
    self.nframe = nframe
    self.out_batch_size = out_batch_size

    # import pickle
    # import sys
    # from PIL import Image
    import glob
    # import os
    # import h5py
    import random

    if c is None:
        sequences = glob.glob(h5dir+"/*.h5")
    else:
        sequences = glob.glob(h5dir+"/*{}*.h5".format(c))
        sequences = [s for s in sequences if "c01" not in s]
    sequences.sort()
    self.list = sequences
        
    if shuffle:
        random.seed(int((time.time()*1000)%1000000))
        random.shuffle(self.list)

    self.total = len(self.list)
    print('Found %d H videos' % len(self.list))

    super(DirectoryIterator_UCF101_FromHdf5, self).__init__(len(self.list)-out_batch_size, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_H = []
    i = 0

    while(len(batch_H) < self.out_batch_size):
        filename = self.list[(current_index+i) % self.total]
        frames = h5py.File(filename, 'r')['frames']
        total_frames, H, W, C = frames.shape

        if H != 240 or W != 320 or C != 3:
            print("weird data", filename, frames.shape)
            i += 1
            continue

        start_frame = np.random.randint(total_frames - self.nframe)
        _H = []
        for f in range(start_frame, start_frame+self.nframe):
            _H.append( frames[f] )
        _H = np.asarray(_H)   # T, H, W, C

        # Random Aug
        # # Rot
        # ri = np.random.randint(0,4)
        # _H = np.transpose(_H, [1,2,3,0])  # h, w, c, t
        # _H = np.rot90(_H, ri)
        # _H = np.transpose(_H, [3,0,1,2])  # T, H, W, C
        # LR flip
        if np.random.random() < 0.5:
            _H = _flip_axis(_H, 2)
        # Temporal flip
        if np.random.random() < 0.5:
            _H = _flip_axis(_H, 0)  # TxHxWxC

        batch_H.append(_H)

        i += 1

    # print(len(batch_H))
    # print(np.asarray(batch_H).shape)
    batch_H = (np.asarray(batch_H) / 255.0).astype(np.float32)   # b, t, h, w, c

    return batch_H.transpose((0,4,1,2,3))   # b, c, t, h, w





#%%
class DirectoryIterator_VSR_Deblur_REDS(Iterator):
  def __init__(self,
               train_dir = '/host/media/ssd1/users/yhjo/dataset/REDS/train/',
            #    listfile = 'vsr_traindata_filelist.pickle',
            #    datafile = 'vsr_traindata_144_nframe31_cpi2_batch16_i10000.pickle',
            #    datadir = '/hdd2/datasets/VSR/train360p_ext/',
               nframe = 3,
               out_batch_size = 16,
               crop_size= 64,
               scale_factor=4,
               shuffle = True,
               seed = None,
               infinite = True):

    self.train_dir = train_dir
    self.nframe = nframe
    self.crop_size = crop_size
    self.out_batch_size = out_batch_size
    self.r = scale_factor

    import glob
    import random


    lr_pngs = []
    blur_pngs = []
    sharp_pngs = []

    subdirs = glob.glob(train_dir+'/X4/*')
    subdirs.sort()
    for sd in subdirs:
        # exclude REDS4
        if sd.split("/")[-1] in ['000','011','015','020']:
            continue

        lrs = glob.glob(sd+'/*.png')
        lrs.sort()
        blurs = glob.glob(sd.replace("X4","train_blur",1)+'/*.png')
        blurs.sort()
        sharps = glob.glob(sd.replace("X4","train_sharp",1)+'/*.png')
        sharps.sort()

        for i in range(len(blurs)-nframe+1):
            ls = []
            bs = []
            ss = []
            for j in range(nframe):
                if blurs[i+j].split("/")[-1] != sharps[i+j].split("/")[-1]:
                    print("DATASET ERROR {} {}".format(blurs[i+j].split("/")[-1], sharps[i+j].split("/")[-1]))
                if lrs[i+j].split("/")[-1] != sharps[i+j].split("/")[-1]:
                    print("DATASET ERROR {} {}".format(lrs[i+j].split("/")[-1], sharps[i+j].split("/")[-1]))
                if lrs[i+j].split("/")[-1] != blurs[i+j].split("/")[-1]:
                    print("DATASET ERROR {} {}".format(lrs[i+j].split("/")[-1], blurs[i+j].split("/")[-1]))
                ls.append( lrs[i+j] )
                bs.append( blurs[i+j] )
                ss.append( sharps[i+j] )
            lr_pngs.append( ls )
            blur_pngs.append( bs )
            sharp_pngs.append( ss )
            # blur_pngs.append( [blurs[i], blurs[i+1], blurs[i+2]] )
            # sharp_pngs.append( [sharps[i], sharps[i+1], sharps[i+2]] )

    if shuffle:
        def shuffle_list(*ls):
            l = list(zip(*ls))
            random.shuffle(l)
            return zip(*l)

        lr_pngs, blur_pngs, sharp_pngs = shuffle_list(lr_pngs, blur_pngs, sharp_pngs)
        
    self.lr_pngs = lr_pngs
    self.blur_pngs = blur_pngs
    self.sharp_pngs = sharp_pngs
    self.total_count = len(blur_pngs)

    print('Found %d trainnig samples' % self.total_count)

    super(DirectoryIterator_VSR_Deblur_REDS, self).__init__(self.total_count, out_batch_size, shuffle, seed, infinite)

  def next(self):
    with self.lock:
        index_array, current_index, current_batch_size = next(self.index_generator)
    # The transformation of images is not under thread lock so it can be done in parallel

    batch_lr = []
    batch_blur = []
    batch_sharp = []
    
    for i in range(int(self.out_batch_size)):
        lrs = self.lr_pngs[(current_index+i) % self.total_count]
        blurs = self.blur_pngs[(current_index+i) % self.total_count]
        sharps = self.sharp_pngs[(current_index+i) % self.total_count]

        L = []
        B = []
        S = []
        for j in range(len(blurs)):
            L.append( _load_img_array(lrs[j]) )
            B.append( _load_img_array(blurs[j]) )
            S.append( _load_img_array(sharps[j]) )
        L_ = np.stack(L, 0)
        B_ = np.stack(B, 0)
        S_ = np.stack(S, 0)  # B H W C

        for j in range(4):
            ls = L_.shape
            sh = np.random.randint(0, ls[1]-self.crop_size+1)
            sw = np.random.randint(0, ls[2]-self.crop_size+1)
            L = L_[:, sh:sh+self.crop_size, sw:sw+self.crop_size]
            B = B_[:, sh*self.r:(sh+self.crop_size)*self.r, sw*self.r:(sw+self.crop_size)*self.r]
            S = S_[:, sh*self.r:(sh+self.crop_size)*self.r, sw*self.r:(sw+self.crop_size)*self.r]

            # Random Aug
            # Rot
            ri = np.random.randint(0,4)
            L = np.transpose(L, [1,2,3,0])  # h, w, c, t
            L = np.rot90(L, ri)
            L = np.transpose(L, [3,0,1,2])  # T, H, W, C

            B = np.transpose(B, [1,2,3,0])  # h, w, c, t
            B = np.rot90(B, ri)
            B = np.transpose(B, [3,0,1,2])  # T, H, W, C

            S = np.transpose(S, [1,2,3,0])  # h, w, c, t
            S = np.rot90(S, ri)
            S = np.transpose(S, [3,0,1,2])  # T, H, W, C

            # LR flip
            if np.random.random() < 0.5:
                L = _flip_axis(L, 2)
                B = _flip_axis(B, 2)
                S = _flip_axis(S, 2)

            # Temporal flip
            if np.random.random() < 0.5:
                L = _flip_axis(L, 0)  # TxHxWxC
                B = _flip_axis(B, 0)  # TxHxWxC
                S = _flip_axis(S, 0)  # TxHxWxC

            batch_lr.append(L)
            batch_blur.append(B)
            batch_sharp.append(S)

    batch_lr = np.stack(batch_lr, 0).astype(np.float32)
    batch_blur = np.stack(batch_blur, 0).astype(np.float32)
    batch_sharp = np.stack(batch_sharp, 0).astype(np.float32)

    return batch_lr.transpose((0,4,1,2,3)), batch_blur.transpose((0,4,1,2,3)), batch_sharp.transpose((0,4,1,2,3))

