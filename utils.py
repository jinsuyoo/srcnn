import tensorflow as tf
import numpy as np
import math

from PIL import Image

from tqdm import tqdm

import os
import h5py

FLAGS = tf.app.flags.FLAGS


# Read image
def imread(fname):
    return Image.open(fname)


# Save image
def imsave(image, path, fname):
    image = image * 255.
    
    image = Image.fromarray(image.astype('uint8'), mode='YCbCr')
    image = image.convert('RGB')
    
    return image.save(os.path.join(path, fname))


# Save ground truth image, bicubic interpolated image and srcnn image
def save_result(path, gt, bicubic, srcnn, i):
    imsave(gt, path, str(i)+ '_gt.png')
    imsave(bicubic, path, str(i) + '_bicubic.png')
    imsave(srcnn, path, str(i) + '_srcnn.png')


# Load sub-images of the dataset
def load_train_data():
    with h5py.File('train.h5', 'r') as f:
        images = np.array(f.get('data'))
        labels = np.array(f.get('label'))
    return images, labels


# Return true if the h5 sub-images file is exists
def exist_train_data():
    return os.path.exists('train.h5')


def prepare_data(config, is_valid=False):
    if is_valid:
        dataset = config.valid_dataset
        path = os.path.join(config.test_dataset_path, dataset)
    else:
        dataset = config.test_dataset
        path = os.path.join(config.test_dataset_path, dataset)

    dir_path = os.path.join(os.getcwd(), path)
    path_gt = os.path.join(dir_path, 'gt')
    path_lr = os.path.join(dir_path, 'bicubic_{:d}x'.format(config.scale))

    # fnames = ['baby_GT.bmp, bird_GT.bmp, ...']
    fnames = os.listdir(path_gt)
    
    inputs = []
    labels = []

    count = 0
    for fname in tqdm(fnames, desc='[*] Generating dataset ... '):
        count += 1
        
        _input = imread(os.path.join(path_lr, fname))
        _label = imread(os.path.join(path_gt, fname))
    
        _input = np.array(_input)
        _label = np.array(_label)
        
        inputs.append(_input / 255.)
        labels.append(_label / 255.)

    if is_valid:
        print('[*] Successfully prepared {:d} valid images !'.format(count))
    else:
        print('[*] Successfully prepared {:d} test images !'.format(count))
        
    return inputs, labels


# Concatenate Y and CrCb channel
def concat_ycrcb(y, crcb):
    return np.concatenate((y, crcb), axis=2)


# Crop border of the image
def crop_border(image):
    padding = int((5+9+1-3)/2)
    if image.ndim == 3:
        h, w, _ = image.shape
    else:
        h, w = image.shape

    return image[padding:h-padding, padding:w-padding]


# Compute Peak Signal to Noise Ratio
# PSNR = 20 * log (MAXi / root(MSE))
def psnr(label, image, max_val=1.):
    h, w, _ = label.shape

    diff = image - label
    rmse = math.sqrt(np.mean(diff ** 2))
    if rmse == 0:
        return 100
    else:
        return 20 * math.log10(max_val / rmse)
