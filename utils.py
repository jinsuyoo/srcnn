import tensorflow as tf
import numpy as np
import math
import scipy
from scipy import ndimage
import imageio

import os
import h5py

FLAGS = tf.app.flags.FLAGS


def make_sub_images(path, is_training=True):
    dataset = os.path.split(path)[-1]
    
    images = load_images(path, is_training)
  
    # the ground truth images {Xi} are prepared as (fsub x fsub x c) pixel
    fsub = FLAGS.image_size
    flabel = FLAGS.label_size
    padding = int((fsub-flabel)/2)

    # Stride 14 for training and same as label size (21) for testing
    if is_training:
        stride = FLAGS.stride
    else:
        stride = flabel

    # Sub-images cropped from the training images
    sub_images = []
    sub_labels = []
    merge_info = []
    low_images = []
    labels = []
    for n in range(len(images)):
        n_sub_w = n_sub_h = 0
        low_image, label = preprocess_image(images[n], is_training)

        low_images.append(low_image)
        labels.append(label)
        #w, h, _ = low_image.shape
        w, h = low_image.shape
        for i in range(0, w-fsub+1, stride):
            n_sub_w += 1
            n_sub_h = 0
            for j in range(0, h-fsub+1, stride):
                n_sub_h += 1
                sub_image = low_image[i:i+fsub, j:j+fsub] 
                sub_label = label[i+padding:i+padding+flabel, j+padding:j+padding+flabel]
                #sub_image = low_image[i:i+fsub, j:j+fsub, :] 
                #sub_label = label[i+padding:i+padding+flabel, j+padding:j+padding+flabel, :]

                # Type casting ... (W, H) -> (W, H, 1)
                sub_image = sub_image.reshape([fsub, fsub, 1])  
                sub_label = sub_label.reshape([flabel, flabel, 1])
                
                sub_images.append(sub_image)
                sub_labels.append(sub_label)
        # Merge information contains required number of sub images
        # in width, height for each to make SR test image 
        merge_info.append([n_sub_w, n_sub_h])

    arr_sub_images = np.asarray(sub_images)
    arr_sub_labels = np.asarray(sub_labels)

    save_data(arr_sub_images, arr_sub_labels, dataset)

    if not is_training:
        return merge_info, low_images, labels


# Merge forward sub-images of the test image
# to get one final result SR image 
def merge_output(sub_images, size):
    h, w = sub_images.shape[1], sub_images.shape[2]
    result = np.zeros((h*size[0], w*size[1], 1))
    for idx, sub_image in enumerate(sub_images):
        i = idx % size[1]
        j = idx // size[1]
        result[j*h:j*h+h, i*w:i*w+w, :] = sub_image
    return result


# Because border of the forward output is cropped,
# pad border of the SR output with bicubic interpolation
def pad_border(result, bicubic):
    temp = bicubic.copy()
    padding = int((FLAGS.image_size-FLAGS.label_size)/2)
    h, w, _ = result.shape
    h_bc, w_bc = temp.shape
    i = h_bc - h
    j = w_bc - w
    temp[padding:padding-i, padding:padding-j] = result[:,:,0]
    return temp


# Save sub-images of the dataset to local with h5 format
def save_data(images, labels, dataset):
    with h5py.File(dataset+'.h5', 'w') as f:
        f.create_dataset('images', data=images)
        f.create_dataset('labels', data=labels)
    

# Compute Peak Signal to Noise Ratio
# PSNR = 20 * log (MAXi / root(MSE))
def compute_psnr(images1, images2):
    max_val = 1.
    psnr = count = 0
    for i in range(len(images1)):
        count += 1
        mse = np.mean((images1[i] - images2[i])**2)
        if mse == 0:
            psnr += 100
        else:
            psnr += 20 * math.log10(max_val / math.sqrt(mse))
    return psnr / count


# Load sub-images of the dataset
def load_data(path):
    path = os.path.split(path)[-1]
    with h5py.File(path+'.h5', 'r') as f:
        images = np.array(f.get('images'))
        labels = np.array(f.get('labels'))
    return images, labels


# Save image
def imsave(image, path):
    return scipy.misc.imsave(path, image.squeeze())


# Return true if the h5 sub-images file is exists
def is_data_ready(dataset):
    return os.path.exists(os.path.split(dataset)[-1]+'.h5')


def save_result(result, low_image, label, path):
    imsave(result, path+'_SRCNN.png')
    imsave(low_image, path+'_bicubic.png')
    imsave(label, path+'_gt.png')
    

# Print 'Hour:Minutes:Seconds'
def print_time(seconds):
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    strtime = '%d:%02d:%02d' % (hour, minutes, seconds) 
    return strtime


def preprocess_image(image, is_training):
    # To down-up scaling image, subtract remainder
    scale = FLAGS.scale
    h, w = image.shape
    h = h - np.mod(h, scale)
    w = w - np.mod(w, scale)
    image = image[0:h, 0:w]
    # Normalize image
    image = image / 255.

    # high_label : high-resolution label
    # low_image  : blurred low-resolution input

    high_label = image
    # blur a sub-image by a Gaussian kernel for training input
    '''
    if is_training:
        blurred_image = ndimage.gaussian_filter(image, sigma=1)
    '''
    # Sub-sample it by the upscaling factor
    low_image = scipy.misc.imresize(image, 1./scale, interp='bicubic', mode='F')
    # Upscale it by the same factor via bicubic interpolation
    low_input = scipy.misc.imresize(low_image, scale/1., interp='bicubic', mode='F')

    return low_input, high_label


# Load images from given path
def load_images(dir_path, is_training):
    images = []
    for filename in os.listdir(dir_path):
        # Read gray-image with YCbCr color space
        img = imageio.imread(os.path.join(dir_path, filename), as_gray=False, pilmode="YCbCr")
        img = img[:,:,0].squeeze()
        # Append image to list
        images.append(np.asarray(img))
    return images
