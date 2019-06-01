import tensorflow as tf
from srcnn import SRCNN

import os


flags = tf.app.flags
flags.DEFINE_integer('epoch', 3000, 'Number of epoch')
flags.DEFINE_integer('batch_size', 128, 'The size of batch images')
flags.DEFINE_integer('image_size', 33, 'The size of sub-image')
flags.DEFINE_integer('label_size', 21, 'The size of label')
flags.DEFINE_integer('stride', 14, 'The stride for pre-process sub-images')
flags.DEFINE_integer('n_channel', 1, 'The number of channel')
flags.DEFINE_integer('scale', 3, 'The up-scale value for training and testing')
flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate of gradient descent algorithm')
flags.DEFINE_float('beta1', 0.9, 'The momentum value of gradient descent algorithm')
flags.DEFINE_string('optimizer', 'Adam', 'The optimizer of gradient descent algorithm')
flags.DEFINE_string('dataset_path', 'SR_dataset', 'The path to dataset directory')
flags.DEFINE_string('train_dataset', '291', 'The name of training dataset')
flags.DEFINE_string('valid_dataset', '91', 'The name of validating dataset')
flags.DEFINE_string('test_dataset', 'Set5', 'The name of testing dataset')
flags.DEFINE_string('checkpoint_path', 'checkpoint', 'The path to checkpoint')
flags.DEFINE_boolean('use_pretrained', False, 'True for use pre-trained model, False for train on your own')
flags.DEFINE_string('result_dir', 'result', 'The path to output images')
flags.DEFINE_boolean('is_training', True, 'True for training, False for testing')
FLAGS = flags.FLAGS


def main(_):
    srcnn = SRCNN(FLAGS)
    if FLAGS.is_training == True:
        srcnn.train()
    elif FLAGS.is_training == False:
        srcnn.test()
    else:
        print('[*] Please give correct [is_training] value ')

if __name__ == '__main__':
    tf.app.run()