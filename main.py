import tensorflow as tf
from srcnn import SRCNN

import os


flags = tf.app.flags
flags.DEFINE_integer('epoch', 10000, 'Number of epoch')
flags.DEFINE_integer('batch_size', 128, 'The size of batch images')
flags.DEFINE_integer('image_size', 33, 'The size of sub-image')
flags.DEFINE_integer('label_size', 21, 'The size of label')

flags.DEFINE_integer('scale', 3, 'The up-scale value for training and testing')

flags.DEFINE_float('learning_rate', 1e-4, 'The learning rate of gradient descent algorithm')
flags.DEFINE_float('beta1', 0.9, 'The momentum value of gradient descent algorithm')

flags.DEFINE_string('train_dataset_path', 'Train', 'The path of train dataset')
flags.DEFINE_string('test_dataset_path', 'Test', 'The path of test dataset')
flags.DEFINE_string('train_dataset', '91', 'The name of training dataset')
flags.DEFINE_string('valid_dataset', 'Set5', 'The name of training dataset')
flags.DEFINE_string('test_dataset', 'Set5', 'The name of testing dataset')

flags.DEFINE_string('checkpoint_path', 'checkpoint', 'The path of checkpoint directory')
flags.DEFINE_boolean('use_pretrained', False, 'True for use pre-trained model, False for train on your own')
flags.DEFINE_string('result_dir', 'result', 'The path to save result images')
flags.DEFINE_boolean('is_training', True, 'True for training, False for testing')
FLAGS = flags.FLAGS


def main(_):
    with tf.Session() as sess:
        srcnn = SRCNN(sess, FLAGS)

        if FLAGS.is_training == True:
            srcnn.train(FLAGS)

        elif FLAGS.is_training == False:
            srcnn.test(FLAGS)

        else:
            print('[*] Please give correct [is_training] value ')

if __name__ == '__main__':
    tf.app.run()