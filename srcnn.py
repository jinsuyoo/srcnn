import tensorflow as tf
import numpy as np

import os
from time import time
from tqdm import tqdm

from utils import *


class SRCNN(object):
    def __init__(self, config):
        self.n_channel = config.n_channel
        # The size of training sub-images is 33
        self.image_size = [None, config.image_size, config.image_size, self.n_channel] 
        # All the convolutional layers have no padding (fsub-f1-f2-f3+3) = (33-5-9-1+3) = 21
        self.label_size = [None, config.label_size, config.label_size, self.n_channel] 

        # The learning rate is 1e-4 for the first two layers, and 1e-5 for the last layer
        self.learning_rate = config.learning_rate
        # 0.9 in paper
        self.beta1 = config.beta1
        # 10000 in paper but 3000 in this code
        self.epoch = config.epoch
        # 128 in paper
        self.batch_size = config.batch_size
        # Adam optimizer
        self.optimizer = config.optimizer
        self.is_training = config.is_training

        self.dataset_path = config.dataset_path
        # 291 dataset for training
        self.train_dataset = os.path.join(self.dataset_path, config.train_dataset)
        # 91 dataset for validating
        self.valid_dataset = os.path.join(self.dataset_path, config.valid_dataset)
        # Set5 dataset for testing
        self.test_dataset = os.path.join(self.dataset_path, config.test_dataset)
        self.result_dir = config.result_dir
        self.checkpoint_path = config.checkpoint_path
        self.use_pretrained = config.use_pretrained

        self.build_model()
    

    def build_model(self):
        self.images = tf.placeholder(tf.float32, self.image_size, name='images')
        self.labels = tf.placeholder(tf.float32, self.label_size, name='labels')

        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, self.n_channel, 64], stddev=0.001), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=0.001), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, self.n_channel], stddev=0.001), name='w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([self.n_channel]), name='b3')
        }

        self.forward = self.model()

        # Loss Function : Mean Square Error
        self.loss = tf.reduce_mean(tf.square(self.labels - self.forward))

        self.saver = tf.train.Saver()


    # Input  : (33 x 33 x n_channel)
    # Layer1 : (9 x 9 x n_channel x 64)
    # Layer2 : (1 x 1 x 64 x 32)
    # Layer3 : (5 x 5 x 32 x n_channel)
    # Output : (21 x 21 x n_channel)
    def model(self):
        conv1 = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID')
        conv1 = conv1 + self.biases['b1']
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID')
        conv2 = conv2 + self.biases['b2']
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID')
        conv3 = conv3 + self.biases['b3']
        return conv3


    def train(self):
        print('[*] SRCNN training will be started ! ')

        if not is_data_ready(self.train_dataset):
            make_sub_images(self.train_dataset)
        if not is_data_ready(self.valid_dataset):
            make_sub_images(self.valid_dataset)

        train_images, train_labels = load_data(self.train_dataset)
        valid_images, valid_labels = load_data(self.valid_dataset)
        
        # Adam optimizer with the standard backpropagation
        # Learning rate 1e-4 for first two layers
        # Learning rate 1e-5 for the last layer
        var_list1 = [self.weights['w1'], self.weights['w2'], self.biases['b1'], self.biases['b2']]
        var_list2 = [self.weights['w3'], self.biases['b3']]
        opt1 = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1)
        opt2 = tf.train.AdamOptimizer(self.learning_rate * 0.1, beta1=self.beta1)
        grads = tf.gradients(self.loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        self.train_op = tf.group(train_op1, train_op2)

        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        init = tf.global_variables_initializer()

        start_time = time()

        # Used to compare current loss and minimum loss    
        min_loss = 777.
      
        with tf.Session() as sess:
            sess.run(init)

            self.load(sess, self.checkpoint_path)

            print('[*] Start training ... Please be patient !')
            for i in tqdm(range(self.epoch), desc='Keep going ! '):
                batch_idxs = len(train_images) // self.batch_size
            
                for idx in range(batch_idxs):
                    batch_images = train_images[idx*self.batch_size : (idx+1)*self.batch_size]
                    batch_labels = train_labels[idx*self.batch_size : (idx+1)*self.batch_size]
            
                    _, err = sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                print('Epoch: [%d], time: [%s], loss: [%.8f]' % (i, print_time(time()-start_time), err))
                
                # Every 50 epoch, validate current model to decide whether save or not
                if (i+1) % 50 == 0:
                    print('[*] Validating model at epoch %d to decide whether to save or not' % (i+1))
                    valid_loss = 0
                    batch_idxs = len(valid_images) // self.batch_size
            
                    for idx in range(batch_idxs):
                        batch_images = valid_images[idx*self.batch_size : (idx+1)*self.batch_size]
                        batch_labels = valid_labels[idx*self.batch_size : (idx+1)*self.batch_size]
            
                        _, err = sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                        valid_loss += err
                    valid_loss = valid_loss / batch_idxs

                    # Update minimum valid loss
                    if min_loss > valid_loss:
                        self.save(sess, self.checkpoint_path, i+1)
                        min_loss = valid_loss
                        print('[*] Minimum loss on valid set updated : %.8f' % min_loss)
                    

    def test(self):
        print('[*] SRCNN testing will be started ! ')

        # Merge information contains required number of sub images
        # in width, height for each to make SR test image 
        merge_info, low_images, labels = make_sub_images(self.test_dataset, self.is_training)
        
        test_images, test_labels = load_data(self.test_dataset)
        
        init = tf.global_variables_initializer()

        results = []
        print('[*] Start testing !')
        with tf.Session() as sess:
            sess.run(init)

            self.load(sess, self.checkpoint_path)

            sub_start = 0
            
            for i, (n_w, n_h) in enumerate(merge_info):
                n_sub = n_w * n_h
                
                images = test_images[sub_start : sub_start+n_sub]
                #labels = test_labels[sub_start : sub_start+n_sub]
                sub_start += n_sub
            
                result = self.forward.eval(feed_dict={self.images: images})
                result = merge_output(result, [n_w, n_h])
                result = pad_border(result, low_images[i])
                results.append(result)
                result_path = self.result_dir
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                result_path = os.path.join(result_path, 'result_%d' % i)
                save_result(result, low_images[i], labels[i], result_path)
            print('[*] PSNR of ground truth and bicubic : %.2f'% compute_psnr(low_images, labels))
            print('[*] PSNR of ground truth and SRCNN   : %.2f'% compute_psnr(results, labels))


    def save(self, sess, path, step):
        model_name = 'srcnn'
        model_dir = 'SRCNN'
        path = os.path.join(path, model_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.saver.save(sess, os.path.join(path, model_name), global_step=step)
        print('[*] Successfully saved checkpoint at %d step' % step)


    def load(self, sess, path):
        if self.use_pretrained:
            model_dir = 'SRCNN_pretrained'
        else:
            model_dir = 'SRCNN'
        path = os.path.join(path, model_dir)
        ckpt_path = tf.train.latest_checkpoint(path)
        if ckpt_path:
            self.saver.restore(sess, ckpt_path)
            print('[*] Successfully loaded checkpoint: %s' % ckpt_path)
        else:
            print('[*] No checkpoint to load ... ')
        