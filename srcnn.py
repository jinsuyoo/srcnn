import tensorflow as tf
import numpy as np

import os
import time
from tqdm import tqdm

from utils import *


class SRCNN(object):
    def __init__(self, sess, config):
        self.sess = sess

        # The size of training sub-images is 33
        # All the convolutional layers have no padding (fsub-f1-f2-f3+3) = (33-5-9-1+3) = 21
        self.image_size = [None, None, None, 1]
        self.label_size = [None, None, None, 1]

        self.scale = config.scale
        
        self.build_model()
    

    def build_model(self):
        self.images = tf.placeholder(tf.float32, self.image_size, name='images')
        self.labels = tf.placeholder(tf.float32, self.label_size, name='labels')

        self.weights = {
            'w1': tf.Variable(tf.random_normal([9, 9, 1, 64], stddev=0.001), name='w1'),
            'w2': tf.Variable(tf.random_normal([1, 1, 64, 32], stddev=0.001), name='w2'),
            'w3': tf.Variable(tf.random_normal([5, 5, 32, 1], stddev=0.001), name='w3')
        }
        self.biases = {
            'b1': tf.Variable(tf.zeros([64]), name='b1'),
            'b2': tf.Variable(tf.zeros([32]), name='b2'),
            'b3': tf.Variable(tf.zeros([1]), name='b3')
        }

        self.forward = self.model()

        # Loss Function : Mean Square Error
        self.loss = tf.reduce_mean(tf.square(self.labels - self.forward))

        # Clip output
        self.result = tf.clip_by_value(self.forward, clip_value_min=0., clip_value_max=1.)

        self.saver = tf.train.Saver()


    # Input  : (33 x 33 x 1)
    # Layer1 : (9 x 9 x 1 x 64)
    # Layer2 : (1 x 1 x 64 x 32)
    # Layer3 : (5 x 5 x 32 x 1)
    # Output : (21 x 21 x 1)
    def model(self):
        conv1 = tf.nn.conv2d(self.images, self.weights['w1'], strides=[1,1,1,1], padding='VALID')
        conv1 = conv1 + self.biases['b1']
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.nn.conv2d(conv1, self.weights['w2'], strides=[1,1,1,1], padding='VALID')
        conv2 = conv2 + self.biases['b2']
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.nn.conv2d(conv2, self.weights['w3'], strides=[1,1,1,1], padding='VALID')
        output = conv3 + self.biases['b3']
    
        return output


    def train(self, config):
        print('[*] SRCNN training will be started ! ')

        if not exist_train_data():
            print('[!] No train data ready .. Please generate train data first with Matlab')
            return
        else:
            train_images, train_labels = load_train_data()
            print('[*] Successfully load train data ! ')
       
        valid_images, valid_labels = prepare_data(config, is_valid=True)

        # Adam optimizer with the standard backpropagation
        # The learning rate is 1e-4 for the first two layers, and 1e-5 for the last layer
        # beta1 is 0.9 in paper
        var_list1 = [self.weights['w1'], self.weights['w2'], self.biases['b1'], self.biases['b2']]
        var_list2 = [self.weights['w3'], self.biases['b3']]
        opt1 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1)
        opt2 = tf.train.AdamOptimizer(config.learning_rate * 0.1, beta1=config.beta1)
        grads = tf.gradients(self.loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        self.train_op = tf.group(train_op1, train_op2)

        #self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Initialize TensorFlow variables
        init = tf.global_variables_initializer()
        self.sess.run(init)

        # Load checkpoint
        self.load(config)

        start_time = time.time()
        bicubic_psnr = []
        print('[*] Start training ... Please be patient !')
        for i in tqdm(range(config.epoch), desc='[*] Keep going ! ', leave=True):
            loss = 0
            batch_idxs = len(train_images) // config.batch_size
            
            for idx in range(batch_idxs):
                batch_images = train_images[idx*config.batch_size : (idx+1)*config.batch_size]
                batch_labels = train_labels[idx*config.batch_size : (idx+1)*config.batch_size]
            
                _, err = self.sess.run([self.train_op, self.loss], feed_dict={self.images: batch_images, self.labels: batch_labels})
                loss += err

            valid_psnr = []
            
            for idx in range(len(valid_images)):
                h, w, _ = valid_images[idx].shape
                valid_input_y = valid_images[idx][:, :, 0]
                valid_label_y = valid_labels[idx][:, :, 0]

                valid_input_y = valid_input_y.reshape([1, h, w, 1])
                valid_label_y = valid_label_y.reshape([1, h, w, 1])

                
                result = self.sess.run(self.result, feed_dict={self.images: valid_input_y, self.labels: valid_label_y})
                
                valid_label_y = crop_border(valid_label_y[0])

                if i == 0:
                        bicubic_psnr.append(psnr(valid_label_y, crop_border(valid_input_y[0])))
                valid_psnr.append(psnr(valid_label_y, result[0]))
                
            print('[*] Epoch: [%d], psnr: [bicubic: %.2f, srcnn: %.2f], loss: [%.8f]' % (i+1, np.mean(bicubic_psnr), np.mean(valid_psnr), loss/batch_idxs))
            
            # Every 50 epoch, validate current model to decide whether save or not
            if (i+1) % 50 == 0:
                self.save(i+1, config)
        print('[*] Training done ! Congrats :) ')
    

    def test(self, config):
        print('[*] SRCNN testing will be started ! ')
        t = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))

        test_images, test_labels = prepare_data(config, is_valid=False)

        init = tf.global_variables_initializer()

        results = []
        bicubic_psnr = []
        test_psnr = []
        print('[*] Start testing !')
        
        self.sess.run(init)
        
        self.load(config)

        sub_start = 0
        for idx in tqdm(range(len(test_images))):
            h, w, _ = test_images[idx].shape
            test_input_y = test_images[idx][:, :, 0]
            test_label_y = test_labels[idx][:, :, 0]

            test_input_cbcr = test_images[idx][:, :, 1:3]
            test_label_cbcr = test_labels[idx][:, :, 1:3]

            test_input_y = test_input_y.reshape([1, h, w, 1])
            test_label_y = test_label_y.reshape([1, h, w, 1])

            test_input_cbcr = test_input_cbcr.reshape([1, h, w, 2])
            test_label_cbcr = test_label_cbcr.reshape([1, h, w, 2])

            result = self.sess.run(self.result, feed_dict={self.images: test_input_y, self.labels: test_label_y})
                
            test_input_y = crop_border(test_input_y[0])
            test_label_y = crop_border(test_label_y[0])

            test_input_cbcr = crop_border(test_input_cbcr[0])
            test_label_cbcr = crop_border(test_label_cbcr[0])

            bicubic_psnr.append(psnr(test_label_y, test_input_y))
            test_psnr.append(psnr(test_label_y, result[0]))

            gt = concat_ycrcb(test_label_y, test_label_cbcr)
            bicubic = concat_ycrcb(test_input_y, test_input_cbcr)
            result = concat_ycrcb(result[0], test_input_cbcr)
            
            path = os.path.join(os.getcwd(), config.result_dir)
            path = os.path.join(path, t)
            if not os.path.exists(path):
                os.makedirs(path)

            save_result(path, gt, bicubic, result, idx)
            
        print('[*] PSNR of ground truth and bicubic : %.2f'% np.mean(bicubic_psnr))
        print('[*] PSNR of ground truth and SRCNN   : %.2f'% np.mean(test_psnr))

        
    def save(self, epoch, config):
        model_name = 'srcnn'
        model_dir = 'SRCNN'
        path = os.path.join(config.checkpoint_path, model_dir)
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.saver.save(self.sess, os.path.join(path, model_name), global_step=epoch+12450)
        print('[*] Save checkpoint at %d epoch' % epoch)


    def load(self, config):
        if config.use_pretrained:
            model_dir = 'SRCNN_pretrained'
        else:
            model_dir = 'SRCNN'
        path = os.path.join(config.checkpoint_path, model_dir)
        ckpt_path = tf.train.latest_checkpoint(path)
        if ckpt_path:
            self.saver.restore(self.sess, ckpt_path)
            print('[*] Load checkpoint: %s' % ckpt_path)
        else:
            print('[*] No checkpoint to load ... ')
        