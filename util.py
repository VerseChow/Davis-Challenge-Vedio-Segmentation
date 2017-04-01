import tensorflow as tf
from glob import glob
from numpy import *
import numpy as np
from scipy.misc import imread, imresize,imshow
from imagenet_classes import class_names

vgg_weights = load('vgg16.npy', encoding='latin1').item()
def load_images(pattern):
    fn = sorted(glob(pattern))
    if 'images' in pattern:
        img = zeros((len(fn), 448, 448, 3), dtype=uint8)
        for k in range(len(fn)):
            img1 = imread(fn[k])
            img1 = imresize(img1, (448, 448,3))
            img[k,...] = img1
    else:
        img = zeros((len(fn), 448, 448), dtype=uint8)
        for k in range(len(fn)):
            pimg = imread(fn[k])
            if len(pimg.shape) == 3:
                img1 = pimg[:,:,0]
                img1 = imresize(img1, (448, 448))
                img[k, ...] = img1
            else:
                img1 = imresize(pimg, (448, 448))
                img[k,...] = img1

        

    return img
def conv_relu_vgg(x, reuse=None, name='conv_vgg'):
    kernel = vgg_weights[name][0]
    bias = vgg_weights[name][1]
    with tf.variable_scope(name):
        x = tf.layers.conv2d(x, kernel.shape[-1], kernel.shape[0],
                padding='same', use_bias=True, reuse=reuse,
                kernel_initializer=tf.constant_initializer(kernel),
                bias_initializer=tf.constant_initializer(bias),
                name='conv2d')
        return tf.nn.relu(x, name='relu')
def upconv_relu(x, num_filters, ksize=3, stride=2, reuse=None, name='upconv'):
    with tf.variable_scope(name):
        x = tf.layers.conv2d_transpose(x, num_filters, ksize, stride,
                padding='same', use_bias=False, reuse=reuse,
                name='conv2d_transpose')
        return tf.nn.relu(x, name='relu')
def build_model(x, y, reuse=None, training=True):
    with tf.variable_scope('OSVOS'):
        
        x = x[..., ::-1] - [103.939, 116.779, 123.68]

        # 224 448
        conv1 = conv_relu_vgg(x, reuse=reuse, name='conv1_1')
        conv1 = conv_relu_vgg(conv1, reuse=reuse, name='conv1_2')

        # 112 224
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name='pool1')
        conv2 = conv_relu_vgg(pool1, reuse=reuse, name='conv2_1')
        conv2 = conv_relu_vgg(conv2, reuse=reuse, name='conv2_2')

        # 56 112
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name='pool2')
        conv3 = conv_relu_vgg(pool2, reuse=reuse, name='conv3_1')
        conv3 = conv_relu_vgg(conv3, reuse=reuse, name='conv3_2')
        conv3 = conv_relu_vgg(conv3, reuse=reuse, name='conv3_3')

        # 28 56
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name='pool3')
        conv4 = conv_relu_vgg(pool3, reuse=reuse, name='conv4_1')
        conv4 = conv_relu_vgg(conv4, reuse=reuse, name='conv4_2')
        conv4 = conv_relu_vgg(conv4, reuse=reuse, name='conv4_3')

        # 14 28
        pool4 = tf.layers.max_pooling2d(conv4, 2, 2, name='pool4')
        conv5 = conv_relu_vgg(pool4, reuse=reuse, name='conv5_1')
        conv5 = conv_relu_vgg(conv5, reuse=reuse, name='conv5_2')
        conv5 = conv_relu_vgg(conv5, reuse=reuse, name='conv5_3')

        # 7 14
        #pool5 = tf.layers.max_pooling2d(conv5, 2, 2, name='pool5')
        #(a)for segmentation
        #
        up1 = upconv_relu(conv1, 1,ksize=3, stride=1, reuse=reuse, name='up1')
        up2 = upconv_relu(conv2, 1,ksize=6, stride=2, reuse=reuse, name='up2')
        up3 = upconv_relu(conv3, 1,ksize=12, stride=4, reuse=reuse, name='up3')
        up4 = upconv_relu(conv4, 1,ksize=24, stride=8, reuse=reuse, name='up4')
        up5 = upconv_relu(conv5, 1, ksize=3, stride=16,reuse=reuse, name='up5')
        k1 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k1')
        k2 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k2')
        k3 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k3')
        k4 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k4')
        k5 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k5')
        
        kup1 = k1*up1
        kup2 = k2*up2
        kup3 = k3*up3
        kup4 = k4*up4
        kup5 = k5*up5
        add12 = tf.add(kup1, kup2, name='add12')
        add123 = tf.add(add12, kup3, name='add123')
        add1234 = tf.add(add123, kup4, name='add1234')
        add12345 = tf.add(add1234, kup5, name='add12345')
        out = tf.sigmoid(add12345,'out')
        logits = tf.reshape(add12345, [-1, 448, 448])
        #loss = -tf.reduce_mean(y*tf.log(out)+(1-y)*tf.log(1-out))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=logits, labels=tf.to_float(y)),name = "loss")
        return logits,loss
        
        
        
        #up1 = tf.concat([up1, conv5], axis=3, name='concat1')
               
        '''(b)for label
        #7^2*512 ->  4096
        #fc1  
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(pool5.get_shape()[1:]))
            kernelf = vgg_weights['fc6'][0]
            bias = vgg_weights['fc6'][1]
            fc1w = tf.Variable(kernelf, name='weights')
            fc1b = tf.Variable(bias, name='biases')
            pool5_flat = tf.reshape(pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            fc1 = tf.nn.relu(fc1l)
            

        # fc2 4096->4096
        with tf.name_scope('fc2') as scope:
            kernelf = vgg_weights['fc7'][0]
            bias = vgg_weights['fc7'][1]
            fc2w = tf.Variable(kernelf, name='weights')
            fc2b = tf.Variable(bias, name='biases')
            fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
            fc2 = tf.nn.relu(fc2l)
            

        # fc3 4096->1000
        with tf.name_scope('fc3') as scope:
            kernelf = vgg_weights['fc8'][0]
            bias = vgg_weights['fc8'][1]       
            fc3w = tf.Variable(kernelf, name='weights')
            fc3b = tf.Variable(bias, name='biases')
            fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
        probs = tf.nn.softmax(fc3l)
        return probs'''   
            