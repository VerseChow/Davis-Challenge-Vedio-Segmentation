#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
from glob import glob
from scipy.misc import imread, imresize,imshow
from sys import stdout

vgg_weights = np.load('vgg16.npy', encoding='latin1').item()

def input_pipeline(filename_queue, batch_size, target_width, target_height):
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string),
        'mask_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
    
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)

    image_shape = [height, width, 3]
    annotation_shape = [height, width, 1]
    
    image = tf.reshape(image, image_shape)
    annotation = tf.reshape(annotation, annotation_shape)
   
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                           target_height=target_height,
                                           target_width=target_width)
    
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                           target_height=target_height,
                                           target_width=target_width)
    

    images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation], 
                                                    batch_size=batch_size, 
                                                    capacity=1000+3*batch_size, 
                                                    num_threads=2, 
                                                    min_after_dequeue=1000)
    
    return images, annotations

def load_edge_image(label_pattern, image_pattern):
    list_of_label = sorted(glob(label_pattern+'/*.png'))
    list_of_image = sorted(glob(image_pattern+'/*.jpg'))
    len_label = 100#len(list_of_label)
    label = zeros((len_label, 448, 448), dtype=uint8)
    img = zeros((len_label, 448, 448, 3), dtype=uint8)
    print('loading the data....')
    for k in range(len_label):
        label1 = imread(list_of_label[k])
        #label1 = imresize(label1, (448, 448))
        label1 = label1/255
        label[k,...] = label1
        base = os.path.basename(list_of_label[k])
        base = os.path.splitext(base)[0]
        matching = [s for s in list_of_image if base in s]
        img1 = imread(matching[0])
        img1 = imresize(img1, (448, 448, 3))
        img[k,...] = img1
        rate = float(k)/float(len_label)*100.0
        stdout.write("\r completing... %.2f %%" % rate)
        stdout.flush()
       
    stdout.write("\n")
    print('finish loading data!')
    return img, label


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
        pool1 = tf.layers.max_pooling2d(conv1, 2, 2, padding='same', name='pool1')
        conv2 = conv_relu_vgg(pool1, reuse=reuse, name='conv2_1')
        conv2 = conv_relu_vgg(conv2, reuse=reuse, name='conv2_2')

        # 56 112
        pool2 = tf.layers.max_pooling2d(conv2, 2, 2, padding='same', name='pool2')
        conv3 = conv_relu_vgg(pool2, reuse=reuse, name='conv3_1')
        conv3 = conv_relu_vgg(conv3, reuse=reuse, name='conv3_2')
        conv3 = conv_relu_vgg(conv3, reuse=reuse, name='conv3_3')

        # 28 56
        pool3 = tf.layers.max_pooling2d(conv3, 2, 2, padding='same', name='pool3')
        conv4 = conv_relu_vgg(pool3, reuse=reuse, name='conv4_1')
        conv4 = conv_relu_vgg(conv4, reuse=reuse, name='conv4_2')
        conv4 = conv_relu_vgg(conv4, reuse=reuse, name='conv4_3')

        # 14 28
        pool4 = tf.layers.max_pooling2d(conv4, 2, 2, padding='same', name='pool4')
        conv5 = conv_relu_vgg(pool4, reuse=reuse, name='conv5_1')
        conv5 = conv_relu_vgg(conv5, reuse=reuse, name='conv5_2')
        conv5 = conv_relu_vgg(conv5, reuse=reuse, name='conv5_3')

        # 7 14
        #pool5 = tf.layers.max_pooling2d(conv5, 2, 2, name='pool5')
        #(a)for segmentation
        #prepare 
        prep2 = tf.layers.conv2d(inputs = conv2, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse, kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name='prep2')
        prep3 = tf.layers.conv2d(inputs = conv3, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse, kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name='prep3')
        prep4 = tf.layers.conv2d(inputs = conv4, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse, kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name='prep4')              
        prep5 = tf.layers.conv2d(inputs = conv5, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse, kernel_initializer=tf.random_normal_initializer(stddev=0.001),
                name='prep5')       
        #upsampling
        up2 = tf.layers.conv2d_transpose(prep2, filters=16, kernel_size = 4, strides = 2,
                padding='same', use_bias=False, reuse=reuse,
                name='up2')
        start1 = (up2.shape[1]-480)//2
        start2 = (up2.shape[2]-854)//2
        end1 = up2.shape[1]-start1
        end2 = up2.shape[2]-start2 
        up2c = up2[:,start1:end1,start2:end2,:]#tf.image.resize_image_with_crop_or_pad(up2, 480, 854)
        # up2c = up2
        up3 = tf.layers.conv2d_transpose(prep3, filters=16, kernel_size = 8, strides = 4,
                padding='valid', use_bias=False, reuse=reuse,
                name='up3')
        # up3c = up3[:,4:,4:,:]
        start1 = (up3.shape[1]-480)//2
        start2 = (up3.shape[2]-854)//2
        end1 = up3.shape[1]-start1
        end2 = up3.shape[2]-start2 
        up3c = up3[:,start1:end1,start2:end2,:]#tf.image.resize_image_with_crop_or_pad(up3, 480, 854)

        up4 = tf.layers.conv2d_transpose(prep4, filters=16, kernel_size = 16, strides = 8,
                padding='valid', use_bias=False, reuse=reuse,
                name='up4')
        # up4c = up4[:,8:,2:,:]
        start1 = (up4.shape[1]-480)//2
        start2 = (up4.shape[2]-854)//2
        end1 = up4.shape[1]-start1
        end2 = up4.shape[2]-start2 
        up4c = up4[:,start1:end1,start2:end2,:]#tf.image.resize_image_with_crop_or_pad(up4, 480, 854)

        up5 = tf.layers.conv2d_transpose(prep5, filters=16, kernel_size = 32, strides = 16,
                padding='valid', use_bias=False, reuse=reuse,
                name='up5')
        # up5c = up5[:, 16:, 10:, :]
        start1 = (up5.shape[1]-480)//2
        start2 = (up5.shape[2]-854)//2
        end1 = up5.shape[1]-start1
        end2 = up5.shape[2]-start2 
        up5c = up5[:,start1:end1,start2:end2,:]#tf.image.resize_image_with_crop_or_pad(up5, 480, 854)
    
        concat_score = tf.concat([up2c, up3c, up4c, up5c], axis=3, name='concat_score')
        logits = tf.layers.conv2d(inputs = concat_score, filters = 1, kernel_size = 1,
                padding='same', use_bias=False, reuse=reuse,
                name='out_prep')  
        # print(out_prep.shape)
        # out1 = tf.sigmoid(out_prep)
        # out = tf.reshape(out1,[-1,480,854,1],name='out')
        # logits = tf.reshape(out_prep, [-1, 480, 854,1])
        #loss = -tf.reduce_mean(y*tf.log(out)+(1-y)*tf.log(1-out))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=logits, labels=y),name = "loss")
        tf.summary.scalar('loss', loss)
        return logits, loss
        
        
        
            
