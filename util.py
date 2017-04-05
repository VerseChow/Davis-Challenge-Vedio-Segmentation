import tensorflow as tf
import os
from glob import glob
from numpy import *
import numpy as np
from scipy.misc import imread, imresize,imshow
from sys import stdout
#from imagenet_classes import class_names


vgg_weights = load('vgg16.npy', encoding='latin1').item()
def load_images(pattern):
    fn = sorted(glob(pattern))
    if 'images' in pattern:
        img = zeros((len(fn), 480, 854, 3), dtype=uint8)
        for k in range(len(fn)):
            img1 = imread(fn[k])
            img1 = imresize(img1, (480, 854,3))
            img[k,...] = img1
    else:
        img = zeros((len(fn), 480, 854), dtype=uint8)
        for k in range(len(fn)):
            pimg = imread(fn[k])
            if len(pimg.shape) == 3:
                img1 = pimg[:,:,0]
                img1 = imresize(img1, (480, 854))
                img[k, ...] = img1
            else:
                img1 = imresize(pimg, (480, 854))
                img[k,...] = img1

    return img

def load_edge_image(label_pattern, image_pattern):
    list_of_label = sorted(glob(label_pattern+'/*.png'))
    list_of_image = sorted(glob(image_pattern+'/*.jpg'))
    len_label = 100#len(list_of_label)
    label = zeros((len_label, 448, 448), dtype=uint8)
    img = zeros((len_label, 448, 448, 3), dtype=uint8)
    print 'loading the data....'
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
    print 'finish loading data!' 
    return img, label

def input_pipeline(fn_seg, fn_img, batch_size):
    reader = tf.WholeFileReader()

    if not len(fn_seg) == len(fn_img):
        raise ValueError('Number of images and segmentations do not match!')

    with tf.variable_scope('segmentation'):
        fn_seg_queue = tf.train.string_input_producer(fn_seg, shuffle=False)
        _, value = reader.read(fn_seg_queue)
        seg = tf.image.decode_png(value, channels=1, dtype=tf.uint8)
        seg = tf.image.resize_images(seg, [480, 854], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        seg = tf.reshape(seg, [480, 854])
        

    with tf.variable_scope('image'):
        fn_img_queue = tf.train.string_input_producer(fn_img, shuffle=False)
        _, value = reader.read(fn_img_queue)
        img = tf.image.decode_jpeg(value, channels=3)
        img = tf.image.resize_images(img, [480, 854], method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, dtype = tf.float32)
    with tf.variable_scope('shuffle'):
        seg, img = tf.train.shuffle_batch([seg, img], batch_size=batch_size,
                                            num_threads=4,
                                            capacity=1000 + 3 * batch_size,
                                            min_after_dequeue=1000)

    return seg/255, img

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
        #prepare 
        prep2 = tf.layers.conv2d(inputs = conv2, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep2')
        prep3 = tf.layers.conv2d(inputs = conv3, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep3')
        prep4 = tf.layers.conv2d(inputs = conv4, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep4')              
        prep5 = tf.layers.conv2d(inputs = conv5, filters = 16, kernel_size = 3, strides = 1,
                padding='same', use_bias=True, reuse=reuse,
                name='prep5')       
        #upsampling
        up2 = tf.layers.conv2d_transpose(prep2, filters=16, kernel_size = 4, strides = 2,
                padding='same', use_bias=False, reuse=reuse,
                name='up2')
        start1 = (up2.shape[1]-480)/2
        start2 = (up2.shape[2]-854)/2
        end1 = up2.shape[1]-start1
        end2 = up2.shape[2]-start2 
        up2c = up2[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up2, 480, 854)
        up3 = tf.layers.conv2d_transpose(prep3, filters=16, kernel_size = 8, strides = 4,
                padding='valid', use_bias=False, reuse=reuse,
                name='up3')
        start1 = (up3.shape[1]-480)/2
        start2 = (up3.shape[2]-854)/2
        end1 = up3.shape[1]-start1
        end2 = up3.shape[2]-start2 
        up3c = up3[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up3, 480, 854)
        up4 = tf.layers.conv2d_transpose(prep4, filters=16, kernel_size = 16, strides = 8,
                padding='valid', use_bias=False, reuse=reuse,
                name='up4')
        start1 = (up4.shape[1]-480)/2
        start2 = (up4.shape[2]-854)/2
        end1 = up4.shape[1]-start1
        end2 = up4.shape[2]-start2 
        up4c = up4[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up4, 480, 854)
        up5 = tf.layers.conv2d_transpose(prep5, filters=16, kernel_size = 32, strides = 16,
                padding='valid', use_bias=False, reuse=reuse,
                name='up5')
        start1 = (up5.shape[1]-480)/2
        start2 = (up5.shape[2]-854)/2
        end1 = up5.shape[1]-start1
        end2 = up5.shape[2]-start2 
        up5c = up5[:,start1:end1,start2:end2,0:16]#tf.image.resize_image_with_crop_or_pad(up5, 480, 854)
        
        # k2 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k2')
        # k3 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k3')
        # k4 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k4')
        # k5 = tf.Variable(tf.random_normal([1],mean = 1.0,stddev = 1.0),name = 'k5')
        
        
        # kup2 = k2*up2c
        # kup3 = k3*up3c
        # kup4 = k4*up4c
        # kup5 = k5*up5c
        
        # add23 = tf.add(kup2, kup3, name='add23')
        # add234 = tf.add(add23, kup4, name='add234')
        # add2345 = tf.add(add234, kup5, name='add2345')
        concat_score = tf.concat([up2c, up3c,up4c,up5c], axis=3, name='concat_score')
        out_prep = tf.layers.conv2d(inputs = concat_score, filters = 1, kernel_size = 1, strides = 1,
                padding='same', use_bias=False, reuse=reuse,
                name='out_prep')  
        out1 = tf.sigmoid(out_prep)
        out = tf.reshape(out1,[-1,480,854],name='out')
        logits = tf.reshape(out_prep, [-1, 480, 854])
        #loss = -tf.reduce_mean(y*tf.log(out)+(1-y)*tf.log(1-out))
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                logits=logits, labels=tf.to_float(y)),name = "loss")
        return out,loss
        
        
        
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
            
