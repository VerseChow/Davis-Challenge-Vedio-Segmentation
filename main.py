from util import *
import tensorflow as tf
import time, os
from numpy import *
from scipy.misc import imsave
from scipy.ndimage.filters import gaussian_filter1d
tf.app.flags.DEFINE_integer('batch_size', 4, 'Number of images in each batch')
tf.app.flags.DEFINE_integer('num_epoch', 100, 'Total number of epochs to run for training')
tf.app.flags.DEFINE_boolean('training', True, 'If true, train the model; otherwise evaluate the existing model')
tf.app.flags.DEFINE_float('init_learning_rate', 1e-4, 'Initial learning rate')
tf.app.flags.DEFINE_float('learning_rate_decay', 0.95, 'Ratio for decaying the learning rate after each epoch')
#tf.app.flags.DEFINE_float('min_learning_rate', 1e-6, 'Minimum learning rate used for training')
tf.app.flags.DEFINE_string('gpu', '0', 'GPU to be used')
config = tf.app.flags.FLAGS
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
x = tf.placeholder(tf.float32, shape=[None, 448, 448, 3], name='x')
y = tf.placeholder(tf.int64, shape=[None, 448, 448], name='y')
# imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
# labels = tf.placeholder(tf.float32, [None, 224, 224, 3])
imgs = tf.placeholder(tf.float32, [None, 448, 448, 3])
labels = tf.placeholder(tf.float32, [None, 448, 448, 3])
img1 = imread('bear.jpg', mode='RGB')
img1 = imresize(img1, (224, 224))
label1 = img1
probs = build_model(imgs,labels)
with tf.Session() as sess:
    init = tf.group(tf.global_variables_initializer(), tf.initialize_local_variables())
    #init = tf.global_variables_initializer()
    sess.run(init)


    #img1 = imread('laska.png', mode='RGB')

    prob = sess.run(probs, feed_dict={imgs: [img1],labels:[label1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]
    for p in preds:
        print class_names[p], prob[p]