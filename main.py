from util import *
import tensorflow as tf
import time, os
from numpy import *
from scipy.misc import imsave,imshow
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
x_val = tf.placeholder(tf.float32, shape=[None, 448, 448, 3], name='x_val')
y_val = tf.placeholder(tf.int64, shape=[None, 448, 448], name='y_val') 
logits, loss = build_model(x, y)
logits_val, loss_val = build_model(x_val, y_val, training=False, reuse=True)
num_param = 0
vars_trainable = tf.trainable_variables()
for var in vars_trainable:
    num_param += prod(var.get_shape()).value
    #print(var.name, var.get_shape())
    tf.summary.histogram(var.name, var)

print('\nTotal nummber of parameters = %d' % num_param)

tf.summary.scalar('loss', loss)
tf.summary.scalar('loss_val', loss_val)
pred_train = tf.to_int64(logits>0.5, name = 'pred_train')
result_train = tf.concat([y, pred_train], axis=2)
result_train = tf.cast(255 * tf.reshape(result_train, [-1, 448, 896, 1]), tf.uint8)

pred_val = tf.to_int64(logits_val>0.5, name = 'pred_train_val')
result_val = tf.concat([y_val, pred_val], axis=2)
result_val = tf.cast(255 * tf.reshape(result_val, [-1, 512, 1024, 1]), tf.uint8)

tf.summary.image('result_train', result_train, max_outputs=config.batch_size)
tf.summary.image('result_val', result_val, max_outputs=config.batch_size)

learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
tf.summary.scalar('learning_rate', learning_rate)

sum_all = tf.summary.merge_all()

t0 = time.time()
if config.training:
    print('\nLoading data from ./data/train')
    path1 = './Data/train/images'
    path2 = './Data/train/labels'
    path3 = './Data/val/images'
    path4 = './Data/val/labels'
    list1 = os.listdir(path1)
    list2 = os.listdir(path2)
    list3 = os.listdir(path3)
    list4 = os.listdir(path4)
    images = load_images(path1+list1[0]+'/*.jpg')
    labels = load_images(path2+list2[0]+'/*.png')
    for i in range(1,len(list1)):
        images1 = load_images(path1+list1[i]+'/*.jpg')
        labels1 = load_images(path2+list2[i]+'/*.png')
        images = np.row_stack((images,images1))
        labels = np.row_stack((labels,labels1)) 

    print('\nLoading data from ./data/val')
    images_val = load_images(path3+list3[0]+'/*.jpg')
    labels_val = load_images(path4+list4[0]+'/*.png')
    for i in range(1,len(list1)):
        images_val1 = load_images(path3+list3[i]+'/*.jpg')
        labels_val1 = load_images(path4+list4[i]+'/*.png')
        images_val = np.row_stack((images_val,images_val1))
        labels_val = np.row_stack((labels_val,labels_val1)) 
else:
    print('\nLoading data from ./data/val')
    #images_val = load_images('./data/val/images/*.png')
    #labels_val = load_images('./data/val/labels/*.png')











'''label mission
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
        print class_names[p], prob[p]'''