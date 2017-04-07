#!/usr/bin/env python
import time, os
import argparse
import numpy as np
import tensorflow as tf
from util import *
import skimage.io as io

def main(config):

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    WIDTH = {'480p': 854, '1080p': 1920}
    HEIGHT = {'480p': 480, '1080p': 1080}
    DB_LEN = {'train': 4044, 'val': 1890}
    t0 = time.time()
    tfrecords_filename = os.path.join('data','DAVIS_'+config.res+'_'+args.set+'.tfrecords')
    filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=config.num_epochs)
    x, y = input_pipeline(filename_queue, config.batch_size, WIDTH[config.res], HEIGHT[config.res])
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    if config.training == 1:
        logits, loss = build_model(x, y)

        learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
        tf.summary.scalar('learning_rate', learning_rate)

        num_param = 0
        vars_trainable = tf.trainable_variables()
        for var in vars_trainable:
            num_param += np.prod(var.get_shape()).value
            tf.summary.histogram(var.name, var)
        print('\nTotal nummber of parameters = %d' % num_param)
        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, var_list=vars_trainable)
    else:
        logits, loss = build_model(x, y, training=False)
    tf.summary.scalar('loss', loss)
    pred_train = tf.sigmoid(logits)
    result_train = tf.concat([y, pred_train], axis=2)
    result_train = tf.cast(result_train*255, tf.uint8)
    tf.summary.image('result_train_img', x, max_outputs=config.batch_size)
    tf.summary.image('result_train', result_train, max_outputs=config.batch_size)

    sum_all = tf.summary.merge_all()
    print('Finished initializing in %.2f seconds.' % (time.time() - t0))

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)

        saver = tf.train.Saver(max_to_keep=10)
        ckpt = tf.train.get_checkpoint_state('./checkpoint')
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join('./checkpoint', ckpt_name))
            print('[*] Success to read {}'.format(ckpt_name))
        else:
            if config.training:
                print('[*] Failed to find a checkpoint. Start training from scratch ...')
            else:
                raise ValueError('[*] Failed to find a checkpoint.')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        writer = tf.summary.FileWriter("./logs", sess.graph)
        t0 = time.time()
        total_count = 0
        if config.training == 1:
            print("Starting training")
            for epoch in range(config.num_epochs):
                lr = config.init_learning_rate * config.learning_rate_decay**epoch

                for k in range(DB_LEN[config.set] // config.batch_size):
                    l_train, _, a = sess.run([loss, train_step, y], feed_dict={learning_rate: lr})

                    if total_count % (20) == 0:
                        writer.add_summary(sess.run(sum_all, feed_dict={learning_rate: lr}), total_count)
                    total_count += 1
                    m, s = divmod(time.time() - t0, 60)
                    h, m = divmod(m, 60)
                    print('Epoch: [%4d/%4d], [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                    % (epoch, config.num_epochs, k, DB_LEN[config.set] // config.batch_size, h, m, s, l_train))

                if epoch % 10 == 0:
                    print('Saving checkpoint ...')
                    saver.save(sess, './checkpoint/Davis.ckpt', global_step=epoch)
        else:
            print("Starting validation")
            for k in range(2079 // config.batch_size):
                summary, l_val = sess.run([sum_all, loss])
                writer.add_summary(summary, total_count)
                total_count += 1
                m, s = divmod(time.time() - t0, 60)
                h, m = divmod(m, 60)
                print('Batch: [%4d/%4d], Time: [%02d:%02d:%02d], loss: %.4f'
                % (k, DB_LEN[config.set] // config.batch_size, h, m, s, l_val))
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DAVIS Video Object Segmentation')
    parser.add_argument('--batch-size', dest='batch_size', help='number of images in each batch', default=4, type=int)
    parser.add_argument('--num-epochs', dest='num_epochs', help='total number of epochs to run for training', default=100, type=int)
    parser.add_argument('--training', dest='training', help='if true, train the model; otherwise evaluate the existing model', default=1, type=int)
    parser.add_argument('--init-lr', dest='init_learning_rate', help='initial learning rate', default=1e-4, type=float)
    parser.add_argument('--lr-decay', dest='learning_rate_decay', help='ratio of decaying the learning rate after each epoch', default=0.95, type=float)
    parser.add_argument('--gpu', dest='gpu', help='GPU device id to be used', default='0', type=str)
    parser.add_argument('--dataset-path', dest='dataset_path', help='path to dataset directory', default='./DAVIS', type=str)
    parser.add_argument('--resolution', dest='res', help='480p or 1080p', default='480p', type=str)
    parser.add_argument('--set', dest='set', help='train or val', default='train', type=str)
    args = parser.parse_args()
    main(args)
