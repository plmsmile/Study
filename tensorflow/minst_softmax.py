#!usr/bin/env python
#-*- coding: utf-8 -*-

'''
minist softmax regression

@author plm
@date 2017-06-27
'''

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


FLAGS = None


def main(_):
    # 导入数据
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

    # 模型
    pixels = 28 * 28
    x = tf.placeholder(dtype=tf.float32, shape=[None, pixels])
    #W = tf.Variable([pixels, 10], dtype=tf.float32)
    #b = tf.Variable([10], dtype=tf.float32)
    W = tf.Variable(tf.zeros([pixels, 10]), dtype=tf.float32)
    b = tf.Variable(tf.zeros([10]), dtype=tf.float32)
    y = tf.add(tf.matmul(x, W), b)
    #y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))

    # loss和优化器
    y_ = tf.placeholder(tf.float32, [None, 10])

    all_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(all_cross_entropy)
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=1))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # sess和初始化
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # 训练
    for _ in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    res = sess.run(
        accuracy,
        feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels})
    print res


if __name__ == '__main__':
    tf.app.run(main=main)

