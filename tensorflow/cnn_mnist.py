#!\usr\bin\env python
# -*- coding: utf-8 -*-

'''
使用cnn实现手写数字识别
'''


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def weight_variable(shape):
    ''' 产生一个权值Variable
    Args:
        shape: 变量的形状，如[5, 5, 1, 32]
    Returns:
        variable: 初始化好的变量
    '''
    # 避免0梯度，使用截断的正态分布，均值是0, 标准差为0.1，即N~(0, 0.01)
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bais_variable(shape, init_value=0.1):
    ''' 产生一个偏置Variable
    Args:
        shape: 偏置的形状，如[10]
        init_value: 初始值，默认为0.1
    Returns:
        variable: 拥有初始值的variable
    '''
    initial = tf.constant(init_value, shape=shape)
    return tf.Variable(initial)


def conv2d(input_x, filter_w, strides=[1, 1, 1, 1], padding='SAME'):
    '''产生2维的卷积
    Args:
        input_x: 需要做卷积的图像，4维tensor。[batch, in_height, in_witdh, in_channels]
                 batch指一次batch的数量
        filter_w: 卷积核，[f_height, f_width, in_channels, out_channels]
                 最后的out_channels是指卷积核的数量，有多少则提取多少张Feature Map
        strides: 步长，[1, stride, stride, 1], 中间分别是步长
        padding: 'SAME'和'VALID'，前者会在图片边缘，不足的补0，后者只会恰当好
    Returns:
        卷积结果，[batch, h, w, channels]
    '''
    return tf.nn.conv2d(input_x, filter_w, strides, padding)


def max_pool(value, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'):
    ''' 最大池化
    Args:
        value: 池化的输入，一般是FM，[batch, height, width, channels]
        ksize: 池化窗口的大小，[1, height, width, 1]，因为不想在batch和channel上池化
        strides: 每一维上滑动的步长，[1, stride, stride, 1]
        padding: 'SAME' or 'VALID'
    Returns:
        池化结果，[batch, h, w, channels]
    '''
    return tf.nn.max_pool(value, ksize, strides, padding)


def go_model():
    ''' 两个卷积层+1个全连接层'''
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

    # 1. 准备数据
    # x, 784个像素
    x = tf.placeholder(tf.float32, [None, 784])
    # 真实标签
    y_ = tf.placeholder(tf.float32, [None, 10])
    # 784还原回28*28像素，-1代表不知道多少个样本，1代表通道
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 2. 第一个卷积层
    # 卷积核大小5*5, 数量32，提取32种特征，1个颜色通道
    w_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bais_variable([32])
    # 卷积操作，默认步长1*1，所以依然28*28
    res_conv1 = conv2d(x_image, w_conv1) + b_conv1
    # relu激活函数非线性处理
    h_conv1 = tf.nn.relu(res_conv1)
    # 池化处理，得到14*14*32
    h_pool1 = max_pool(h_conv1)

    # 3. 第二个卷积层
    # 大小5*5，数量64，前面输入32
    w_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bais_variable([64])
    res_conv2 = conv2d(h_pool1, w_conv2) + b_conv2
    h_conv2 = tf.nn.relu(res_conv2)
    # 池化，得到7*7*64
    h_pool2 = max_pool(h_conv2)

    # 4. 全连接层，把7*7*64打平，全连接到1024个神经元上
    w_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bais_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # 5. Dropout&Softmax
    keep_prob = tf.placeholder(tf.float32)
    # 减轻过拟合，以一些将一部分节点设置为0，实质上创造了很多新的随机样本
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    w_fc2 = weight_variable([1024, 10])
    b_fc2 = bais_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # 6. 定义一些信息
    all_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(all_cross_entropy)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    # axis=0 列，axis=1 行
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 7. 训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            feed_dict = {x: batch[0], y_: batch[1], keep_prob: 0.5}
            if i % 100 == 0:
                feed_dict[keep_prob] = 1.0
                train_accuracy = accuracy.eval(feed_dict)
                print ('step %d, accuracy %g' % (i, train_accuracy))
            train_step.run(feed_dict=feed_dict)

        feed_dict = {
            x: mnist.test.images,
            y_: mnist.test.labels,
            keep_prob: 1.0}
        res = accuracy.eval(feed_dict=feed_dict)
        print ('test accuracy %g' % res)


if __name__ == '__main__':
    go_model()
