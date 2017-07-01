#!/usr/bin/env python
#-*-coding: utf-8 -*-

'''
多层感知机，数字识别

@author plm
@date 2017-06-29
'''


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf



def weight_variable(shape):
    '''产生一个权值Variable
    Args:
        shape:[]w的形状 
    Returns:
        Variable with initial 参数
    '''
    # 截断的正态分布，避免0梯度，方差为0.1
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    '''产生一个bias variable
    Args:
        shape: bias shape []
    Returns:
        tf.Variable
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)



def mlp_main():
    '''mlp'''
    ## 1. 定义算法公式
    # data
    mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)
    # 各个层的神经元数量
    in_units = 784  # 输入层，28*28
    h1_units = 300  # 隐含层，此模型200-1000隐含层节点区别不大
    out_units = 10  # 输出层，10类
    # 隐含层和输出层的参数
    h1_w = weight_variable([in_units, h1_units])
    h1_b = bias_variable([h1_units])
    out_w = weight_variable([h1_units, out_units])
    out_b = bias_variable([out_units])

    # 输入x
    x = tf.placeholder(tf.float32, [None, in_units])
    # dropout的保留比例，会以概率p随机将一部分节点设置为0
    keep_prob = tf.placeholder(tf.float32)
    
    # 隐含层
    h1_v = tf.matmul(x, h1_w) + h1_b 
    # 使用relu作为激活函数，解决梯度弥散问题，适合深层神经网络
    # 1. 1-4%神经元被激活 2. 多层反向传播梯度不会减少 3. 单侧抑制
    h1 = tf.nn.relu(h1_v)
    h1_drop = tf.nn.dropout(h1, keep_prob)
    
    # 输出层
    out_v = tf.matmul(h1_drop, out_w) + out_b
    # 多分类
    y = tf.nn.softmax(out_v)    # 预测的y

    ## 2. 定义loss(交叉熵)选择优化器优化
    y_ = tf.placeholder(tf.float32, [None, 10]) # 真实的y
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_ * tf.log(y), axis=1))
    all_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
    cross_entropy = tf.reduce_mean(all_cross_entropy)
    train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)

    ## 3. 训练
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    for i in range(3000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        train_step.run({x:batch_xs, y_: batch_ys, keep_prob:0.75})

    ## 4. 测试
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    res = accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    print res

if __name__ == '__main__':
    mlp_main()
