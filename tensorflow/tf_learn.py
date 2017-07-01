# ~/usr/bin/env python
# encoding=utf8

'''
tensorflow first demo
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 去掉警告

import tensorflow as tf
import numpy as np


def get_nodes():
    """ 返回两个constant node"""
    node1 = tf.constant(3.0, tf.float32)
    node2 = tf.constant(4.0)  # 默认也是tf.float32
    return node1, node2


def constant_demo():
    """两个constant demo"""
    node1, node2 = get_nodes()
    print node1, node2
    sess = tf.Session()
    res = sess.run([node1, node2])
    print res


def constant_add():
    """ 一个操作也是一个node"""
    node1, node2 = get_nodes()
    node3 = tf.add(node1, node2)
    print node3
    sess = tf.Session()
    res = sess.run(node3)
    print res


def placeholder_demo():
    a = tf.placeholder(tf.float32)
    b = tf.placeholder(tf.float32)
    adder_node = a + b
    # adder_node = tf.add(a, b)    # 简写
    sess = tf.Session()
    res1 = sess.run(adder_node, {a: 3, b: 4.5})
    res2 = sess.run(adder_node, {a: [1, 2], b: [3, 4]})
    print res1, res2
    add_and_triple = adder_node * 3
    res3 = sess.run(add_and_triple, {a: 3, b: 4})
    print res3


def test_variables():
    """ 变量可以给任意的值 """
    W = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    x = tf.placeholder(tf.float32)
    linear_model = W * x + b
    # 初始化变量值
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # 输入x计算模型的值
    res = sess.run(linear_model, {x: [1, 2, 3, 4]})
    print res

    # 计算误差
    y = tf.placeholder(tf.float32)
    squared_deltas = tf.square(linear_model - y)
    loss = tf.reduce_sum(squared_deltas)
    res_loss = sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print res_loss

    # 为w和b重新赋值，再运行，误差为0
    fixW = tf.assign(W, [-1.])
    fixb = tf.assign(b, [1.])
    sess.run([fixW, fixb])
    res_loss_new = sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
    print res_loss_new


def lr_demo():
    '''线性回归，梯度下降，训练，完整demo'''
    # 模型参数
    w = tf.Variable([.3], tf.float32)
    b = tf.Variable([-.3], tf.float32)
    # 输入和输出
    x = tf.placeholder(tf.float32)
    linear_model = w * x + b
    y = tf.placeholder(tf.float32)
    # 误差
    loss = tf.reduce_sum(tf.square(linear_model - y))
    # 优化器 梯度下降法
    optimizer = tf.train.GradientDescentOptimizer(0.01)  # 学习率
    train = optimizer.minimize(loss)
    # 训练数据
    x_train = [1, 2, 3, 4]
    y_train = [0, -1, -2, -3]
    # 训练
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)  # 参数初始化
    for i in range(1000):   # 这个训练次数怎么确定？？
        sess.run(train, {x: x_train, y: y_train})

    # 评估结果
    cur_w, cur_b, cur_loss = sess.run([w, b, loss], {x: x_train, y: y_train})
    print ("w: %s, b: %s, loss: %s" % (cur_w, cur_b, cur_loss))


def lr_contrib_learn():
    '''使用高层抽象来实现线性回归'''
    features = [tf.contrib.layers.real_valued_column("x", dimension=1)]
    estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
    x = np.array([1., 2., 3., 4.])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn(
        {'x': x}, y, batch_size=4, num_epochs=1000)

    estimator.fit(input_fn=input_fn, steps=1000)
    print (estimator.evaluate(input_fn=input_fn))


def lr_model(features, labels, mode):
    ''' 使用高层抽象建立自己的线性模型'''
    # 线性模型
    w = tf.get_variable('w', [1], dtype=tf.float64)
    b = tf.get_variable('b', [1], dtype=tf.float64)
    y = w * features['x'] + b
    # loss sub-graph
    loss = tf.reduce_sum(tf.square(y - labels))
    # train sub-graph
    global_step = tf.train.get_global_step()
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = tf.group(optimizer.minimize(loss), tf.assign_add(global_step, 1))
    # ModelFnOps 把这些subgraphs连接起来完成一个功能
    return tf.contrib.learn.ModelFnOps(
        mode=mode, predictions=y, loss=loss, train_op=train)


def lr_self_model():
    x = np.array([1., 2., 3., 4., ])
    y = np.array([0., -1., -2., -3.])
    input_fn = tf.contrib.learn.io.numpy_input_fn(
        {'x': x}, y, batch_size=4, num_epochs=1000)
    # estimator
    estimator = tf.contrib.learn.Estimator(model_fn=lr_model)
    # train
    estimator.fit(input_fn=input_fn, steps=1000)
    # evaluate our model
    print estimator.evaluate(input_fn=input_fn, steps=10)


def test_div():
    '''div test'''
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    res = tf.div(x, y)
    xv = np.array([[2, 4], [4, 8]])
    # yv = np.array([[1, 2], [1, 2]])
    yv = np.array([2])
    sess = tf.Session()
    resv = sess.run(res, {x:xv, y:yv})
    print x.shape
    print resv


def test_convd():
    pass


if __name__ == '__main__':
    # constant_demo()
    # constant_add()
    # placeholder_demo()
    # test_variables()
    # lr_demo()
    # lr_contrib_learn()
    # lr_self_model()
    test_div()
