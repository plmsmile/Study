#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


N_INPUT = 4
N_HIDDEN = 100
N_OUTPUT = N_INPUT
BETA = tf.constant(3.0)
LAMBDA = tf.constant(0.0001)
EPSILON = 0.0001
RHO = 0.1


def diff(input_data, output_data):
    ''' input and output's diff '''
    separate_loss = tf.pow(tf.subtract(output_data, input_data), 2)
    loss = tf.reduce_sum(separate_loss)
    return loss


def main(_):
    
    weights = {
        'hidden': tf.Variable(tf.random_normal([N_INPUT, N_HIDDEN]), name = 'w_hidden'),
        'out': tf.Variable(tf.random_normal([N_HIDDEN, N_OUTPUT]), name = 'w_out')
    }

    biases = {
        'hidden': tf.Variable(tf.random_normal([N_HIDDEN], name = 'b_hidden')),
        'out': tf.Variable(tf.random_normal([N_OUTPUT]), name = "b_out")
    }

    def KLD(p, q):
        '''散度'''
        invrho = tf.subtract(tf.constant(1.), p)
        invrhohat = tf.subtract(tf.constant(1.), q)
        t1 = tf.multiply(p, tf.log(tf.div(p, q)))
        t2 = tf.multiply(invrho, tf.log(tf.div(invrho, invrhohat)))
        addrho = tf.add(t1, t2)
        return tf.reduce_sum(addrho)

    
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, N_INPUT], name = 'x_input')

    with tf.name_scope('hidden_layer'):
        hidden_tmp = tf.add(tf.matmul(x, weights['hidden']), biases['hidden'])
        hidden_res = tf.sigmoid(hidden_tmp)

    with tf.name_scope('output_layer'):
        out_tmp = tf.add(tf.matmul(hidden_res, weights['out']), biases['out'])
        out_res = tf.nn.softmax(out_tmp)

    with tf.name_scope('loss'):
        cost_j = tf.reduce_sum(tf.pow(tf.subtract(out_res, x), 2))

    with tf.name_scope('cost_sparse'):
        # kl divergence
        rho_hat = tf.div(tf.reduce_sum(hidden_res), N_HIDDEN)
        cost_sparse = tf.multiply(BETA, KLD(RHO, rho_hat))

    with tf.name_scope('cost_reg'):
        t = tf.add(tf.nn.l2_loss(weights['hidden']), tf.nn.l2_loss(weights['out']))
        cost_reg = tf.multiply(LAMBDA, t)

    with tf.name_scope('cost'):
        cost = tf.add(tf.add(cost_j, cost_reg), cost_sparse)


    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        input_data = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]], float)
        
        for i in range(1000):
            sess.run(optimizer, {x:input_data})
            if i % 100 == 0:
                fetch = [out_res, cost_sparse]
                tmp = sess.run(fetch, {x:input_data})
                print tmp
                print i, sess.run(diff(tmp[0], input_data))

        tmp = sess.run(out_res, {x:input_data})
        print tmp


if __name__ == '__main__':
    tf.app.run()














