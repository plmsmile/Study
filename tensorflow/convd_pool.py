#!usr/bin/env python
# -*- coding: utf-8 -*-

'''
test con2d, max_pool

@author plm
@date 2017-06-28
'''

import tensorflow as tf


def test_max_pool():
    a = tf.constant([
         [1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0],
         [8.0, 7.0, 6.0, 5.0],
         [4.0, 3.0, 2.0, 1.0],
    ])
    a = tf.reshape(a, [1, 4, 4, 1])

    pooling = tf.nn.max_pool(a, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    with tf.Session() as sess:
        print("image:")
        image = sess.run(a)
        print image.shape
        print (image)
        print("reslut:")
        result = sess.run(pooling)
        print result.shape
        print (result[0])


if __name__ == '__main__':
    test_max_pool()
