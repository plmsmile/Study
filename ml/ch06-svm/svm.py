#!/usr/bin/env python
# encoding=utf-8

'''
SVM支持向量机

@author:PLM
@date:2017-05-08
'''

import numpy as np
import numpy.matlib


def load_dataset(file_name):
    ''' 加载文件中的数据，点集合label
    Args:
        file_name: 文件名，含路径
    Returns:
        point_list: 数据点的列表
        label_list: 标签列表
    '''
    point_list = []
    label_list = []
    f = open(file_name)
    for line in f.readlines():
        x, y, label = line.strip().split('\t')
        point_list.append((float(x), float(y)))
        label_list.append(int(label))
    return point_list, label_list


def select_x_rand(i, m):
    ''' 随机选择一个不等于i的整数，在[0,m)之间
    Args:
        i: 不选择的数
        m: 范围，[0,m)
    Returns:
        x: [0,m)间不等于i的整数
    '''
    x = i
    while x == i:
        x = np.random.randint(0, m)
    return x


def check_a(a, max_a, min_a):
    ''' 调整大于max_a或小于min_a的alpha值 
    Args:
        a: 当前的alpha值
        max_a: 最大的alpha值
        min_a: 最小的alpha值
    Returns:
        b: 调整后的a值
    '''
    b = a
    if a > max_a:
        b = max_a
    if a < min_a:
        b = min_a
    return b


def smo_simple(point_list, label_list, max_a, error_rate, iter_num):
    ''' 简版smo，计算一系列alpha和b
    Args:
        point_list: (x, y)点列表
        label_list: 数据点对应的类别标签，-1或1'
        max_a: alpha的最大值，松弛变量，控制最大化间隔，保证大部分点的函数间隔小于1.0 
        error_rate: 容错率
        iter_num: 迭代次数
    Returns:
        alpha: alpha矩阵，m*1
        b: 常数b
    '''
    # 点 m*n
    data_mat = np.mat(point_list)
    # 标签 m*1
    label_mat = np.mat(label_list).transpose()
    # m个数据
    m = data_mat.shape[0]
    # n个特征
    n = data_mat.shape[1]
    # 常数b
    b = 0
    # alpha矩阵 m*1
    # a_mat = np.mat(np.zeros((m,1)))
    a_mat = np.matlib.zeros((m,1))
    iter_count = 0
    while iter_count < 1:
        a_pairs_changed = False
        for i in range(1):
            # a点*label, 100*1矩阵
            a_label = np.multiply(a_mat, label_mat)
            # 1*2 转置为2*1
            d_i = data_mat[i,:].T
            print data_mat.shape
            print d_i.shape
            # 矢量乘，m*2 2*1 = m*1
            data_d_i = data_mat * d_i
            print data_d_i.shape
        iter_count += 1

def test_svm():
    points, labels = load_dataset('./testSet.txt')
    smo_simple(points, labels, 0.6, 0.001, 40)


if __name__ == '__main__':
    test_svm()
