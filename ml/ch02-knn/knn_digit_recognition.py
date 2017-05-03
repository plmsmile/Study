#!/usr/bin/env python
# coding=utf-8

import numpy as np
import os

import kNN


def img2vector(filename):
    """
    把数字文件32*32转换为1*1024的向量
    Args:
        filename: 文件地址
    Returns:
        num_vec: 数字组成的向量
    """
    num_vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        line = line.strip()
        for j in range(32):
            idx = i * 32 + j
            num_vec[0, idx] = line[j]
    return num_vec


def load_training_data(folder_name):
    """
    加载训练数据
    Args:
        folder_name: 文件夹的地址
    Returns:

    """
    file_list = os.listdir(folder_name)
    m = len(file_list)
    # 数据矩阵
    mat = np.zeros((m, 1024))
    # 标记list
    label_list = []
    for i in range(m):
        file_name = file_list[i]
        vec = img2vector("{}/{}".format(folder_name, file_name))
        label_list.append(file_name[0])
        mat[i, ...] = vec[0, ...]
    return mat, label_list


def classify_knn(inx, data_mat, labels, k):
    """ knn分类算法
    1. 距离计算 2. 选择距离最小的k个点 3. 排序选择分类最多的label
    Args:
        inx: 要分类的数据inx，1*n的矩阵
        data_mat: 已知的数据矩阵，m*n
        labels: data_mat对应的label
        k: 最后从前k个中进行选择
    Returns:
        res_label: 最终分类的label
    """
    m = data_mat.shape[0]
    # 把inx转换成m*n的矩阵，复制m行
    inx_t = np.tile(inx, (m, 1))
    # 1. inx与data的欧式距离
    # 差 平方
    squared_diff = (data_mat - inx_t) ** 2
    # 按行求平方和，即inx与每个样本的距平方。0列 1行
    squared_dis = squared_diff.sum(1)
    # 求得距离，开方
    dis = squared_dis ** 0.5
    # 对距离进行排序，得到索引
    sorted_dis_index = np.argsort(dis)

    # 2. 选择距离最小的k个点
    # 存放x可能的label和数量
    x_labels = {}
    for i in range(k):
        label = labels[sorted_dis_index[i]]
        x_labels[label] = x_labels.get(label, 0) + 1

    # 3. 选择分类最多的label
    sorted_x_labels = sorted(
        x_labels.items(),
        key=lambda d: d[1],
        reverse=True)
    return sorted_x_labels[0][0]


def go_knn_digit_recognition():
    """
    数字识别主程序
    """
    training_folder = './trainingDigits/'
    test_folder = './testDigits/'
    train_mat, train_labels = load_training_data(training_folder)
    test_file_list = os.listdir(test_folder)
    tm = len(test_file_list)
    error_count = 0
    for i in range(tm):
        t_filename = test_file_list[i]
        t_vec = img2vector("{}/{}".format(test_folder, t_filename))
        res = classify_knn(t_vec, train_mat, train_labels, 3)
        if res != t_filename[0]:
            error_count += 1
    print '%d / %d = %f' % (error_count, tm, error_count / float(tm))


if __name__ == '__main__':
    go_knn_digit_recognition()
