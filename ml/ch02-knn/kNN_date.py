#!/usr/bin/env python
# coding=utf-8

import numpy as np


def load_file(filename):
    """从文件中加载数据到矩阵中
    Args:
        filename: 数据文件
    Returns:
        data_mat: 数据矩阵 n*3
        label_vec: data_mat对应的label
    """
    fr = open(filename)  # 默认是r打开
    line_count = len(fr.readlines())
    # 数据矩阵
    data_mat = np.zeros((line_count, 3))
    label_vec = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()     # 去掉回车字符
        data_list = line.split('\t')
        # 把数据填充到data_mat和label_vec中
        data_mat[index, :] = data_list[0:3]
        label_vec.append(int(data_list[-1]))
        index += 1
    return data_mat, label_vec


def auto_norm(data_mat):
    """
    归一化特征值 new_val=(old_val-min)/(max-min)
    Args:
        data_mat: 数据矩阵
    Returns:
        norm_data_mat: 归一化后的数据矩阵
        ranges: 变化范围
        min_vals: 最小值
    """
    # 按列来取最小值，最大值，0:列，1:行
    min_vals = data_mat.min(0)
    max_vals = data_mat.max(0)
    ranges = max_vals - min_vals
    norm_data_mat = np.zeros(np.shape(data_mat))
    num = data_mat.shape[0]
    # 减去min 把min_vals复制num行
    norm_data_mat = data_mat - np.tile(min_vals, (num, 1))
    # 除以ranges
    norm_data_mat = norm_data_mat / np.tile(ranges, (num, 1))
    return norm_data_mat, ranges, min_vals


def go_knn_date():
    """ knn_date约会主函数
    """
    raw_data_mat, label_vec = load_file('./datingTestSet2.txt')
    data_mat, ranges, min_vals = auto_norm(raw_data_mat)
    print raw_data_mat
    print data_mat
    print ranges, min_vals


if __name__ == '__main__':
    go_knn_date()
