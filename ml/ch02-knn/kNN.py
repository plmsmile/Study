#!/usr/bin/env python
# coding=utf-8

import numpy as np


def create_data():
    """初始化已知数据
    Returns:
        group: 数据
        labels: group点的label

    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inx, data, labels, k):
    """ k-近邻算法
    1. 距离计算 2. 选择距离最小的k个点 3. 排序
    Args:
        inx: 输入值
        data: 已知的数据集
        labels: data对应的label
        k: neighbor的数量
    Returns:
        分类结果
    """
    data_size = data.shape[0]
    # 把inx转换为与data相同的矩阵，(data_size,2) 复制了data_size行
    t_inx = np.tile(inx, (data_size, 1))
    # 求inx与data的差别矩阵
    diff_mat = t_inx - data
    # 平方
    sq_diff_mat = diff_mat ** 2
    # 计算x与每一个data的距离
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = np.sqrt(sq_distances)
    # 对距离进行排序，得到index
    sorted_distance_indices = np.argsort(distances)

    # 选择距离最小的k个点
    # label_dict, 记录每个label的数量
    label_count = {}
    for i in range(k):
        vote_i_label = labels[sorted_distance_indices[i]]
        label_count[vote_i_label] = label_count.get(vote_i_label, 0) + 1
    # 对label_count进行排序，选择count最多的作为返回结果
    sorted_label_count = sorted(
        label_count.items(),
        key=lambda d: d[1],
        reverse=True)
    return sorted_label_count[0][0]


def test_classify0():
    """测试classify0
    """
    data, labels = create_data()
    inx = [1, 1.3]
    k = 3
    print inx, classify0(inx, data, labels, k)


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


def test_knn_date():
    """ 测试约会数据
    """
    raw_data_mat, label_vec = load_file('./datingTestSet2.txt')
    data_mat, ranges, min_vals = auto_norm(raw_data_mat)
    m = data_mat.shape[0]
    test_ratio = 0.1
    test_num = int(m * test_ratio)
    error_count = 0
    for i in range(test_num):
        res = classify0(
            data_mat[i, :], data_mat[test_num:m, :], label_vec[test_num:m], 3)
        print 'No %d, real=%s, predict=%s' % (i, label_vec[i], res)
        if res != label_vec[i]:
            error_count += 1
    print 'error_count = %d, error_rate = %f' % (error_count, error_count / float(test_num))


def go_knn_date(input_val, data_filename):
    """
    使用knn算法预测input_val属于哪一类人
    Args:
       input_val: 要判断的人
       data_filename: 已经收集的数据文件
    Returns: 无
    """
    # 加载已有数据，并进行归一化
    raw_data_mat, label_vec = load_file(data_filename)
    data_mat, ranges, min_vals = auto_norm(raw_data_mat)
    # 对input_val归一化
    input_val = input_val - min_vals
    input_val = input_val / ranges
    # 调用classify0进行预测
    res_list = ['不感兴趣', '有一点魅力', '很大的魅力']
    res = classify0(input_val, data_mat, label_vec, 3)
    print res_list[res - 1]


if __name__ == '__main__':

    test_knn_date()
    #go_knn_date([7000, 15, 0.5], './datingTestSet2.txt')
