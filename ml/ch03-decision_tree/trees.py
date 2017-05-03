#!/usr/bin/env python
# coding=utf-8

"""
决策树
"""

import numpy as np
import math


def cal_shannon_ent(dataset):
    """
    计算香农熵 求和 -p*log(p, 2), p=p(xi)
    1. 求每个label的数量 2. 求和
    Args:
        dataset: 数据集，矩阵形式
    Returns:
        shannon_ent: 香农熵
    """
    m = dataset.shape[0]
    # 1. 计算每个label的数量
    label_counts = {}
    for vec in dataset:
        label = vec[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    # 2. 计算香农熵
    shannon_ent = 0.0
    for label in label_counts:
        prob = label_counts[label] / float(m)
        shannon_ent -= prob * math.log(prob, 2)
    return shannon_ent


def create_dataset():
    """
    创建数据集, 和labels
    Returns:
        dataset: 创建的数据集
    """
    #dt = np.dtype([('feat1', 'i1'), ('feat2', 'i1'), ('feat3', 'S20')])
    # dataset = np.array([(1, 1, 'yes'),
    #                    (1, 1, 'maybe'),
    #                    (1, 0, 'no'),
    #                    (0, 1, 'no'),
    #                    (0, 1, 'no')],
    #                   dtype=dt)
    # dataset = np.array([[1, 1, 'yes'],
    #                    [1, 1, 'maybe'],
    #                    [1, 0, 'maybe'],
    #                    [1, 0, 'no'],
    #                    [0, 1, 'no']])
    #labels = ['nosurfacing', 'flipper']
    dataset = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 1, 0]])
    labels = ['nosurfacing', 'flipper']
    return dataset, labels


def split_dataset(dataset, feat_idx, feat_value):
    """
    按照特征划分数据集
    Args:
        dataset: 数据集
        feat_idx: 要划分的特征的index
        feat_value: 该特征的值
    Returns:
        target_dataset: 满足该特征值的数据集，不包含特征这一列
    """
    m = dataset.shape[0]
    dataset_list = []
    for i in range(m):
        vec = dataset[i]
        if vec[feat_idx] == feat_value:
            # 去掉特征这一项
            data_line = []
            data_line.extend(vec[:feat_idx])
            data_line.extend(vec[feat_idx + 1:])
            dataset_list.append(data_line)
    return np.array(dataset_list)


def choose_best_feature_to_split(dataset):
    """
    选择最好的数据集划分方式，信息增益最大, base_ent-ent
    Args:
        dataset: 输入数据集, m*n，第n列是特征 
    Returns:
        best_feat_idx: 最好特征的idx
    """
    feat_num = dataset.shape[1] - 1
    best_info_gain = 0
    best_feat_idx = -1
    m = float(dataset.shape[0])
    # 基础熵
    base_entropy = cal_shannon_ent(dataset)
    for i in range(feat_num):
        feat_value_list = [line[i] for line in dataset]
        uniq_values = set(feat_value_list)
        new_entropy = 0.0
        for value in uniq_values:
            sub_dataset = split_dataset(dataset, i, value)
            # print sub_dataset
            prob = sub_dataset.shape[0] / m
            t_ent = cal_shannon_ent(sub_dataset)
            # print 'prob = %f, t_ent = %f' % (prob, t_ent)
            new_entropy += prob * t_ent
        info_gain = base_entropy - new_entropy
        # print 'i = %d, new = %f, info_gain = %f' % (i, new_entropy, info_gain)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feat_idx = i
    return best_feat_idx


def major_label(label_list):
    """
    投票表决
    Args:
        label_list: 分类列表
    Returns:
        返回数量最多的分类
    """
    label_counts = {}
    for label in label_list:
        label_counts[label] = label_counts.get(label, 0) + 1
    sorted_label_count = sorted(label_counts.items(), key = lambda d: d[1], reverse=True)
    return sorted_label_count[0][0]
    

def create_tree(dataset, feat_names):
    """
    递归创建决策树
    Args:
        dataset: 数据集，最后一列为label，前面的列为特征
        feat_names: 特征名称列表，与dataset的前n-1列顺序对应上 
    Returns:
        决策树
    """
    label_list = [line[-1] for line in dataset]
    # 1. 所有的label都相同，则返回唯一的这个label
    if label_list.count(label_list[0]) == len(label_list):
        return label_list[0]
    # 2. 特征已经用完，全是label，则投票选举
    if dataset.shape[1] == 1:
        return major_label(label_list)
    # 3. 递归构建树，按照最好特征分类
    best_feat_idx = choose_best_feature_to_split(dataset)
    feat_name_list = feat_names[:]
    current_feat_name = feat_name_list[best_feat_idx] 
    decision_tree = {current_feat_name:{}} 
    # 删除当前feat_name
    del(feat_name_list[best_feat_idx])
    feat_value_list = [example[best_feat_idx] for example in dataset]
    uniq_feat_values = set(feat_value_list)
    for value in uniq_feat_values:
        sub_feat_name_list = feat_name_list[:]
        sub_dataset = split_dataset(dataset, best_feat_idx, value)
        decision_tree[current_feat_name][value] = create_tree(sub_dataset, sub_feat_name_list)
    return decision_tree 


def classify(tree, feat_name_list, test_vec):
    ''' 使用决策树进行分类
    Args:
        tree: 构建好的决策树
        feat_name_list: 特征名称列表，与tree的每一层的名称对应
        test_vec: 要测试的向量
    Returns:
        label: 最终的分类结果
    '''
    cur_feat_name = tree.keys()[0] 
    level = feat_name_list.index(cur_feat_name)
    cur_check_value = test_vec[level]
    children = tree[cur_feat_name]
    label = -1
    for key, value in children.items():
        if cur_check_value == key:
            if isinstance(value, dict):
                label = classify(value, feat_name_list, test_vec)
            else:
                label = value
    return label


def go_decision_tree():
    """
    决策树的主程序
    """
    dataset, feat_names = create_dataset()
    decision_tree = create_tree(dataset, feat_names)
    test_vec = [1, 0]
    label = classify(decision_tree, feat_names, test_vec)
    print label
    #prin choose_best_feature_to_split(dataset)
    # target_dataset = split_dataset(dataset, 0, 1)
    # >print dataset
    #print target_dataset
    # print dataset
    # print cal_shannon_ent(dataset)


if __name__ == '__main__':
    go_decision_tree()
