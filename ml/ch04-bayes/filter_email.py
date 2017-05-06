#!usr/bin/env python
# coding=utf-8

'''
使用朴素贝叶斯过滤垃圾邮件

@author: PLM
@date: 2017-05-06
'''

import re
import random

import bayes as bys

def parse_str(big_str):
    ''' 解析文本为单词列表
    Args:
        big_str: 长文本
    Returns:
        单词列表
    '''
    # 以任何非单词字符切割
    word_list = re.split(r'\W*', big_str)
    # 只保留长度大于3的单词，并且全部转化为小写
    return [word.lower() for word in word_list if len(word) > 2]


def load_dataset(spam_dir, ham_dir):
    ''' 从文件夹中加载文件
    Args:
        spam_dir: 垃圾邮件文件夹
        ham_dir: 正常邮件文件夹
    Returns:
        email_list: 邮件列表
        class_list: 分类好的列表
    '''
    email_list = []
    class_list = []
    txt_num = 25    # 每个文件夹有25个文件
    for i in range(1, txt_num + 1):
        for j in range(2):
            file_dir = spam_dir if j == 1 else ham_dir
            f = open(('{}/{}.txt').format(file_dir, i))
            f_str = f.read()
            f.close()
            words = parse_str(f_str)
            email_list.append(words)    # 邮件列表
            class_list.append(j)        # 分类标签，1垃圾邮件，0非垃圾邮件
    return email_list, class_list


def get_train_test_indices(data_num):
    ''' 划分训练集和测试集
    Args:
        data_num: 数据集的数量
    Returns:
        train_indices: 训练集的索引列表
        test_indices: 测试集的索引列表
    '''
    train_indices = range(data_num)
    test_ratio = 0.3        # 测试数据的比例
    test_num = int(data_num * test_ratio)
    test_indices = random.sample(train_indices, test_num)
    for i in test_indices:
        train_indices.remove(i)
    return train_indices, test_indices


def go_bayes_email():
    ''' 贝叶斯垃圾邮件过滤主程序
    Returns:
        error_rate: 错误率
    '''
    # 源数据
    email_list, class_list = load_dataset('email/spam', 'email/ham')
    # 总的词汇表
    vocab_list = bys.get_vocab_list(email_list)
    # 训练数据，测试数据的索引列表
    data_num = len(email_list)
    train_indices, test_indices = get_train_test_indices(data_num)
    # 训练数据的矩阵和分类列表
    train_mat = []
    train_class = []
    for i in train_indices:
        vec = bys.get_doc_vec(email_list[i], vocab_list)
        train_mat.append(vec)
        train_class.append(class_list[i])
    # 训练数据
    p0_vec, p1_vec, p1 = bys.train_nb0(train_mat, train_class)
    
    # 测试数据
    error_count = 0
    for i in test_indices:
        vec = bys.get_doc_vec(email_list[i], vocab_list)
        res = bys.classify_nb(vec, p0_vec, p1_vec, p1)
        if res != class_list[i]:
            error_count += 1
    error_rate = error_count / float(data_num)
    print 'error=%d, rate=%s, test=%d, all=%d' % (error_count, error_rate, len(test_indices),
                    data_num)
    return error_rate


def test_bayes_email():
    ''' 执行多次go_bayes_email，计算平均错误率 '''
    times = 100
    error_rate_sum = 0.0
    for i in range(10):
        error_rate_sum += go_bayes_email()
    print 'average_rate = %s' % (error_rate_sum / 10)

if __name__ == '__main__':
    # go_bayes_email()
    test_bayes_email()
