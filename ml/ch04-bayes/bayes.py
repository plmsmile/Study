#!usr/bin/env python
# coding=utf-8

'''
朴素贝叶斯

@author PLM
@date: 2017-05-05
'''

import numpy as np
import math


def load_post_dataset():
    ''' 创造一些文章数据
    Returns:
        post_list: 文章列表，已经把每篇文章切分为多个单词了
        class_list: 文章对应的分类，1-侮辱性的言论，0-合法言论
    '''
    post_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_list = [0, 1, 0, 1, 0, 1]
    return post_list, class_list


def get_vocab_list(post_list):
    ''' 从数据集中获取所有的不重复的词汇列表
    Args:
        post_list: 多个文章的列表，一篇文章：由单词组成的list
    Returns:
        vocab_list: 单词列表
    '''
    vocab_set = set([])
    for post in post_list:
        vocab_set = vocab_set | set(post)
    return list(vocab_set)


def get_doc_vec(doc, vocab_list, is_bag = False):
    ''' 获得一篇doc的文档向量
    词集模型：每个词出现为1，不出现为0。每个词出现1次
    词袋模型：每个词出现次数，可以多次出现。
    Args:
        vocab_list: 总的词汇表
        doc: 一篇文档，由word组成的list
        is_bag: 是否是词袋模型，默认为Fasle
    Returns:
        doc_vec: 文档向量，1出现，0未出现
    '''
    doc_vec = [0] * len(vocab_list)
    for word in doc:
        if word in vocab_list:
            idx = vocab_list.index(word)
            if is_bag == False:         # 词集模型
                doc_vec[idx] = 1
            else:
                doc_vec[idx] += 1       # 词袋模型
        else:
            print '词汇表中没有 %s ' % word
    return doc_vec


def train_nb0(train_mat, class_list):
    ''' 朴素贝叶斯训练算法，二分类问题
    Args:
        train_mat: 训练矩阵，文档向量组成的矩阵
        class_list: 每一篇文档对应的分类结果
    Returns:
        p0_vec: c0中各个word占c0总词汇的概率
        p1_vec: c1中各个word占c1总词汇的概率
        p1: 文章是c1的概率
    '''
    # 文档数目，单词数目
    doc_num = len(train_mat)
    word_num = len(train_mat[0])
    # 两个类别的总单词数量
    c0_word_count = 2.0
    c1_word_count = 2.0
    # 向量累加
    c0_vec_sum = np.ones(word_num)
    c1_vec_sum = np.ones(word_num)
    for i in range(doc_num):
        if class_list[i] == 0:
            c0_word_count += sum(train_mat[i])
            c0_vec_sum += train_mat[i]
        else:
            c1_word_count += sum(train_mat[i])
            c1_vec_sum += train_mat[i]
    # c1_num = class_list.count(1)
    c1_num = sum(class_list)
    p1 = c1_num / float(doc_num)
    p0_vec = c0_vec_sum / c0_word_count
    # map(lambda x: math.log(x), p0_vec)
    p1_vec = c1_vec_sum / c1_word_count
    # 由于后面做乘法会下溢出，所以取对数做加法
    for i in range(word_num):
        p0_vec[i] = math.log(p0_vec[i])
        p1_vec[i] = math.log(p1_vec[i])
    return p0_vec, p1_vec, p1


def classify_nb(w_vec, p0_vec, p1_vec, p1):
    ''' 使用朴素贝叶斯分类
    Args:
        w_vec: 要测试的向量
        p0_vec: c0中所有词汇占c0的总词汇的概率
        p1_vec: c1中所有词汇占c1的总词汇的概率
        p1: 文章为类型1的概率，即P(c1)
    '''
    # P(w|c0)*P(c0) = P(w1|c0)*...*P(wn|c0)*P(c0)
    # 由于下溢出，所以上文取了对数，来做加法
    w_p0 = sum(w_vec * p0_vec) + math.log(1 - p1)
    w_p1 = sum(w_vec * p1_vec) + math.log(p1)
    if w_p0 > w_p1:
        return 0
    return 1


def test_bayes():
    ''' 测试函数，判断doc是否是侮辱性邮件 '''
    
    post_list, class_list = load_post_dataset()
    vocab_list = get_vocab_list(post_list)
    train_mat = []
    for post in post_list:
        post_vec = get_doc_vec(post, vocab_list)
        train_mat.append(post_vec)
    p0_vec, p1_vec, p1 = train_nb0(train_mat, class_list)
    doc1 = ['love', 'my', 'dalmation']
    doc2 = ['stupid', 'garbage']
    doc1_vec = get_doc_vec(doc1, vocab_list)
    doc2_vec = get_doc_vec(doc2, vocab_list)
    doc1_class = classify_nb(doc1_vec, p0_vec, p1_vec, p1)
    doc2_class = classify_nb(doc2_vec, p0_vec, p1_vec, p1)
    print doc1, doc1_class
    print doc2, doc2_class


def main():
    test_bayes()


if __name__ == '__main__':
    main()
