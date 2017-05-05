#!usr/bin/env python
# coding=utf-8

'''
朴素贝叶斯

@author PLM
@date: 2017-05-05
'''

import numpy as np


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


def get_doc_vec(vocab_list, doc):
    ''' 获得一篇doc的文档向量
    词汇表中的单词出现为1，不出现为0
    Args:
        vocab_list: 总的词汇表
        doc: 一篇文档，由word组成的list
    Returns:
        doc_vec: 文档向量，1出现，0未出现
    '''
    doc_vec = [0] * len(vocab_list)
    for word in doc:
        if word in vocab_list:
            idx = vocab_list.index(word)
            doc_vec[idx] = 1
        else:
            print '词汇表中没有 %s ' % word
    return doc_vec


def train_nb0(train_mat, class_list):
    ''' 朴素贝叶斯训练算法，二分类问题
    Args:
        train_mat: 训练矩阵，文档向量组成的矩阵
        class_list: 每一篇文档对应的分类结果
    Returns:
        
    '''


def test_bayes():
    ''' 测试函数 '''
    post_list, class_list = load_post_dataset()
    vocab_list = get_vocab_list(post_list)
    post = post_list[0]
    post_vec = get_doc_vec(vocab_list, post)
    print post
    print post_vec


def main():
    test_bayes()


if __name__ == '__main__':
    main()











