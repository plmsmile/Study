#!/usr/bin/env python
#-*-coding:utf-8 -*-

'''
word to vector

@author plm
@date 2017-07-01
'''

import os
import urllib
import zipfile
import collections
import numpy as np
import random
import math
import tensorflow as tf


class WordUtil(object):
    ''' 和word相关的工具类'''

    def __init__(self, vocab_size):
        ''' 初始化
        '''
        # 单词的word的index
        self.__data_index = 0
        # 有效单词列表的长度
        self.__vocab_size = vocab_size

    def maybe_download(self, download_url, filename, expected_bytes):
        ''' 下载文件
        Args:
            download_url: 下载地址
            filename: 要下载的文件的名字
            expected_bytes: 文件的字节数量
        Reuturns:
            filename: 当前的下载的文件名字
        '''
        if not os.path.exists(filename):
            filename, _ = urllib.urlretrieve(download_url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print ('Found and verified %s' % filename)
        else:
            print (
                'expected %s, but real %s' %
                (expected_bytes, statinfo.st_size))
            raise Exception('file bytes miss')

    def read_data(self, filename):
        ''' 读取数据
        Args:
            filename: 本地的zip文件
        Returns:
            data: 单词列表
        '''
        with zipfile.ZipFile(filename, 'r') as f:
            data = f.read(f.namelist()[0]).split()
        return data

    def build_dataset(self, words):
        ''' 构建数据集
        Args:
            words: 单词列表
        Returns:
            word_code: 所有word的编码，top的单词：某个顺序；其余的：0
            topword_id: topword-id
            id_topword: id-word
            topcount: 包含所有word的一个Counter对象
        '''
        # 获取top50000频数的单词
        unk = 'UNK'
        topcount = [[unk, -1]]
        topcount.extend(
            collections.Counter(words).most_common(
                self.__vocab_size - 1))
        topword_id = {}
        # topword: 1-vocab_size编码
        for word, _ in topcount:
            topword_id[word] = len(topword_id)
        # 构建单词的编码。top单词：某个顺序；其余单词：0
        word_code = []
        unk_count = 0
        for w in words:
            if w in topword_id:
                c = topword_id[w]
            else:
                c = 0
                unk_count += 1
            word_code.append(c)
        topcount[0][1] = unk_count
        id_topword = dict(zip(topword_id.values(), topword_id.keys()))
        return word_code, topword_id, id_topword, topcount

    def generate_batch(self, batch_size, single_num, skip_window, word_code):
        '''产生训练样本。Skip-Gram模型，从当前推测上下文
        如 i love you. (love, i), (love, you)
        Args:
            batch_size: 每一个batch的大小，即多少个()
            single_num: 对单个单词生成多少个样本
            skip_window: 单词最远可以联系的距离
            word_code: 所有单词，单词以code形式表示
        Returns:
            batch: 目标单词
            labels: 语境单词
        '''
        # 条件判断
        # 确保每个batch包含了一个词汇对应的所有样本
        assert batch_size % single_num == 0
        # 样本数量限制
        assert single_num <= 2 * skip_window

        # batch label
        batch = np.ndarray(shape=(batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
        # 目标单词和相关单词
        span = 2 * skip_window + 1
        word_buffer = collections.deque(maxlen=span)
        for _ in range(span):
            word_buffer.append(word_code[self.__data_index])
            self.__data_index = (self.__data_index + 1) % len(word_code)

        # 遍历batchsize/samplenums次，每次一个目标词汇，一次samplenums个语境词汇
        for i in range(batch_size // single_num):
            target = skip_window                # 当前的单词
            targets_to_void = [skip_window]     # 已经选过的单词+自己本身
            # 为当前单词选取样本
            for j in range(single_num):
                while target in targets_to_void:
                    target = random.randint(0, span - 1)
                targets_to_void.append(target)
                batch[i * single_num + j] = word_buffer[skip_window]
                labels[i * single_num + j, 0] = word_buffer[target]
            # 当前单词已经选择完毕，输入下一个单词，skip_window单词也成为下一个
            self.__data_index = (self.__data_index + 1) % len(word_code)
            word_buffer.append(word_code[self.__data_index])
        return batch, labels


def go_model():
    ''' skip-gram model'''
    # 频率top50000个单词
    vocab_size = 50000
    # 一批样本的数量
    batch_size = 128
    # 将单词转化为稠密向量的维度
    embedding_size = 128
    # 为单词找相邻单词，向左向右最多能取得范围
    skip_window = 1
    # 每个单词的语境单词数量
    single_num = 2

    # 验证单词的数量
    valid_size = 16
    # 验证单词从频数最高的100个单词中抽取
    valid_window = 100
    # 从100个中随机选择16个
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    # 负样本的噪声数量
    noise_num = 64

    graph = tf.Graph()
    with graph.as_default():
        # 输入数据
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        with tf.device('/cpu:0'):
            # 随机生成单词的词向量，50000*128
            embeddings = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
            # 查找输入inputs对应的向量
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)
            nce_weights = tf.Variable(
                tf.truncated_normal([vocab_size, embedding_size],
                                    stddev=1.0 / math.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocab_size]))
        # 为每个batch计算nceloss
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels = train_labels,
                                             inputs=embed,
                                             num_sampled=noise_num,
                                             num_classes=vocab_size))
        # sgd
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # 计算embeddings的L2范式，各元素的平方和然后求平方根，防止过拟合
        norm = tf.sqrt(
            tf.reduce_sum(
                tf.square(embeddings),
                axis=1,
                keep_dims=True))
        # 标准化
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(
                                    normalized_embeddings, valid_dataset)
        # valid单词和所有单词的相似度计算，向量相乘
        similarity = tf.matmul(
            valid_embeddings,
            normalized_embeddings,
            transpose_b=True)

        init = tf.global_variables_initializer()

    # 词汇工具
    wu = WordUtil(vocab_size)
    words = wu.read_data('text8.zip')
    word_code, topword_id, id_topword, topcount = wu.build_dataset(words)

    num_steps = 100001
    with tf.Session(graph=graph) as sess:
        init.run()
        print('Initialized')
        avg_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = wu.generate_batch(
                    batch_size, single_num, skip_window,  word_code)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}
            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            avg_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    avg_loss /= 2000
                print ("avg loss at step %s : %s" % (step, avg_loss)) 
                avg_loss = 0
            
            if step % 10000 == 0:
                # 相似度，16*50000
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = id_topword[valid_examples[i]]
                    # 选相似的前8个
                    top_k = 8
                    # 排序，获得id
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log_str = "Nearest to %s: " % valid_word
                    for k in range(top_k):
                        close_word = id_topword[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print log_str
        final_embeddings = normalized_embeddings.eval()



if __name__ == '__main__':
    #filename = maybe_download('text8.zip', 31344016)
    vocab_size = 50000
    wu = WordUtil(vocab_size)
    words = wu.read_data('text8.zip')
    word_code, topword_id, id_topword, topcount= wu.build_dataset(words)
    batch, labels = wu.generate_batch(8, 2, 1, word_code)
    print type(topword_id), type(id_topword)
    for i in range(8):
        print batch[i], labels[i][0], id_topword[batch[i]], id_topword[labels[i][0]]
    #go_model()
