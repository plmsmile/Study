#!/usr/bin/env python
#-*-coding: utf8-*-

'''
nltk first learn
@author plm
@create 2017-06-21
'''

from nltk.book import *

def test_book():
    ''' book api'''
    b1 = text1     # nltk.text.Text，实际上是一个List，b1[2]
    b2 = text2
    # find the word
    word = 'monstrous'
    b2.concordance(word)
    # find similar words
    b1.similar(word)
    b2.similar(word)
    # common context
    b2.common_contexts(['monstrous', 'very'])
    # length
    whole_len = len(b2)
    # 包括标点符号
    distinct_word_num = len(set(b2))
    print distinct_word_num
    word_count = b2.count(word)
    print word_count


def text_list():
    '''文本是一个list'''
    b1 = text1
    b1[2]
    b1.index('awaken')
    # 可以进行切片访问
    print b1[2:5]


def simple_count():
    '''简单的统计'''
    b1 = text1
    # 统计出每个词语出现的数量
    fdist1 = FreqDist(b1)
    vacabulary = fdist1.keys()
    print len(vacabulary)
    print vacabulary[:10]


def get_words(text):
    '''获取text中所有的单词'''
    words = [w.lower() for w in text if w.isalpha()]
    words = set(words)
    return words

def lexical_diversity(text):
    ''' 词汇复杂性 '''
    return len(text) / len(set(text))


def find_useful_words(text):
    '''找出一个text中比较有用的词汇
    长度大于7并且出现此处高预7次的词
    Args:
        text:一篇文章
    '''
    fdist = FreqDist(text)
    vacabs = set(text)
    words = [w for w in vacabs if len(w) > 7 and fdist[w] > 7]
    return words


def usual_func(s):
    '''常用的词汇比较运算符'''
    s.startswith(t)
    s.endswith(t)
    'hello' in s
    s.islower()
    s.isupper()
    s.isalpha() # 都是字母
    s.isalnum() # 都是字母或者数字
    s.isdigit() # 都是数字
    s.istitle() # 是否首字母大写


if __name__ == '__main__':
    # test_book()
    # text_list()
    # simple_count()
    # print find_useful_words(text2)
    print get_words(text3)
