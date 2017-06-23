#!/usr/bin/env python
#-*-coding: utf8 -*-

'''
get corpus
@author plm
@create 2017-06-22
'''


import nltk as nk


def test_gutenberg():
    '''gutenberg corpus'''
    fileids = nk.corpus.gutenberg.fileids()
    file_name = fileids[0]
    # 获得words
    words = nk.corpus.gutenberg.words(file_name)
    # 通过words创建text
    text = nk.Text(words)
    print file_name, len(text)
    print text[:20]
    text.concordance('clever')


def show_corpusinfo(corpus):
    '''显示一些信息
    Args:
        corpus: nkl.corpus.xxx
    '''
    for fileid in corpus.fileids():
        # raw 语料库的原始内容
        chars = corpus.raw(fileid)
        # words 整个语料库中的词汇
        words = corpus.words(fileid)
        # 句子
        sents = corpus.sents(fileid)
        # 唯一的词汇
        vocabs = set(w.lower() for w in words)
        # 平均词长，句长，词汇多样性
        avg_word_len = int(len(chars) / len(words))
        avg_sent_len = int(len(words) / len(sents))
        avg_vocab_use = int(len(words) / len(vocabs))
        print ('char:%d, words:%d, sents:%d, vocabs:%d, %d, %d, %d,  %s'
               % (len(chars), len(words), len(sents), len(vocabs),
                  avg_word_len, avg_sent_len, avg_vocab_use, fileid))


def test_webtext():
    '''网络和聊天文本'''
    web_fileids = nk.corpus.webtext.fileids()
    chat_room = nk.corpus.nps_chat('10-19-20s_706posts.xml')
    print chat_room[123]


def test_brown():
    '''布朗语料库'''
    categories = nk.corpus.brown.categories()
    fileids = nk.corpus.brown.fileids()
    print categories
    # 类别的所有words
    words_category = nk.corpus.brown.words(categories=categories[0:2])
    # fileid的所有words
    words_fileids = nk.corpus.brown.words(fileids=fileids[0:3])
    print len(words_category), len(words_fileids)
    words_news = nk.corpus.brown.words(categories='news')
    fdist = nk.FreqDist([w.lower() for w in words_news])
    modals = ('can', 'could', 'may', 'might', 'must', 'will')
    for m in modals:
        print m, fdist.get(m, None)


def basic_corpus_funcs():
    '''基本语料库函数'''
    fileids()                   # 语料库中的文件
    fileids([categories])       # 指定分类读文件
    categories()                # 分类
    categories([fileids])       # 指定文件的分类
    raw()                       # 原始内容
    raw(fileids=[f1, f2, f3])  # 指定文件
    raw(categories=[c1, c2])
    words()                     # 词汇
    words(fileids=['f'])
    words(categories=['news'])
    sents()                     # 句子
    sents(fileids=['f1'])
    sents(categories=['news'])
    abspath(fileid)             # 文件在磁盘中的路径
    open(fileid)                # 打开文件流
    root()                      # 语料库的根路径
    readme()                    # 语料库中README的内容


def test_reuters():
    '''路透社语料库'''
    fileids = nk.corpus.reuters.fileids()
    categories = nk.corpus.reuters.categories()


def test_cfdist(corpus):
    '''条件概率分布'''
    cond_samples = []
    genres = corpus.categories()[:2]
    for genre in genres: 
        for word in corpus.words(categories=genre):
            cond_samples.append((genre, word))
    cfdist = nk.ConditionalFreqDist(cond_samples)
    print cfdist.conditions() 


if __name__ == '__main__':
    # test_gutenberg()
    # show_corpusinfo(nk.corpus.webtext)
    # test_webtext()
    # test_brown()
    test_cfdist(nk.corpus.brown)



