#!/usr/bin/env python
#-*-coding: utf8 -*-

'''
get corpus
@author plm
@create 2017-06-22
'''


import nltk as nk
from nltk.corpus import wordnet as wn

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
    print cfdist[genres[0]]['could']


def get_english_vocabs():
    '''获得英语所有的词汇'''
    return set(w.lower() for w in nk.corpus.words.words())


def unusual_words(text):
    '''获取text中的unusual words
    不足：单词的变形也被认为是unusual words
    Args:
        text: 一段文本
    '''
    text_vocabs = set(w.lower() for w in text if w.isalpha())
    en_vocabs = get_english_vocabs()
    unusual = text_vocabs.difference(en_vocabs)
    return sorted(unusual)


def get_useful_words(text):
    '''获取除stopwords以外的词语'''
    stop_words = get_stopwords()
    words = [w for w in text if w not in stop_words]
    return words


def get_text(corpus):
    '''从corpus中获取一个text'''
    fileids = corpus.fileids()
    words = corpus.words(fileids = fileids[0])    
    return words


def get_stopwords():
    '''停用语料库，如the a 等'''
    return nk.corpus.stopwords.words('english')


def compose_words(alphas):
    '''使用字母组建单词
    Args:
        alphas: 所能使用的字母
    Returns:
        words: 所能组成的单词列表
    '''
    puzzle_letters = nk.FreqDist(alphas)
    obligatory = alphas[0]
    english_vocabs = get_english_vocabs()
    words = []
    for w in english_vocabs:
        if len(w) >= 6 and obligatory in w and nk.FreqDist(w) <= puzzle_letters:
            words.append(w)
    return words


def get_common_words(language):
    '''获得一种语言里的常用的词汇
    Args:
        language: 语言
    Returns:
        words: 该语言的常用词汇
    '''
    return nk.corpus.swadesh.words(language)


def get_translate_dict(src_lang, dst_lang):
    '''常用词的词汇翻译
    Args:
        src_lang: 源语言
        dst_lang: 目标语言
    Returns:
        translate_dict: 翻译后的dict
    '''
    translate_dict = dict(nk.corpus.swadesh.entries([src_lang, dst_lang]))
    return translate_dict


def test_wordnet():
    '''wordnet'''
    syn_sets = wn.synsets('love')
    synset = syn_sets[0]
    print synset
    print synset.lemma_names()
    print wn.synset('love.n.01').lemma_names()
    print synset.definition()
    print synset.examples()
    print wn.synset('car.n.01').lemma_names()




if __name__ == '__main__':
    # test_gutenberg()
    # show_corpusinfo(nk.corpus.webtext)
    # test_webtext()
    # test_brown()
    # test_cfdist(nk.corpus.brown)
    # print unusual_words(get_text(nk.corpus.brown))
    # text = get_text(nk.corpus.brown)
    # print len(text), len(get_useful_words(text))
    # print compose_words('regivvonl')
    # print len(get_common_words('en'))

    #translate_dict = get_translate_dict('en', 'fr')
    #print len(translate_dict)
    test_wordnet()

