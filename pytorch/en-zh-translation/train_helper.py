#!/usr/bin/env python
# encoding=utf-8
'''
模型训练方法
@author PLM
@date 2017-10-16
'''
from __future__ import print_function

import random
import torch

import data_helper as dh
from data_helper import get_variable
from masked_cross_entropy import masked_cross_entropy
import show

def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder,
         encoder_optimizer, decoder_optimizer, train_conf):
    '''训练一批数据'''
    batch_size = len(input_lengths)
    # 1. zero grad
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # 2. 输入encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths)
    # 3. decoder 默认输入
    decoder_input = decoder.create_input_seq(batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    # 4. 输入到decoder
    max_target_len = max(target_lengths)
    all_decoder_outputs = get_variable(torch.zeros(max_target_len, batch_size, decoder.output_size))
    
    use_teacher_forcing = random.random() < train_conf['teacher_forcing_ratio']
    
    for t in range(max_target_len):
        output, decoder_hidden, attn_weights = decoder(decoder_input, decoder_hidden, encoder_outputs)
        all_decoder_outputs[t] = output
        # 喂真实lable，应该喂output的结果
        if False:
            decoder_input = target_batches[t]
        else:
            # 从output中找到两个最符合的单词
            words = parse_output(output)
            decoder_input = get_variable(torch.LongTensor(words))
    
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths
    )
    loss.backward()
    
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), train_conf['clip'])
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), train_conf['clip'])
    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.data[0], ec, dc


def evaluate(input_seq, input_lang, target_lang, encoder, decoder, target_maxlen=25):
    ''' 验证一条句子
    Args:
        input_seq: 输入的一个句子，不包含EOS_token
        target_maxlen: 翻译目标句子的最大长度，不包括EOS_token的长度
    Returns:
        decoded_words: 翻译后的词语
        decoder_attentions: Attention [目标句子长度，原句子长度]
    '''
    batch_size = 1
    seq_wordids = dh.indexes_from_sentence(input_lang, input_seq)
    # 已经自动加上EOS_token的长度。
    input_length = len(seq_wordids)
    # batch=1，转换为[1, s]
    input_lengths = [input_length]
    input_batches = [seq_wordids]
    input_batches = get_variable(torch.LongTensor(input_batches))
    # encoder输入是[s, b]
    input_batches = input_batches.transpose(0, 1)
    
    # 非训练模式，避免dropout
    encoder.train(False)
    decoder.train(False)
    
    # 过encoder，准备decoder数据
    #print ('input_batches:', input_batches.size())
    #print ('input_lengths:', input_lengths)
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)
    decoder_input = decoder.create_input_seq(batch_size)
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # 最终结果
    decoded_words = []
    decoder_attentions = torch.zeros(target_maxlen + 1, input_length)
    
    # 过decoder
    for di in range(target_maxlen):
        # (b, s)=(1,s), s是输入句子的长度
        decoder_output, decoder_hidden, attn_weights = \
            decoder(decoder_input, decoder_hidden, encoder_outputs)
        # word信息
        word_id = parse_output(decoder_output)[0]
        word = target_lang.index2word[word_id]
        decoded_words.append(word)
        # attention
        decoder_attentions[di] += attn_weights.data.squeeze(0)
        
        if word_id == dh.EOS_token:
            break
        # 当前单词作为下一个的输入
        decoder_input = get_variable(torch.LongTensor([word_id]))
    
    # 改变encoder的模式
    encoder.train(True)
    decoder.train(True)
    res = decoder_attentions[:di+1,:]
    print ('input_length:{}, di={}, size={}'.format(input_length, di, res.size()))
    return decoded_words, res


def evaluate_randomly(pairs, input_lang, target_lang, encoder, decoder,
                      print_res=False, show_attention=False, show_in_visdom=False):
    ''' 随机翻译一条句子，并且打印结果 '''
    [input_sentence, target_sentence] = random.choice(pairs)

    evaluate_sentence(input_sentence, input_lang, target_lang,
                      encoder, decoder, target_sentence=target_sentence, print_res=print_res,
                      show_attention=show_attention, show_in_visdom=show_in_visdom)


def parse_output(output):
    ''' 解析得到output中的words信息，id表示
    Args:
        output: [batch_size, output]
    Returns:
        word_ids: [batch_size] 翻译出来的word_id信息
    '''
    word_ids = []
    batch_size = output.size()[0]
    for i in range(batch_size):
        topv, topi = output[i].data.topk(1)
        word_ids.append(topi[0])
    return word_ids


def evaluate_sentence(input_sentence, input_lang, target_lang, encoder, decoder, print_res=False,
                      target_sentence=None, show_attention=False, show_in_visdom=False):
    '''翻译并评估一条句子'''
    output_words, attentions = evaluate(input_sentence, input_lang, target_lang, 
                                       encoder, decoder)
    output_sentence = ' '.join(output_words)
    if print_res:
        print('>', input_sentence)
        if target_sentence is not None:
            print('=', target_sentence)
        print('< ', output_sentence)
    
    if show_attention:
        show.show_attention(input_sentence, output_words, attentions, 
                            target_sentence=target_sentence, show_in_visdom=show_in_visdom)
    
