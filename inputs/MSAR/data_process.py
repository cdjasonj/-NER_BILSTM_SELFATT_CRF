"""
转换成json ， train 划分0.15作为验证集
"""
import numpy as np
import json
import codecs
import random
import pandas as pd
import csv
import jieba



def process_train_data():
    #ns LOC  nr
    train_data = []
    with open('./train1.txt',encoding='utf-8') as fr:
        for line in fr:
            _line = line.strip().split(' ')
            dic = {}
            label = []
            words = []
            for tokens in _line :
                words.append(tokens.strip().split('/')[0])
                _label = tokens.strip().split('/')[1]
                if _label == 'ns':
                    label.append('LOC')
                elif _label =='nr':
                    label.append('PER')
                elif _label =='nt':
                    label.append('ORG')
                else:
                    label.append('O')
            dic['words'] = words
            dic['label'] = label
            train_data.append(dic)
    return train_data

def data_process2(train_data):
    """
    train_data[1] : [words] [label]
    :param train_data:
    :return:
    """
    new_train_data = []
    for data in train_data:
        dic = {}
        words = data['words']
        label = data['label']
        text = ''
        radicals = []
        NER_BIO = []
        for index,word in enumerate(words):
            text += word
            _label = label[index]
            if _label == 'O':
                NER_BIO+=['O']*len(word)
            else:
                NER_BIO+=['B'+'-'+_label]
                NER_BIO+=['I'+'-'+_label]*(len(word)-1)
        for word in text:

            _radical = radical_data.get(str(word))

            if not _radical:
                _radical = []
                radicals.append(_radical)
            else:
                radicals.append(_radical[0])

        bichar_text = get_bichar(text)
        trichar_text = get_trichar(text)
        jieba_words = jieba.cut(text)
        dic['text'] = text
        dic['radicals'] = radicals
        dic['words'] = list(jieba_words)
        dic['NER_BIO'] = NER_BIO
        dic['bichar'] = bichar_text
        dic['trichar'] = trichar_text
        new_train_data.append(dic)
    return new_train_data

def process_test_data():
    test_data = []
    with open('./testright1.txt',encoding='utf-8') as fr:
        for line in fr:
            dic = {}
            _line = line.strip().split(' ')
            words = []
            label = []
            for tokens in _line:
                words.append(tokens.strip().split('/')[0])
                _label = tokens.strip().split('/')[1]
                if _label == 'ns':
                    label.append('LOC')
                elif _label =='nr':
                    label.append('PER')
                elif _label =='nt':
                    label.append('ORG')
                else:
                    label.append('O')
                    dic['words'] = words
                    dic['label'] = label
            if dic:
              test_data.append(dic)
    return test_data


def split_data(data):
    # 划分数据集，0.1当验证集
    train_data, dev_data = [], []
    idx = list(range(len(data)))
    np.random.shuffle(idx)
    train_dev_ids = random.sample(idx,round(len(data)*0.9))
    dev_ids = random.sample(train_dev_ids,round(len(data)*0.1))
    train_ids = list(set(train_dev_ids)-set(dev_ids))

    # print(len(train_ids))
    # print(len(dev_ids))
    train_data = [data[id] for id in train_ids]
    dev_data = [data[id] for id in dev_ids]
    return train_data,dev_data


def collect_char2id(datasets, save_file):
    chars = {}
    for data in datasets:
        for word in data['text']:
            for char in str(word):
                chars[char] = chars.get(char, 0) + 1
    chars = {i:j for i,j in chars.items()  }
    id2char = {i + 1: j for i, j in enumerate(chars)}  # padding: 0
    char2id = {j: i for i, j in id2char.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2char, char2id], f, indent=4, ensure_ascii=False)


def collect_word2id(datasets, save_file):
    #这里做个词频， 0 PAD 1 UNK
    #用了char级别得语言模型就不做词频了
    words = {}
    for data in datasets:
        for word in data['text']:
            words[word] = words.get(word, 0) + 1
    words = {i:j for i,j in words.items()}
    id2word = {i + 1 : j for i, j in enumerate(words)}  # padding 0 , UNK 1, bichar pad 2
    id2word[2] = '<bicharPAD>'
    word2id = {j: i for i, j in id2word.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2word, word2id], f, indent=4, ensure_ascii=False)


def collect_BIO2id(datasets, save_file):
    BIOs = {}
    for data in datasets:
        for bio in data['NER_BIO']:
            if bio != 'O':
                BIOs[bio] = BIOs.get(bio, 0) + 1

    id2BIO = {i + 1: j for i, j in enumerate(BIOs)}  # padding:0
    id2BIO[0] = "O"
    BIO2id = {j: i for i, j in id2BIO.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2BIO, BIO2id], f, indent=4, ensure_ascii=False)

def collect_radical2id(datasets,save_file):
    radicals = {}
    for data in datasets:
        for _radicals in data['radicals']:
            for rad in _radicals:
                radicals[rad] = radicals.get(rad,0)+1

    radicals = {i:j for i,j in radicals.items() }
    id2radical = {i + 1: j for i, j in enumerate(radicals)}  #padding:0, UNK2
    radical2id = {j: i for i, j in id2radical.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2radical, radical2id], f, indent=4, ensure_ascii=False)


def collect_bichar(datasets, save_file):
    #这里做个词频， 0 PAD 1 UNK
    #用了char级别得语言模型就不做词频了
    bichar = {}
    for data in datasets:
        for _bichar in data['bichar']:
            bichar[_bichar] = bichar.get(_bichar, 0) + 1

    bichar = {i:j for i,j in bichar.items()}
    id2bichar = {i + 1: j for i, j in enumerate(bichar)}  # padding 0, bichar pad 2
    bichar2id = {j: i for i, j in id2bichar.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2bichar, bichar2id], f, indent=4, ensure_ascii=False)

def collect_trichar(datasets, save_file):
    #这里做个词频， 0 PAD 1 UNK
    #用了char级别得语言模型就不做词频了
    trichar = {}
    for data in datasets:
        for _trichar in data['trichar']:
            trichar[_trichar] = trichar.get(_trichar, 0) + 1

    trichar = {i:j for i,j in trichar.items()}
    id2trichar = {i + 1: j for i, j in enumerate(trichar)}  # padding 0, bichar pad 2
    trichar2id = {j: i for i, j in id2trichar.items()}
    with codecs.open(save_file, 'w', encoding='utf-8') as f:
        json.dump([id2trichar, trichar2id], f, indent=4, ensure_ascii=False)

def get_bichar(text):
    """
    w1,w2,w3,w4,w5,w6,w7
    bichar : w1w2,w2w3,w4w5,w5w6,w7w0
    :param text:
    :return:
    """
    new_text = []
    for index,char in enumerate(text):
        if index != len(text) - 1:  #没有达到最后一个
            new_text.append(char+text[index+1])
        else:
            new_text.append(char+'$')
    return new_text

def get_trichar(text):
    """
    w1,w2,w3,w4,w5,w6,w7
    wpadw1w2 w1w2w3 w2w3w4 w4w5w6 w5w6w7 w6w7wpad
    :param text:
    :return:
    """
    new_text = []
    for index, char in enumerate(text):
        if len(text) == 1:
            new_text.append('$$$')
        elif index == 0: #[第一个]
            new_text.append( '$'+char + text[index + 1])
        elif index != len(text) - 1:  # 没有达到最后一个
            new_text.append(text[index-1]+char + text[index + 1])
        else:
            new_text.append(text[index-1] + char + '$')
    return new_text

def get_radical():
    data = {}
    for i in ['./chaizi-ft.txt', './chaizi-jt.txt']:
        with open(i, 'rt',encoding='utf-8') as fd:
            for line in fd:
                item_list = line.strip().split('\t')
                key = item_list[0]
                value = [i.strip().split() for i in item_list[1:]]

                data[key] = value
    return data

radical_data = get_radical()
train_data = data_process2(process_train_data())
test_data = data_process2(process_test_data())
train_data,dev_data = split_data(train_data)

all_data = train_data+test_data+dev_data

collect_char2id(all_data,'./char2id.json')
collect_word2id(all_data,'./word2id.json')
collect_BIO2id(all_data,'./BIO2id.json')
collect_radical2id(all_data,'./radical2id.json')
collect_bichar(all_data,'./bichar2id.json')
collect_trichar(all_data,'./trichar2id.json')

with codecs.open('./test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=4, ensure_ascii=False)
with codecs.open('./train_data.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, indent=4, ensure_ascii=False)
with codecs.open('./dev_data.json', 'w', encoding='utf-8') as f:
    json.dump(dev_data, f, indent=4, ensure_ascii=False)
