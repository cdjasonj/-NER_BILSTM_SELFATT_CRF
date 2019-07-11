import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tools.read_properties import read_properties
import json

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)

    return [x + [0] * (ML - len(x)) for x in X]

def char_pad(datas,maxlen_sentence,maxlen_word):
    #word_leve pad for char input
    #use the maxlen of batch data of words to pad the char levels and use the maxlen of batch datas to pad the sentence level inputs
    """
    :param datas: [batch_size,None,None]
    :return: [batch_size,maxlen of sentence , maxlen of words]
    """
    # L = [len(x) for x in TEXT_word]
    # maxlen_sentence = max(L)
    new_data = []
    for sentence in datas:
        _sentence = []
        for word in sentence:
            if len(word) < maxlen_word:
                word+=[0]*(maxlen_word - len(word))
            else:
                word = word[:maxlen_word]
            _sentence.append(word)

        pad_word = [0]*maxlen_word
        if len(_sentence) < maxlen_sentence:
            for i in range(maxlen_sentence - len(_sentence)):
                _sentence.append(pad_word)
        else:
            _sentence = _sentence[:maxlen_sentence]
        new_data.append(_sentence)
    return new_data

def get_forward_LMtarget(TEXT_WORD):
    """

    TEXT_WORD[0] = [w1,w2,w3,w4,w5,w6]
    LMtarget[0] = [W2,W3,W4,W5,W6,END] END -1
    :param TEXT_WORD:
    :return:
    """
    LMtarget = []
    for text in TEXT_WORD:
        _text = text[1:]
        _text.append(3)
        LMtarget.append(_text)
    return LMtarget

def get_backward_LMtarget(TEXT_WORD):
    """

    TEXT_WORD[0] = [w1,w2,w3,w4,w5,w6]
    LMtarget[0] = [START,W1,W2,W3,W4,W5]
    :param TEXT_WORD:
    :return:
    """
    LMtarget = []
    for text in TEXT_WORD:
        _target = []
        _target.append(2)
        LMtarget.append(_target+text[:-2])
    return LMtarget

def get_text_char(text,char2id,maxlen_word):
    """

    :param text: [w1,w2,w3..,]
    :return:
    """
    text_char = []
    for word in text:
        word = str(word)
        if len(word) > maxlen_word: #word超长
            text_char += [char2id.get(char, 1) for char in word[:maxlen_word]]
        else:
            text_char += [char2id.get(char,1) for char in word]
            text_char += [0] * (maxlen_word - len(word))
    return text_char

class data_generator():
    def __init__(self,data,word2id,char2id,BIO2id,maxlen_sentence,maxlen_word,batch_size=8):
        self.data = data
        self.batch_size = batch_size
        self.word2id = word2id
        self.char2id = char2id
        self.BIO2id = BIO2id
        self.maxlen_word = maxlen_word
        self.maxlen_sentence = maxlen_sentence
        self.steps = len(self.data)//self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True :
            index = list(range(len(self.data)))
            np.random.shuffle(index)
            #WORD[w1,w2,...,wn]
            #char[c11,c12,..,c1n<pad>,..,<pad>,.....,..]
            TEXT_WORD,TEXT_CHAR,BIO,FOWARD_LM,BACKWARD_LM = [],[],[],[],[]
            for idx in index:
                _data = self.data[idx]
                text = _data['text']
                bio = _data['BIOS']
                _text_word = [self.word2id.get(str(word),1) for word in text] #1UNK,0PAD
                _bio = [self.BIO2id.get(b) for b in bio]
                _text_char = get_text_char(text,self.char2id,self.maxlen_word)

                TEXT_WORD.append(_text_word)
                TEXT_CHAR.append(_text_char)
                # first in word dimensions for sentence maxlen ,then ,in char dimensions for maxlen_word
                BIO.append(_bio)

                if len(TEXT_WORD) == self.batch_size or idx == index[-1]:

                    FOWARD_LM = get_forward_LMtarget(TEXT_WORD)
                    BACKWARD_LM = get_backward_LMtarget(TEXT_WORD)
                    TEXT_WORD = pad_sequences(TEXT_WORD,maxlen=self.maxlen_sentence,padding='post',value=0)
                    TEXT_CHAR = pad_sequences(TEXT_CHAR,maxlen=self.maxlen_word*self.maxlen_sentence,padding='post',value=0)
                    FOWARD_LM = pad_sequences(FOWARD_LM, maxlen=self.maxlen_sentence, padding='post', value=0)
                    BACKWARD_LM = pad_sequences(BACKWARD_LM, maxlen=self.maxlen_sentence, padding='post', value=0)
                    # TEXT_WORD = np.array(seq_padding(TEXT_WORD))
                    # BIO = np.array(seq_padding(BIO))
                    BIO = pad_sequences(BIO,maxlen=self.maxlen_sentence,padding='post',value=0)

                    yield [TEXT_WORD,TEXT_CHAR,BIO,FOWARD_LM,BACKWARD_LM],None
                    TEXT_WORD, TEXT_CHAR, BIO, FOWARD_LM, BACKWARD_LM =[],[],[],[],[]

def load_data(eval_data,word2id,char2id,BIO2id,maxlen_sentence,maxlen_word):
    #only for ner prediction now , then i will compelet the function for joint extraction
    #load data for predict

    TEXT_WORD,TEXT_CHAR, BIO = [],[], []
    for data in eval_data:
        text = data['text']
        bio = data['BIOS']
        _text_word = [word2id.get(word,1) for word in text] # 0pad 1UNK

        _text_char = get_text_char(text, char2id,maxlen_word)
        _bio = [BIO2id.get(b) for b in bio]
        TEXT_WORD.append(_text_word)
        TEXT_CHAR.append(_text_char)
        BIO.append(_bio)

    FOWARD_LM = get_forward_LMtarget(TEXT_WORD)
    BACKWARD_LM = get_backward_LMtarget(TEXT_WORD)
    TEXT_WORD = pad_sequences(TEXT_WORD, maxlen=maxlen_sentence, padding='post', value=0)
    FOWARD_LM = pad_sequences(FOWARD_LM, maxlen=maxlen_sentence, padding='post', value=0)
    BACKWARD_LM = pad_sequences(BACKWARD_LM, maxlen=maxlen_sentence, padding='post', value=0)
    # TEXT_WORD = np.array(seq_padding(TEXT_WORD))
    TEXT_CHAR = pad_sequences(TEXT_CHAR, maxlen=maxlen_word * maxlen_sentence, padding='post', value=0)
    BIO = pad_sequences(BIO, maxlen=maxlen_sentence, padding='post', value=0)
    return TEXT_WORD,TEXT_CHAR,BIO,FOWARD_LM,BACKWARD_LM

class base_data_generator():
    def __init__(self,data,word2id,BIO2id,maxlen_sentence,batch_size=128):
        self.data = data
        self.batch_size = batch_size
        self.word2id = word2id
        self.BIO2id = BIO2id
        self.maxlen_sentence = maxlen_sentence
        self.steps = len(self.data)//self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True :
            index = list(range(len(self.data)))
            np.random.shuffle(index)
            TEXT_WORD,BIO = [],[]
            for idx in index:
                _data = self.data[idx]
                text = _data['text']
                bio = _data['BIOS']
                _text_word = [self.word2id.get(str(word),1) for word in text] #1UNK,0PAD
                _bio = [self.BIO2id.get(b) for b in bio]
                TEXT_WORD.append(_text_word)
                # first in word dimensions for sentence maxlen ,then ,in char dimensions for maxlen_word
                BIO.append(_bio)
                if len(TEXT_WORD) == self.batch_size or idx == index[-1]:
                    # TEXT_WORD = pad_sequences(TEXT_WORD,maxlen=self.maxlen_sentence,padding='post',value=0)
                    TEXT_WORD = np.array(seq_padding(TEXT_WORD))
                    BIO = np.array(seq_padding(BIO))
                    # BIO = pad_sequences(BIO,maxlen=self.maxlen_sentence,padding='post',value=0)
                    yield [TEXT_WORD,BIO ],None
                    TEXT_WORD,BIO =[],[]

def base_load_data(eval_data,word2id,BIO2id,maxlen_sentence):
    #only for ner prediction now , then i will compelet the function for joint extraction
    #load data for predict

    TEXT_WORD, BIO = [], []
    for data in eval_data:
        text = data['text']
        bio = data['BIOS']
        _text_word = [word2id.get(word,1) for word in text] # 0pad 1UNK
        _text_char = []  # 2 dimmensions

        _bio = [BIO2id.get(b) for b in bio]
        TEXT_WORD.append(_text_word)
        BIO.append(_bio)

    TEXT_WORD = np.array(seq_padding(TEXT_WORD))
    BIO = pad_sequences(BIO, maxlen=30, padding='post', value=0)
    return TEXT_WORD,BIO