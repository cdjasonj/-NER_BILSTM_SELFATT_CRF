import numpy as np
import sys
import gc
import time
import os
import random
import logging

import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from keras_self_attention import SeqSelfAttention
from .keraslayers.ChainCRF import ChainCRF

class BiLSTM:
    def __init__(self,char_embedding,bichar_embedding,trichar_embedding,params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training
        self.models = None
        self.modelSavePath = None
        self.resultsSavePath = None

        # self.word_embedding = word_embedding
        self.char_embedding = char_embedding
        self.bichar_embedding = bichar_embedding
        self.trichar_embedding = trichar_embedding

        defaultParams = {'dropout': (0.25,0.25), 'LSTM-Size': (200,),
                         'optimizer': 'adam', 'clipvalue': 0, 'clipnorm': 1,'n_class_labels':7}
        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

    def build_model(self):

        tokens_input = Input(shape=(None,), dtype='int32', name='chars_input')
        bichar_input = Input(shape=(None,),dtype='int32',name='bichar_input')
        trichar_input = Input(shape=(None,),dtype='int32',name='trichar_input')

        inputNodes = [tokens_input,bichar_input,trichar_input]

        char_embedding = Embedding(input_dim=self.params['char2id_size']+2, output_dim=self.params['char_embedding_size'],weights=[self.char_embedding]
                                   , trainable=False, name='char_embedding')(tokens_input)

        bichar_embedding = Embedding(input_dim=self.params['bichar2id_size']+2,output_dim=self.params['bichar_embedding_size'],weights=[self.bichar_embedding]
                                   ,trainable=True,name='bichar_embedding')(bichar_input)
        trichar_embedding = Embedding(input_dim=self.params['trichar2id_size']+2,output_dim=self.params['trichar_embedding_size'],weights=[self.trichar_embedding]
                                   ,trainable=True,name='bichar_embedding')(trichar_input)

        embedding = Concatenate(axis=-1)([char_embedding,bichar_embedding,trichar_embedding])

        # Add LSTMs
        shared_layer = embedding
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]),
                                             name='char_shared_varLSTM_' + str(cnt))(shared_layer)
            cnt += 1


        self_att = SeqSelfAttention()(shared_layer)
        lstm_att = Concatenate(axis=-1)([shared_layer,self_att])

        output = lstm_att
        output = TimeDistributed(Dense(self.params['n_class_labels'], activation=None))(output)
        crf = ChainCRF()
        output = crf(output)
        lossFct = crf.sparse_loss

        # :: Parameters for the optimizer ::
        optimizerParams = {}
        if 'clipnorm' in self.params and self.params['clipnorm'] != None and self.params['clipnorm'] > 0:
            optimizerParams['clipnorm'] = self.params['clipnorm']

        if 'clipvalue' in self.params and self.params['clipvalue'] != None and self.params['clipvalue'] > 0:
            optimizerParams['clipvalue'] = self.params['clipvalue']

        if self.params['optimizer'].lower() == 'adam':
            opt = Adam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'nadam':
            opt = Nadam(**optimizerParams)
        elif self.params['optimizer'].lower() == 'rmsprop':
            opt = RMSprop(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adadelta':
            opt = Adadelta(**optimizerParams)
        elif self.params['optimizer'].lower() == 'adagrad':
            opt = Adagrad(**optimizerParams)
        elif self.params['optimizer'].lower() == 'sgd':
            opt = SGD(lr=0.1, **optimizerParams)

        model = Model(inputs=inputNodes, outputs=[output])
        model.compile(loss=lossFct, optimizer=opt)

        model.summary(line_length=125)

        return model