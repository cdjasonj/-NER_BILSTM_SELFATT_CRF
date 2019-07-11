import numpy as np
import sys
import gc
import time
import os
import random
import logging
import math
import keras
from keras.optimizers import *
from keras.models import Model
from keras.layers import *

from .keraslayers.ChainCRF import ChainCRF



class BiLSTM:
    def __init__(self,params=None):
        # modelSavePath = Path for storing models, resultsSavePath = Path for storing output labels while training

        # Hyperparameters for the network
        defaultParams = {'dropout': (0.25, 0.25), 'classifier': ['Softmax'], 'LSTM-Size': (100,), 'customClassifier': {},
                         'optimizer': 'adam','maxRadicalLen':'20',
                         'useTaskIdentifier': False, 'clipvalue': 0, 'clipnorm': 1,
                         'earlyStopping': 5, 'miniBatchSize': 32, 'n_class_labels': 4}

        if params != None:
            defaultParams.update(params)
        self.params = defaultParams

    def build_model(self):

        tokens_input = Input(shape=(None,), dtype='int32', name='chars_input')
        word_input = Input(shape=(None,), dtype='int32', name='words_input')
        radical_input = Input(shape=(None,self.params['maxlen_rad']),name='radical_input',dtype='int32')

        inputNodes = [tokens_input, word_input,radical_input]

        char_embedding = Embedding(input_dim=self.params['char2id_size']+1, output_dim=self.params['char_embedding_size']
                                   , trainable=True, name='char_embedding')(tokens_input)

        word_embedding = Embedding(input_dim=self.params['word2id_size']+1,output_dim=self.params['word_embedding_size']
                                   ,trainable=True,name='word_embedding')(word_input)

        radical_embedding = TimeDistributed(
            Embedding(input_dim=self.params['rad2id_size']+1, output_dim=self.params['rad_embedding_size'],
                      trainable=True, mask_zero=True), name='rad_embedding')(radical_input)

        radical_embedding = TimeDistributed(Bidirectional(LSTM(self.params['rad_hidden_size'], return_sequences=False)), name="char_lstm")(
            radical_embedding) #返回最后一个作为全局特征

        embedding = Concatenate(axis=-1)([char_embedding,word_embedding,radical_embedding])
        # radical_embedding = Dense(self.params['char_embedding_size'])(radical_embedding)
        # word_embedding = Dense(self.params['char_embedding_size'])(word_embedding)
        # char_embedding = Add(axis=-1)([char_embedding, word_embedding,radical_embedding])

        # Add LSTMs
        shared_layer = embedding
        logging.info("LSTM-Size: %s" % str(self.params['LSTM-Size']))
        cnt = 1
        for size in self.params['LSTM-Size']:
            if isinstance(self.params['dropout'], (list, tuple)):
                shared_layer = Bidirectional(LSTM(size, return_sequences=True, dropout=self.params['dropout'][0], recurrent_dropout=self.params['dropout'][1]),
                                             name='shared_varLSTM_' + str(cnt))(shared_layer)
            cnt += 1

        output = shared_layer
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