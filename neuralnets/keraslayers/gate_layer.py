import keras
from keras import backend as K
from keras.layers import *

class Gate_Add_Lyaer(keras.layers.Layer):
    """
    gate add mechanism for word_char embedding
    z =  sigmoid(W(1)tanh(W(2)word_embedding + W(3)char_att))
    word_char_embedding = z*word_embedding + (1-z)char_att

    """
    def __init__(self,**kwargs):
        """

        :param word_embedding:  shape [batch,sentence,dim of word_embedding]
        :param char_att:  shape [batch,sentence,dim of char_embedding]
        :param kwargs:
        """
        super(Gate_Add_Lyaer,self).__init__(**kwargs)
        self.supports_masking = True

    def build(self, input_shape):

        self.W1 = self.add_weight(name='W1',shape=(input_shape[0][-1],input_shape[0][-1]),initializer='glorot_normal') #[dim,dim]
        # self.b1 = self.add_weight(name='b1',shape=(self.input_shape[0][-1]),initializer='glorot_normal')

        self.W_word = self.add_weight(name='W1',shape=(input_shape[1][-1],input_shape[0][-1]),initializer='glorot_normal') #[dim,dim]
        # self.b_word= self.add_weight(name='b2',shape=(self.input_shape[0][-1]),initializer='glorot_normal')

        self.W_bichar = self.add_weight(name='W1',shape=(input_shape[2][-1],input_shape[0][-1]),initializer='glorot_normal') #[dim,dim]
        # self.b_bichar = self.add_weight(name='b3',shape=(self.input_shape[0][-1]),initializer='glorot_normal')

        self.W4 = self.add_weight(name='W1',shape=(input_shape[0][-1],input_shape[0][-1]),initializer='glorot_normal') #[dim,dim]
        # self.b4 = self.add_weight(name='b4',shape=(self.input_shape[0][-1]),initializer='glorot_normal')
        super(Gate_Add_Lyaer, self).build(input_shape)

    def call(self,inputs,mask=None):
        # inputs[0]:word_embedding ,inputs[1]:char_embedding
        # word_embedding_reshaped = K.reshape(inputs[0],shape=(-1,word_embedding_shape[-1])) #[batch*sentence,dim of word embedding]
        # char_embedding_reshaped = K.reshape(inputs[1],shape=(-1,char_embedding_shape[-1])) #[batch*sentence, dim of char embedding]
        char_embedding = inputs[0]
        word_embedding = inputs[1]
        bichar_embedding = inputs[2]

        aux_embedding = K.dot(K.dot(word_embedding,self.W_word)+ K.dot(bichar_embedding,self.W_bichar),self.W4)

        g = K.sigmoid(K.dot(char_embedding,self.W1))
        embedding = g*aux_embedding + (1-g)*word_embedding
        return embedding

    def compute_mask(self, inputs, mask=None):
        return mask


    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2])
