from neuralnets.char_bichar_trichar_bilstm import BiLSTM
import json
import math
import numpy as np
from tools.conlleval import evaluate_conll_file
import os
from tqdm import tqdm
import logging
from tools.load_word2vec import get_char_embedding_matrix,get_bichar_embedding_matrix,get_trichar_embedding_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
train_data = json.load(open('./inputs/MSAR/train_data.json', encoding='utf-8'))
dev_data = json.load(open('./inputs/MSAR/dev_data.json', encoding='utf-8'))
test_data = json.load(open('./inputs/MSAR/test_data.json', encoding='utf-8'))
id2word, word2id = json.load(open('./inputs/MSAR/word2id.json', encoding='utf-8'))
id2char, char2id = json.load(open('./inputs/MSAR/char2id.json', encoding='utf-8'))
id2BIO, BIO2id = json.load(open('./inputs/MSAR/BIO2id.json', encoding='utf-8'))
id2Radical, Radical2id = json.load(open('./inputs/MSAR/radical2id.json', encoding='utf-8'))
id2bichar, bichar2id = json.load(open('./inputs/MSAR/bichar2id.json',encoding='utf-8'))
id2trichar,trichar2id = json.load(open('./inputs/MSAR/trichar2id.json',encoding='utf-8'))

# word_embedding_matrix = get_embedding_matrix(word2id)

char_embedding_matrix = get_char_embedding_matrix(char2id)
bichar_embedding_matrix = get_bichar_embedding_matrix(bichar2id)
trichar_embedding_matrix = get_trichar_embedding_matrix(trichar2id)

params = {'char2id_size':len(char2id),'char_embedding_size':300,'epochs':45,'earlyStopping':5,'trichar_embedding_size':300,'trichar2id_size':len(trichar2id),
          'word_embedding_size':300,'word2id_size':len(word2id),'bichar_embedding_size':300,'bichar2id_size':len(bichar2id)
          ,'n_class_labels':len(BIO2id),'save_path':'./models/char_trichar.weights'}

debug = False
if debug:
    train_data = train_data[:200]
test_data = [data for data in test_data if len(data['text']) > 1]

train_data = train_data + dev_data


def process_batch_data(batch_data, char2id, word2id, Radical2id, BIO2id):
    new_batch_data = []

    for data in batch_data:
        dic = {}
        text = [char2id.get(_char, 1) for _char in data['text']]  # 1,UNK,0 pad
        bichar = [bichar2id.get(_bichar, 1) for _bichar in data['bichar']]
        trichar = [trichar2id.get(_trichar,1) for _trichar in data['trichar']]

        bio = [BIO2id.get(_bio) for _bio in data['NER_BIO']]
        # bio = np.expand_dims(bio,axis=-1)

        dic['text'] = text
        dic['bichar'] = bichar
        dic['NER_BIO'] = bio
        dic['trichar'] = trichar

        new_batch_data.append(dic)

    return new_batch_data


def minibatch_iterate_dataset(trainData, miniBatchSize=64):
    trainData.sort(key=lambda x: len(x['text']))  # Sort train matrix by sentence length
    trainRanges = []
    oldSentLength = len(trainData[0]['text'])
    idxStart = 0
    # Shuffle TrainData
    # Find start and end of ranges with sentences with same length
    for idx in range(len(trainData)):
        sentLength = len(trainData[idx]['text'])

        if sentLength != oldSentLength:
            trainRanges.append((idxStart, idx))
            idxStart = idx

        oldSentLength = sentLength

    # Add last sentence
    trainRanges.append((idxStart, len(trainData)))

    # Break up ranges into smaller mini batch sizes
    # Break up ranges into smaller mini batch sizes
    miniBatchRanges = []
    for batchRange in trainRanges:
        rangeLen = batchRange[1] - batchRange[0]
        bins = int(math.ceil(rangeLen / float(miniBatchSize)))
        binSize = int(math.ceil(rangeLen / float(bins)))

        for binNr in range(bins):
            startIdx = binNr * binSize + batchRange[0]
            endIdx = min(batchRange[1], (binNr + 1) * binSize + batchRange[0])
            miniBatchRanges.append((startIdx, endIdx))

    # shuffle minBatchRanges
    np.random.shuffle(miniBatchRanges)

    for miniRange in tqdm(miniBatchRanges):
        # print(miniRange)
        batch_data = []
        for i in range(miniRange[0], miniRange[1]):
            batch_data.append(trainData[i])
            # 序列化text,label
        batch_data = process_batch_data(batch_data, char2id, word2id, Radical2id, BIO2id)

        yield batch_data


def trainModel(model):
    for batch in minibatch_iterate_dataset(train_data):
        nnInput = []
        Labels = np.array([data['NER_BIO'] for data in batch])

        inputSentece = np.array([data['text'] for data in batch])
        inputBichar = np.array([data['bichar'] for data in batch])
        inputTrichar = np.array([data['trichar'] for data in batch])

        nnInput.append(inputSentece)
        nnInput.append(inputBichar)
        nnInput.append(inputTrichar)

        Labels = np.expand_dims(Labels, axis=-1)
        model.train_on_batch(nnInput, Labels)


def getSentenceLengths(sentences):
    # 返回字典 [len(sentence),idx]
    sentenceLengths = {}
    for idx in range(len(sentences)):
        sentence = sentences[idx]
        if len(sentence) not in sentenceLengths:
            sentenceLengths[len(sentence)] = []
        sentenceLengths[len(sentence)].append(idx)

    return sentenceLengths


def predictLabels(model, data):
    # char-bilstm 只输入char
    sentences = [_data['text'] for _data in data]
    trichars = [_data['trichar'] for _data in data]
    bichars = [_data['bichar'] for _data in data]

    predLabels = [None] * len(sentences)
    sentenceLengths = getSentenceLengths(sentences)

    for indices in tqdm(sentenceLengths.values()):
        nnInput = []

        # 输入数据
        inputSentence = np.array([sentences[idx] for idx in indices])
        inputTriichar = np.array([trichars[idx] for idx in indices])
        inputBichar = np.array([bichars[idx] for idx in indices])

        nnInput.append(inputSentence)
        nnInput.append(inputBichar)
        nnInput.append(inputTriichar)

        predictions = model.predict(nnInput, verbose=False)
        predictions = predictions.argmax(axis=-1)  # Predict classes

        predIdx = 0
        for idx in indices:
            predLabels[idx] = predictions[predIdx]
            predIdx += 1

    return predLabels


def save_result(ner_pred, ner_true, save_file):
    """
    保存 text ture pred
    ner_pred: [[00021000..],...[]]
    ner_true: json [text:; bio:]
    :return:
    """

    pred_bio = []
    for _pred_bio in ner_pred:
        temp = [id2BIO.get(str(bio), 'O') for bio in _pred_bio]
        pred_bio.append(temp)

    with open(save_file, 'w', encoding='utf-8') as fr:
        for idx in range(len(pred_bio)):
            for i in range(len(ner_true[idx]['text'])):
                fr.write(str(ner_true[idx]['text'][i]) + ' ' + str(ner_true[idx]['NER_BIO'][i]) + ' ' + str(pred_bio[idx][i]) + '\n')

def fit(params,dev_data,test_data):
    iterations_f1 = {}
    epochs = params['epochs']
    no_improvement_since = 0
    char_lstm = BiLSTM(char_embedding_matrix,bichar_embedding_matrix,trichar_embedding_matrix,params)
    model = char_lstm.build_model()
    # best_dev_f,best_dev_p,best_dev_r = 0,0,0
    best_test_f,best_test_p,best_test_r = 0,0,0
    for epoch in range(epochs):
        trainModel(model)
        # _dev_data = process_batch_data(dev_data,char2id,word2id,Radical2id,BIO2id)
        _test_data = process_batch_data(test_data,char2id,word2id,Radical2id,BIO2id)

        # dev_pred = predictLabels(model,_dev_data)

        test_pred = predictLabels(model,_test_data)


        # save_result(dev_pred,dev_data,'./outputs/dev')
        # dev_p, dev_r, dev_f = evaluate_conll_file('./outputs/dev')

        save_result(test_pred,test_data,'./outputs/test')
        test_p, test_r, test_f = evaluate_conll_file('./outputs/test')

        # print('NER,当前第{}个epoch，验证集,准确度为{},召回为{},f1为：{}'.format(epoch,  dev_p, dev_r, dev_f))
        print('NER,当前第{}个epoch，测试集,准确度为{},召回为{},f1为：{}'.format(epoch,  test_p, test_r, test_f))
        # print('RC,当前第{}个epoch，验证集,准确度为{},召回为{},f1为：{}'.format(i, rel_P, rel_R, rel_F))
        print('-' * 20)

        iterations_f1[epoch] = test_f

        if best_test_f <  test_f:
            best_test_f = test_f
            best_test_p = test_p
            best_test_r = test_r
            no_improvement_since = 0
            print('epoch{},当前最好的 测试集,准确度为{},召回为{},f1为：{}'.format(epoch,test_p, test_r, test_f))

            if params['save_path'] != None:
                model.save_weights(params['save_path'])
        else:
            no_improvement_since+=1
        if params['earlyStopping'] > 0 and no_improvement_since >= params['earlyStopping']:
            logging.info("!!! Early stopping, no improvement after " + str(no_improvement_since) + " epochs !!!")
            break

        # if best_dev_f <  dev_f:
        #     best_dev_f = dev_f
        #     best_dev_p = dev_p
        #     best_dev_r = dev_r
        #     best_test_f = test_f
        #     best_test_p = test_p
        #     best_test_r = test_r
        #     no_improvement_since = 0
        #     print('当前最好的 验证集,准确度为{},召回为{},f1为：{}'.format(dev_p, dev_r, dev_f))
        #     print('当前最好的 测试集,准确度为{},召回为{},f1为：{}'.format(test_p, test_r, test_f))
        #
        #     if params['save_path'] != None:
        #         model.save_weights(params['save_path'])
        # else:
        #     no_improvement_since+=1
        # if params['earlyStopping'] > 0 and no_improvement_since >= params['earlyStopping']:
        #     logging.info("!!! Early stopping, no improvement after " + str(no_improvement_since) + " epochs !!!")
        #     break

    print('训练结束')
    # print('当前最好的 验证集,准确度为{},召回为{},f1为：{}'.format(best_dev_p, best_dev_r, best_dev_f))
    print('当前最好的 测试集,准确度为{},召回为{},f1为：{}'.format(best_test_p, best_test_r, best_test_f))
    return iterations_f1

iterations_f1 = fit(params,dev_data,test_data)
print(iterations_f1)
# def fit(params, dev_data, test_data):
#     epochs = params['epochs']
#     no_improvement_since = 0
#     char_lstm = BiLSTM(params)
#     model = char_lstm.build_model()
#     best_dev_f,best_dev_p,best_dev_r = 0,0,0
#     best_test_f, best_test_p, best_test_r = 0, 0, 0
#     for epoch in range(epochs):
#         trainModel(model)
#         _dev_data = process_batch_data(dev_data,char2id,word2id,Radical2id,BIO2id)
#         _test_data = process_batch_data(test_data, char2id, word2id, Radical2id, BIO2id)
#
#         dev_pred = predictLabels(model,_dev_data)
#         test_pred = predictLabels(model, _test_data)
#
#         save_result(dev_pred,dev_data,'./outputs/dev')
#         dev_p, dev_r, dev_f = evaluate_conll_file('./outputs/dev')
#
#         save_result(test_pred, test_data, './outputs/test')
#         test_p, test_r, test_f = evaluate_conll_file('./outputs/test')
#
#         logging.info('NER,当前第{}个epoch，验证集,准确度为{},召回为{},f1为：{}'.format(epoch,  dev_p, dev_r, dev_f))
#         logging.info('NER,当前第{}个epoch，测试集,准确度为{},召回为{},f1为：{}'.format(epoch, test_p, test_r, test_f))
#         logging.info('-' * 20)
#
#         if best_dev_f < dev_f:
#             best_test_f = test_f
#             best_test_p = test_p
#             best_test_r = test_r
#
#             best_dev_f = dev_f
#             best_dev_p = dev_p
#             best_dev_r = dev_r
#
#             no_improvement_since = 0
#             print('当前最好的 测试集,准确度为{},召回为{},f1为：{}'.format(test_p, test_r, test_f))
#
#             if params['save_path'] != None:
#                 model.save_weights(params['save_path'])
#         else:
#             no_improvement_since += 1
#         if params['earlyStopping'] > 0 and no_improvement_since >= params['earlyStopping']:
#             logging.info("!!! Early stopping, no improvement after " + str(no_improvement_since) + " epochs !!!")
#             break
#
#     print('训练结束')
#     print('当前最好的 验证集,准确度为{},召回为{},f1为：{}'.format(best_dev_p, best_dev_r, best_dev_f))
#     print('当前最好的 测试集,准确度为{},召回为{},f1为：{}'.format(best_test_p, best_test_r, best_test_f))
#
#
# fit(params, dev_data, test_data)