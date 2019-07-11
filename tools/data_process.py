

def char_pad(datas,maxlen_sentence,maxlen_word):
    #word_leve pad for char input
    #use the maxlen of batch data of words to pad the char levels and use the maxlen of batch datas to pad the sentence level inputs
    """
    :param datas: [batch_size,None,None]
    :return: [batch_size,maxlen of sentence , maxlen of words]
    """
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

