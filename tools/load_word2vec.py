import numpy as np

def _load_embed(file):
    def get_coefs(word, *arr):
        return word, np.asarray(arr)[:300]
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file, encoding='utf-8'))

    return embeddings_index

def _load_embedding_matrix(word_index, embedding):
    embed_word_count = 0
    # nb_words = min(max_features, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(size=(nb_words+2, 300))

    for word, i in word_index.items():

#        if i >= max_features: continue
        if word not in embedding:
            word = word.lower()
        if word.islower and word not in embedding:
            word = word.title()
        embedding_vector = embedding.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            embed_word_count += 1
    print('词向量的覆盖率为{}'.format(embed_word_count / len(word_index)))
    return embedding_matrix


def _load_bichar_embedding_matrix(word_index, embedding):
    embed_word_count = 0
    # nb_words = min(max_features, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(size=(nb_words+2, 300))

    for word, i in word_index.items():

        #        if i >= max_features: continue
        if word not in embedding:
            word = word.lower()
        if word.islower and word not in embedding:
            word = word.title()

        char1 = str(word[0])
        char2 = str(word[1])

        char1_vector = embedding.get(char1)
        char2_vector = embedding.get(char2)

        if char1_vector is not None and char2_vector is not None:
            char1_vector = np.array(char1_vector,dtype='float32')
            char2_vector = np.array(char2_vector, dtype='float32')

            embedding_vecotr = np.mean([char1_vector,char2_vector],axis=0)
            embedding_matrix[i] = embedding_vecotr
            embed_word_count += 1
    print('词向量的覆盖率为{}'.format(embed_word_count / len(word_index)))
    return embedding_matrix

def _load_trichar_embedding_matrix(word_index, embedding):
    embed_word_count = 0
    # nb_words = min(max_features, len(word_index))
    nb_words = len(word_index)
    embedding_matrix = np.random.normal(size=(nb_words+2, 300))

    for word, i in word_index.items():

        #        if i >= max_features: continue
        if word not in embedding:
            word = word.lower()
        if word.islower and word not in embedding:
            word = word.title()

        char1 = str(word[0])
        char2 = str(word[1])
        char3 = str(word[2])


        char1_vector = embedding.get(char1)
        char2_vector = embedding.get(char2)
        char3_vector = embedding.get(char3)

        if char1_vector is not None and char2_vector is not None and char3_vector is not None:
            char1_vector = np.array(char1_vector,dtype='float32')
            char2_vector = np.array(char2_vector, dtype='float32')
            char3_vector = np.array(char3_vector, dtype='float32')

            embedding_vecotr = np.mean([char1_vector,char2_vector,char3_vector],axis=0)

            embedding_matrix[i] = embedding_vecotr
            embed_word_count += 1
    print('词向量的覆盖率为{}'.format(embed_word_count / len(word_index)))
    return embedding_matrix


def get_char_embedding_matrix(word_index):
    embedding_dir =  '../inputs/embedding_matrix/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    embedding = _load_embed(embedding_dir)
    embedding_matrix = _load_embedding_matrix(word_index, embedding)

    return embedding_matrix

def get_word_embedding_matrix(word_index):
    embedding_dir = '../inputs/embedding_matrix/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    embedding = _load_embed(embedding_dir)
    embedding_matrix = _load_embedding_matrix(word_index, embedding)

    return embedding_matrix

def get_bichar_embedding_matrix(word_index):
    embedding_dir =  '../inputs/embedding_matrix/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    embedding = _load_embed(embedding_dir)
    embedding_matrix = _load_bichar_embedding_matrix(word_index, embedding)

    return embedding_matrix

def get_trichar_embedding_matrix(word_index):
    embedding_dir = '../inputs/embedding_matrix/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5'
    embedding = _load_embed(embedding_dir)
    embedding_matrix = _load_trichar_embedding_matrix(word_index, embedding)

    return embedding_matrix
