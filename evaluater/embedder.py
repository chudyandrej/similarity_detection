import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict
from unidecode import unidecode
from keras.preprocessing.text import Tokenizer
from tensorflow.python.lib.io import file_io

import json


def tekonizing(data_inputs, method, save_path):
    if method == "unidecode":
        data_inputs = [list(map(lambda value: list(map(lambda char: unidecode(char), value)), data_input))
                       for data_input in data_inputs]

    t = Tokenizer(char_level=True)
    [t.fit_on_texts(data_input) for data_input in data_inputs]
    if method == 'ord':
        result = ([list(map(lambda value: list(map(lambda char: ord(char), value)), data_input))
                  for data_input in data_inputs]), None
    elif method == 'tokenizer' or method == "unidecode":
        result = ([t.texts_to_sequences(data_input) for data_input in data_inputs]), len(t.index_word)
    else:
        assert False, "Unknown tekonizing method"

    with file_io.FileIO(save_path, mode='w+') as fp:
        json.dump(t.word_index, fp)
    return result


def convert_to_vec_onehot(encoder_model, data, max_seq_len, max_count_tokens):
    """Converting string data array to array of embedding vectors

    Args:
        encoder_model (keras.engine.training.Model): Description
        data (Array of STRINGS): Column values
        max_seq_len
        max_count_tokens

    Returns:
        (Array of Numpy): Array of embedding vectors
    """
    assert (len(data) > 0), "Input data are empty!"
    assert (max_count_tokens > 0), "Count of tokens can not be 0 or negative"

    encoder_input_data = np.zeros((len(data), max_seq_len, max_count_tokens), dtype='float32')

    for i, input_text in enumerate(data):
        for t, char in enumerate(input_text):
            # If character not be include in training vec2vec it will be replace by ' '
            print(tokenizer(char, full_unicode=False))
            encoder_input_data[i, t, tokenizer(char, full_unicode=False)] = 1.

    input_seq = encoder_model.predict(encoder_input_data)
    vec_s1 = input_seq[0]
    vec_s2 = input_seq[1]
    assert (vec_s1.shape == vec_s2.shape), "Shape of vec1 and vec2 is not equal!"

    vectors = []
    for index in range(len(vec_s1)):
        tmp = np.concatenate([vec_s1[index], vec_s2[index]], axis=0)
        vectors.append(tmp)

    return vectors


def convert_to_vec_tok(encoder_model, data, max_seq_len, full_unicode=False):
    """Converting string data array to array of embedding vectors

    Args:
        :param max_seq_len:
        :param data:
        :param encoder_model:
        :param full_unicode:

    Returns:
        (Array of Numpy): Array of embedding vectors

    """
    assert (len(data) > 0), "Input data are empty!"
    assert (max_seq_len > 0), "Count of tokens can not be 0 or negative"

    encoder_input_data = np.zeros((len(data), max_seq_len), dtype='int')

    for i, input_text in enumerate(data):
        for t, char in enumerate(input_text):
            # If character not be include in training vec2vec it will be replace by ' '
            encoder_input_data[i, t] = tokenizer(char, full_unicode)

    output = encoder_model.predict(encoder_input_data)

    return output


def convert_to_vec_tok_over_columns(encoder_model, data, max_seq_len, full_unicode=False):
    """Converting string data array to array of embedding vectors

    Args:
        :param max_seq_len:
        :param data:
        :param encoder_model:
        :param full_unicode:

    Returns:
        (Array of Numpy): Array of embedding vectors

    """
    assert (len(data) > 0), "Input data are empty!"
    assert (max_seq_len > 0), "Count of tokens can not be 0 or negative"

    encoder_input_data = np.zeros((data.shape[0], data.shape[1], max_seq_len), dtype='int')

    for i, quantiles in enumerate(data):
        for k, input_text in enumerate(quantiles):
            for t, char in enumerate(input_text):
                encoder_input_data[i, k, t] = tokenizer(char, full_unicode)
    output = encoder_model.predict(encoder_input_data)
    return output


def create_column_embedding_by(uid_embedding, ag_method):
    uid_embedding = list(uid_embedding)
    assert (len(uid_embedding) > 0), "Input data are empty!"

    uid_embeddings_index = defaultdict(list)
    [uid_embeddings_index[uid].append(embedding) for uid, embedding in uid_embedding]

    uid_embedding = []
    for uid, embeddings in uid_embeddings_index.items():
        if ag_method == "mean":
            vector = np.average(np.array(embeddings), axis=0)
        else:
            vector = np.sum(np.array(embeddings), axis=0)

        uid_embedding.append((uid, vector))

    return uid_embedding


def create_column_embedding_by_mrc(type_embedding):
    """Create column embedding from value embeddings by more representations of class by selected value encodings

    Args:
        type_embedding (Array): Array of Tuples
        weights (None, optional): Array of numbers with same size as value_embeddings.

    Returns:
        Array of Tuple: [(column_name, Numpy), ...]

    """
    type_embedding = list(type_embedding)
    assert (len(type_embedding) > 0), "Input data are empty!"

    class_embeddings_index = defaultdict(list)
    [class_embeddings_index[key].append(embedding) for key, embedding in type_embedding]
    class_embeddings_index = dict(class_embeddings_index)

    class_embedding = []
    for column_name, embs in class_embeddings_index.items():
        [class_embedding.append((column_name, e)) for e in embs]
    return class_embedding


def tokenizer_0_96(char):
    """Reduce he alphabet replace all white symbols as space

    Args:
        char (STRING): Char

    Returns:
        NUMBER: Code <0,94>
    """
    code = ord(char)
    if 0 <= code <= 31 or code == 127:    # Is white
        code = 0
    else:
        code -= 32

    return code

# ------------------------- PRIVATE ----------------------------------


def job(uid, embeddings, ag_method):
    # print(embeddings)
    vector = []
    if ag_method == "mean":
        vector = np.average(np.array(embeddings), axis=0)
    elif ag_method == "sum":
        vector = np.sum(np.array(embeddings), axis=0)
    return uid, vector


def tokenizer(char, full_unicode):
    """Reduce he alphabet replace all white symbols as space
    Args:
        :param char:
        :param full_unicode:

    Returns:
        NUMBER: Code <0,94>

    """
    assert (char is not None), "Input data are empty!"
    code = ord(char)

    if full_unicode:
        return code

    if 0 <= code <= 31 or code == 127:    # Is white
        code = 0
    else:
        code -= 32

    return code
