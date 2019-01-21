import numpy as np
from joblib import Parallel, delayed
from collections import defaultdict


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


def create_column_embedding_by_avg(type_embedding):
    """Create column embedding from value embeddings by vector AVERAGE

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

    class_embedding = Parallel(n_jobs=-1)(delayed(job)(column_name, embeddings)
                                          for column_name, embeddings in class_embeddings_index.items())
    return class_embedding


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


# ------------------------- PRIVATE ----------------------------------


def job(column_name, embeddings):

    mean_embedding = np.average(np.array(embeddings), axis=0)
    return column_name, mean_embedding


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
