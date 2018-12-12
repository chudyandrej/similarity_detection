
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dot, Concatenate, Convolution1D, GlobalMaxPooling1D, AlphaDropout
from collections import defaultdict
import tensorflow as tf
from unidecode import unidecode
import numpy as np
import pandas as pd
import random


DATA_SIZE = 600000
MAX_TEXT_SEQUENCE_LEN = 64
TOKEN_COUNT = 95
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 256  # Latent dimensionality of the encoding space.
DENSE_DIM = 1024
EMBEDDING_SIZE = 128
CNN_LAYERS = [[256, 10], [256, 7], [256, 5], [256, 3]]
DROPOUT = 0.2


def create_model():

    def embedder():
        # Embedding layers
        inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name='sent_input', dtype='int64')
        emb = Embedding(TOKEN_COUNT + 1, EMBEDDING_SIZE, input_length=MAX_TEXT_SEQUENCE_LEN)(inputs)
        # Convolution layers
        convolution_output = []
        for num_filters, filter_width in CNN_LAYERS:
            conv = Convolution1D(filters=num_filters,
                                 kernel_size=filter_width,
                                 activation='tanh',
                                 name='Conv1D_{}_{}'.format(num_filters, filter_width))(emb)
            pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))(conv)
            convolution_output.append(pool)
        x = Concatenate()(convolution_output)
        # Fully connected layers
        x = Dense(DENSE_DIM, activation='selu', kernel_initializer='lecun_normal')(x)
        x = AlphaDropout(DROPOUT)(x)
        x = Dense(256, activation='selu', kernel_initializer='lecun_normal')(x)
        return Model(inputs=inputs, outputs=x)

    inputs1 = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name='sent_input1', dtype='int64')
    inputs2 = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name='sent_input2', dtype='int64')
    embedding_model = embedder()
    output1 = embedding_model(inputs1)
    output2 = embedding_model(inputs2)
    output = Dot(1, normalize=True, name="dotLayer")([output1, output2])

    # Build and compile model
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    print("CharCNNKim model built: ")
    model.summary()
    return model


def generate_random_fit(input_texts, types):
    """Generator for filtering

    Args:
        input_texts (Array): Encoder data
        types (Array): Type

    Yields:
        TYPE: Batch of data
    """
    column_index = make_index_by_firs_value(list(zip(types, input_texts)))

    while True:
        batch_data = get_pairs_with(True, BATCH_SIZE // 2, column_index)
        batch_data += get_pairs_with(False, BATCH_SIZE // 2, column_index)

        match, pairs = zip(*batch_data)
        batch_data1, batch_data2 = zip(*pairs)

        encoder_input_data1 = convert_to_vec(batch_data1)
        encoder_input_data2 = convert_to_vec(batch_data2)

        distances = np.array(list(map(lambda x: int(x), match)))

        yield [encoder_input_data1, encoder_input_data2], distances


def load_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    input_texts = map(lambda x: unidecode(str(x)), df['value'].values)
    input_texts = map(lambda x: x.replace('\t', '  '), input_texts)
    input_texts = list(map(lambda x: x[:MAX_TEXT_SEQUENCE_LEN - 1], input_texts))

    types = df['type'].values
    print("Loaded " + str(len(input_texts)))
    return input_texts, types


def tokenizer(char):
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


def convert_to_vec(batch_data):
    """Convert value input to vector form encoder

    Args:
        batch_data (ARRAY)

    Returns:
        (np.Array, np.Array, np.Array): Vectors with one-hot encoding for Encoder, Decoder and Target
    """
    encoder_data = np.zeros((len(batch_data), MAX_TEXT_SEQUENCE_LEN), dtype='int32')

    for i, input_text in enumerate(batch_data):
        for t, char in enumerate(input_text):
            encoder_data[i, t] = tokenizer(char)

    return encoder_data


def get_pairs_with(is_same_class, count, column_index):
    """Get pairs of data with the same or different class

    Args:
        is_same_class (BOOL): I want pairs wit equaly class or not
        count (NUMBER): Count of pairs
        column_index (Array((enc_data, deco_data, class))): Description

    Returns:
        Array((first, second)): Pairs
    """

    batch_data = []
    while count >= len(batch_data):
        columns = list(column_index.keys()).copy()
        column_key = columns[random.randint(0, len(columns) - 1)]
        column_data = column_index[column_key].copy()

        if len(column_data) <= 1:
            print("continue")
            continue

        first = column_data[random.randint(0, len(column_data) - 1)]

        if is_same_class:
            column_data.remove(first)
            second = column_data[random.randint(0, len(column_data) - 1)]
        else:

            columns.remove(column_key)
            column_key = columns[random.randint(0, len(columns) - 1)]
            column_data = column_index[column_key]
            second = column_data[random.randint(0, len(column_data) - 1)]

        batch_data.append((is_same_class, (first, second)))
    return batch_data


def make_index_by_firs_value(data):
    result = defaultdict(list)
    for k, v in data:
        result[k].append(v)
    return dict(result)
