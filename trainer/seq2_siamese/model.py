
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Dot
from collections import defaultdict
import tensorflow as tf
from unidecode import unidecode
import numpy as np
import pandas as pd
import random


MAX_TEXT_SEQUENCE_LEN = 64
TOKEN_COUNT = 95
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
TRAINING_SAMPLES = 600000
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 256  # Latent dimensionality of the encoding space.


def create_model():
    encoder_inputs_1 = Input(shape=(None, TOKEN_COUNT), name="encoder_Input_1")
    encoder_inputs_2 = Input(shape=(None, TOKEN_COUNT), name="encoder_Input_2")

    # Define an input sequence and process it.
    encoder = LSTM(LSTM_DIM, return_state=True, name="encoder")
    encoder_outputs1, state_h1, state_c1 = encoder(encoder_inputs_1)
    encoder_states1 = [state_h1, state_c1]
    encoder_states1_concat = Concatenate(-1)(encoder_states1)
    encoder_outputs2, state_h2, state_c2 = encoder(encoder_inputs_2)
    encoder_states2 = [state_h2, state_c2]
    encoder_states2_concat = Concatenate(-1)(encoder_states2)

    output = Dot(1, normalize=True, name="dotLayer")([encoder_states1_concat, encoder_states2_concat])
    model = Model([encoder_inputs_1, encoder_inputs_2], output)
    return model


def generate_random_fit(input_texts, types):
    """Generator for fiting

    Args:
        input_texts (Array): Encder data
        target_texts (Array): Decoder data
        types (Array): Type

    Yields:
        TYPE: Batch of data
    """

    column_index = make_index_by_firs_value(zip(types, input_texts))

    while True:
        batch_data = get_pairs_with(True, BATCH_SIZE // 2, column_index)
        batch_data += get_pairs_with(False, BATCH_SIZE // 2, column_index)

        match, pairs = zip(*batch_data)
        batch_data1, batch_data2 = zip(*pairs)

        encoder_input_data1 = convert_to_vec(batch_data1)
        encoder_input_data2 = convert_to_vec(batch_data2)
        # print(np.array_equal(encoder_input_data1[0], encoder_input_data2[0]))

        distances = np.array(list(map(lambda x: int(x), match)))

        yield [encoder_input_data1, encoder_input_data2], distances


def load_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    input_texts = map(lambda x: unidecode(str(x)), df['value'].values)
    input_texts = map(lambda x: x.replace('\t', '  '), input_texts)
    input_texts = list(map(lambda x: x[:MAX_TEXT_SEQUENCE_LEN - 1], input_texts))
    types = df['type'].values

    return input_texts, types


def tokening(char):
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
    encoder_input_data = np.zeros((len(batch_data), MAX_TEXT_SEQUENCE_LEN, TOKEN_COUNT), dtype='float32')

    for i, input_text in enumerate(batch_data):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, tokening(char)] = 1.
    return encoder_input_data


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
    while count > len(batch_data):
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
