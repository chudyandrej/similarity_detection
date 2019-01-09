
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Dot
from collections import defaultdict
import tensorflow as tf
from unidecode import unidecode
import numpy as np
import pandas as pd
import random
import pickle

MAX_TEXT_SEQUENCE_LEN = 100
TOKEN_COUNT = 95
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 512  # Latent dimensionality of the encoding space.


def create_model():
    encoder_inputs = Input(shape=(None, TOKEN_COUNT), name="encoder_Input")
    decoder_inputs = Input(shape=(None, TOKEN_COUNT), name="decoder_Input")

    # Define an input sequence and process it.
    encoder = LSTM(LSTM_DIM, return_state=True, name="encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, name="decoder")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(TOKEN_COUNT, activation='softmax', name="dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def generate_random_fit(input_texts, target_texts):
    """Generator for fiting

    Args:
        input_texts (Array): Encder data
        target_texts (Array): Decoder data
        types (Array): Type

    Yields:
        TYPE: Batch of data
    """

    data_batches = cut_data_to_batches(input_texts, target_texts)

    while True:
        for data_slide in data_batches:

            encoder_input_data, decoder_input_data, decoder_target_data = convert_to_vec(data_slide)
            yield [encoder_input_data, decoder_input_data], decoder_target_data


def load_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    input_texts = map(lambda x: unidecode(str(x)), df['value'].values)
    input_texts = map(lambda x: x.replace('\t', '  '), input_texts)
    input_texts = list(map(lambda x: x[:MAX_TEXT_SEQUENCE_LEN - 1], input_texts))

    target_texts = list(map(lambda x: "\t" + x, input_texts))
    return input_texts, target_texts


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
    decoder_input_data = np.zeros((len(batch_data), MAX_TEXT_SEQUENCE_LEN, TOKEN_COUNT), dtype='float32')
    decoder_target_data = np.zeros((len(batch_data), MAX_TEXT_SEQUENCE_LEN, TOKEN_COUNT), dtype='float32')

    for i, (input_text, target_text) in batch_data:
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, tokening(char)] = 1.
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, tokening(char)] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, tokening(char)] = 1.

    return encoder_input_data, decoder_input_data, decoder_target_data


def cut_data_to_batches(input_texts, target_texts):
    data = list(zip(input_texts, target_texts))
    result = []

    for i in range(len(data) // BATCH_SIZE):
        slide = list(enumerate(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE]))
        result.append(slide)
    return result
