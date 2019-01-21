
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Concatenate, Dot, Embedding
from keras.preprocessing.sequence import pad_sequences

from collections import defaultdict
import tensorflow as tf
from unidecode import unidecode
import numpy as np
import pandas as pd
import random
import pickle

MAX_TEXT_SEQUENCE_LEN = 64
TOKEN_COUNT = 95
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 512  # Latent dimensionality of the encoding space.
ENCODER_OUTPUT_DIM = 128


def embedding_convert_model():
    encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input")
    embedding = Embedding(65536, output_dim=ENCODER_OUTPUT_DIM)
    encoder_output = embedding(encoder_inputs)
    model = Model(encoder_inputs, encoder_output)
    return model


def create_model():
    encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN, ENCODER_OUTPUT_DIM), name="encoder_Input")
    decoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN, ENCODER_OUTPUT_DIM), name="decoder_Input")

    # Define an input sequence and process it.
    encoder = LSTM(LSTM_DIM, return_state=True, name="encoder")
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, name="decoder")
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    decoder_dense = Dense(ENCODER_OUTPUT_DIM,  name="dense")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model


def generate_random_fit(input_texts, target_texts, embedd_model):
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

            d1 = embedd_model.predict(encoder_input_data)
            d2 = embedd_model.predict(decoder_input_data)
            d3 = embedd_model.predict(decoder_target_data)

            yield [d1, d2], d3


def load_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    input_texts = df['value'].values
    input_texts = map(str, input_texts)
    input_texts = map(str.strip, input_texts)
    input_texts = list(map(lambda x: str(x).replace('\t', '  '), input_texts))
    # input_texts = (x[::-1] for x in input_texts)

    target_texts = list(map(lambda x: "\t" + x, input_texts))

    return input_texts, target_texts


def convert_to_vec(batch_data):
    """Convert value input to vector form encoder

    Args:
        batch_data (ARRAY)

    Returns:
        (np.Array, np.Array, np.Array):
    """
    input_texts, target_texts = zip(*batch_data)
    input_texts = list(map(lambda x: [ord(y) for y in x], input_texts))
    target_texts = list(map(lambda x: [ord(y) for y in x], target_texts))

    input_texts = pad_sequences(input_texts, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='pre')
    target_texts = pad_sequences(target_texts, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='pre')

    return input_texts, target_texts, input_texts.copy()


def cut_data_to_batches(input_texts, target_texts):
    data = list(zip(input_texts, target_texts))
    np.random.shuffle(data)
    result = []

    for i in range(len(data) // BATCH_SIZE):

        slide = list(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        result.append(slide)
    np.random.shuffle(result)
    return result
