
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.preprocessing.sequence import pad_sequences

from collections import defaultdict
import tensorflow as tf
from unidecode import unidecode
from keras.utils import plot_model
import numpy as np
import pandas as pd
import random
import pickle

MAX_TEXT_SEQUENCE_LEN = 64
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 256  # Latent dimensionality of the encoding space.
ENCODER_OUTPUT_DIM = 128


def embedding_convert_model():
    encoder_inputs = Input(shape=(None, MAX_TEXT_SEQUENCE_LEN), name="encoder_Input")
    embedding = TimeDistributed(Embedding(65536, output_dim=ENCODER_OUTPUT_DIM))(encoder_inputs)
    model = Model(encoder_inputs, embedding)
    return model


def create_model():
    enbedding_input = Input(shape=(MAX_TEXT_SEQUENCE_LEN, ENCODER_OUTPUT_DIM), name="encoder_input")
    encodder = LSTM(LSTM_DIM, activation='relu', recurrent_dropout=0.3)(enbedding_input)
    x = RepeatVector(MAX_TEXT_SEQUENCE_LEN)(encodder)
    x = LSTM(LSTM_DIM, activation='relu', return_sequences=True, recurrent_dropout=0.3)(x)
    x = TimeDistributed(Dense(ENCODER_OUTPUT_DIM))(x)
    model = Model(enbedding_input, x)
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
    input_d_batches, target_d_batches = cut_data_to_batches(input_texts, target_texts)
    input_encodded_batches = embedd_model.predict(input_d_batches)
    target_encodded_batches = embedd_model.predict(target_d_batches)

    while True:
        for i in range(target_encodded_batches.shape[0]):
            yield input_encodded_batches[i], target_encodded_batches[i]


def load_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    input_texts = df['value'].values
    input_texts = map(str, input_texts)
    input_texts = map(str.strip, input_texts)
    input_texts = list(map(lambda x: x[:MAX_TEXT_SEQUENCE_LEN], input_texts))
    target_texts = list(map(lambda x: x[::-1], input_texts))

    #TRANSFORM
    input_tokenized = list(map(lambda x: [ord(y) for y in x], input_texts))
    target_tokenized = list(map(lambda x: [ord(y) for y in x], target_texts))

    input_paded = pad_sequences(input_tokenized, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='pre')
    target_paded = pad_sequences(target_tokenized, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='post')

    return input_paded, target_paded


def cut_data_to_batches(input_texts, target_texts):
    data = list(zip(input_texts, target_texts))
    np.random.shuffle(data)
    input_d, target_d = zip(*data)

    r_input_d = []
    r_target_d = []
    for i in range(len(data) // BATCH_SIZE):
        r_input_d.append(input_d[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        r_target_d.append(target_d[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
    return np.array(r_input_d), np.array(r_target_d)
