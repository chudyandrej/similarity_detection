
# Company: Ataccama Software. s.r.o.
# Author: Andrej Chudy
# Date created: 9/19/2018
# Python Version: 3.6.5

__version__ = '0.0.2'

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import keras.backend as K

import random
import pickle
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from keras.utils import plot_model
from scipy.sparse import *
from scipy import *

from data import SeqDataObject

# Data constants

CORE_PATH = "../../../"
DATA_PATH = CORE_PATH + 'input_data/aggregate_value.csv'

# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 300  # Number of epochs to train for.
LSTM_DIM = 256  # Latent dimensionality of the encoding space.


def cut_data_to_batches(input_texts, target_texts):
    data = list(zip(input_texts, target_texts))
    result = []

    for i in range(int(len(data) / BATCH_SIZE)):
        slide = list(enumerate(data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]))
        result.append(slide)
    return result


def generate_fit_vectors(data):
    data_batches = cut_data_to_batches(data.input_texts, data.target_texts)

    while True:
        for data_slide in data_batches:
            encoder_input_data = np.zeros((len(data_slide), data.max_seq_length, data.size_tokens), dtype='float32')
            decoder_input_data = np.zeros((len(data_slide), data.max_seq_length, data.size_tokens), dtype='float32')
            decoder_target_data = np.zeros((len(data_slide), data.max_seq_length, data.size_tokens), dtype='float32')
            for i, (input_text, target_text) in data_slide:
                for t, char in enumerate(input_text):
                    encoder_input_data[i, t, ord(char)] = 1.
                for t, char in enumerate(target_text):
                    # decoder_target_data is ahead of decoder_input_data by one timestep
                    decoder_input_data[i, t, ord(char)] = 1.
                    if t > 0:
                        # decoder_target_data will be ahead by one timestep
                        # and will not include the start character.
                        decoder_target_data[i, t - 1, ord(char)] = 1.

            yield [encoder_input_data, decoder_input_data], decoder_target_data


def categorical_crossentropy(y_true, y_pred):
    print("=============================")
    print(y_true)
    print(y_pred)
    result = K.categorical_crossentropy(y_true, y_pred)
    print(result)
    print("=============================")

    return result


if __name__ == "__main__":
    data = SeqDataObject(DATA_PATH, LSTM_DIM)

    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, data.size_tokens))
    encoder = LSTM(LSTM_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, data.size_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(data.size_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    data.save()
    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Run training
    model.compile(optimizer='rmsprop', loss=categorical_crossentropy)
    history = model.fit_generator(generate_fit_vectors(data),
                                  steps_per_epoch=int(len(data.input_texts) / BATCH_SIZE),
                                  epochs=EPOCHS,
                                  # validation_split=n,
                                  callbacks=[
        ModelCheckpoint(f'{CORE_PATH}output_data/autoencode_model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min'),
        EarlyStopping(monitor='loss', patience=6, verbose=1)
    ]
    )
    with open(f'{CORE_PATH}output_data/history.pkl', 'wb') as f:
        pickle.dump(history.history, f)

    print("Training success!")
