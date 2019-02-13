
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K
from keras.preprocessing.text import Tokenizer


import tensorflow as tf
import numpy as np
import pandas as pd
from evaluater.embedder import tekonizing
import trainer.custom_components as cc

MAX_TEXT_SEQUENCE_LEN = 64
TOKEN_COUNT = 95
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 128  # Latent dimensionality of the encoding space.
ENCODER_OUTPUT_DIM = 256

tokenizer = Tokenizer(char_level=True)


def create_model_embedding(embedding_size):
    encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input")
    decoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="decoder_Input")
    target = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="target_Input")

    embedding = Embedding(embedding_size, output_dim=ENCODER_OUTPUT_DIM, trainable=False, name="embedding_layer")
    embedded_encoder_input = embedding(encoder_inputs)
    embedded_decoder_input = embedding(decoder_inputs)
    emedded_target = embedding(target)

    # Define an input sequence and process it.
    encoder = LSTM(LSTM_DIM, return_state=True, dropout=0.2, recurrent_dropout=0.2, name="encoder")
    encoder_outputs, state_h, state_c = encoder(embedded_encoder_input)
    encoder_states = [state_h, state_c]

    decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2,
                        name="decoder")
    decoder_outputs, _, _ = decoder_lstm(embedded_decoder_input, initial_state=encoder_states)

    decoder_dense = Dense(ENCODER_OUTPUT_DIM, activation='sigmoid', name="dense")
    decoder_output = decoder_dense(decoder_outputs)
    output = cc.CustomRegularization()([emedded_target, decoder_output])
    model = Model([encoder_inputs, decoder_inputs, target], output)
    return model


def create_model_onehot_layer(embedding_size):
    encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input")
    decoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="decoder_Input")
    target = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="target_Input")

    embedding = cc.OneHot(input_dim=embedding_size, input_length=MAX_TEXT_SEQUENCE_LEN)
    embedded_encoder_input = embedding(encoder_inputs)
    embedded_decoder_input = embedding(decoder_inputs)
    emedded_target = embedding(target)

    # Define an input sequence and process it.
    encoder = LSTM(LSTM_DIM, return_state=True, dropout=0.2, recurrent_dropout=0.2, name="encoder")
    encoder_outputs, state_h, state_c = encoder(embedded_encoder_input)
    encoder_states = [state_h, state_c]

    decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2,
                        name="decoder")
    decoder_outputs, _, _ = decoder_lstm(embedded_decoder_input, initial_state=encoder_states)

    decoder_dense = Dense(2100, activation='softmax', name="dense")
    decoder_output = decoder_dense(decoder_outputs)

    output = cc.CustomRegularization(loss_function="categorical_crossentropy")([emedded_target, decoder_output])
    model = Model([encoder_inputs, decoder_inputs, target], output)
    return model


def generate_batches(data):
    data_batches = cut_data_to_batches(data)

    while True:
        for data_slide in data_batches:
            input, imput_decoder, target = zip(*data_slide)

            input_paded = pad_sequences(input, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='pre')
            input_decoder_paded = pad_sequences(imput_decoder, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre',
                                                padding='post')
            target_paded = pad_sequences(target, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='post')
            yield [input_paded, input_decoder_paded, target_paded], target_paded


def load_and_preprocess_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    train_data = df['value'].values
    train_data = map(str, train_data)
    train_data = list(map(str.strip, train_data))
    train_data = list(map(lambda x: x[:MAX_TEXT_SEQUENCE_LEN - 1], train_data))

    input_texts = list(map(lambda x: x[::-1], train_data))
    input_decoder_texts = list(map(lambda x: '\t' + x, train_data))
    target_texts = train_data

    tokenized_data, count_chars = tekonizing([input_texts, input_decoder_texts, target_texts], method="ord")
    return tokenized_data, count_chars


def cut_data_to_batches(data):
    np.random.shuffle(data)
    result = []
    for i in range(len(data) // BATCH_SIZE):
        slide = list(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        result.append(slide)

    return result


