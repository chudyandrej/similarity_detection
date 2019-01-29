
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding, Lambda, GRU
from keras.preprocessing.sequence import pad_sequences
from unidecode import unidecode
from keras import backend as K



import tensorflow as tf
import numpy as np
import pandas as pd
from evaluater.embedder import tokenizer_0_96
import trainer.custom_components as cc

MAX_TEXT_SEQUENCE_LEN = 64
TOKEN_COUNT = 95
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
GRU_DIM = 128  # Latent dimensionality of the encoding space.
ENCODER_OUTPUT_DIM = 256


def create_model_fullunicode():
    encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input")
    decoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="decoder_Input")
    target = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="target_Input")

    embedding = Embedding(65536, output_dim=ENCODER_OUTPUT_DIM, trainable=False, name="embedding_layer")
    embedded_encoder_input = embedding(encoder_inputs)
    embedded_decoder_input = embedding(decoder_inputs)
    emedded_target = embedding(target)

    encoder = GRU(GRU_DIM, return_state=True)
    encoder_outputs, state_h = encoder(embedded_encoder_input)

    decoder_gru = GRU(GRU_DIM, return_sequences=True)
    decoder_outputs = decoder_gru(embedded_decoder_input, initial_state=state_h)
    decoder_dense = Dense(ENCODER_OUTPUT_DIM, activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)

    output = cc.CustomRegularization(loss_function="mean_squared_error")([emedded_target, decoder_outputs])
    model = Model([encoder_inputs, decoder_inputs, target], output)
    return model


# def create_model_unidecode():
#     encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input")
#     decoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="decoder_Input")
#     target = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="target_Input")
#
#     embedding = Embedding(95, output_dim=ENCODER_OUTPUT_DIM, trainable=False, name="embedding_layer")
#     embedded_encoder_input = embedding(encoder_inputs)
#     embedded_decoder_input = embedding(decoder_inputs)
#     emedded_target = embedding(target)
#
#     # Define an input sequence and process it.
#     encoder = LSTM(LSTM_DIM, return_state=True, dropout=0.2, recurrent_dropout=0.2, name="encoder")
#     encoder_outputs, state_h, state_c = encoder(embedded_encoder_input)
#     encoder_states = [state_h, state_c]
#
#     decoder_lstm = LSTM(LSTM_DIM, return_sequences=True, return_state=True, dropout=0.2, recurrent_dropout=0.2,
#                         name="decoder")
#     decoder_outputs, _, _ = decoder_lstm(embedded_decoder_input, initial_state=encoder_states)
#
#     decoder_dense = Dense(ENCODER_OUTPUT_DIM, activation='sigmoid', name="dense")
#     decoder_output = decoder_dense(decoder_outputs)
#     output = cc.CustomRegularization()([emedded_target, decoder_output])
#     model = Model([encoder_inputs, decoder_inputs, target], output)
#     return model


def create_model_onehot_layer():
    encoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input", dtype="int32")
    decoder_inputs = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="decoder_Input", dtype="int32")
    target = Input(shape=(MAX_TEXT_SEQUENCE_LEN,), name="target_Input", dtype="int32")

    embedding = cc.OneHot(input_dim=2000, input_length=MAX_TEXT_SEQUENCE_LEN)
    embedded_encoder_input = embedding(encoder_inputs)
    embedded_decoder_input = embedding(decoder_inputs)
    emedded_target = embedding(target)

    encoder = GRU(GRU_DIM, return_state=True)
    encoder_outputs, state_h = encoder(embedded_encoder_input)

    decoder_gru = GRU(GRU_DIM, return_sequences=True)
    decoder_outputs = decoder_gru(embedded_decoder_input, initial_state=state_h)
    decoder_dense = Dense(ENCODER_OUTPUT_DIM, activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)

    output = cc.CustomRegularization(loss_function="mean_squared_error")([emedded_target, decoder_outputs])
    model = Model([encoder_inputs, decoder_inputs, target], output)
    return model


def generate_batches(data):
    data_batches = cut_data_to_batches(data)

    while True:
        for data_slide in data_batches:
            input_paded, input_decoder_paded, target_paded = convert_to_vec(data_slide, full_unicode=False)
            yield [input_paded, input_decoder_paded, target_paded], target_paded


def load_data(data_path):
    df = pd.read_csv(tf.gfile.Open(data_path))
    train_data = df['value'].values
    train_data = map(str, train_data)
    train_data = list(map(str.strip, train_data))
    train_data = list(map(lambda x: x[:MAX_TEXT_SEQUENCE_LEN - 1], train_data))

    input_texts = list(map(lambda x: x[::-1], train_data))

    input_decoder_texts = list(map(lambda x: '\t' + x, train_data))

    target_texts = train_data

    return input_texts, input_decoder_texts, target_texts


def convert_to_vec(batch_data, full_unicode=True):

    if full_unicode:
        tokenizer_custom = ord
    else:
        batch_data = map(lambda x: (unidecode(x[0]), unidecode(x[1]), unidecode(x[2])), batch_data)
        tokenizer_custom = tokenizer_0_96

    input_texts, imput_decoder_texts, target_texts = zip(*batch_data)
    input_tokenized = list(map(lambda x: [tokenizer_custom(y) for y in x], input_texts))
    imput_decoder_tokenized = list(map(lambda x: [tokenizer_custom(y) for y in x], imput_decoder_texts))
    target_tokenized = list(map(lambda x: [tokenizer_custom(y) for y in x], target_texts))

    input_paded = pad_sequences(input_tokenized, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='pre')
    input_decoder_paded = pad_sequences(imput_decoder_tokenized, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='post')
    target_paded = pad_sequences(target_tokenized, maxlen=MAX_TEXT_SEQUENCE_LEN, truncating='pre', padding='post')
    return input_paded, input_decoder_paded, target_paded


def cut_data_to_batches(data):
    np.random.shuffle(data)
    result = []
    for i in range(len(data) // BATCH_SIZE):
        slide = list(data[i*BATCH_SIZE:(i+1)*BATCH_SIZE])
        result.append(slide)

    return result


