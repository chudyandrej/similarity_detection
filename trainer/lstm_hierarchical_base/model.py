
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Bidirectional, Lambda
from keras import backend as K
from collections import defaultdict
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from  sdep import pairs_generator
import os

from evaluater.load_models import load_seq2seq_embedder


MAX_TEXT_SEQUENCE_LEN = 64
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 256  # Latent dimensionality of the encoding space.
TRAINING_SAMPLES = 500000


def create_model(quantile_shape):
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    # Ensemble Joint Model
    left_input = Input(shape=quantile_shape, name='left_input')
    right_input = Input(shape=quantile_shape, name='right_input')

    value_encoder = load_seq2seq_embedder(model_path, emb_path)
    left_value_encoded = TimeDistributed(value_encoder, trainable=False)(left_input)
    right_value_encoded = TimeDistributed(value_encoder, trainable=False)(right_input)

    quantile_encoder = Bidirectional(LSTM(128, dropout=0.20, recurrent_dropout=0.20), name='quantile_encoder')
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)

    # If cosine similary wanted, use the other comparer instead
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    comparer = Lambda(function=l2_similarity, name='comparer')
    output = comparer([left_encoded, right_encoded])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    return joint_model


def l1_similarity(x):
    return K.exp(-1 * K.sum(K.abs(x[0] - x[1]), axis=-1, keepdims=True))


def l2_similarity(x):
    return K.exp(-1 * K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True)))


def generate_random_fit(ev):
    """Generator for fitting

    Args:
        input_texts (Array): Encoder data
        types (Array): Type

    Yields:
        TYPE: Batch of data
    """
    train_profiles, _ = ev.get_train_dataset()
    while True:
        left, right, label = next(pairs_generator(train_profiles, BATCH_SIZE))
        left = np.array(list(map(lambda x: preprocess_quantiles(x.quantiles, MAX_TEXT_SEQUENCE_LEN), left)))
        right = np.array(list(map(lambda x: preprocess_quantiles(x.quantiles, MAX_TEXT_SEQUENCE_LEN), right)))
        yield [left, right], label


def preprocess_quantiles(quantiles, pad_maxlen):
    quantiles = map(str, quantiles)
    quantiles = map(str.strip, quantiles)
    quantiles = map(lambda x: x[:pad_maxlen], quantiles)
    quantiles = (x[::-1] for x in quantiles)
    quantiles = list(map(lambda x: [ord(y) for y in x], quantiles))
    quantiles = pad_sequences(quantiles, maxlen=pad_maxlen, truncating='pre', padding='pre')
    return quantiles


