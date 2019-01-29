
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Bidirectional, Lambda, Dense, Dot, Convolution1D
from keras.layers import GlobalMaxPooling1D, Concatenate, AlphaDropout, GRU
from keras import backend as K
from collections import defaultdict
from collections import Counter
from keras.utils import plot_model

from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sdep import pairs_generator
import os

from evaluater.load_models import load_seq2seq_embedder


MAX_TEXT_SEQUENCE_LEN = 64
# Model constants
BATCH_SIZE = 1024  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
GRU_DIM = 64  # Latent dimensionality of the encoding space.
TRAINING_SAMPLES = 500000


def create_model_base(quantile_shape):
    # Ensemble Joint Model
    left_input = Input(shape=quantile_shape, name='left_input')
    right_input = Input(shape=quantile_shape, name='right_input')

    value_embedder = Embedding(input_dim=65536, output_dim=128, name='value_embedder', trainable=False)
    left_embedded = TimeDistributed(value_embedder,  trainable=False)(left_input)
    right_embedded = TimeDistributed(value_embedder,  trainable=False)(right_input)

    value_encoder = GRU(GRU_DIM, dropout=0.2, recurrent_dropout=0.20, name='value_encoder')
    left_value_encoded = TimeDistributed(value_encoder)(left_embedded)
    right_value_encoded = TimeDistributed(value_encoder)(right_embedded)

    quantile_encoder = Bidirectional(GRU(GRU_DIM, dropout=0.20, recurrent_dropout=0.20), name='quantile_encoder')
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)

    # If cosine similary wanted, use the other comparer instead
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    comparer = Lambda(function=euclidean_distance, name='comparer')
    output = comparer([left_encoded, right_encoded])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    return joint_model


def l1_similarity(x):
    return K.exp(-1 * K.sum(K.abs(x[0] - x[1]), axis=-1, keepdims=True))


def l2_similarity(x):
    return K.exp(-1 * K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True)))


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return shape1[0], 1


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
        left, right, label, weights = next(pairs_generator(train_profiles, BATCH_SIZE))

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

