
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Bidirectional, Lambda, Dense, Dot, Convolution1D
from keras.layers import GlobalMaxPooling1D, Concatenate, AlphaDropout, SeparableConv1D, MaxPooling1D
from keras.layers import GlobalAveragePooling1D

from keras import backend as K


import numpy as np
import sdep
import os

from keras.models import Model, load_model
from evaluater.preprocessing import preprocess_values_standard
import trainer.custom_components as cc


MAX_TEXT_SEQUENCE_LEN = 64
# Model constants
BATCH_SIZE = 512  # Batch size for training.
EPOCHS = 1000  # Number of epochs to train for.
LSTM_DIM = 128  # Latent dimensionality of the encoding space.
TRAINING_SAMPLES = 200000


def model1():
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq-hot1549903432/model.h5"
    model = load_model(model_path, custom_objects={
        "CustomRegularization": cc.CustomRegularization,
        "zero_loss": cc.zero_loss
    })
    value_encoder = Model(model.inputs[0], model.layers[4].output[1])
    for layer in value_encoder.layers:
        layer.trainable = False

    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

    # Value embedding model trained as seq2seq.
    left_value_encoded = TimeDistributed(value_encoder)(left_input)
    right_value_encoded = TimeDistributed(value_encoder)(right_input)

    quantile_encoder = Bidirectional(LSTM(LSTM_DIM, dropout=0.20, recurrent_dropout=0.20), name='quantile_encoder')
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)

    # If cosine similary wanted, use the other comparer instead
    comparer = Lambda(cc.l2_similarity)
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    # comparer = Lambda(function=l2_similarity, name='comparer')
    # comparer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    output = comparer([left_encoded, right_encoded])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    joint_model.summary(line_length=120)
    return joint_model


def model2():

    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq-hot1549903432/model.h5"
    model = load_model(model_path, custom_objects={
        "CustomRegularization": cc.CustomRegularization,
        "zero_loss": cc.zero_loss
    })
    for layer in model.layers:
        layer.trainable = False

    value_encoder = Model(model.inputs[0], model.layers[4].output[1])

    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

    # Value embedding model trained as seq2seq.
    left_value_encoded = TimeDistributed(value_encoder)(left_input)
    left_encoded = Lambda(lambda xin: K.mean(xin, axis=1))(left_value_encoded)

    right_value_encoded = TimeDistributed(value_encoder)(right_input)
    right_encoded = Lambda(lambda xin: K.mean(xin, axis=1))(right_value_encoded)

    dense1 = Dense(256, activation="relu")
    left_encoded = dense1(left_encoded)
    right_encoded = dense1(right_encoded)

    dense2 = Dense(256, activation="relu")
    left_encoded = dense2(left_encoded)
    right_encoded = dense2(right_encoded)

    # If cosine similary wanted, use the other comparer instead
    comparer = Lambda(cc.euclidean_distance)
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    # comparer = Lambda(function=l2_similarity, name='comparer')
    # comparer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    output = comparer([left_encoded, right_encoded])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    joint_model.summary()
    return joint_model


def model3():
    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

    value_embedder = Embedding(input_dim=65536, output_dim=128, name='value_embedder')
    left_embedded = TimeDistributed(value_embedder)(left_input)
    right_embedded = TimeDistributed(value_embedder)(right_input)

    value_encoder = LSTM(128, dropout=0.2, recurrent_dropout=0.20, name='value_encoder')
    left_value_encoded = TimeDistributed(value_encoder)(left_embedded)
    right_value_encoded = TimeDistributed(value_encoder)(right_embedded)

    quantile_encoder = Bidirectional(LSTM(128, dropout=0.20, recurrent_dropout=0.20), name='quantile_encoder')
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)

    # If cosine similary wanted, use the other comparer instead
    comparer = Lambda(cc.euclidean_distance)
    # comparer = Lambda(function=l2_similarity, name='comparer')
    output = comparer([left_encoded, right_encoded])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    joint_model.summary(line_length=120)

    return joint_model


def model4(quantile_shape):
    CNN_LAYERS = [[256, 10], [256, 7], [256, 5], [256, 3]]

    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    # Ensemble Joint Model
    left_input = Input(shape=quantile_shape, name='left_input')
    right_input = Input(shape=quantile_shape, name='right_input')

    # Value embedding model trained as seq2seq.
    value_encoder = load_seq2seq_embedder(model_path, emb_path)
    left_value_encoded = TimeDistributed(value_encoder, trainable=False)(left_input)
    right_value_encoded = TimeDistributed(value_encoder, trainable=False)(right_input)

    # Convolution layers
    convolution_output1 = []
    convolution_output2 = []

    for num_filters, filter_width in CNN_LAYERS:
        conv = Convolution1D(filters=num_filters,
                             kernel_size=filter_width,
                             activation='tanh',
                             name='Conv1D_{}_{}'.format(num_filters, filter_width))
        c1 = conv(left_value_encoded)
        c2 = conv(right_value_encoded)
        pool = GlobalMaxPooling1D(name='MaxPoolingOverTime_{}_{}'.format(num_filters, filter_width))
        p1 = pool(c1)
        p2 = pool(c2)
        convolution_output1.append(p1)
        convolution_output2.append(p2)
    x1 = Concatenate()(convolution_output1)
    x2 = Concatenate()(convolution_output2)
    dense1 = Dense(LSTM_DIM, activation='selu', kernel_initializer='lecun_normal')
    x1 = dense1(x1)
    x2 = dense1(x2)
    x1 = AlphaDropout(0.2)(x1)
    x2 = AlphaDropout(0.2)(x2)
    dense2 = Dense(256, activation='selu', kernel_initializer='lecun_normal')
    x1 = dense2(x1)
    x2 = dense2(x2)
    # If cosine similary wanted, use the other comparer instead
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    comparer = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)
    # comparer = Lambda(function=l2_similarity, name='comparer')
    output = comparer([x1, x2])
    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    # plot_model(joint_model, to_file='model.png')
    joint_model.summary(line_length=120)
    return joint_model


def generate_random_fit(train_profiles):
    """Generator for fitting

    Args:
        input_texts (Array): Encoder data
        types (Array): Type

    Yields:
        TYPE: Batch of data
    """
    while True:
        left, right, label, weights = next(sdep.pairs_generator(train_profiles, BATCH_SIZE))

        left = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, MAX_TEXT_SEQUENCE_LEN), left)))
        right = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, MAX_TEXT_SEQUENCE_LEN), right)))

        yield [left, right], label


