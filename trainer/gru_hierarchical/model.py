import os

from keras.models import Model, load_model
from keras.layers import Input, LSTM, Embedding, TimeDistributed, Bidirectional, Lambda, Dense, Dot, Convolution1D
from keras.layers import GlobalMaxPooling1D, Concatenate, AlphaDropout, GRU, Dropout, CuDNNGRU
import trainer.custom_components as cc


MAX_TEXT_SEQUENCE_LEN = 64
# Model constants
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
GRU_DIM = 128  # Latent dimensionality of the encoding space.
SAMPLES = 200000
DROP_RATE = 0.45



def model1():
    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

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
    comparer = Lambda(function=cc.euclidean_distance, name='comparer')
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
    right_value_encoded = TimeDistributed(value_encoder)(right_input)

    quantile_encoder = Bidirectional(GRU(GRU_DIM, dropout=0.20, recurrent_dropout=0.20), name='quantile_encoder')
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)

    comparer = Lambda(function=cc.euclidean_distance, name='comparer')
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    output = comparer([left_encoded, right_encoded])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    joint_model.summary()
    return joint_model


def model3():
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq-hot1549903432/model.h5"
    model = load_model(model_path, custom_objects={
        "CustomRegularization": cc.CustomRegularization,
        "zero_loss": cc.zero_loss
    })
    for layer in model.layers:
        layer.trainable = False

    value_encoder = Model(model.inputs[0], model.layers[4].output[1])

    print(value_encoder.output)
    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

    # Value embedding model trained as seq2seq.
    left_value_encoded = TimeDistributed(value_encoder)(left_input)
    right_value_encoded = TimeDistributed(value_encoder)(right_input)

    quantile_encoder = Bidirectional(CuDNNGRU(units=GRU_DIM, return_sequences=True), merge_mode='concat', weights=None)
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)

    context_layer = cc.AttentionWithContext(return_coefficients=False)
    col_att_vec_left = context_layer(left_encoded)
    col_att_vec_right = context_layer(right_encoded)

    dropout_1 = Dropout(0.45)
    col_att_vec_dr_left = dropout_1(col_att_vec_left)
    col_att_vec_dr_right = dropout_1(col_att_vec_right)

    comparer = Lambda(function=cc.euclidean_distance, name='comparer')
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    output = comparer([col_att_vec_dr_left, col_att_vec_dr_right])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    joint_model.summary()
    return joint_model


def model4():

    def value_encodder():

        input_layer = Input(shape=(64,), name='input')
        embedded_value = Embedding(input_dim=67000, output_dim=128, name='value_embedder', trainable=False)(input_layer)
        embedded_value = Dropout(0.45)(embedded_value)
        encoded_value = Bidirectional(
            CuDNNGRU(units=128, return_sequences=False), merge_mode='concat', weights=None)(embedded_value)
        encoded_value = Dropout(0.45)(encoded_value)
        model = Model(input_layer, encoded_value)
        print(model.output)
        return model

    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

    # Value embedding model trained as seq2seq.
    value_encoder = value_encodder()
    left_value_encoded = TimeDistributed(value_encoder)(left_input)
    right_value_encoded = TimeDistributed(value_encoder)(right_input)

    quantile_encoder = Bidirectional(CuDNNGRU(units=GRU_DIM, return_sequences=True), merge_mode='concat',
                                        weights=None, name="bidirectional_quantile")
    left_encoded = quantile_encoder(left_value_encoded)
    right_encoded = quantile_encoder(right_value_encoded)
    context_layer = cc.AttentionWithContext(return_coefficients=False)
    col_att_vec_left = context_layer(left_encoded)
    col_att_vec_right = context_layer(right_encoded)

    dropout_1 = Dropout(0.45)
    col_att_vec_dr_left = dropout_1(col_att_vec_left)
    col_att_vec_dr_right = dropout_1(col_att_vec_right)

    comparer = Lambda(function=cc.euclidean_distance, name='comparer')
    # comparer = Dot(axes=1, normalize=True, name='comparer')
    output = comparer([col_att_vec_dr_left, col_att_vec_dr_right])

    # Compile and train Joint Model
    joint_model = Model(inputs=[left_input, right_input], outputs=output)
    joint_model.summary()
    return joint_model




