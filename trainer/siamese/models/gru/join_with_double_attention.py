
from keras.models import Model
from keras.layers import Input,  Embedding, TimeDistributed, Bidirectional, Lambda, GRU, CuDNNGRU, Dropout

import trainer.custom_components as cc

# Model constants
MAX_TEXT_SEQUENCE_LEN = 64
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
GRU_DIM = 128  # Latent dimensionality of the encoding space.
SAMPLES = 200000
DROP_RATE = 0.45


def build_model():
    def value_encoder():
        input_layer = Input(shape=(64,), name='input')
        embedded_value = Embedding(input_dim=65536, output_dim=200, name='value_embedder', trainable=False)(input_layer)
        embedded_value = Dropout(0.45)(embedded_value)
        encoded_value = Bidirectional(
            CuDNNGRU(units=50, return_sequences=True), merge_mode='concat', weights=None)(embedded_value)
        encoded_value = cc.AttentionWithContext(return_coefficients=False)(encoded_value)
        encoded_value = Dropout(0.45)(encoded_value)
        model = Model(input_layer, encoded_value)
        print(model.output)
        return model

    # Ensemble Joint Model
    left_input = Input(shape=(11, 64), name='left_input')
    right_input = Input(shape=(11, 64), name='right_input')

    # Value embedding model trained as seq2seq.
    value_encoder = value_encoder()
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
    joint_model.name = "join_with_double_attention"
    return joint_model



