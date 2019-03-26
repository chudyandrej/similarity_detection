
import os

from keras.models import Model
from keras.layers import Input,  Embedding, TimeDistributed, Bidirectional, Lambda, GRU

import trainer.custom_components as cc

# Model constants
MAX_TEXT_SEQUENCE_LEN = 64
BATCH_SIZE = 64  # Batch size for training.
EPOCHS = 100  # Number of epochs to train for.
GRU_DIM = 128  # Latent dimensionality of the encoding space.
SAMPLES = 200000
DROP_RATE = 0.45


def build_model():
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
    joint_model.compile(optimizer="adam", loss=cc.contrastive_loss)
    joint_model.name = "join_model"

    return joint_model
