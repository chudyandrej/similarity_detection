import os
from keras.layers import *
from keras.models import *
from ..siamese import Siamese

import custom_components as cc
from preprocessor.encoder import Encoder


class GruHierSiamWithFixValueEnc(Siamese):
    def __init__(self, value_model: Model, encoder: Encoder, max_seq_len: int, lstm_dim: int, enc_output_dim: int, dropout, version):
        self.value_model = value_model
        self.lstm_dim = lstm_dim
        self.enc_output_dim = enc_output_dim
        self.dropout = dropout
        self.encoder = encoder
        self.version = version

        self.output_space = f"{super().OUTPUT_ROOT}/{type(self).__name__}/{self.version}"
        super().__init__(encoder=encoder, max_seq_len=max_seq_len, output_path=self.output_space)

    def build_model(self):
        # Ensemble Joint Model
        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        for layer in self.value_model.layers:
            layer.trainable = False

        # Value embedding model trained as seq2seq.
        left_value_encoded = TimeDistributed(self.value_model)(left_input)
        right_value_encoded = TimeDistributed(self.value_model)(right_input)
        quantile_encoder = Bidirectional(LSTM(units=self.lstm_dim, return_sequences=True, dropout=self.dropout,
                                              recurrent_dropout=self.dropout), name="bidirectional_quantile")

        left_encoded = quantile_encoder(left_value_encoded)
        right_encoded = quantile_encoder(right_value_encoded)

        comparer = Lambda(function=cc.euclidean_distance, name='comparer')
        # comparer = Dot(axes=1, normalize=True, name='comparer')
        output = comparer([left_encoded, right_encoded])

        # Compile and train Joint Model
        joint_model = Model(inputs=[left_input, right_input], outputs=output)
        joint_model.compile(optimizer="adam", loss=cc.contrastive_loss)
        joint_model.name = self.version
        joint_model.summary()
        return joint_model

    def load_encoder(self):
        model = load_model(f"{self.output_space}/model.h5", custom_objects={
            "euclidean_distance": cc.euclidean_distance,
            "contrastive_loss": cc.contrastive_loss

        })
        print(model.layers)
        exit()
        model: Model = Model(model.inputs[0], model.layers[6].get_output_at(0))
        model.summary()
        return model
