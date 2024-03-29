
import os
import json

import tensorflow as tf
from keras.layers import *
from keras.models import *
from tasks.siamese.siamese import Siamese

import custom_components as cc
from preprocessor.encoder import Encoder

from ...seq2seq import CuDNNGRUSeq2seqWithGpt2Encoder
from ...computing_model import ComputingModel
from preprocessor.encoder import BytePairEncoding


class HierSiamJointlyWithSeq2Encoder(Siamese):
    def __init__(self, rnn_type: str, attention: bool, value_compute_obj: ComputingModel, enc_out_dim, max_seq_len,
                 rnn_dim, dropout, version, name):
        self.rnn_dim = rnn_dim
        self.dropout = dropout
        self.version = version
        self.rnn_type = rnn_type
        self.attention = attention
        self.enc_out_dim = enc_out_dim
        self.value_compute_obj = value_compute_obj

        self.output_space = f"{super().OUTPUT_ROOT}/{name}/{self.version}"

        super().__init__(encoder=self.value_compute_obj.get_encoder(), max_seq_len=max_seq_len,
                         output_path=self.output_space)

    def build_model(self):

        # Ensemble Joint Model
        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        # Value embedding model trained as seq2seq.
        value_model = self.value_compute_obj.load_encoder()
        for layer in value_model.layers:
            layer.trainable = False

        left_value_encoded = TimeDistributed(value_model)(left_input)
        right_value_encoded = TimeDistributed(value_model)(right_input)

        quantile_encoder = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                                        return_sequences=self.attention)

        left_encoded = quantile_encoder(left_value_encoded)
        right_encoded = quantile_encoder(right_value_encoded)

        if self.attention:
            context_layer = cc.AttentionWithContext(return_coefficients=False)
            left_encoded = context_layer(left_encoded)
            right_encoded = context_layer(right_encoded)

        # dropout_1 = Dropout(self.dropout)
        # left_encoded = dropout_1(left_encoded)
        # right_encoded = dropout_1(right_encoded)

        comparer = Lambda(function=cc.euclidean_distance, name='comparer')
        # comparer = Dot(axes=1, normalize=True, name='comparer')
        output = comparer([left_encoded, right_encoded])

        # Compile and train Joint Model
        joint_model = Model(inputs=[left_input, right_input], outputs=output)
        joint_model.compile(optimizer="adam", loss=cc.contrastive_loss)

        joint_model.summary()
        return joint_model

    def load_encoder(self):
        model = self.load_model()
        if self.attention:
            model: Model = Model(model.inputs[0], model.layers[7].get_output_at(0))
        else:
            model: Model = Model(model.inputs[0], model.layers[4].get_output_at(0))

        model.summary()
        return model

    def load_model(self):
        return load_model(f"{self.output_space}/model.h5", custom_objects={
            "euclidean_distance": cc.euclidean_distance,
            "contrastive_loss": cc.contrastive_loss,
            "AttentionWithContext": cc.AttentionWithContext

        })



























