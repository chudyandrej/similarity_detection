
import os
import json

import tensorflow as tf
from keras.layers import *
from keras.models import *
from tasks.triplet import Triplet

import custom_components as cc
from ...computing_model import ComputingModel


class HierTripletWithSeq2Encoder(Triplet):
    def __init__(self, rnn_type: str, attention: bool, value_compute_obj: ComputingModel,
                 enc_out_dim, max_seq_len, rnn_dim, dropout, version, name):
        self.rnn_dim = rnn_dim
        self.dropout = dropout
        self.version = version
        self.rnn_type = rnn_type
        self.attention = attention
        self.enc_out_dim = enc_out_dim
        self.value_compute_obj = value_compute_obj

        self.output_space = f"{super().OUTPUT_ROOT}/{name}/{self.version}"
        super().__init__(encoder=self.value_compute_obj.get_encoder(), max_seq_len=max_seq_len, output_path=self.output_space)

    def build_model(self):

        anchor_input = Input(shape=(11, self.max_seq_len), name='anchor_input')
        positive_input = Input(shape=(11, self.max_seq_len), name='positive_input')
        negative_input = Input(shape=(11, self.max_seq_len), name='negative_input')

        # Value embedding model trained as seq2seq.
        value_model = self.value_compute_obj.load_encoder()
        for layer in value_model.layers:
            layer.trainable = False

        anchor_values = TimeDistributed(value_model)(anchor_input)
        positive_values = TimeDistributed(value_model)(positive_input)
        negataive_values = TimeDistributed(value_model)(negative_input)

        quantile_lstm = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                                     return_sequences=self.attention)
        anchor_quantiles = quantile_lstm(anchor_values)
        positive_quantiles = quantile_lstm(positive_values)
        negative_quantiles = quantile_lstm(negataive_values)

        if self.attention:
            context_layer = cc.AttentionWithContext(return_coefficients=False)
            anchor_quantiles = context_layer(anchor_quantiles)
            positive_quantiles = context_layer(positive_quantiles)
            negative_quantiles = context_layer(negative_quantiles)

        quantile_norm = Lambda(lambda x: K.l2_normalize(x, axis=-1))
        anchor_normed = quantile_norm(anchor_quantiles)
        positive_normed = quantile_norm(positive_quantiles)
        negative_normed = quantile_norm(negative_quantiles)

        triplet_concat = Concatenate()
        triplet_concated = triplet_concat([anchor_normed, positive_normed, negative_normed])

        triplet_reshape = Reshape(target_shape=(3, self.rnn_dim*2))
        triplet_reshaped = triplet_reshape(triplet_concated)

        inputs = [anchor_input, positive_input, negative_input]
        model = Model(inputs=inputs, outputs=triplet_reshaped)
        model.compile(loss=cc.triplet_loss, optimizer='adam')

        model.summary()
        return model

    def load_encoder(self):
        model = self.load_model()
        if self.attention:
            model: Model = Model(model.inputs[0], model.layers[7].get_output_at(0))
        else:
            model: Model = Model(model.inputs[0], model.layers[6].get_output_at(0))

        model.summary()
        return model

    def load_model(self):
        return load_model(f"{self.output_space}/model.h5", custom_objects={
            "euclidean_distance": cc.euclidean_distance,
            "triplet_loss": cc.triplet_loss,
            "AttentionWithContext": cc.AttentionWithContext

        })


























