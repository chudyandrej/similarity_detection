import os
import json

import tensorflow as tf
from keras.layers import *
from keras.models import *
from tasks.triplet import Triplet

import custom_components as cc
from preprocessor.encoder import Encoder


class HierTripletJointlyWithGpt2Encoder(Triplet):
    def __init__(self, rnn_type: str, attention: bool, encoder: Encoder, enc_out_dim, max_seq_len, rnn_dim, dropout,
                 version):
        self.rnn_dim = rnn_dim
        self.attention = attention
        self.rnn_type = rnn_type
        self.enc_out_dim = enc_out_dim
        self.dropout = dropout
        self.version = version

        self.output_space = f"{super().OUTPUT_ROOT}/{rnn_type}{type(self).__name__}/{self.version}"
        if attention:
            self.output_space = f"{super().OUTPUT_ROOT}/{rnn_type}{type(self).__name__}WithAttention/{self.version}"

        super().__init__(encoder=encoder, max_seq_len=max_seq_len, output_path=self.output_space)

    def build_model(self):
        with open(super().GPT2_CONFIG_PATH, 'r') as reader:
            config = json.load(reader)

        def value_level():
            input_layer = Input(shape=(self.max_seq_len,), name='input')
            embedded_value = Embedding(input_dim=config['n_vocab'], output_dim=config['n_embd'], mask_zero=False,
                                       name='Embed-Token', trainable=False)(input_layer)
            rnn = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                               return_sequences=self.attention)
            rnn_output = rnn(embedded_value)

            if self.attention:
                rnn_output = cc.AttentionWithContext(return_coefficients=False)(rnn_output)
            model = Model(input_layer, rnn_output)
            return model

        anchor_input = Input(shape=(11, self.max_seq_len), name='anchor_input')
        positive_input = Input(shape=(11, self.max_seq_len), name='positive_input')
        negative_input = Input(shape=(11, self.max_seq_len), name='negative_input')

        value_encoder = value_level()
        value_encoder.get_layer(name='Embed-Token').set_weights([
            tf.train.load_variable(super().GPT2_CHECKPOINT_PATH, 'model/wte:0'),
        ])
        anchor_values = TimeDistributed(value_encoder)(anchor_input)
        positive_values = TimeDistributed(value_encoder)(positive_input)
        negataive_values = TimeDistributed(value_encoder)(negative_input)

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

        model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=triplet_reshaped)
        model.compile(loss=cc.triplet_loss, optimizer='adam')
        model.summary()
        return model

    def load_encoder(self):
        model = self.load_model()
        print(model.layers)

        if self.attention:
            model: Model = Model(model.inputs[0], model.layers[7].get_output_at(0))
        else:
            model: Model = Model(model.inputs[0], model.layers[6].get_output_at(0))
        model.summary()
        return model

    def load_model(self):
        return load_model(f"{self.output_space}/model.h5", custom_objects={
            "triplet_loss": cc.triplet_loss,
            "AttentionWithContext": cc.AttentionWithContext
        })


