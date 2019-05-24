
import os
import json

import tensorflow as tf
from keras.layers import *
from keras.models import *
from tasks.siamese.siamese import Siamese

import custom_components as cc
from preprocessor.encoder import Encoder


class MeanHierSiamJointlyWithGpt2Encoder(Siamese):
    def __init__(self, rnn_type: str, attention: bool, encoder: Encoder, enc_out_dim, max_seq_len, rnn_dim, dropout,
                 version, name):
        self.rnn_dim = rnn_dim
        self.dropout = dropout
        self.version = version
        self.rnn_type = rnn_type
        self.attention = attention
        self.enc_out_dim = enc_out_dim

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
            embedded_value = Dropout(self.dropout)(embedded_value)
            rnn = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                               return_sequences=self.attention)
            x = rnn(embedded_value)
            if self.attention:
                x = cc.AttentionWithContext(return_coefficients=False)(x)
            x = Dropout(self.dropout)(x)
            model = Model(input_layer, x)
            return model

        # Ensemble Joint Model
        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        # Value embedding model trained as seq2seq.
        value_model = value_level()
        value_model.get_layer(name='Embed-Token').set_weights([
            tf.train.load_variable(super().GPT2_CHECKPOINT_PATH, 'model/wte:0'),
        ])

        left_value_encoded = TimeDistributed(value_model)(left_input)
        right_value_encoded = TimeDistributed(value_model)(right_input)

        left_mean = Lambda(lambda xin: K.mean(xin, axis=1))(left_value_encoded)
        right_mean = Lambda(lambda xin: K.mean(xin, axis=1))(right_value_encoded)

        comparer = Lambda(function=cc.euclidean_distance, name='comparer')
        output = comparer([left_mean, right_mean])

        # Compile and train Joint Model
        joint_model = Model(inputs=[left_input, right_input], outputs=output)
        joint_model.compile(optimizer="adam", loss=cc.contrastive_loss)
        joint_model.name = self.version
        joint_model.summary()
        return joint_model

    def load_encoder(self):
        model = self.load_model()

        model: Model = Model(model.inputs[0], model.layers[4].get_output_at(0))
        model.summary()
        return model

    def load_model(self):
        return load_model(f"{self.output_space}/model.h5", custom_objects={
            "euclidean_distance": cc.euclidean_distance,
            "contrastive_loss": cc.contrastive_loss,
            "AttentionWithContext": cc.AttentionWithContext
        })


































