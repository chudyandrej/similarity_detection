import os
from keras.layers import *
from keras.models import *
from tasks.siamese.siamese import Siamese

import custom_components as cc
from preprocessor.encoder import Encoder


class HierSiamJointly(Siamese):

    def __init__(self, rnn_type: str, attention: bool, encoder: Encoder, enc_out_dim, max_seq_len, rnn_dim, dropout,
                 version, name):
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
        def value_level():
            input_layer = Input(shape=(self.max_seq_len,), name='input')
            embedded_value = Embedding(input_dim=self.encoder.get_vocab_size()+1, output_dim=self.enc_out_dim,
                                       name='value_embedder', trainable=True)(input_layer)
            embedding = Dropout(self.dropout)(embedded_value)

            rnn = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                               return_sequences=self.attention)
            rnn_output = rnn(embedding)

            if self.attention:
                rnn_output = cc.AttentionWithContext(return_coefficients=False)(rnn_output)
            rnn_output = Dropout(self.dropout)(rnn_output)

            model = Model(input_layer, rnn_output)
            return model

        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        # Value embedding model trained as seq2seq.
        value_encoder = value_level()
        left_value_encoded = TimeDistributed(value_encoder)(left_input)
        right_value_encoded = TimeDistributed(value_encoder)(right_input)

        quantile_encoder = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                                        return_sequences=self.attention)

        left_encoded = quantile_encoder(left_value_encoded)
        right_encoded = quantile_encoder(right_value_encoded)

        if self.attention:
            context_layer = cc.AttentionWithContext(return_coefficients=False)
            left_encoded = context_layer(left_encoded)
            right_encoded = context_layer(right_encoded)

        dropout_1 = Dropout(self.dropout)
        col_att_vec_dr_left = dropout_1(left_encoded)
        col_att_vec_dr_right = dropout_1(right_encoded)

        comparer = Lambda(function=cc.euclidean_distance, name='comparer')
        # comparer = Dot(axes=1, normalize=True, name='comparer')
        output = comparer([col_att_vec_dr_left, col_att_vec_dr_right])

        # Compile and train Joint Model
        joint_model = Model(inputs=[left_input, right_input], outputs=output)
        joint_model.compile(optimizer="adam", loss=cc.contrastive_loss)

        joint_model.summary()
        return joint_model

    def load_encoder(self):
        model = self.load_model()
        if self.attention:
            model: Model = Model(model.inputs[0], model.layers[5].get_output_at(0))
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
