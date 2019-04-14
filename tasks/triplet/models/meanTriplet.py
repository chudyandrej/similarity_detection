import os
from keras.layers import *
from keras.models import *
from tasks.triplet import Triplet

import custom_components as cc
from preprocessor.encoder import Encoder


class MeanTriplet(Triplet):
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

        anchor_input = Input(shape=(11, self.max_seq_len), name='anchor_input')
        positive_input = Input(shape=(11, self.max_seq_len), name='positive_input')
        negative_input = Input(shape=(11, self.max_seq_len), name='negative_input')

        value_embed = Embedding(input_dim=self.encoder.get_vocab_size()+1, output_dim=self.enc_out_dim)
        anchor_embedded = TimeDistributed(value_embed)(anchor_input)
        positive_embedded = TimeDistributed(value_embed)(positive_input)
        negative_embedded = TimeDistributed(value_embed)(negative_input)

        value_lstm = self.get_rnn(rnn_type=self.rnn_type, rnn_dim=self.rnn_dim, dropout=self.dropout,
                                  return_sequences=self.attention)
        anchor_values = TimeDistributed(value_lstm)(anchor_embedded)
        positive_values = TimeDistributed(value_lstm)(positive_embedded)
        negataive_values = TimeDistributed(value_lstm)(negative_embedded)
        
        quantile_mean = Lambda(lambda x: K.mean(x, axis=1))
        anchor_quantiles = quantile_mean(anchor_values)
        positive_quantiles = quantile_mean(positive_values)
        negative_quantiles = quantile_mean(negataive_values)

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
        model = load_model(f"{self.output_space}/model.h5", custom_objects={
            "triplet_loss": cc.triplet_loss,
            "AttentionWithContext": cc.AttentionWithContext
        })
        print(model.layers)
        exit()
        model.summary()
        return model
