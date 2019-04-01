import os
from keras.layers import *
from keras.models import *
from ..siamese import Siamese

from preprocessor.encoder import Encoder
import custom_components as cc


class GruHierSiamWithFixValueEncAndAttention(Siamese):
    def __init__(self, value_embedder: Model, encoder: Encoder, max_seq_len, lstm_dim, dropout, version):

        self.value_embedder = value_embedder
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.version = version

        self.output_space = f"{super().OUTPUT_ROOT}/{type(self).__name__}/{self.version}"
        super().__init__(encoder=encoder, max_seq_len=max_seq_len, output_path=self.dropout)

    def build_model(self):
        # Ensemble Joint Model
        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        for layer in self.value_embedder.layers:
            layer.trainable = False

        # Value embedding model trained as seq2seq.
        left_value_encoded = TimeDistributed(self.value_embedder)(left_input)
        right_value_encoded = TimeDistributed(self.value_embedder)(right_input)

        quantile_encoder = Bidirectional(LSTM(units=self.lstm_dim, return_sequences=True), merge_mode='concat',
                                         weights=None, name="bidirectional_quantile")
        left_encoded = quantile_encoder(left_value_encoded)
        right_encoded = quantile_encoder(right_value_encoded)

        context_layer = cc.AttentionWithContext(return_coefficients=False)
        col_att_vec_left = context_layer(left_encoded)
        col_att_vec_right = context_layer(right_encoded)

        dropout_1 = Dropout(self.dropout)
        col_att_vec_dr_left = dropout_1(col_att_vec_left)
        col_att_vec_dr_right = dropout_1(col_att_vec_right)

        comparer = Lambda(function=cc.euclidean_distance, name='comparer')
        # comparer = Dot(axes=1, normalize=True, name='comparer')
        output = comparer([col_att_vec_dr_left, col_att_vec_dr_right])

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
