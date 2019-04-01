import os
from keras.layers import *
from keras.models import *
from ..siamese import Siamese
from preprocessor.encoder import Encoder
import custom_components as cc


class LstmHierSiamJointly(Siamese):
    def __init__(self, encoder: Encoder, emb_out_dim,  max_seq_len: int, lstm_dim, dropout, version):
        self.lstm_dim = lstm_dim
        self.emb_out_dim = emb_out_dim
        self.dropout = dropout
        self.version = version

        self.output_space = f"{super().OUTPUT_ROOT}/{type(self).__name__}/{self.version}"
        super().__init__(encoder=encoder, max_seq_len=max_seq_len, output_path=self.output_space )

        # Ensemble Joint Model

    def build_model(self):
        def value_level():
            input_layer = Input(shape=(self.max_seq_len,), name='input')
            embedded_value = Embedding(input_dim=self.encoder.get_vocab_size(), output_dim=self.emb_out_dim,
                                       name='value_embedder', trainable=True)(input_layer)
            embedded_value = Dropout(self.dropout)(embedded_value)
            encoded_value = Bidirectional(LSTM(units=self.lstm_dim, return_sequences=True, recurrent_dropout=self.dropout,
                                               dropout=self.dropout))(embedded_value)
            model = Model(input_layer, encoded_value)
            return model

        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        # Value embedding model trained as seq2seq.
        value_encoder = value_level()
        left_value_encoded = TimeDistributed(value_encoder)(left_input)
        right_value_encoded = TimeDistributed(value_encoder)(right_input)

        quantile_encoder = Bidirectional(LSTM(units=self.lstm_dim, return_sequences=True, recurrent_dropout=self.dropout,
                                              dropout=self.dropout), name="bidirectional_quantile")
        left_encoded = quantile_encoder(left_value_encoded)
        right_encoded = quantile_encoder(right_value_encoded)

        comparer = Lambda(function=cc.euclidean_distance, name='comparer')
        # comparer = Dot(axes=1, normalize=True, name='comparer')
        output = comparer([left_encoded, right_encoded])

        # Compile and train Joint Model
        joint_model = Model(inputs=[left_input, right_input], outputs=output)
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
