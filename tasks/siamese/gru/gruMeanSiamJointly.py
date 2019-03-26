
import os
from keras.layers import *
from keras.models import *
from ..siamese import Siamese

import trainer.custom_components as cc


class GruHierSiamJointly(Siamese):
    def __init__(self, encoder, max_seq_len, gru_dim, dropout, version):
        super().__init__(encoder=encoder, max_seq_len=max_seq_len)
        self.gru_dim = gru_dim
        self.dropout = dropout
        self.version = version

        self.output_space = f"{super().OUTPUT_ROOT}/{type(self).__name__}/{self.version}"
        os.makedirs(self.output_space, exist_ok=True)

    def get_output_space(self):
        return self.output_space

    def value_embedder(self):
        input_layer = Input(shape=(self.max_seq_len,), name='input')
        embedded_value = Embedding(input_dim=65536, output_dim=200, name='value_embedder', trainable=False)(
            input_layer)
        embedded_value = Dropout(self.dropout)(embedded_value)
        encoded_value = Bidirectional(
            CuDNNGRU(units=self.gru_dim, return_sequences=True), merge_mode='concat', weights=None)(embedded_value)
        encoded_value = Dropout(self.dropout)(encoded_value)
        model = Model(input_layer, encoded_value)
        return model

    def build_model(self):
        # Ensemble Joint Model
        left_input = Input(shape=(11, self.max_seq_len), name='left_input')
        right_input = Input(shape=(11, self.max_seq_len), name='right_input')

        # Value embedding model trained as seq2seq.
        left_value_encoded = TimeDistributed(self.value_embedder)(left_input)
        right_value_encoded = TimeDistributed(self.value_embedder)(right_input)

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
        model = load_model(f"{self.output_space}/model.h5", custom_objects={
            "euclidean_distance": cc.euclidean_distance,
            "contrastive_loss": cc.contrastive_loss

        })
        print(model.layers)
        exit()
        model: Model = Model(model.inputs[0], model.layers[6].get_output_at(0))
        model.summary()
        return model

































