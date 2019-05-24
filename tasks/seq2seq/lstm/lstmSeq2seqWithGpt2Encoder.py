
import tensorflow as tf

from keras.layers import *
from keras.models import *
from keras.callbacks import *
from preprocessor.encoder import Encoder

import custom_components as cc
from tasks.seq2seq.seq2seq import Seq2seq


class LstmSeq2seqWithGpt2Encoder(Seq2seq):
    def __init__(self, lstm_dim, dropout, max_seq_len, version, encoder: Encoder, name):
        self.lstm_dim = lstm_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.version = version
        self.encoder = encoder

        self.output_space = f"{super().OUTPUT_ROOT}/{name}/{self.version}"
        super().__init__(encoder, self.max_seq_len, self.output_space)

    def build_model(self):
        with open(super().GPT2_CONFIG_PATH, 'r') as reader:
            config = json.load(reader)

        encoder_inputs = Input(shape=(self.max_seq_len,), name="encoder_Input", dtype="int32")
        decoder_inputs = Input(shape=(self.max_seq_len,), name="decoder_Input", dtype="int32")
        target = Input(shape=(self.max_seq_len,), name="target_Input", dtype="int32")

        embedding = Embedding(input_dim=config['n_vocab'], output_dim=config['n_embd'], mask_zero=False,
                              name='Embed-Token', trainable=False)

        embedded_encoder_input = embedding(encoder_inputs)
        embedded_decoder_input = embedding(decoder_inputs)
        embedded_target = embedding(target)

        encoder = LSTM(self.lstm_dim, return_state=True, recurrent_dropout=self.dropout, dropout=self.dropout)
        encoder_outputs, state_h, state_c = encoder(embedded_encoder_input)

        decoder_gru = LSTM(self.lstm_dim, return_sequences=True, recurrent_dropout=self.dropout, dropout=self.dropout)
        decoder_outputs = decoder_gru(embedded_decoder_input, initial_state=[state_h, state_c])
        decoder_dense = Dense(config['n_embd'], activation='tanh')
        decoder_outputs = decoder_dense(decoder_outputs)

        output = Concatenate(axis=1)([embedded_target, decoder_outputs])
        output = Reshape((2, self.max_seq_len, config['n_embd']))(output)

        model = Model([encoder_inputs, decoder_inputs, target], output)

        model.get_layer(name='Embed-Token').set_weights([
            tf.train.load_variable(super().GPT2_CHECKPOINT_PATH, 'model/wte:0'),
        ])
        model.summary()
        model.compile(optimizer='adam', loss=cc.mean_squared_error_from_pred)
        model.name = self.version
        return model

    def load_encoder(self):
        model = self.load_model()
        model: Model = Model(model.inputs[0], model.layers[4].output[2])
        model.summary()
        return model

    def load_model(self):
        return load_model(f"{self.output_space}/model.h5", custom_objects={
            "mean_squared_error_from_pred": cc.mean_squared_error_from_pred,
            "EmbeddingRet": cc.EmbeddingRet
        })
