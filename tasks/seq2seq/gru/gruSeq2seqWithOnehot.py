
from keras.layers import *
from keras.models import *
import custom_components as cc

from preprocessor.encoder import Encoder
from ..seq2seq import Seq2seq


class GruSeq2seqWithOnehot(Seq2seq):

    def __init__(self, gru_dim, dropout, max_seq_len, version, encoder: Encoder):
        self.gru_dim = gru_dim
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.version = version
        self.output_space = f"{super().OUTPUT_ROOT}/{type(self).__name__}/{self.version}"
        super().__init__(encoder, self.max_seq_len, self.output_space)

    def build_model(self):
        encoder_inputs = Input(shape=(self.max_seq_len,), name="encoder_Input", dtype="int32")
        decoder_inputs = Input(shape=(self.max_seq_len,), name="decoder_Input", dtype="int32")
        target = Input(shape=(self.max_seq_len,), name="target_Input", dtype="int32")

        embedding = cc.OneHot(input_dim=self.encoder.get_vocab_size(), input_length=self.max_seq_len)
        embedded_encoder_input = embedding(encoder_inputs)
        embedded_decoder_input = embedding(decoder_inputs)
        emedded_target = embedding(target)

        encoder = GRU(self.gru_dim, dropout=self.dropout, recurrent_dropout=self.dropout, return_state=True)
        encoder_outputs, state_h = encoder(embedded_encoder_input)

        decoder_gru = GRU(self.gru_dim, dropout=self.dropout, recurrent_dropout=self.dropout, return_sequences=True)
        decoder_outputs = decoder_gru(embedded_decoder_input, initial_state=state_h)
        decoder_dense = Dense(self.encoder.get_vocab_size(), activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        output = Concatenate(axis=1)([emedded_target, decoder_outputs])
        output = Reshape((2, self.max_seq_len, self.encoder.get_vocab_size()))(output)

        model = Model([encoder_inputs, decoder_inputs, target], output)
        model.compile(optimizer='adam', loss=cc.categorical_crossentropy_form_pred)
        model.summary()

        return model

    def load_encoder(self):
        model = self.load_model()
        model: Model = Model(model.inputs[0], model.layers[4].output[1])
        model.summary()
        return model

    def load_model(self):
        return load_model(f"{self.output_space}/model.h5", custom_objects={
            "mean_squared_error_from_pred": cc.categorical_crossentropy_form_pred,
            "OneHot": cc.OneHot
        })
