
from keras.layers import *
from keras.models import *
import trainer.custom_components as cc
from tasks.seq2seq.gru.onehot_encoder.config import Config


from ..seq2seq import Seq2seq


class GruSeq2seqWithOnehot(Seq2seq):

    def __init__(self, input_dim, gru_dim, max_seq_len, version, encoder):
        super().__init__(encoder)
        self.input_dim = input_dim
        self.gru_dim = gru_dim
        self.max_seq_len = max_seq_len
        self.version = version
        self.output_space = f"{super().OUTPUT_ROOT}/{type(self).__name__}/{self.version}"

    def build_model(self):
        encoder_inputs = Input(shape=(Config.MAX_TEXT_SEQUENCE_LEN,), name="encoder_Input", dtype="int32")
        decoder_inputs = Input(shape=(Config.MAX_TEXT_SEQUENCE_LEN,), name="decoder_Input", dtype="int32")
        target = Input(shape=(Config.MAX_TEXT_SEQUENCE_LEN,), name="target_Input", dtype="int32")

        embedding = cc.OneHot(input_dim=self.input_dim, input_length=Config.MAX_TEXT_SEQUENCE_LEN)
        embedded_encoder_input = embedding(encoder_inputs)
        embedded_decoder_input = embedding(decoder_inputs)
        emedded_target = embedding(target)

        encoder = GRU(self.gru_dim, return_state=True)
        encoder_outputs, state_h = encoder(embedded_encoder_input)

        decoder_gru = GRU(self.gru_dim, return_sequences=True)
        decoder_outputs = decoder_gru(embedded_decoder_input, initial_state=state_h)
        decoder_dense = Dense(self.input_dim, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)

        output = Concatenate(axis=1)([emedded_target, decoder_outputs])
        output = Reshape((2, self.max_seq_len, self.input_dim))(output)

        model = Model([encoder_inputs, decoder_inputs, target], output)
        model.compile(optimizer='adam', loss=cc.categorical_crossentropy_form_pred)
        model.summary()

        return model

    def load_encoder(self):
        model = load_model(f"{self.output_space}/model.h5", custom_objects={
            "mean_squared_error_from_pred": cc.mean_squared_error_from_pred,
            "zero_loss": cc.zero_loss,
            "EmbeddingRet": cc.EmbeddingRet
        })
        model: Model = Model(model.inputs[0], model.layers[4].output[1])
        model.summary()
        return model

    def get_output_space(self):
        pass

