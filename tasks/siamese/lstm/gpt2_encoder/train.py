
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.callbacks import *
import pandas as pd

import trainer.custom_components as cc
from .config import Config

from preprocessor.preprocessor import DataPreprocessorSeq2seq
from preprocessor.encoder.bpe import BytePairEncoding


def build_model_v1():
    with open(Config.GPT2_CONFIG_PATH, 'r') as reader:
        config = json.load(reader)

    encoder_inputs = Input(shape=(64,), name="encoder_Input", dtype="int32")
    decoder_inputs = Input(shape=(64,), name="decoder_Input", dtype="int32")
    target = Input(shape=(64,), name="target_Input", dtype="int32")

    embedding = cc.EmbeddingRet(input_dim=config['n_vocab'], output_dim=config['n_embd'],
                                mask_zero=False, name='Embed-Token', trainable=False)

    embedded_encoder_input, _ = embedding(encoder_inputs)
    embedded_decoder_input, _ = embedding(decoder_inputs)
    embedded_target, _ = embedding(target)

    encoder = LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder(embedded_encoder_input)

    decoder_gru = LSTM(128, return_sequences=False)
    decoder_outputs = decoder_gru(embedded_decoder_input, initial_state=state_c)
    decoder_dense = Dense(config['n_embd'], activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)

    output = Concatenate(axis=1)([embedded_target, decoder_outputs])
    output = Reshape((2, 64, config['n_embd']))(output)

    model = Model([encoder_inputs, decoder_inputs, target], output)

    model.get_layer(name='Embed-Token').set_weights([
        tf.train.load_variable(Config.GPT2_CHECKPOINT_PATH, 'model/wte:0'),
    ])
    model.summary()
    model.compile(optimizer='adam', loss=cc.zero_loss)
    model.name = "V1"
    return model


def build_model_v2():
    with open(Config.GPT2_CONFIG_PATH, 'r') as reader:
        config = json.load(reader)

    encoder_inputs = Input(shape=(64,), name="encoder_Input", dtype="int32")
    decoder_inputs = Input(shape=(64,), name="decoder_Input", dtype="int32")
    target = Input(shape=(64,), name="target_Input", dtype="int32")

    embedding = cc.EmbeddingRet(input_dim=config['n_vocab'], output_dim=config['n_embd'],
                                mask_zero=False, name='Embed-Token', trainable=False)

    embedded_encoder_input, _ = embedding(encoder_inputs)
    embedded_decoder_input, _ = embedding(decoder_inputs)
    embedded_target, _ = embedding(target)

    encoder = LSTM(256, return_state=True, recurrent_dropout=0.4, dropout=0.4)
    encoder_outputs, state_h, state_c = encoder(embedded_encoder_input)

    decoder_gru = LSTM(256, return_sequences=True, return_state=True, recurrent_dropout=0.4, dropout=0.4)
    decoder_outputs, _, _ = decoder_gru(embedded_decoder_input, initial_state=[state_h, state_c])
    decoder_dense = Dense(config['n_embd'], activation='sigmoid')
    decoder_outputs = decoder_dense(decoder_outputs)

    output = Concatenate(axis=1)([embedded_target, decoder_outputs])
    output = Reshape((2, 64, config['n_embd']))(output)

    model = Model([encoder_inputs, decoder_inputs, target], output)

    model.get_layer(name='Embed-Token').set_weights([
        tf.train.load_variable(Config.GPT2_CHECKPOINT_PATH, 'model/wte:0'),
    ])
    model.summary()
    model.compile(optimizer='adam', loss=cc.mean_squared_error_from_pred)
    model.name = "V2"
    return model


def train_pipeline():
    model = build_model_v2()
    training_space = Config.OUTPUT_SPACE + "/" + model.name
    os.makedirs(training_space, exist_ok=True)

    string_values = pd.read_csv(tf.gfile.Open(Config.DATA_PATH))['value'].values

    # Preprocess data
    preprocessor = DataPreprocessorSeq2seq(string_values, encoder=BytePairEncoding())
    input_coder, input_decoder, target = preprocessor.get_training_data(Config.MAX_TEXT_SEQUENCE_LEN)

    model.fit(x=[input_coder, input_decoder, target],
              y=target,
              epochs=500,
              batch_size=64,
              validation_split=0.3,
              callbacks=[
                  cc.ModelCheckpointMLEngine(training_space+"/model.h5", monitor='val_loss',
                                             verbose=1, save_best_only=True, mode='min'),
                  EarlyStopping(monitor='val_loss', patience=15, verbose=1),
                  TensorBoard(log_dir=training_space+'/training_log', write_graph=True,
                              embeddings_freq=0)
                       ])


if __name__ == '__main__':
    train_pipeline()
