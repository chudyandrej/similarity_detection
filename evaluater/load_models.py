from keras.models import Model, load_model

from keras.layers import Concatenate, Input, TimeDistributed, LSTM, Bidirectional, Embedding


def load_seq2seq_siamese(model_path):
    model = load_model(model_path)
    # encoder
    encoder_inputs = model.get_layer(name="encoder_Input_1").output
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name="encoder").get_output_at(0)
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    return encoder_model


def load_seq2_siamese(model_path):

    model = load_model(model_path)
    # encoder
    encoder_inputs = model.get_layer(name="encoder_Input_1").output
    encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name="encoder").get_output_at(0)
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    return encoder_model


def load_seq2seq(model_path):
    """Load Vec2Vec model and divide to encoder and decoder. With model is loaded
    settenigs piclke file.

    Args:
        model_path (STRING): Path to .h5  file

    Returns:
        (keras.engine.training.Model): Encoder model
    """
    model = load_model(model_path)
    # encoder
    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)
    return encoder_model


def load_cnn_kim(model_path):
    model = load_model(model_path)
    return model.layers[2]


def load_cnn_tcn(model_path):
    model = load_model(model_path)
    return model.layers[2]


def load_seq2seq_embedder(model_path, embedder_path):
    embedder = load_model(embedder_path)
    seq2seq = load_model(model_path)

    encoder_inputs = embedder.input
    x = embedder.layers[1](encoder_inputs)
    _ = seq2seq.layers[2](x)
    encoder_outputs, state_h_enc, state_c_enc = seq2seq.layers[2].get_output_at(1)
    encoder_states = [state_h_enc, state_c_enc]
    x = Concatenate()(encoder_states)
    encoder_model = Model(encoder_inputs, x)
    return encoder_model


def load_hierarchy_lstm_model(model_path, embedder_path, quantile_shape=(11, 64)):
    seq2seq = load_seq2seq_embedder(model_path, embedder_path)

    input_net = Input(shape=quantile_shape, name='input')
    encoder = TimeDistributed(seq2seq)(input_net)
    quantile_encoder = Bidirectional(LSTM(128, dropout=0.50, recurrent_dropout=0.50))
    _ = quantile_encoder(encoder)
    encoder_outputs = quantile_encoder.output
    hierarchy_encoder_model = Model(input_net, encoder_outputs)
    return hierarchy_encoder_model


def load_hierarchy_lstm_base_model(quantile_shape=(11, 64)):
    # Ensemble Joint Model
    net_input = Input(shape=quantile_shape, name='left_input')

    value_embedder = Embedding(input_dim=65536, output_dim=128, name='value_embedder')
    embedded = TimeDistributed(value_embedder)(net_input)

    value_encoder = LSTM(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
    value_encoded = TimeDistributed(value_encoder)(embedded)

    quantile_encoder = Bidirectional(LSTM(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')
    encoded = quantile_encoder(value_encoded)

    # Compile and train Joint Model
    model = Model(inputs=net_input, outputs=encoded)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary(line_length=120)
    return model
