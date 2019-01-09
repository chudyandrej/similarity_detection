from keras.models import Model, load_model


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


# def load_seq2seq(model_path):
#     model = load_model(model_path)
#     # encoder
#     encoder_inputs = model.get_layer(name="encoder_Input_1").output
#     encoder_outputs, state_h_enc, state_c_enc = model.get_layer(name="encoder").get_output_at(0)
#     encoder_states = [state_h_enc, state_c_enc]
#     encoder_model = Model(encoder_inputs, encoder_states)
#     return encoder_model


def load_seq2seq(model_path):
    """Load Vec2Vec model and divide to encoder and decoder. With model is loaded
    settenigs piclke file.

    Args:
        model_path (STRING): Path to .h5  file

    Returns:
        (keras.engine.training.Model, keras.engine.training.Model): Encoder and  decoder_model
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
