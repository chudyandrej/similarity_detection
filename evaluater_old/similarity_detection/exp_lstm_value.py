import os
import json
import sdep
import pickle

from keras.preprocessing.sequence import pad_sequences

import argparse
from evaluater.embedder import tokenizer_0_96
from unidecode import unidecode
import evaluater.load_models as lm
import evaluater.embedder as em

import trainer.custom_components as cc
from keras.models import Model, load_model

CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/similarity_detection/pickles/"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def preprocess_values(values, pad_maxlen, char_index=None):
    values = map(lambda x: str(x), values)
    values = map(str.strip, values)
    values = (x[::-1] for x in values)
    if char_index is None:
        values = list(map(lambda x: [ord(y) for y in x], values))
    else:
        # values = map(lambda x: unidecode(x), values)
        values = map(lambda x: [char_index[y] for y in x], values)
    # print(list(values)[:100])
    values = pad_sequences(list(values), maxlen=pad_maxlen, truncating='pre', padding='pre')
    return values


def calculate_value_vectors(model, authority_evaluator, max_seqence_len):
    test_profiles = authority_evaluator.get_test_dataset()
    print(str(len(test_profiles)) + " classes!")
    class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
    tokened_data = preprocess_values(map(lambda x: x[1], class_values), max_seqence_len)
    value_embeddings = model.predict(tokened_data)
    class_embeddings = list(map(lambda x: x[0], class_values))
    print(str(len(value_embeddings)) + " values for activation.")
    print(str(len(class_embeddings)) + " classes for data.")
    return class_embeddings, value_embeddings


# ====================================================================
#                           EXPERIMENT 1.0
# This experiment testing seq2seq model with embedding layer as independent
# model and then construct to one encoder.

# Experiment with LSTM seq2seq with Enbedding layer independent.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT:{'total': 63585, 0: 21639, None: 16546, 1: 4923, 2: 2507, 3: 1613,
# 4: 1279, 5: 1082, 6: 945, 7: 772, 8: 678, 9: 597, 10: 566}
# Percentage of found labels on first 3 index : 45%
# ====================================================================
def experiment_seq2seq_lstm_embedder():
    def load_h5(model_src, embedder_src):
        from keras.layers import Concatenate
        embedder = load_model(embedder_src)
        seq2seq = load_model(model_src)

        encoder_inputs = embedder.input
        x = embedder.layers[1](encoder_inputs)
        _ = seq2seq.layers[2](x)
        encoder_outputs, state_h_enc, state_c_enc = seq2seq.layers[2].get_output_at(1)
        encoder_states = [state_h_enc, state_c_enc]
        x = Concatenate()(encoder_states)
        return Model(encoder_inputs, x)

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_lstm_embedder.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, train_size=0.5)

    encoder_model = load_h5(model_path, emb_path)
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    print(str(len(test_profiles)) + " classes!")
    class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
    tokened_data = preprocess_values(map(lambda x: x[1], class_values), max_text_seqence_len)
    value_embeddings = encoder_model.predict(tokened_data)
    class_embeddings = list(map(lambda x: x[0], class_values))
    print(str(len(value_embeddings)) + " values for activation.")
    print(str(len(class_embeddings)) + " classes for data.")

    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), ag_method="mean")
    pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 2.0
# This experiment testing seq2seq model with embedding layer as independent
# model and then construct to one encoder.

# Experiment with LSTM seq2seq with Enbedding layer independent.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT:
# ====================================================================
def experiment_seq2seq_lstm_onehot():
    def load_h5(model_src):
        from keras.layers import Concatenate

        model = load_model(model_src, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        print(model.layers)
        exit()
    
        encoder = Model(model.inputs[0], Concatenate()([model.layers[4].output[1], model.layers[4].output[2]]))
        encoder.summary()
        return encoder

    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_lstm_onehot.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/lstm_seq2seq_onehot_tokenizer1549359842/" \
                                                          "lstm_seq2seq_onehot-model.h5"
    char_index_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/lstm_seq2seq_onehot_tokenizer1549359842/" \
                                                               "lstm_seq2seq_onehot-model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, train_size=0.5)
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset(data_src="s3")
    print(str(len(test_profiles)) + " classes!")
    class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
    tokened_data = preprocess_values(map(lambda x: x[1], class_values), 64, char_index=char_index)
    value_embeddings = encoder_model.predict(tokened_data)
    class_embeddings = list(map(lambda x: x[0], class_values))
    print(str(len(value_embeddings)) + " values for activation.")
    print(str(len(class_embeddings)) + " classes for data.")

    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), ag_method="mean")
    pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


if __name__ == '__main__':
    experiment_seq2seq_lstm_onehot()
