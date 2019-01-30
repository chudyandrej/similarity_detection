import os
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


def preprocess_values(values, pad_maxlen, full_unicode=True):
    values = map(lambda x: str(x), values)
    values = map(str.strip, values)
    values = (x[::-1] for x in values)
    if full_unicode:
        values = list(map(lambda x: [ord(y) for y in x], values))
    else:
        values = map(lambda x: unidecode(x), values)
        values = map(lambda x: [tokenizer_0_96(y) for y in x], values)
    # print(list(values)[:100])
    values = pad_sequences(list(values), maxlen=pad_maxlen, truncating='pre', padding='pre')
    return values


def calculate_value_vectors(model, authority_evaluator, max_seqence_len):
    test_profiles = authority_evaluator.get_test_dataset()
    print(str(len(test_profiles)) + " classes!")
    class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
    tokened_data = preprocess_values(map(lambda x: x[1], class_values), max_seqence_len, full_unicode=False)
    value_embeddings = model.predict(tokened_data)
    class_embeddings = list(map(lambda x: x[0], class_values))
    print(str(len(value_embeddings)) + " values for activation.")
    print(str(len(class_embeddings)) + " classes for data.")
    return class_embeddings, value_embeddings


# ====================================================================
#                           EXPERIMENT 1.0
# Experiment with GRU seq2seq with Enbedding layer jointly.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRIANING_DATA=./data/s3+cvut_data.csv
# RESULT: {'total': 63585, 0: 22708, None: 15725, 1: 5499, 2: 2615,
# 3: 1833, 4: 1443, 5: 1149, 6: 930, 7: 738, 8: 664, 9: 614, 10: 538)
# Percentage of found labels on first 3 index : 48%
# ====================================================================
def experiment_seq2seq_gru_embedder_jointly():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_embedder_jointly.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_embedder1548800105/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "mean")

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.1
# Experiment with GRU seq2seq with Enbedding layer jointly.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRIANING_DATA=./data/s3+cvut_data.csv
# RESULT: {'total': 699435, None: 263634, 0: 172336, 1: 47177, 2: 26564,
# 3: 18721, 4: 14783, 5: 12164, 6: 10012, 7: 9057, 8: 7570, 9: 6551, 10: 6142}
# Percentage of found labels on first 3 index : 35%
# ====================================================================
def experiment_seq2seq_gru_embedder_jointly_1():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_embedder_jointly_1.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_embedder1548800105/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by_mrc(list(zip(class_embeddings, value_embeddings)))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.2
# Experiment with GRU seq2seq with Enbedding layer jointly.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRIANING_DATA=./data/s3+cvut_data.csv
# RESULT: {'total': 63585, 0: 22708, None: 15729, 1: 5499, 2: 2613, 3: 1834,
# 4: 1444, 5: 1149, 6: 929, 7: 738, 8: 664, 9: 614, 10: 538)
# Percentage of found labels on first 3 index : 48%
# ====================================================================
def experiment_seq2seq_gru_embedder_jointly_2():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_embedder_jointly_2.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_embedder1548800105/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "sum")

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.3
# Experiment with GRU seq2seq with Enbedding layer jointly.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRIANING_DATA=./data/s3+cvut_data.csv
# RESULT:
# ====================================================================
def experiment_seq2seq_gru_embedder_jointly_3():
    def load_h5(path):
        from keras.layers import Concatenate
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        concat = Concatenate()([model.layers[4].output[0], model.layers[4].output[1]])
        encoder = Model(model.inputs[0], concat)
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_embedder_jointly_3.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_embedder1548800105/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "mean")

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 2
# Value embedding of seq2seq with embedder layer over 75k characters.
#        Percentage of found labels on first 3 index : 50%
# ====================================================================
def experiment_seq2seq_embedder(recompute):

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_embedder.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH + experiment_name) and not recompute:
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH + experiment_name, "rb"))
    else:
        encoder_model = lm.load_seq2seq_embedder(model_path, emb_path)
        print("Model successfully loaded. ")
        test_profiles = ev.get_test_dataset()
        print(str(len(test_profiles)) + " classes!")
        class_values = [(profile, value) for profile in test_profiles for value in profile.quantiles]
        tokened_data = preprocess_values(map(lambda x: x[1], class_values), max_text_seqence_len)
        value_embeddings = encoder_model.predict(tokened_data)
        class_embeddings = list(map(lambda x: x[0], class_values))
        print(str(len(value_embeddings)) + " values for activation.")
        print(str(len(class_embeddings)) + " classes for data.")

        class_embedding = em.create_column_embedding_by_avg(list(zip(class_embeddings, value_embeddings)))
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


if __name__ == '__main__':
    experiment_seq2seq_gru_embedder_jointly_3()


