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
from evaluater.preprocessing import preprocess_values_standard
from keras.models import Model, load_model


CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/similarity_detection/pickles/"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def calculate_value_vectors(model, authority_evaluator, max_seqence_len):

    return uids, embeddings


# ====================================================================
#                           EXPERIMENT 1.0
# Mean aggregation Ht output as embedding vector from coder.

# Experiment with GRU seq2seq with Enbedding layer jointly.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
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
# Change indexing method in space aggregation of vectros not used
# every column have 11 representations (by size of quantile).

# Experiment with GRU seq2seq with Enbedding layer jointly.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
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
# Vector aggregation method SUM

# Experiment with GRU seq2seq with Enbedding layer.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
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
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.99)

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
# Concat Ht and STATE (in GRU the same)

# Experiment with GRU seq2seq with Enbedding layer.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT:{'total': 63585, 0: 22708, None: 15725, 1: 5498, 2: 2615, 3: 1834,
# 4: 1443, 5: 1149, 6: 930, 7: 738, 8: 664, 9: 614, 10: 538}
# Percentage of found labels on first 3 index : 48%
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
    encoder_model.summary()
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "mean")

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.4
# Select STATE output from GRU as embedding vector

# Experiment with GRU seq2seq with Enbedding layer.
# TOKEN_COUNT = 65536, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 256
# loss_function= mean_squared_error, TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT:{'total': 63585, 0: 22709, None: 15725, 1: 5497, 2: 2616, 3: 1833,
# 4: 1443, 5: 1149, 6: 930, 7: 738, 8: 664, 9: 614, 10: 538}
# Percentage of found labels on first 3 index : 48%
# ====================================================================
def experiment_seq2seq_gru_embedder_jointly_4():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[0])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_embedder_jointly_4.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_embedder1548800105/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "mean")

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 2.0
# Experiment with GRU seq2seq with one-hot layer.

# TOKEN_COUNT = 2100, GRU_DIM = 128, loss_function=categorical_crossentropy,
# TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT:{'total': 63585, 0: 24514, None: 14367, 1: 5596, 2: 2643, 3: 1759,
# 4: 1376, 5: 1019, 6: 902, 7: 742, 8: 668, 9: 573, 10: 536}
# Percentage of found labels on first 3 index : 51%
# ====================================================================
def experiment_seq2seq_gru_onehot():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_onehot.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_one-hot1548841391/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")
    class_embeddings, value_embeddings = calculate_value_vectors(encoder_model, ev, 64)
    print("Clustering value vectors to column representation")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "mean")

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 2.1
# Experiment with GRU seq2seq with one-hot layer.

# TOKEN_COUNT = 2100, GRU_DIM = 256, loss_function=categorical_crossentropy,
# TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT:{'total': 63585, 0: 25282, None: 14353, 1: 5569, 2: 2545,
# 3: 1824, 4: 1271, 5: 1059, 6: 913, 7: 703, 8: 589, 9: 561, 10: 510}
# Percentage of found labels on first 3 index : 52%
# ====================================================================
def experiment_seq2seq_gru_onehot_1():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru_onehot_1.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq_one-hot1548861624/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', data_source="s3", neighbors=5, train_size=0.99)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")

    # Get data
    test_profiles = ev.get_test_dataset()
    # Transform to value level
    uid_values = [(profile['uid'], value) for profile in test_profiles for value in profile['quantiles']]
    tokened_data = preprocess_values_standard(map(lambda x: x[1], uid_values), 64)
    embeddings = encoder_model.predict(tokened_data)
    uids = list(map(lambda x: x[0], uid_values))

    print("Clustering value vectors to column representation")
    uid_embedding = em.create_column_embedding_by(list(zip(uids, embeddings)), "mean")
    uid_profile_index = dict(map(lambda profile: (profile['uid'], profile), test_profiles))
    profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(uid_embedding)))
    ev.evaluate_embeddings(profile_embedding)


if __name__ == '__main__':
    experiment_seq2seq_gru_onehot_1()


