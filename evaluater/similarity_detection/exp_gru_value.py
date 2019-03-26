import os
import numpy as np
import keras
import codecs
import datetime
import evaluater.embedder as em

import trainer.custom_components as cc
from evaluater.preprocessing import preprocess_values_standard
from keras.models import Model, load_model
from typing import List, Optional, Tuple

from sdep import AuthorityEvaluator, Profile   # Needed
from preprocessor.preprocessor import DataPreprocessorSeq2seq


# ====================================================================
def exp1():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = sim_detection_seq2seq_gru_onehot_1.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/outcome/vec2vec_gpt2_embedding/training/model2/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50,
                            results_file="./results/"+experiment_name+".txt")
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    uid_values: List[Tuple[Tuple, str]] = [(profile.uid, value) for profile in test_profiles
                                           for value in profile.quantiles]


    tokened_data: np.array = preprocess_values_standard(map(lambda x: x[1], uid_values), 64)
    embeddings = model.predict(tokened_data)
    uids = list(map(lambda x: x[0], uid_values))

    print("Clustering value vectors to column representation")
    uid_embedding = em.create_column_embedding_by(list(zip(uids, embeddings)), ag_method)
    uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
    profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(uid_embedding)))
    ev.evaluate_embeddings(profile_embedding)


if __name__ == '__main__':




























































def computing_body_by_interface(model: keras.engine.training.Model, test_profiles: List[Profile],
                                ev: AuthorityEvaluator, ag_method='mean'):
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    uid_values: List[Tuple[Tuple, str]] = [(profile.uid, value) for profile in test_profiles
                                           for value in profile.quantiles]

    tokened_data: np.array = preprocess_values_standard(map(lambda x: x[1], uid_values), 64)
    embeddings = model.predict(tokened_data)
    uids = list(map(lambda x: x[0], uid_values))

    print("Clustering value vectors to column representation")
    uid_embedding = em.create_column_embedding_by(list(zip(uids, embeddings)), ag_method)
    uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
    profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(uid_embedding)))
    ev.evaluate_embeddings(profile_embedding)


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
    ev = AuthorityEvaluator(username='andrej', neighbors=100, train_size=0.5)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)


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
    ev = AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.5)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)


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
    ev = AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.99)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)


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
    ev = AuthorityEvaluator(username='andrej', neighbors=100, train_size=0.5)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)


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
    ev = AuthorityEvaluator(username='andrej', neighbors=100, train_size=0.5)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)


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
    ev = AuthorityEvaluator(username='andrej', neighbors=100, train_size=0.5)
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)


# ====================================================================
#                           EXPERIMENT 2.1
# Experiment with GRU seq2seq with one-hot layer.

# TOKEN_COUNT = 2100, GRU_DIM = 256, loss_function=categorical_crossentropy,
# TRAINING_DATA=./data/s3+cvut_data.csv
# RESULT [OLD_S3_PROFILE]:{'total': 63585, 0: 25336, None: 21520, 1: 5726, 2: 2733, 3: 1684, 4: 1314, 5: 1096, 6: 815,
# 7: 671, 8: 557, 9: 487, 10: 405, 11: 352, 12: 268, 13: 238, 14: 134, 15: 130, 16: 66, 17: 39, 18: 14}
# Percentage of found labels on first 3 index : 53%
# RESULT [NEW_S3_PROFILE]:{'total': 73758, 0: 37016, None: 13843, 1: 8179, 2: 3845, 3: 2514, 4: 1816, 5: 1447,
# 6: 1077, 7: 932, 8: 730, 9: 629, 10: 483, 11: 418, 12: 277, 13: 253, 14: 143, 15: 100, 16: 36, 17: 16, 18: 4}
# Percentage of found labels on first 3 index : 66%
# ====================================================================
def sim_detection_seq2seq_gru_onehot_1():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = sim_detection_seq2seq_gru_onehot_1.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/gru_seq2seq-hot1549903432/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50,
                            results_file="./results/"+experiment_name+".txt")
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    computing_body_by_interface(encoder_model, test_profiles, ev)





# ====================================================================
#                           EXPERIMENT 8
# [GPT2 fix]-[DENSE linear]
# ====================================================================
def exp8():
    def load_h5(model_src):
        model = load_model(model_src, custom_objects={
            "euclidean_distance": cc.euclidean_distance,
            "contrastive_loss": cc.contrastive_loss
        })
        encoder = Model(model.inputs[0], model.layers[3].get_output_at(0))
        encoder.summary()
        return encoder

    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    date = datetime.datetime.now().strftime("%m-%d_%H:%M")
    eval_space = os.environ['PYTHONPATH'].split(":")[0] + '/outcome/GPT2/eval_'+date
    training_space = os.environ['PYTHONPATH'].split(":")[0] + '/outcome/GPT2/training'
    os.makedirs(eval_space)

    # -------------- LOAD DATA  --------------------
    ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean", results_file=eval_space)
    test_profiles = ev.get_test_dataset(data_src="s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    final_tune_model = load_h5(training_space+"/model.h5")
    print("Model successfully loaded. ")
    profile2embedding = get_index_profile2embedding_gpt2(test_profiles, training_space)
    embeddings = [profile2embedding[prof] for prof in test_profiles]
    final_embeddings = final_tune_model.predict(np.array(embeddings))
    print("Processed " + str(len(final_embeddings)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(list(zip(test_profiles, final_embeddings)))


# ====================================================================
# ====================================================================
def exp11():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = sim_detection_seq2seq_gru_onehot_1.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/outcome/vec2vec_gpt2_embedding/training/model2/model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50,
                            results_file="./results/"+experiment_name+".txt")
    test_profiles: List[Profile] = ev.get_test_dataset("s3")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    uid_values: List[Tuple[Tuple, str]] = [(profile.uid, value) for profile in test_profiles
                                           for value in profile.quantiles]

    tokened_data: np.array = preprocess_values_standard(map(lambda x: x[1], uid_values), 64)
    embeddings = model.predict(tokened_data)
    uids = list(map(lambda x: x[0], uid_values))

    print("Clustering value vectors to column representation")
    uid_embedding = em.create_column_embedding_by(list(zip(uids, embeddings)), ag_method)
    uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
    profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(uid_embedding)))
    ev.evaluate_embeddings(profile_embedding)


if __name__ == '__main__':
    sim_detection_seq2seq_gru_onehot_1()


