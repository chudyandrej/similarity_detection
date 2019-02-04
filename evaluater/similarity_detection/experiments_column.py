import os
import sdep
import pickle

from keras.models import Model, load_model
from keras.layers import Concatenate, Input, TimeDistributed, LSTM, Bidirectional, Embedding, GRU

import numpy as np
from unidecode import unidecode

from evaluater.embedder import tokenizer_0_96
from evaluater.computing import compute_neighbors
from evaluater.similarity_detection.evaluation import evaluate_similarity_index
from keras.preprocessing.sequence import pad_sequences

# import data_preparation.cvut as dt
import evaluater.load_models as lm
import evaluater.embedder as em
import trainer.custom_components as cc


CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/similarity_detection/pickles/"

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


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


# ====================================================================
#                           EXPERIMENT 1.0
# This is base-line experiment. Hierarchy LSTM with random initialised
# weights.

# Experiment with hierarchy LSTM and Enbedding layer.
# TOKEN_COUNT = 67000, LSTM_DIM = 128, ENCODER_OUTPUT_DIM = 128
# RESULT:{'total': 63585, 0: 21043, None: 16475, 1: 5406, 2: 2735,
# 3: 1844, 4: 1452, 5: 1085, 6: 939, 7: 781, 8: 741, 9: 680, 11: 534, 10: 527}
# Percentage of found labels on first 3 index : 45%
# ====================================================================
def experiment_lstm_hierarchy_base():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = Embedding(input_dim=67000, output_dim=128, name='value_embedder')
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = LSTM(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(LSTM(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_lstm_hierarchy_base.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.1
# This is base-line experiment. Hierarchy LSTM with random initialised
# weights.

# Experiment with hierarchy LSTM and Enbedding layer.
# TOKEN_COUNT = 67000, LSTM_DIM = 128, ENCODER_OUTPUT_DIM = 128
# RESULT:{'total': 63585, 0: 21022, None: 16510, 1: 5414, 2: 2736, 3: 1864,
# 4: 1445, 5: 1091, 6: 928, 7: 798, 8: 726, 9: 669, 11: 543, 10: 533}
# Percentage of found labels on first 3 index : 45%
# ====================================================================
def experiment_lstm_hierarchy_base_1():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = Embedding(input_dim=2100, output_dim=128, name='value_embedder')
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = LSTM(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(LSTM(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_lstm_hierarchy_base_1.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.2
# This is base-line experiment. Hierarchy LSTM with random initialised
# weights.

# Experiment with hierarchy LSTM and One-hot layer.
# TOKEN_COUNT = 2100, LSTM_DIM = 128
# RESULT:{'total': 63585, 0: 21174, None: 16708, 1: 5422, 2: 2674, 3: 1880,
# 4: 1332, 5: 1129, 6: 887, 7: 813, 8: 695, 9: 636, 10: 594}
# Percentage of found labels on first 3 index : 46%
# ====================================================================
def experiment_lstm_hierarchy_base_2():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = cc.OneHot(input_dim=2100, input_length=64)
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = LSTM(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(LSTM(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_lstm_hierarchy_base_2.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 1.3
# This is base-line experiment. Hierarchy LSTM with random initialised
# weights.

# Experiment with hierarchy LSTM and One-hot layer.
# TOKEN_COUNT = 2100, LSTM_DIM = 256
# RESULT:{'total': 63585, 0: 21980, None: 15888, 1: 5656, 2: 2675,
# 3: 1826, 4: 1362, 5: 1110, 6: 931, 7: 797, 8: 652, 9: 596, 10: 585}
# Percentage of found labels on first 3 index : 47%
# ====================================================================
def experiment_lstm_hierarchy_base_3():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = cc.OneHot(input_dim=2100, input_length=64)
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = LSTM(256, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(LSTM(256, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_lstm_hierarchy_base_3.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)


# ======================================================================================================================
# ======================================================================================================================


# ====================================================================
#                           EXPERIMENT 2.0
# This is base-line experiment. Hierarchy GRU with random initialised
# weights.

# Experiment with hierarchy GRU and Enbedding layer.
# TOKEN_COUNT = 67000, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 128
# RESULT:{'total': 63585, None: 20512, 0: 17113, 1: 4837, 2: 2608,
# 3: 1724, 4: 1386, 5: 1071, 6: 921, 7: 855, 9: 686, 8: 681, 10: 569}
# Percentage of found labels on first 3 index : 38%
# ====================================================================
def experiment_gru_hierarchy_base():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = Embedding(input_dim=67000, output_dim=128, name='value_embedder')
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = GRU(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(GRU(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_gru_hierarchy_base.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 2.1
# This is base-line experiment. Hierarchy GRU with random initialised
# weights.

# Experiment with hierarchy GRU and Enbedding layer.
# TOKEN_COUNT = 2100, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 128
# RESULT:{'total': 63585, None: 20597, 0: 17112, 1: 4852, 2: 2612, 3: 1737,
# 4: 1372, 5: 1079, 6: 923, 7: 837, 8: 697, 9: 681, 10: 564}
# Percentage of found labels on first 3 index : 38%
# ====================================================================
def experiment_gru_hierarchy_base_1():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = Embedding(input_dim=2100, output_dim=128, name='value_embedder')
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = GRU(128, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(GRU(128, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_gru_hierarchy_base_1.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 2.2
# This is base-line experiment. Hierarchy GRU with random initialised
# weights.

# Experiment with hierarchy GRU and Enbedding layer.
# TOKEN_COUNT = 2100, GRU_DIM = 128, ENCODER_OUTPUT_DIM = 128
# RESULT:{'total': 63585, None: 20396, 0: 17341, 1: 4848, 2: 2506,
# 3: 1730, 4: 1379, 5: 1043, 6: 929, 7: 795, 8: 700, 9: 640, 10: 574}
# Percentage of found labels on first 3 index : 38%
# ====================================================================
def experiment_gru_hierarchy_base_2():
    def load_h5():
        net_input = Input(shape=(11, 64), name='left_input')
        value_embedder = cc.OneHot(input_dim=2100, input_length=64)
        embedded = TimeDistributed(value_embedder)(net_input)

        value_encoder = GRU(256, dropout=0.50, recurrent_dropout=0.50, name='value_encoder')
        value_encoded = TimeDistributed(value_encoder)(embedded)
        quantile_encoder = Bidirectional(GRU(256, dropout=0.50, recurrent_dropout=0.50), name='quantile_encoder')

        encoded = quantile_encoder(value_encoded)
        return Model(inputs=net_input, outputs=encoded)
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_gru_hierarchy_base_2.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    encoder_model = load_h5()
    encoder_model.summary()
    print("Model successfully loaded. ")
    test_profiles = ev.get_test_dataset()
    quantiles_data = np.array(list(map(lambda x: preprocess_values(x.quantiles, 64), test_profiles)))
    print("Model is invoking ... ")
    embedding_vectors = encoder_model.predict(quantiles_data)
    print("Processed " + str(len(embedding_vectors)) + " value embeddings")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(test_profiles))))
    ev.evaluate_embeddings(test_profiles, embedding_vectors)

























# ====================================================================
#                           EXPERIMENT 1
# ====================================================================
def experiment_seq2seq_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_onehot(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=250, radius=0.15,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 2
# ====================================================================
def experiment_seq2_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2_siamese1543913096-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_onehot(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.1,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 3
# ====================================================================
def experiment_seq2seq():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_onehot(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 4
# ====================================================================
def experiment_cnn_kim():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_kim.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_kim1544535197-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_cnn_kim(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_tok(encoder_model, dataclass.df["value"].values, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 5
# ====================================================================
def experiment_cnn_tck():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_tck.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_tcn1544609078-model.h5"
    print("Experiment " + experiment_name + " running ...")

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_cnn_tcn(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_tok(encoder_model, dataclass.df["value"].values, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 6
# ====================================================================
def experiment_cnn_tck2():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_tck.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_tcn1544609078-model.h5"
    print("Experiment " + experiment_name + " running ...")
    class_embeddings = {}
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embeddings = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = lm.load_cnn_tcn(model_path)
        print("Checkpoint not found. Calculating...")
        dataclass.df.value = dataclass.df.value.map(lambda x: unidecode(str(x))[:63])
        value_embedding = em.convert_to_vec_tok(encoder_model, dataclass.df["value"].values, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embeddings = em.create_column_embedding_by_mrc(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embeddings, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes = []
    embedding_vectors = []
    for key, embeddings in class_embeddings.items():
        for embedding in embeddings:
            classes.append(key)
            embedding_vectors.append(embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")
    index = evaluate_similarity_index(similarity_index)
    return index


# ====================================================================
#                           EXPERIMENT 7
# ====================================================================
def experiment_seq2seq_siamese_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")

        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 8
# ====================================================================
def experiment_seq2_siamese_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2_siamese_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2_siamese1543913096-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found :-)")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        # test_data = list(filter(lambda x: x.dtype == "object", test_data))
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 9
# ====================================================================
def experiment_seq2seq_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        # test_data = list(filter(lambda x: x.dtype == "object", test_data))
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 10
# ====================================================================
def experiment_cnn_kim_sdep():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_kim_sdep.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_kim1544535197-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_cnn_kim(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_tok(encoder_model, quantiles, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 11
# ====================================================================
def experiment_cnn_tck_sdeq():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_cnn_tck_sdeq.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/cnn_tcn1544609078-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_cnn_tcn(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_tok(encoder_model, quantiles, 64)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 12
# ====================================================================
def experiment_seq2seq_sdep_2():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_sdep_2.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        # test_data = list(filter(lambda x: x.dtype == "object", test_data))
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_mrc(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 13
# ====================================================================
def experiment_seq2seq_siamese_sdep_2():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese_sdep_2.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()
        quantiles = [data.quantiles for data in test_data]
        quantiles = [unidecode(str(q))[:63] for q in quantiles]
        encoder_model = lm.load_seq2_siamese(model_path)
        print("Checkpoint not found. Calculating...")

        value_embedding = em.convert_to_vec_onehot(encoder_model, quantiles, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_mrc(list(zip(test_data, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)


# ====================================================================
#                           EXPERIMENT 14
# ====================================================================
def experiment_seq2seq_sdep_3():
    max_text_seqence_len = 100
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_sdep_3.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1547040526-model.h5"
    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        test_data = ev.get_test_dataset()

        values = []
        classes = []
        for data in test_data:
            values += [unidecode(str(q)[:max_text_seqence_len]) for q in data.quantiles]
            classes += [data]*len(data.quantiles)

        encoder_model = lm.load_seq2seq(model_path)
        print("Checkpoint not found. Calculating...")
        value_embedding = em.convert_to_vec_onehot(encoder_model, values, max_text_seqence_len, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding_by_avg(list(zip(classes, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)







# ====================================================================
#                           EXPERIMENT 16
# ====================================================================
def experiment_seq2seq_hierarchy_lstm():
    def preprocess_quantiles(quantiles, pad_maxlen):
        quantiles = map(str, quantiles)
        quantiles = map(str.strip, quantiles)
        quantiles = (x[::-1] for x in quantiles)
        quantiles = list(map(lambda x: [ord(y) for y in x], quantiles))
        quantiles = pad_sequences(quantiles, maxlen=pad_maxlen, truncating='pre', padding='pre')
        return quantiles

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_hierarchy_lstm.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/model.h5"
    emb_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_embedding_2/embedding_model.h5"

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20, train_size=0.7)

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        encoder_model = lm.load_hierarchy_lstm_model(model_path, emb_path)
        print("Checkpoint not found. Calculating...")
        test_profiles = ev.get_test_dataset()
        print(str(len(test_profiles)) + " classes!")

        quantiles = np.array(list(map(lambda x: preprocess_quantiles(x.quantiles, max_text_seqence_len), test_profiles)))
        embedding_vectors = encoder_model.predict(quantiles)

        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        class_embedding = list(zip(test_profiles, embedding_vectors))
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)




# ====================================================================
#                           EXPERIMENT 18
# This experiment testing non-training lstm hierarchy model in testing
# data from s3.
# ====================================================================
def experiment_seq2seq_hierarchy_lstm_trained(recompute):
    max_index = 0
    def preprocess_quantiles(quantiles, pad_maxlen):
        quantiles = map(str, quantiles)
        quantiles = map(str.strip, quantiles)
        quantiles = (x[::-1] for x in quantiles)
        quantiles = list(map(lambda x: [ord(y) for y in x], quantiles))
        quantiles = pad_sequences(quantiles, maxlen=pad_maxlen, truncating='pre', padding='pre')
        return quantiles

    max_text_seqence_len = 64
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_hierarchy_lstm_trained.__name__

    print("Experiment " + experiment_name + " running ...")
    ev = sdep.AuthorityEvaluator(username='andrej', neighbors=100, radius=20)
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/gru_hierarchical1548766779/model.h5"

    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name) and not recompute:
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        encoder_model = lm.load_hierarchy_model(model_path)
        # encoder_model = lm.load_hierarchy_seq2seq_convolution_model(model_path)

        print("Checkpoint not found. Calculating...")

        test_data = ev.get_test_dataset()

        quantiles_data = np.array(list(map(lambda x: preprocess_quantiles(x.quantiles, max_text_seqence_len),
                                           test_data)))
        embedding_vectors = encoder_model.predict(quantiles_data)

        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        class_embedding = list(zip(test_data, embedding_vectors))
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    ev.evaluate_embeddings(classes, embedding_vectors)



if __name__ == '__main__':
    experiment_gru_hierarchy_base_2()
