import os
import pickle
import numpy as np
from unidecode import unidecode

from evaluater.computing import compute_neighbors
from evaluater.similarity_detection.evaluation import evaluate_similarity_index
import data_preparation.cvut as dt
import evaluater.load_models as lm
import evaluater.embedder as em
from evaluater.key_analyzer.profile_class import Profile

CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/similarity_detection/pickles/"

def print_similar_domain(similarity_index, dataclass):
    for column_name_base, similar in similarity_index.items():
        tmp1 = list(filter(lambda x: x[1] == column_name_base, dataclass.df.values))[:5]

        max_len = max(list(map(lambda x: len(str(x[0])), tmp1)))
        if max_len > 30 or len(similar) == 0:
            continue
        print("======================================================")
        print(str(list(map(lambda x: x[0], tmp1))))
        print("----->")
        for column_name in similar[:5]:
            if len(similar) == 0:
                continue
            tmp2 = list(filter(lambda x: x[1] == column_name, dataclass.df.values))[:5]
            print(list(map(lambda x: x[0], tmp2)))


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
        value_embedding = em.convert_to_vec(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=250, radius=0.15,
                                         mode="radius+kneighbors")
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
        value_embedding = em.convert_to_vec(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.1,
                                         mode="radius+kneighbors")

    # dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
    # print_similar_domain(similarity_index, dataclass)
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
        value_embedding = em.convert_to_vec(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    # dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
    # print_similar_domain(similarity_index, dataclass)
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
        value_embedding = em.convert_to_vec(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=100, radius=0.3, mode="radius+kneighbors")

    # dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
    # print_similar_domain(similarity_index, dataclass)
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
        value_embedding = em.convert_to_vec(encoder_model, dataclass.df["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = em.create_column_embedding(list(zip(dataclass.df["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes), n_neighbors=100, radius=0.3,
                                         mode="radius+kneighbors")

    print_similar_domain(similarity_index, dt.CvutDataset(dt.SelectData.profile_similarity_basic))
    index = evaluate_similarity_index(similarity_index)
    return index
