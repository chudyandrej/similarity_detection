
import os
import pickle
import argparse
import numpy as np
from unidecode import unidecode

import evaluater.load_models as model_loader
from evaluater.embedder import convert_to_vec, create_column_embedding
from evaluater.computing import compute_neighbors
from evaluater.similarity_detection.evaluation import evaluate_similarity_index
import data_preparation.cvut as dt
from evaluater.key_analyzer.profile_class import Profile

CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + \
                  "/evaluater/similarity_detection/pickles/"


def compute_embedding_pipeline(experiment_name, dataclass, encoder_model):
    print("Checkpoint not found ... Calculating...")
    data = dataclass.df
    data.value = data.value.map(lambda x: unidecode(str(x))[:63])

    print("Checkpoint not found. Converting values to vectors...")
    value_embedding = convert_to_vec(encoder_model, data["value"].values, 64, 95)
    print("Processed " + str(len(value_embedding)) + " value embeddings")
    class_embedding = create_column_embedding(list(zip(data["type"].values, value_embedding)))
    print("Column embedding calculated.")
    pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))
    return class_embedding


def print_similar_domain(similarity_index, dataclass):
    for column_name_base, similar in similarity_index.items():
        tmp1 = list(filter(lambda x: x[1] == column_name_base, dataclass.df.values))[:5]
        for column_name in similar:
            if len(similar) == 0:
                continue

            tmp2 = list(filter(lambda x: x[1] == column_name, dataclass.df.values))[:5]
            print(list(map(lambda x: x[0], tmp1)))
            print(list(map(lambda x: x[0], tmp2)))
            print()


def experiment_seq2seq_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] +\
                 "/data/models/seq2seq_siamese1543504392-model.h5"
    print("Experiment " + experiment_name + " running ...")
    # -------------- COMPUTING EXPERIMENT BODY --------------------
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
        encoder_model = model_loader.load_seq2_siamese(model_path)
        class_embedding = compute_embedding_pipeline(experiment_name, dataclass, encoder_model)

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=250, radius=0.15, mode="radius+kneighbors")
    return evaluate_similarity_index(similarity_index)


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
        encoder_model = model_loader.load_seq2_siamese(model_path)
        class_embedding = compute_embedding_pipeline(experiment_name, dataclass, encoder_model)

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=100, radius=0.3, mode="radius+kneighbors")

    # dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
    # print_similar_domain(similarity_index, dataclass)
    return evaluate_similarity_index(similarity_index)


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
        encoder_model = model_loader.load_seq2seq(model_path)
        class_embedding = compute_embedding_pipeline(experiment_name, dataclass, encoder_model)

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    print("Count of classes: " + str(len(set(classes))))
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=100, radius=0.3, mode="radius+kneighbors")

    # dataclass = dt.CvutDataset(dt.SelectData.profile_similarity_basic)
    # print_similar_domain(similarity_index, dataclass)
    return evaluate_similarity_index(similarity_index)


if __name__ == '__main__':
    states = {}
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',
                        required=True,
                        type=int,
                        help='Experiment selection')
    parse_args, _ = parser.parse_known_args()
    if parse_args.exp == 1:
        states = experiment_seq2seq_siamese()
    elif parse_args.exp == 2:
        states = experiment_seq2_siamese()
    elif parse_args.exp == 3:
        states = experiment_seq2seq()
    else:
        print("Bad experiment code")
    print(states)

