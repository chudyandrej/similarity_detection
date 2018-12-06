import os
import pickle
import argparse
import numpy as np
from unidecode import unidecode


from evaluater.embedder import convert_to_vec, create_column_embedding
from evaluater.load_models import load_seq2seq_siamese, load_seq2_siamese, load_seq2seq
from evaluater.key_analyzer.evaluation import evaluate_stats
from evaluater.computing import compute_neighbors
import data_preparation.cvut as dataset
from evaluater.key_analyzer.profile_class import Profile

PROF_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/profiles.pkl"
PK_FK_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/foreign_keys.pkl"
CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/key_analyzer/pickles/"


def compute_embedding_pipeline(experiment_name, dataclass, encoder_model):

    print("Checkpoint not found ... Calculating...")
    data = dataclass.df
    data.value = data.value.map(lambda x: unidecode(str(x))[:63])

    value_embedding = convert_to_vec(encoder_model, data["value"].values, 64, 95)
    print("Vectoring success!")
    class_embedding = create_column_embedding(list(zip(data["type"].values, value_embedding)))
    pickle.dump(class_embedding, open(CHECKPOINT_PATH + experiment_name, "wb"))
    print("Saved!")
    return class_embedding


def experiment_seq2seq_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
    encoder_model = load_seq2seq_siamese(model_path)
    data_object = dataset.CvutDataset(dataset.SelectData.load_key_analyzer)

    print("Loading testing data Success!")
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found :)")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        class_embedding = compute_embedding_pipeline(experiment_name, data_object, encoder_model)

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    candFK_candPK = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=200, radius=0.15, mode="radius+kneighbors")
    return evaluate_stats(candFK_candPK, data_object.pk_fk, data_object.profiles)


def experiment_seq2_siamese():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2_siamese.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2_siamese1543913096-model.h5"
    encoder_model = load_seq2_siamese(model_path)
    data_object = dataset.CvutDataset(dataset.SelectData.load_key_analyzer)

    print("Loading testing data Success!")
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found :)")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        class_embedding = compute_embedding_pipeline(experiment_name, data_object, encoder_model)

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    candFK_candPK = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=200, radius=0.15, mode="radius+kneighbors")
    return evaluate_stats(candFK_candPK, data_object.pk_fk, data_object.profiles)


def experiment_seq2seq():
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"
    encoder_model = load_seq2seq(model_path)
    data_object = dataset.CvutDataset(dataset.SelectData.load_key_analyzer)

    print("Loading testing data Success!")
    if os.path.exists(CHECKPOINT_PATH+experiment_name):
        print("Checkpoint found :)")
        class_embedding = pickle.load(open(CHECKPOINT_PATH+experiment_name, "rb"))
    else:
        class_embedding = compute_embedding_pipeline(experiment_name, data_object, encoder_model)

    # -------------- EVALUATE EXPERIMENT --------------------
    classes, embedding_vectors = zip(*class_embedding)
    candFK_candPK = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=200, radius=0.15, mode="radius+kneighbors")
    return evaluate_stats(candFK_candPK, data_object.pk_fk, data_object.profiles)


if __name__ == "__main__":
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
