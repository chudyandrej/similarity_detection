
import os
import pickle
import numpy as np
import pandas as pd
from unidecode import unidecode

from evaluater.load_models import load_seq2seq_siamese
from evaluater.embedder import convert_to_vec, create_column_embedding, evaluate_neighbors
from data_preparation.cvut import CvutDataset

MODEL_PATH = "../../data/models/seq2seq_siamese1543504392-model.h5"
DATA_PATH = "../../data/cvutProfiles_gnumbers.csv"


def run_evaluation():
    dataclass = CvutDataset()
    data = dataclass.load_for_similarity_case()
    data = data.applymap(lambda x: unidecode(str(x))[:63])

    encoder_model = load_seq2seq_siamese(MODEL_PATH)

    PATH = "./pickles/runtume_checkpoint.pickle"
    if os.path.exists(PATH):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(PATH, "rb"))
    else:
        print("Checkpoint not found calculating ...")
        value_embedding = convert_to_vec(encoder_model, data["value"].values, 64, 95)
        class_embedding = create_column_embedding(zip(data["type"].values, value_embedding))
        pickle.dump(class_embedding, open(PATH, "wb"))

    classes, embedding_vectors = zip(*class_embedding)
    candFK_candPK = evaluate_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=200, radius=0.15, mode="radius+kneighbors")

