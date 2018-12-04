import os
import pickle
import numpy as np


from evaluater.embedder import convert_to_vec, create_column_embedding, evaluate_neighbors
from evaluater.load_models import load_seq2seq_siamese
from evaluater.key_analyzer.data_analytics import load_dataset
from evaluater.key_analyzer.evaluation import evaluate_stats
from evaluater.key_analyzer.profile_class import Profile

MODEL_PATH = "../../data/models/seq2seq_siamese1543504392-model.h5"
PROF_PATH = "./pickles/profiles.pkl"
PK_FK_PATH = "./pickles/foreign_keys.pkl"


def run_evaluation():
    encoder_model = load_seq2seq_siamese(MODEL_PATH)
    data, fk_pk = load_dataset(PROF_PATH, PK_FK_PATH, full_load=False)
    print("Loading testing data Success!")

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
    return evaluate_stats(candFK_candPK, fk_pk)


if __name__ == "__main__":
    result = run_evaluation()
    print(result)
