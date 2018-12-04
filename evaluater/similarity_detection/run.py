
import os
import pickle
import numpy as np
from unidecode import unidecode

from evaluater.load_models import load_seq2seq_siamese
from evaluater.embedder import convert_to_vec, create_column_embedding
from evaluater.computing import compute_neighbors
from evaluater.similarity_detection.evaluation import evaluate_similarity_index
from data_preparation.cvut import CvutDataset
from evaluater.key_analyzer.profile_class import Profile

MODEL_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq_siamese1543504392-model.h5"
DATA_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvutProfiles_gnumbers.csv"
CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + \
                  "/evaluater/similarity_detection/pickles/runtime_checkpoint.pickle"

def run_evaluation():
    dataclass = CvutDataset()
    data = dataclass.load_for_similarity_case()
    data = data.applymap(lambda x: unidecode(str(x))[:63])

    encoder_model = load_seq2seq_siamese(MODEL_PATH)

    if os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint found ...")
        class_embedding = pickle.load(open(CHECKPOINT_PATH, "rb"))
    else:
        print("Checkpoint not found. Converting values to vectors...")
        value_embedding = convert_to_vec(encoder_model, data["value"].values, 64, 95)
        print("Processed " + str(len(value_embedding)) + " value embeddings")
        class_embedding = create_column_embedding(list(zip(data["type"].values, value_embedding)))
        print("Column embedding calculated.")
        pickle.dump(class_embedding, open(CHECKPOINT_PATH, "wb"))

    print("Finishing pipeline ... ")
    classes, embedding_vectors = zip(*class_embedding)
    similarity_index = compute_neighbors(np.array(embedding_vectors), np.array(classes),
                                       n_neighbors=200, radius=0.15, mode="radius+kneighbors")
    return evaluate_similarity_index(similarity_index)


if __name__ == '__main__':
    states = run_evaluation()
    print(states)

