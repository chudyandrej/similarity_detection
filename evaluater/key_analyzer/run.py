import os
import pickle
import argparse
import numpy as np

from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences

from unidecode import unidecode
import trainer.custom_components as cc
import evaluater.embedder as em
import data_preparation.cvut as dataset


PROF_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/profiles.pkl"
PK_FK_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/foreign_keys.pkl"
CHECKPOINT_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/evaluater/key_analyzer/pickles/"


def preprocess_values(values, pad_maxlen, full_unicode=True):
    values = map(lambda x: str(x)[:pad_maxlen], values)
    values = map(str.strip, values)
    values = (x[::-1] for x in values)
    if full_unicode:
        values = list(map(lambda x: [ord(y) for y in x], values))
    else:
        # TODO
        values = map(lambda x: unidecode(x), values)
        # values = map(lambda x: [tokenizer_0_96(y) for y in x], values)
    # print(list(values)[:100])
    values = pad_sequences(list(values), maxlen=pad_maxlen, truncating='pre', padding='pre')
    return values


def experiment_seq2seq_gru():
    def load_h5(path):
        model = load_model(path, custom_objects={
            "CustomRegularization": cc.CustomRegularization,
            "zero_loss": cc.zero_loss
        })
        encoder = Model(model.inputs[0], model.layers[4].output[1])
        return encoder
    # -------------- SET PARAMETERS OF EXPERIMENT --------------------
    experiment_name = experiment_seq2seq_gru.__name__
    model_path = os.environ['PYTHONPATH'].split(":")[0] + "/data/models/seq2seq1544020916-model.h5"

    # -------------- LOAD DATA AND MODEL --------------------
    print("Experiment " + experiment_name + " running ...")
    data_object = dataset.CvutDataset(dataset.SelectData.load_key_analyzer)
    encoder_model = load_h5(model_path)
    encoder_model.summary()
    print("Model successfully loaded. ")

    # -------------- COMPUTING --------------------
    data = data_object.df
    values = preprocess_values(data['value'].values, 64)
    value_embeddings = encoder_model.predict(values)
    class_embeddings = data['type'].values
    print(str(len(value_embeddings)) + " values for activation.")
    print(str(len(class_embeddings)) + " classes for data.")
    class_embedding = em.create_column_embedding_by(list(zip(class_embeddings, value_embeddings)), "mean")

    # -------------- EVALUATE EXPERIMENT --------------------
    print("Count of classes: " + str(len(set(class_embedding))))
    return evaluate_stats(class_embedding, data_object.pk_fk, data_object.profiles)


    # -------------- EVALUATE EXPERIMENT --------------------




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
