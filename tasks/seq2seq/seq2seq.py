import os
import numpy as np
import pandas as pd
import tensorflow as tf

from typing import List
from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences

from abc import abstractmethod
from typing import List, Tuple

import evaluater.embedder as em
import trainer.custom_components as cc
from sdep import AuthorityEvaluator, Profile   # Needed
from ..computing_model import ComputingModel


class Seq2seq(ComputingModel):

    def __init__(self, encoder, max_seq_len):
        self.encoder = encoder
        self.max_seq_len = max_seq_len

    @abstractmethod
    def get_output_space(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def load_encoder(self):
        pass

    def train_model(self):
        model: Model = self.build_model()
        string_values = pd.read_csv(tf.gfile.Open(self.DATA_PATH))['value'].values

        # Preprocess data
        input_coder, input_decoder, target = self.preprocess_data_for_training(string_values, 64)

        model.fit(x=[input_coder, input_decoder, target],
                  y=target,
                  epochs=500,
                  batch_size=64,
                  validation_split=0.3,
                  callbacks=[
                      cc.ModelCheckpointMLEngine(self.get_output_space() + "/model.h5", monitor='val_loss', verbose=1,
                                                 save_best_only=True, mode='min'),
                      EarlyStopping(monitor='val_loss', patience=6, verbose=1),
                      TensorBoard(log_dir=self.get_output_space() + '/training_log', write_graph=True, embeddings_freq=0)
                  ])
        self.evaluate_model()

    def evaluate_model(self):
        print("Experiment FIX GPT2 encoder running ...")

        ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50,
                                results_file=self.get_output_space())
        test_profiles: List[Profile] = ev.get_test_dataset("s3")

        # -------------- COMPUTING EXPERIMENT BODY --------------------
        uid_values: List[Tuple[Tuple, str]] = [(profile.uid, value) for profile in test_profiles
                                               for value in profile.quantiles]
        nn_input: np.array = self.preprocess_profiles(test_profiles)
        nn_input = np.reshape(nn_input, (nn_input.shape[0] * nn_input.shape[1], nn_input.shape[2]))

        nn_output = self.load_encoder().predict(nn_input, verbose=1)

        print("Clustering value vectors to column representation")
        uids = list(map(lambda x: x[0], uid_values))
        uid_embedding = em.create_column_embedding_by(list(zip(uids, nn_output)), ag_method="mean")
        uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
        profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

        # -------------- EVALUATE EXPERIMENT --------------------
        print("Count of classes: " + str(len(uid_embedding)))
        ev.evaluate_embeddings(profile_embedding)

    def preprocess_profiles(self, profiles: List[Profile]):
        result = []
        for profile in profiles:
            values = map(lambda x: str(x)[:self.max_seq_len], profile.quantiles)
            values = map(str.strip, values)
            values = [self.encoder.encode(value)[:self.max_seq_len] for value in values]
            values = (x[::-1] for x in values)
            values = pad_sequences(list(values), maxlen=self.max_seq_len, truncating='pre', padding='pre')
            result.append(values)
        return np.array(result)

    def preprocess_data_for_training(self, row_src: List[str], max_seq_len: int):
        self.max_seq_len = max_seq_len
        strings = map(str, row_src)
        strings = list(map(str.strip, strings))
        strings = list(set(strings))

        embedded_strings = [self.encoder.encode(text)[:max_seq_len - 1] for text in strings]

        input_coder = pad_sequences(list(map(lambda x: x[::-1], embedded_strings)), maxlen=max_seq_len, padding='pre')
        input_decoder = pad_sequences(embedded_strings, maxlen=max_seq_len, padding='post')
        input_decoder = np.roll(input_decoder, 1)
        target = pad_sequences(embedded_strings, maxlen=max_seq_len, padding='post')
        return input_coder, input_decoder, target
