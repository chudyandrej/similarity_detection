import os
import numpy as np

from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences

from abc import abstractmethod
from typing import List, Tuple

import trainer.custom_components as cc
from sdep import AuthorityEvaluator, Profile, pairs_generator   # Needed
from ..computing_model import ComputingModel


class Siamese(ComputingModel):

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

        ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.5, valid_size=0.2,
                                results_file=self.get_output_space())
        train_profiles, valid_profile = ev.get_train_dataset(data_src="s3")

        [left_val, right_val], label_val = next(self.fit_profile_generator(valid_profile, batch_size=50000))
        self.build_model().fit_generator(
            self.fit_profile_generator(train_profiles, batch_size=64),
            validation_data=([left_val, right_val], label_val),
            steps_per_epoch=200000//64,
            epochs=200,
            callbacks=[
                cc.ModelCheckpointMLEngine(self.get_output_space() + "/model.h5", monitor='val_loss', verbose=1,
                                           save_best_only=True, mode='min'),
                TensorBoard(log_dir=self.get_output_space() + '/training_log', write_graph=True, embeddings_freq=0)
            ])

        self.evaluate_model()

    def evaluate_model(self):
        ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean", train_size=0.5, valid_size=0.2,
                                results_file=self.get_output_space())

        test_prof = ev.get_test_dataset(data_src="s3")
        quantiles_data = np.array([self.preprocess_profiles(prof, pad_max_len=self.max_seq_len) for prof in test_prof])

        embedding_vectors = self.load_encoder().predict(quantiles_data, verbose=1)
        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        ev.evaluate_embeddings(list(zip(test_prof, embedding_vectors)))

    def preprocess_profiles(self, profiles: List[Profile], pad_max_len: int):
        values = map(lambda x: x.quantiles, profiles)
        values = [self.encoder.encode(value)[:pad_max_len] for value in values]
        values = map(lambda x: str(x)[:pad_max_len], values)
        values = map(str.strip, values)
        values = (x[::-1] for x in values)
        values = pad_sequences(list(values), maxlen=pad_max_len, truncating='pre', padding='pre')
        return values

    def fit_profile_generator(self, profiles: List[Profile], batch_size: int):

        profile_generator = pairs_generator(profiles, batch_size, self.max_seq_len)

        while True:
            lefts_profiles, rights_profiles, labels, weights = next(profile_generator)
            left = np.array([self.preprocess_profiles(prof, pad_max_len=self.max_seq_len) for prof in lefts_profiles])
            right = np.array([self.preprocess_profiles(prof, pad_max_len=self.max_seq_len) for prof in lefts_profiles])
            yield [left, right], labels
