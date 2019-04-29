import os
import json
import numpy as np
from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Bidirectional, CuDNNGRU, LSTM

from keras.preprocessing.sequence import pad_sequences

from abc import abstractmethod
from typing import List, Tuple

import custom_components as cc
from evaluation import AuthorityEvaluator, pairs_generator
from ..computing_model import ComputingModel
from preprocessor.encoder import Encoder


class Siamese(ComputingModel):
    OUTPUT_ROOT = f"{ComputingModel.OUTPUT_ROOT}/siamese"

    def __init__(self, encoder: Encoder, max_seq_len, output_path):

        self.encoder = encoder
        self.max_seq_len = max_seq_len
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        super().__init__(self.output_path)

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def load_encoder(self):
        pass

    def train_model(self):
        model: Model = self.build_model()

        ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.99, valid_size=0.2,
                                results_file=self.output_path)
        train_profiles, valid_profiles = ev.split_profiles(ev.s3_profiles, train_size=0.8)

        [left, right], labels = next(pairs_generator(profiles=valid_profiles, batch_size=20000, preprocess_profile=self.preprocess_profile))

        time_callback = cc.TimeHistory()
        hist = model.fit_generator(generator=pairs_generator(train_profiles, 128, self.preprocess_profile),
                                   steps_per_epoch=1000,
                                   epochs=200,
                                   validation_data=([left, right], labels),
                                   callbacks=[
                                       cc.ModelCheckpointMLEngine(self.output_path + "/model.h5", monitor='val_loss',
                                                                  verbose=1,
                                                                  save_best_only=True, mode='min'),
                                       EarlyStopping(monitor='val_loss', patience=8, verbose=1),
                                       TensorBoard(log_dir=self.output_path + '/training_log', write_graph=True),
                                       time_callback

                                   ])

        hist.history['times'] = time_callback.times

        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        hist.history['model_topology'] = "\n".join(string_list)

        with open(self.output_path + '/training_hist.json', 'w') as f:
            json.dump(hist.history, f)

    def evaluate_model(self):
        encoder_model = self.load_encoder()

        ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean", results_file=self.output_path)
        test_profiles = ev.cvut_profiles
        quantiles_data = np.array(list(map(lambda x: self.preprocess_profile(x), test_profiles)))
        embedding_vectors = encoder_model.predict(quantiles_data, verbose=1)
        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        ev.evaluate_embeddings(test_profiles, embedding_vectors)

    def preprocess_profile(self, profile):
        values = profile.quantiles
        values = map(str, values)
        values = map(str.strip, values)
        values = map(lambda x: x[::-1], values)
        values = list(map(lambda x: self.encoder.encode(x), values))
        values = pad_sequences(values, maxlen=self.max_seq_len)
        return np.array(values)

    def get_encoder(self):
        return self.encoder

    @staticmethod
    def get_rnn(rnn_type, rnn_dim, dropout, return_sequences):
        if rnn_type == "Gru":
            return Bidirectional(CuDNNGRU(units=rnn_dim, return_sequences=return_sequences))
        elif rnn_type == "Lstm":
            return Bidirectional(LSTM(units=rnn_dim, return_sequences=return_sequences, recurrent_dropout=dropout))
        else:
            raise ValueError('Unknown type of network!')


