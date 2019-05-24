import os
import json
import numpy as np

from keras import Model
from keras.callbacks import EarlyStopping, TensorBoard
from keras.preprocessing.sequence import pad_sequences

from abc import abstractmethod
from typing import List, Tuple

import preprocessor.embedder as em
import custom_components as cc
from evaluation import AuthorityEvaluator
from ..computing_model import ComputingModel
from preprocessor.encoder import Encoder


class Seq2seq(ComputingModel):
    OUTPUT_ROOT = f"{ComputingModel.OUTPUT_ROOT}/seq2seq"

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

        # Preprocess data
        input_coder, input_decoder, target = self.preprocess_data_for_training()

        # input_coder = input_coder[:100]
        # input_decoder = input_decoder[:100]
        # target = target[:100]

        time_callback = cc.TimeHistory()
        hist = model.fit(x=[input_coder, input_decoder, target],
                         y=target,
                         epochs=300,
                         batch_size=64,
                         validation_split=0.3,
                         verbose=1,
                         callbacks=[
                             cc.ModelCheckpointMLEngine(self.output_path + "/model.h5", monitor='val_loss',
                                                        verbose=1, save_best_only=True, mode='min'),
                             EarlyStopping(monitor='val_loss', patience=6, verbose=1),
                             TensorBoard(log_dir=self.output_path + '/training_log', write_graph=True),
                             time_callback
                         ])
        hist.history['times'] = time_callback.times

        string_list = []
        model.summary(print_fn=lambda x: string_list.append(x))
        hist.history['model_topology'] = "\n".join(string_list)

        with open(self.output_path+'/training_hist.json', 'w') as f:
            json.dump(hist.history, f)

        # self.evaluate_model()

    def evaluate_model(self):
        model = self.load_encoder()

        ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50,
                                results_file=self.output_path)
        test_profiles: List = ev.cvut_profiles
        # -------------- COMPUTING EXPERIMENT BODY --------------------
        uid_values: List[Tuple[Tuple, str]] = [(profile.uid, value) for profile in test_profiles
                                               for value in profile.quantiles]
        nn_input: np.array = self.preprocess_profiles(test_profiles)

        nn_input = np.reshape(nn_input, (nn_input.shape[0] * nn_input.shape[1], nn_input.shape[2]))

        nn_output = model.predict(nn_input, verbose=1)

        print("Clustering value vectors to column representation")
        uids = list(map(lambda x: x[0], uid_values))
        uid_embedding = em.create_column_embedding_by(list(zip(uids, nn_output)), ag_method="mean")
        uid_profile_index = dict(map(lambda profile: (profile.uid, profile), test_profiles))
        profile_embedding = [(uid_profile_index[uid], embedding) for uid, embedding in uid_embedding]

        # -------------- EVALUATE EXPERIMENT --------------------
        print("Count of classes: " + str(len(uid_embedding)))
        profile, embedding = zip(*profile_embedding)
        ev.evaluate_embeddings(profile, embedding)

    def preprocess_profiles(self, profiles):
        result = []
        for profile in profiles:
            if len(profile.quantiles) != 11:
                continue

            values = map(lambda x: str(x)[:self.max_seq_len], profile.quantiles)
            values = map(str.strip, values)
            values = [self.encoder.encode(value)[:self.max_seq_len] for value in values]
            values = (x[::-1] for x in values)
            values = pad_sequences(list(values), maxlen=self.max_seq_len, truncating='pre', padding='pre')
            result.append(values)
        return np.array(result)

    def preprocess_data_for_training(self):
        ev = AuthorityEvaluator(username='andrej', neighbors=20, train_size=0.50, results_file=self.output_path)
        training_profile = ev.s3_profiles
        training_values = [value for profile in training_profile for value in profile.quantiles]
        training_values = map(str, training_values)
        training_values = list(map(str.strip, training_values))
        training_values = list(set(training_values))

        print(f"Training on {len(training_profile)} profiles => {len(training_values)} values!")

        embedded_strings = [self.encoder.encode(text)[:self.max_seq_len - 1] for text in training_values]
        input_coder = pad_sequences(list(map(lambda x: x[::-1], embedded_strings)), maxlen=self.max_seq_len, padding='pre')
        input_decoder = np.roll(pad_sequences(embedded_strings, maxlen=self.max_seq_len, padding='post'), 1)
        target = pad_sequences(embedded_strings, maxlen=self.max_seq_len, padding='post')
        return input_coder, input_decoder, target

    def get_encoder(self):
        return self.encoder
