import os
import pickle
import numpy as np
import regex as re
from keras_gpt_2 import load_trained_model_from_checkpoint
from evaluation import AuthorityEvaluator

from preprocessor.encoder.bpe import BytePairEncoding
from ..computing_model import ComputingModel


# Options for load GPT2
model_folder = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


class GPT2(ComputingModel):

    def __init__(self, version):
        self.version = version

        self.model = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        self.bpe = BytePairEncoding()
        self.embeddings = []
        self.output_space = f"{super().OUTPUT_ROOT}/pretrain/{type(self).__name__}/{self.version}"
        os.makedirs(self.output_space, exist_ok=True)
        super().__init__(self.output_space)


    def build_model(self):
        pass

    def load_encoder(self):
        pass

    def load_model(self):
        pass

    def train_model(self):
        pass

    def get_encoder(self):
        pass

    def evaluate_model(self):
        ev = AuthorityEvaluator(username='andrej', neighbors=20, metric="euclidean", results_file=self.output_path)
        test_profiles = ev.cvut_profiles[:10]
        embedding_vectors = self.calculate_embeddings(test_profiles)
        print("Processed " + str(len(embedding_vectors)) + " value embeddings")
        ev.evaluate_embeddings(list(zip(test_profiles, embedding_vectors)))

    def calculate_embeddings(self, profiles):
        result_embeddings = []

        quantiles_texts = list(map(lambda x: self.convert_quantile_to_text(x) , profiles))
        encodes = [self.bpe.encode(text) for text in quantiles_texts]
        text_lens = [len(encode) for encode in encodes]
        max_len = max(text_lens)
        input_data = [encode + [0] * (max_len - len(encode)) for encode in encodes]
        output_data = self.model.predict(np.array(input_data), verbose=1)
        print(output_data.shape)
        exit()
        for i in range(len(output_data)):
            result_embeddings.append(output_data[i][text_lens[i] - 1].copy())

        return result_embeddings

    def convert_quantile_to_text(self, prof_obj):
        text = ""
        for value in prof_obj.quantiles:
            value = str(value).replace("$", "€")
            text += value + " $ "
        return text


