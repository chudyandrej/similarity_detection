import os
from abc import abstractmethod


class ComputingModel(object):
    DATA_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/s3+cvut_data.csv"

    GPT2_CONFIG_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'hparams.json')
    GPT2_CHECKPOINT_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'model.ckpt')
    GPT2_ENCODER_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'encoder.json')
    GPT2_VOCAB_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'vocab.bpe')
    OUTPUT_ROOT = f"{os.environ['PYTHONPATH'].split(':')[0]}/outcome/seq2seq"

    @abstractmethod
    def get_output_space(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def load_encoder(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass
