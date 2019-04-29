import os
import sh
import json
import numpy as np
from abc import abstractmethod
from keras.utils import plot_model
from keras.models import Model
from keras.models import load_model


class ComputingModel(object):
    DATA_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/s3+cvut_data.csv"

    GPT2_CONFIG_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'hparams.json')
    GPT2_CHECKPOINT_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'model.ckpt')
    GPT2_ENCODER_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'encoder.json')
    GPT2_VOCAB_PATH = os.path.join(os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M', 'vocab.bpe')
    OUTPUT_ROOT = f"{os.environ['PYTHONPATH'].split(':')[0]}/outcome"

    def __init__(self, output_path):
        self.output_path = output_path

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def load_encoder(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def get_encoder(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    def print_training_stats(self):
        print(self.output_path)
        try:
            with open(f"{self.output_path}/training_hist.json") as json_file:
                data = json.load(json_file)
                print({
                    "epoch": len(data["times"]),
                    "epoch_time": np.mean(data["times"]),
                })
                print(sh.tail("-n 1", f"{self.output_path}/results.txt", _iter=True))

        except:
            pass

    def make_plots(self):
        plot_model(self.load_encoder(), to_file=f"{self.output_path}/encoder_model.png", show_shapes=True)
        model: Model = self.load_model()
        plot_model(model, to_file=f"{self.output_path}/model.png", show_shapes=True)
        plot_model(model.layers[2].layer, to_file=f"{self.output_path}/model_inside.png", show_shapes=True)










