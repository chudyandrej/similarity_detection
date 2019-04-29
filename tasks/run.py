import os

from .seq2seq import *

from .gpt2 import GPT2
from preprocessor.encoder import *

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def run():
    model = GPT2(version="v1")
    model.evaluate_model()


if __name__ == '__main__':
    run()
