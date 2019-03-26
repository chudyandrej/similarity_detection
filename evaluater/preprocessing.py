from keras.preprocessing.sequence import pad_sequences
from typing import List, Optional, Tuple
from keras_gpt_2 import load_trained_model_from_checkpoint, get_bpe_from_files, generate


import numpy as np
import pickle
import os


# Options for load GPT2
model_folder = os.environ['PYTHONPATH'].split(":")[0] + '/data/models/117M'
config_path = os.path.join(model_folder, 'hparams.json')
checkpoint_path = os.path.join(model_folder, 'model.ckpt')
encoder_path = os.path.join(model_folder, 'encoder.json')
vocab_path = os.path.join(model_folder, 'vocab.bpe')


def preprocess_values_standard(values: List[str], pad_maxlen: int, char_index: dict = None):
    """
    Standard preprocess pipeline into encoder. Cut to max size, revert,
    tokenizing by index or ascii value and padding.
    :param values: (Array) String values
    :param pad_maxlen: (Number)Maximal length of sequence
    :param char_index: (Dict)(optional) Index for convert chars to tokens
    :return: (Array) Preprocessed values prepare for input to neural network
    """
    values = map(lambda x: str(x)[:pad_maxlen], values)
    # values = map(str.strip, values)
    values = (x[::-1] for x in values)
    if char_index is None:
        values = list(map(lambda x: [ord(y) for y in x], values))
    else:
        values = map(lambda x: [char_index[y] for y in x], values)
    values = pad_sequences(list(values), maxlen=pad_maxlen, truncating='pre', padding='pre')
    return values


