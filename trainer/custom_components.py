import keras
from keras import backend as K
from keras import objectives
from keras.layers import Lambda, Bidirectional, CuDNNGRU, GRU
from keras.layers import Layer
from keras import initializers, regularizers, constraints
from typing import List, Optional, Tuple
import numpy as np
from keras.callbacks import *
from evaluater.preprocessing import preprocess_values_standard
import sdep

import keras
import warnings
import numpy as np
from tensorflow.python.lib.io import file_io





##################################################
#             FIT GENERATORS
##################################################
def fit_generator_profile_pairs(profiles: List[sdep.Profile],
                                max_text_sequence_len: int,
                                batch_size: int,
                                neg_ratio: int = 2,
                                get_raw_profiles=True):
    """
    Fit generator for generate similar and unsimilar pairs for training of siamese model with similarity distance
    optimization
    :param profiles: List of profiles
    :param max_text_sequence_len: Max sequence len
    :param batch_size:
    :param neg_ratio: default 2 = equal positive and  negative, 4 => 1:4 POS: NEG
    """
    while True:
        left, right, label, weights = next(sdep.pairs_generator(profiles, batch_size, neg_ratio=neg_ratio))

        left = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, max_text_sequence_len), left)))
        right = np.array(list(map(lambda x: preprocess_values_standard(x.quantiles, max_text_sequence_len), right)))

        yield [left, right], label


def fit_generator_profile_pairs_with_dict(profiles: List[sdep.Profile],
                                          profile_vector_dict: dict,
                                          batch_size: int,
                                          neg_ratio: int = 2):
    """
    Fit generator for generate similar and unsimilar pairs for training of siamese model with similarity distance
    optimization
    :param profiles: List of profiles
    :param profile_vector_dict:
    :param batch_size:
    :param neg_ratio: default 2 = equal positive and  negative, 4 => 1:4 POS: NEG
    """
    profile_generator = sdep.pairs_generator(profiles, batch_size, neg_ratio=neg_ratio)
    while True:
        left, right, label, weights = next(profile_generator)
        left = np.array([profile_vector_dict[prof] for prof in left])
        right = np.array([profile_vector_dict[prof] for prof in right])

        yield [left, right], label


##################################################
#             SIMILARITY FUNCTIONS
##################################################




##################################################
#             LOSS FUNCTIONS
##################################################



