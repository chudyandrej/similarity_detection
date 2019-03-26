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



def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


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

def l1_similarity(x):
    return K.exp(-1 * K.sum(K.abs(x[0] - x[1]), axis=-1, keepdims=True))


def l2_similarity(x):
    return K.exp(-1 * K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True)))


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


##################################################
#             LOSS FUNCTIONS
##################################################

def l2_loss(y_true, y_pred):
    return K.mean(y_true * K.square(1 - y_pred) + (1 - y_true) * K.square(y_pred))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 5
    sqaure_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * sqaure_pred + (1 - y_true) * margin_square)


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


def mean_squared_error_from_pred(y_true, y_pred):
    y_true = y_pred[:, 0]
    y_pred = y_pred[:, 1]
    return objectives.mean_squared_error(y_true, y_pred)


def categorical_crossentropy_form_pred(y_true, y_pred):
    y_true = y_pred[:, 0]
    y_pred = y_pred[:, 1]
    return objectives.categorical_crossentropy(y_true, y_pred)


##################################################
#                 LAYERS
##################################################

class CustomRegularization(Layer):
    def __init__(self, loss_function="mean_squared_error", **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)
        self.loss_function = loss_function

    def call(self, x, mask=None):
        target = x[0]
        pred = x[1]
        tmp = 0
        if self.loss_function == 'categorical_crossentropy':
            tmp = objectives.categorical_crossentropy(target, pred)
        elif self.loss_function == 'mean_squared_error':
            tmp = objectives.mean_squared_error(target, pred)

        loss = K.sum(tmp)
        self.add_loss(loss, x)
        return tmp

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)


def OneHot(input_dim=None, input_length=None):
    # Check if inputs were supplied correctly
    if input_dim is None or input_length is None:
        raise TypeError("input_dim or input_length is not set")

    # Helper method (not inlined for clarity)
    def _one_hot(x, num_classes):
        return K.one_hot(K.cast(x, 'uint8'),
                          num_classes=num_classes)

    # Final layer representation as a Lambda layer
    return Lambda(_one_hot,
                  arguments={'num_classes': input_dim},
                  input_shape=(input_length,))


def bidir_gru(my_seq, n_units, in_GPU):
    '''
    just a convenient wrapper for bidirectional RNN with GRU units
    # regardless of whether training is done on GPU, can be loaded on CPU
    # see: https://github.com/keras-team/keras/pull/9112
    '''
    if in_GPU:
        return Bidirectional(CuDNNGRU(units=n_units, return_sequences=True), merge_mode='concat', weights=None)(my_seq)
    else:
        return Bidirectional(GRU(units=n_units, activation='tanh', dropout=0.0, recurrent_dropout=0.0, implementation=1,
                                 return_sequences=True, reset_after=True, recurrent_activation='sigmoid'),
                             merge_mode='concat', weights=None)(my_seq)


# this script was taken from https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, return_coefficients=False,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.return_coefficients = return_coefficients
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a

        if self.return_coefficients:
            return [K.sum(weighted_input, axis=1), a]
        else:
            return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[-1], 1)]
        else:
            return input_shape[0], input_shape[-1]


class EmbeddingRet(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return [
            super(EmbeddingRet, self).compute_output_shape(input_shape),
            (self.input_dim, self.output_dim),
        ]

    def compute_mask(self, inputs, mask=None):
        return [
            super(EmbeddingRet, self).compute_mask(inputs, mask),
            None,
        ]

    def call(self, inputs):
        return [
            super(EmbeddingRet, self).call(inputs),
            self.embeddings,
        ]

##################################################
#                 CALLBACKS
##################################################


class ModelCheckpointMLEngine(keras.callbacks.Callback):
    """Save the model after every epoch.
    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).
    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.
    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpointMLEngine, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            if self.filepath.startswith('gs://'):
                                print("Gcloud_save")
                                save_model_to_cloud(self.model, self.filepath)
                            else:
                                self.model.save(filepath, overwrite=True)

                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    if self.filepath.startswith('gs://'):
                        print("Gcloud_save")
                        save_model_to_cloud(self.model, self.filepath)
                    else:
                        self.model.save(filepath, overwrite=True)


def save_model_to_cloud(model, filepath):
    filename = filepath.split("/")[-1]
    print(filename)
    model.save(filename)
    with file_io.FileIO(filename, mode='rb') as inputFile:
        with file_io.FileIO(filepath, mode='wb+') as outFile:
            outFile.write(inputFile.read())
