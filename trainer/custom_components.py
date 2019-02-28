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
def fit_generator_profile_pairs(profiles: List[sdep.Profile], max_text_sequence_len: int, batch_size: int, neg_ratio: int = 2):
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


##################################################
#                 CALLBACKS
##################################################
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        # self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        # logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        # logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        # self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        # self.history.setdefault('iterations', []).append(self.trn_iterations)

        # for k, v in logs.items():
        #    self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


class CyclicMT(Callback):
    def __init__(self, base_mt=0.85, max_mt=0.95, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicMT, self).__init__()

        self.base_mt = base_mt
        self.max_mt = max_mt
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.cmt_iterations = 0.
        self.trn_iterations = 0.
        # self.history = {}

        self._reset()

    def _reset(self, new_base_mt=None, new_max_mt=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_mt != None:
            self.base_mt = new_base_mt
        if new_max_mt != None:
            self.max_mt = new_max_mt
        if new_step_size != None:
            self.step_size = new_step_size
        self.cmt_iterations = 0.

    def cmt(self):
        cycle = np.floor(1 + self.cmt_iterations / (2 * self.step_size))
        x = np.abs(self.cmt_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_mt + (-self.max_mt + self.base_mt) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_mt + (-self.max_mt + self.base_mt) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.cmt_iterations)

    def on_train_begin(self, logs={}):
        # logs = logs or {}

        if self.cmt_iterations == 0:
            K.set_value(self.model.optimizer.momentum, self.base_mt)
        else:
            K.set_value(self.model.optimizer.momentum, self.cmt())

    def on_batch_end(self, epoch, logs=None):

        # logs = logs or {}
        self.trn_iterations += 1
        self.cmt_iterations += 1

        # self.history.setdefault('mt', []).append(K.get_value(self.model.optimizer.momentum))
        # self.history.setdefault('iterations', []).append(self.trn_iterations)

        # for k, v in logs.items():
        #    self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.momentum, self.cmt())


class PerClassAccHistory(Callback):
    '''
    a note about the confusion matrix:
    Cij = nb of obs known to be in group i and predicted to be in group j. So:
    - the nb of right predictions is given by the diagonal
    - the total nb of observations for a group is given by summing the corresponding row
    - the total nb of predictions for a group is given by summing the corresponding col
    accuracy is (nb of correct preds)/(total nb of preds)
    # https://developers.google.com/machine-learning/crash-course/classification/accuracy
    '''

    def __init__(self, my_n_cats, my_rd, my_n_steps):
        super().__init__()
        self.my_n_cats = my_n_cats
        self.my_rd = my_rd
        self.my_n_steps = my_n_steps

    def on_train_begin(self, logs={}):
        self.per_class_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        cmat = np.zeros(shape=(self.my_n_cats, self.my_n_cats))
        for repeat in range(self.my_n_steps):
            docs, labels = self.my_rd.__next__()
            preds_floats = self.model.predict(docs)
            y_pred = np.argmax(np.array(preds_floats), axis=1)
            y_true = np.argmax(labels, axis=1)
            cmat = np.add(cmat, confusion_matrix(y_true, y_pred))
            if repeat % round(self.my_n_steps / 5) == 0:
                print(repeat)
        accs = list(np.round(1e2 * cmat.diagonal() / cmat.sum(axis=0), 2))
        self.per_class_accuracy.append(accs)


class TimeHistory(Callback):
    def on_train_begin(self, logs=None):
        self.times = []

    def on_epoch_begin(self, batch, logs=None):
        self.epoch_time_start = time()

    def on_epoch_end(self, batch, logs=None):
        self.times.append(round(time() - self.epoch_time_start, 2))


class LossHistory(Callback):
    '''
    records the average loss on the full *training* set so far
    the loss returned by logs is just that of the current batch!
    '''

    def on_train_begin(self, logs=None):
        self.losses = []
        self.loss_avg = []

    def on_batch_end(self, batch, logs=None):
        self.losses.append(round(float(logs.get('loss')), 6))
        self.loss_avg.append(round(np.mean(self.losses, dtype=np.float64), 6))

    def on_epoch_end(self, batch, logs={}):
        self.losses = []


class LRHistory(Callback):
    ''' records the current learning rate'''

    def on_train_begin(self, logs=None):
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        my_lr = K.eval(self.model.optimizer.lr)
        self.lrs.append(my_lr)


class MTHistory(Callback):
    ''' records the current momentum'''

    def on_train_begin(self, logs=None):
        self.mts = []

    def on_batch_end(self, batch, logs=None):
        my_mt = K.eval(self.model.optimizer.momentum)
        self.mts.append(my_mt)
