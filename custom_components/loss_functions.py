
from keras import backend as K
from keras import objectives


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