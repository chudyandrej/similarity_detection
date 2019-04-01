from keras import backend as K


def l1_similarity(x):
    return K.exp(-1 * K.sum(K.abs(x[0] - x[1]), axis=-1, keepdims=True))


def l2_similarity(x):
    return K.exp(-1 * K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True)))


def euclidean_distance(x):
    sum_square = K.sum(K.square(x[0] - x[1]), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))
