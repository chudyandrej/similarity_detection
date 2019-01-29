from keras import backend as K
from keras.engine.topology import Layer
from keras import objectives


##################################################
#             SIMILARITY FUNCTIONS
##################################################

def l1_similarity(x):
    return K.exp(-1 * K.sum(K.abs(x[0] - x[1]), axis=-1, keepdims=True))


def l2_similarity(x):
    return K.exp(-1 * K.sqrt(K.sum(K.square(x[0] - x[1]), axis=-1, keepdims=True)))


##################################################
#             LOSS FUNCTIONS
##################################################

def l2_loss(y_true, y_pred):
    return K.mean(y_true * K.square(1 - y_pred) + (1 - y_true) * K.square(y_pred))


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def zero_loss(y_true, y_pred):
    return K.zeros_like(y_pred)


##################################################
#                 LAYERS
##################################################

class CustomRegularization(Layer):
    def __init__(self, **kwargs):
        super(CustomRegularization, self).__init__(**kwargs)

    def call(self, x, mask=None):
        target = x[0]
        pred = x[1]
        mse = objectives.mean_squared_error(target, pred)
        loss = K.sum(mse)
        self.add_loss(loss, x)
        return mse

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0],1)

