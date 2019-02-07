from keras.preprocessing.sequence import pad_sequences


def preprocess_values_standard(values, pad_maxlen, char_index=None):
    """
    Standard preprocess pipeline into encoder. Cut to max size, revert,
    tokenizing by index or ascii value and padding.
    :param values: (Array) String values
    :param pad_maxlen: (Number)Maximal length of sequence
    :param char_index: (Dict)(optional) Index for convert chars to tokens
    :return: (Array) Preprocessed values prepare for input to neural network
    """
    values = map(lambda x: str(x)[:pad_maxlen], values)
    values = map(str.strip, values)
    values = (x[::-1] for x in values)
    if char_index is None:
        values = list(map(lambda x: [ord(y) for y in x], values))
    else:
        values = map(lambda x: [char_index[y] for y in x], values)
    values = pad_sequences(list(values), maxlen=pad_maxlen, truncating='pre', padding='pre')
    return values
