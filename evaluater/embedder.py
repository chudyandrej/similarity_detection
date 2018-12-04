import numpy as np
from sklearn.neighbors import NearestNeighbors


def convert_to_vec(encoder_model, data, max_seq_len, max_count_tokens):
    """Converting string data array to array of embedding vectors

    Args:
        encoder_model (keras.engine.training.Model): Description
        data (Array of STRINGS): Column values
        max_seq_len
        max_count_tokens

    Returns:
        (Array of Numpy): Array of embedding vectors
    """

    encoder_input_data = np.zeros((len(data), max_seq_len, max_count_tokens), dtype='float32')

    for i, input_text in enumerate(data):
        for t, char in enumerate(input_text):
            # If character not be include in training vec2vec it will be replace by ' '
            encoder_input_data[i, t, tokenizer(char)] = 1.

    input_seq = encoder_model.predict(encoder_input_data)
    vec_s1 = input_seq[0]
    vec_s2 = input_seq[1]
    assert (vec_s1.shape == vec_s2.shape), "Shape of vec1 and vec2 is not equal!"

    vectors = []
    for index in range(len(vec_s1)):
        tmp = np.concatenate([vec_s1[index], vec_s2[index]], axis=0)
        vectors.append(tmp)

    return vectors


def create_column_embedding(type_embedding, weights=None):
    """Create column embedding from value embeddings by vector AVERAGE

    Args:
        type_embedding (Array): Array of Tuples
        weights (None, optional): Array of numbers with same size as value_embeddings.

    Returns:
        Array of Tuple: [(column_name, Numpy), ...]

    """

    columns_name_set = list(set(list(map(lambda x: x[0], type_embedding))))
    if weights is not None and len(weights) == len(type_embedding):
        weights = list(map(lambda x: int(x), weights))
        data = list(zip(type_embedding, weights))
    else:
        data = type_embedding
        weights = None

    class_embedding = []
    for column_name in columns_name_set:
        class_embedding.append(job(column_name, data, weights))

    return class_embedding


def evaluate_neighbors(embedding_vectors, classes, n_neighbors=100, radius=0.15, metric="braycurtis", mode="kneighbors"):

    neigh_model = NearestNeighbors(n_neighbors=n_neighbors, radius=radius, n_jobs=-1, metric=metric)
    neigh_model.fit(embedding_vectors)
    if mode == "kneighbors":
        neighbor_indexes = neigh_model.kneighbors(embedding_vectors, return_distance=False)
    elif mode == "radius":
        neighbor_indexes = neigh_model.radius_neighbors(embedding_vectors, return_distance=False)
    else:
        neighbor_indexes1 = neigh_model.kneighbors(embedding_vectors, return_distance=False)
        neighbor_indexes2 = neigh_model.radius_neighbors(embedding_vectors, return_distance=False)
        neighbor_indexes = list(map(lambda x: np.intersect1d(x[0], x[1]), zip(neighbor_indexes1, neighbor_indexes2)))
    index = {}

    for i in range(len(neighbor_indexes)):
        neighbor_classes = classes[neighbor_indexes[i]]
        neighbor_classes = list(filter(lambda x: x != classes[i], neighbor_classes))
        index[classes[i]] = neighbor_classes

    return index

# ------------------------- PRIVATE ----------------------------------


def job(column_name, data, weights_for_values):
    class_data = list(filter(lambda x: x[0] == column_name, data))
    if weights_for_values is not None:
        _, class_embeddings, weights_for_values = zip(*class_data)
    else:
        _, class_embeddings = zip(*class_data)

    mean_embedding = np.average(np.array(class_embeddings), axis=0, weights=weights_for_values)
    return column_name, mean_embedding


def tokenizer(char):
    """Reduce he alphabet replace all white symbols as space

    Args:
        char (STRING): Char

    Returns:
        NUMBER: Code <0,94>
    """
    code = ord(char)
    if 0 <= code <= 31 or code == 127:    # Is white
        code = 0
    else:
        code -= 32

    return code
