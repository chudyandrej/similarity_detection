from sklearn.neighbors import NearestNeighbors
import numpy as np


def compute_neighbors(embedding_vectors, classes, n_neighbors=100, radius=0.15, metric="braycurtis", mode="kneighbors"):

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
