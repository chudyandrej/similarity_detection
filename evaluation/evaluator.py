from typing import List, Optional, Tuple
import random
import logging
import os
import pickle
from collections import namedtuple
import numpy as np
from collections import defaultdict

from .similarity_task import evaluate_similarity
from sklearn.model_selection import train_test_split


def create_logger():
    logger_local = logging.getLogger(__file__)
    logger_local.setLevel(logging.INFO)
    logger_local.propagate = False

    formatter = logging.Formatter('[%(filename)s][%(asctime)s] %(message)s')
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger_local.addHandler(handler)
    return logger_local

seed = 42
logger = create_logger()
random.seed(seed)
np.random.seed(seed)


class AuthorityEvaluator:
    CVUT_SRC = os.environ['PYTHONPATH'].split(":")[0] + "/data/profiles/cvut_data_partition.pkl"
    S3_SRC = os.environ['PYTHONPATH'].split(":")[0] + "/data/profiles/s3_data_partition.pkl"

    def __init__(self, username, train_size=0.5, valid_size=0.1, test_size=None,
                 results_file='', metric='l2', neighbors=20, radius=5):

        self.test_size = test_size
        self.train_size = train_size
        self.valid_size = valid_size
        self.results_file = results_file + '/results.txt'
        self.metric = metric
        self.neighbors = neighbors
        self.radius = radius
        self.seed = seed
        self.logger = logger
        self.training_profiles = []
        self.valid_profiles = []

        self.cvut_profiles = [namedtuple("Profile", d.keys())(*d.values()) for d in pickle.load(open(self.CVUT_SRC, "rb"))]
        self.s3_profiles = [namedtuple("Profile", d.keys())(*d.values()) for d in pickle.load(open(self.S3_SRC, "rb"))]

        logger.info('Data successfully loaded')
        logger.info('CVUT dataset contains ' + str(len(self.cvut_profiles)) + ' profiles!')
        logger.info('S3 dataset contains ' + str(len(self.s3_profiles)) + ' profiles!')

    def split_profiles(self, profiles, train_size):
        logger.info(f'Splitting profiles')
        all_tables = sorted(set(x.uid[0] for x in profiles))
        train_tables, valid_tables = train_test_split(all_tables, train_size=train_size)

        train_profiles = [x for x in profiles if x.uid[0] in set(train_tables)]
        valid_profiles = [x for x in profiles if x.uid[0] in set(valid_tables)]
        return train_profiles, valid_profiles

    def evaluate_embeddings(self, profile_embedding: List[Tuple[namedtuple, np.array]]):
        """
        Evaluation function for similarity detection task. Function build NearestNeighbors index over all uid_embedding
        data. This data consist from profile and column vector (his embedding). This function try to find
        partitions for every column representation.
        :param profile_embedding: (Dict, np.Vector)
        """
        evaluate_similarity(self, profile_embedding)


def get_profile_similarity(profile1, profile2):
    left_values = set(map(str, profile1.quantiles))
    right_values = set(map(str, profile2.quantiles))

    inter_values = left_values & right_values
    union_values = left_values | right_values

    similarity = len(inter_values) / len(union_values)
    return similarity


def triplet_generator(profiles, batch_size, preprocess_profile):
    # Bucketing partitions into columns
    column_buckets = defaultdict(list)
    for profile in profiles:
        column_uid = profile.uid[:2]
        column_buckets[column_uid].append(profile)
    column_buckets = list(column_buckets.values())

    while True:
        anchors = []
        positives = []
        negatives = []

        while len(anchors) < batch_size:
            bucket_1, bucket_2 = random.sample(column_buckets, 2)
            anchor, positive = random.sample(bucket_1, 2)

            negative = random.choice(bucket_2)

            if anchor.dtype != negative.dtype: continue
            if get_profile_similarity(anchor, negative) > 0.2: continue

            anchors.append(preprocess_profile(anchor))
            positives.append(preprocess_profile(positive))
            negatives.append(preprocess_profile(negative))

        yield anchors, positives, negatives


def pairs_generator(profiles, batch_size, preprocess_profile):

    column_buckets = defaultdict(list)

    for profile in profiles:
        column_uid = profile.uid[:2]
        column_buckets[column_uid].append(profile)
    column_buckets = list(column_buckets.values())

    while True:
        left = []
        right = []
        labels = []
        seen_pairs = set()

        while len(labels) < batch_size:

            bucket_1, bucket_2 = random.sample(column_buckets, 2)
            if len(bucket_1) < 2 or len(bucket_2) < 2:
                continue

            positive_left, positive_right = random.sample(bucket_1, 2)
            negative_left = random.choice(bucket_1)
            negative_right = random.choice(bucket_2)

            if negative_left.dtype != negative_right.dtype:
                continue

            if get_profile_similarity(negative_left, negative_right) > 0.2:
                continue

            positive_left_hash = hash(str(positive_left.quantiles))
            positive_right_hash = hash(str(positive_right.quantiles))
            negative_left_hash = hash(str(negative_left.quantiles))
            negative_right_hash = hash(str(negative_right.quantiles))

            if (positive_left_hash, positive_right_hash) in seen_pairs or (negative_left_hash, negative_right_hash) in seen_pairs:
                continue

            seen_pairs.add((positive_left_hash, positive_right_hash))
            seen_pairs.add((negative_left_hash, negative_right_hash))

            left += [preprocess_profile(positive_left), preprocess_profile(negative_left)]
            right += [preprocess_profile(positive_right), preprocess_profile(negative_right)]
            labels += [1, 0]

        yield [np.array(left), np.array(right)], labels


if __name__ == '__main__':
    def preprocess_profile_p(profile):
        return profile

    a = AuthorityEvaluator(username='andrej')
    gen = triplet_generator(a.s3_profiles, 20, preprocess_profile_p)

    for i in range(10000):
        anchors, positives, negatives = next(gen)




