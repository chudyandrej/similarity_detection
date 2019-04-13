from sklearn.neighbors import NearestNeighbors
from collections import Counter
from typing import List, Optional
import codecs
import numpy as np
from collections import defaultdict


def evaluate_similarity(eval_class, profile_embedding):
    eval_class.logger.info('Evaluating embeddings ...')
    profiles, embeddings = zip(*profile_embedding)
    eval_class.logger.info('Building index ...')
    index = NearestNeighbors(metric=eval_class.metric, n_jobs=-1).fit(embeddings)

    eval_class.logger.info('Querying index ...')
    distances, indices = index.kneighbors(embeddings, n_neighbors=eval_class.neighbors)

    eval_class.logger.info('Evaluating embeddings ...')
    evaluation = Counter()
    results_file = codecs.open(eval_class.results_file, 'w', 'utf-8')
    true_first_distances = []
    false_first_distances = []

    for query_idx, idxs_distances in enumerate(zip(indices, distances)):
        query_profile = profiles[query_idx]
        query_label: (str, str) = query_profile.uid[:2]

        # Print query profile information
        results_file.write(str("=" * 120) + '\n')
        results_file.write('Query UID: ' + str(query_profile.uid) + '\n')
        results_file.write('Query Quantiles: ' + str(query_profile.quantiles) + '\n\n')

        distances = process_suggestions(query_idx, idxs_distances, profiles, results_file)

        distances = {k: np.mean(v) for k, v in distances.items()}
        distances = distances.items()
        distances = sorted(distances, key=lambda x: x[1])
        # Print true and predicted columns (labels)
        results_file.write('True: ' + str(query_label) + '\n')
        i = 0
        for label, distance in distances[:5]:
            results_file.write('Predicted: ' + str(label) + ', Similarity:' + str(distance) + '\n')
            if i == 0 and query_label == label:
                true_first_distances.append(distance)
            else:
                false_first_distances.append(distance)
            i += 1

        # Get the index of the true positive
        result_labels = [dist[0] for dist in distances]
        index = result_labels.index(query_label) if query_label in result_labels else None
        evaluation[index] += 1

    results_file.write("\n\n")
    results_file.write("==============================")
    results_file.write("\n\n")

    evaluation['total'] = sum(evaluation.values())
    count = Counter(evaluation)
    eval_class.logger.info('Evaluation: ' + str(count))
    sum_indexes = sum([count[index] for index in range(3)])
    percentage = sum_indexes / count["total"] * 100
    eval_class.logger.info('Percentage of found labels on first 3 index : ' + str(int(percentage)) + '%')
    results_file.write(str(count))
    results_file.write("Result: " + str(percentage))


def process_suggestions(query_idx: int, idxs_distances: (int, float), profiles, results_file):
    distances = defaultdict(list)
    (result_idx, result_distance) = idxs_distances
    for i in range(len(result_idx)):
        if query_idx == result_idx[i]:
            continue

        result_profile = profiles[result_idx[i]]
        result_label = result_profile.uid[:2]

        # Print result profile information
        results_file.write('Result Distance: ' + str(result_distance[i]) + '\n')
        results_file.write('Result UID: ' + str(result_profile.uid) + '\n')
        results_file.write('Result Quantiles: ' + str(result_profile.quantiles) + '\n\n')

        # Calculate similarity as complement
        distances[result_label].append(result_distance[i])

    return distances
