import numpy as np


def evaluate_similarity_index(similarity_index):
    recall_list = []
    precision_list = []
    count_selected = []
    tps = []

    for class_name, n_class_names in similarity_index.items():

        n_class_names_true = list(filter(lambda x: x != class_name and x[:-2] == class_name[:-2], n_class_names))
        tp = len(n_class_names_true)
        tps.append(tp)
        count_selected.append(len(n_class_names))
        recall = tp / 1
        precision = 0
        if len(n_class_names) != 0:
            precision = tp / len(n_class_names)

        recall_list.append(recall)
        precision_list.append(precision)

    sum_recall = sum(tps) / (len(tps))
    sum_precision = sum(tps) / sum(count_selected)

    stats = {
        "avg_recall": np.average(recall_list),
        "avg_precision": np.average(precision_list),
        "sum_recall": sum_recall,
        "sum_precision": sum_precision,
        "max_selected": max(count_selected),
        "min_selected": min(count_selected),
        "avg_selected": np.average(count_selected)
    }
    return stats
