from ast import literal_eval
import numpy as np

from data_preparation.cvut import filter_cand_key_by_prof_id


def evaluate_stats(candidate_key_index, fk_pk, profiles):
    recall_list = []
    precision_list = []
    count_selected = []
    tps = []
    candidate_keys_count = 0
    for cfk, cand_pks in candidate_key_index.items():
        cand_pks = filter_cand_key_by_prof_id(cfk, cand_pks, profiles)
        if len(cand_pks) == 0:
            continue

        candidate_keys_count += len(cand_pks)
        cfk = literal_eval(cfk)

        if cfk not in fk_pk.keys():
            continue
        true_pk = fk_pk[cfk]

        tp = 1 if str(true_pk) in cand_pks else 0
        recall = tp / 1
        precision = tp / len(cfk)

        tps.append(tp)
        count_selected.append(len(cand_pks))
        recall_list.append(recall)
        precision_list.append(precision)

    sum_recall = sum(tps) / (len(tps))
    sum_precision = sum(tps) / sum(count_selected)

    result = {
        "count_of_cand_keys": candidate_keys_count,
        "avg_recall": np.average(recall_list),
        "avg_precision": np.average(precision_list),
        "sum_recall": sum_recall,
        "sum_precision": sum_precision,
        "max_selected": max(count_selected),
        "min_selected": min(count_selected),
        "avg_selected": np.average(count_selected)
    }
    return result
