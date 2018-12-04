from HLL import HyperLogLog
from unidecode import unidecode
from ast import literal_eval


import numpy as np
import pandas as pd
import pickle

PROFILES = []


def load_dataset(prof_path, pk_fk_path, full_load=True):
    """

    Args:

    Returns:
        (DataFrame, [(fk, pk)]): DataFrame (value, class) , maping foreign keys -> relebamt primary key
    """
    pk_fk = pickle.load(open(pk_fk_path, "rb"))
    profiles = pickle.load(open(prof_path, "rb"))
    global PROFILES
    PROFILES = profiles

    pk_fk = filter_keys(profiles, pk_fk)
    pk_fk = dict(list(pk_fk))
    df = pd.DataFrame({"value": [], "type": ""})
    if full_load:
        profiles_data = dict(list(map(lambda x: (
            x[0], x[1].most_frequent[:50] + x[1].least_frequent[-50:]),
            profiles.items())))

        for key, prof_val in profiles_data.items():
            values = list(map(lambda x: unidecode(str(x[0])), prof_val))

            df = pd.concat([df, pd.DataFrame(
                {"value": values, "type": str(key)})], ignore_index=True)

    return df, pk_fk


def filter_cand_key_by_prof_id(cfk, cpks):
    if len(PROFILES) == 0:
        assert "First must run load dataset"
        return

    cfk_profile = PROFILES[literal_eval(cfk)]
    cpk_filtered = []
    for cpk in cpks:
        profile_cpks = PROFILES[literal_eval(cpk)]
        if is_candidate_key(cfk_profile, profile_cpks):
            cpk_filtered.append(cpk)
    return cpk_filtered


# -------------------------------PRIVATE-------------------------------------


def filter_keys(profiles, foreign_keys):
    filtered_keys = set()
    for fk_uid, pk_uid in foreign_keys:
        fk_profile = profiles[fk_uid]
        pk_profile = profiles[pk_uid]
        if is_candidate_key(fk_profile, pk_profile):
            filtered_keys.add((fk_uid, pk_uid))

    return filtered_keys


def clip_div(x, y, min_value=0, max_value=1):
    return clip_value(x / y, min_value=min_value, max_value=max_value)


def clip_value(value, min_value=0, max_value=1):
    return max(min(value, max_value), min_value)


def range_sim(profile_1, profile_2):
    if profile_1.max_value < profile_2.min_value:
        return 0
    if profile_2.max_value < profile_1.min_value:
        return 0

    min_value = clip_value(profile_1.min_value,
                           profile_2.min_value, profile_2.max_value)
    max_value = clip_value(profile_1.max_value,
                           profile_2.min_value, profile_2.max_value)
    profile_1_range = max_value - min_value
    profile_2_range = profile_2.max_value - profile_2.min_value
    range_sim = (profile_1_range + 1) / (profile_2_range + 1)
    return clip_value(range_sim, 0, 1)


def mean_sim(profile_1, profile_2):
    if profile_1.mean_value == profile_2.mean_value:
        return 1
    if profile_2.max_value == profile_2.min_value:
        return 0
    if profile_1.max_value < profile_2.min_value:
        return 0
    if profile_2.max_value < profile_1.min_value:
        return 0

    mean_diff = abs(profile_1.mean_value - profile_2.mean_value)
    profile_2_range = profile_2.max_value - profile_2.min_value
    mean_sim = 1 - (mean_diff / profile_2_range)
    return clip_value(mean_sim, 0, 1)


def bi_range_sim(profile_1, profile_2):
    max_mins_value = max(profile_1.min_value, profile_2.min_value)
    min_maxs_value = min(profile_1.max_value, profile_2.max_value)
    min_mins_value = min(profile_1.min_value, profile_2.min_value)
    max_maxs_value = max(profile_1.max_value, profile_2.max_value)

    inner_range = min_maxs_value - max_mins_value + 1
    outer_range = max_maxs_value - min_mins_value + 1
    return inner_range / outer_range


def coverage(profile_1, profile_2):
    hyper_size = int(np.log2(profile_1.hyper.size()))
    union_hyper = HyperLogLog(hyper_size, seed=42)
    union_hyper.merge(profile_1.hyper)
    union_hyper.merge(profile_2.hyper)

    cardinality_1 = profile_1.hyper.cardinality()
    cardinality_2 = profile_2.hyper.cardinality()
    union_cardinality = union_hyper.cardinality()

    # Inclusion–exclusion principle |PK ∩ FK| == |FK| + |PK| - |FK U PK|
    inter_cardinality = cardinality_1 + cardinality_2 - union_cardinality
    coverage_1 = clip_div(inter_cardinality, cardinality_1, 0, 1)
    coverage_2 = clip_div(inter_cardinality, cardinality_2, 0, 1)
    return coverage_1, coverage_2


def is_candidate_key(fk_profile, pk_profile):
    if fk_profile == pk_profile:
        return False
    if fk_profile.resolved_type != pk_profile.resolved_type:
        return False

    if not fk_profile.is_fk_candidate():
        return False
    if not pk_profile.is_pk_candidate():
        return False
    if fk_profile.unique_count <= 1:
        return False

    fk_coverage, pk_coverage = coverage(fk_profile, pk_profile)
    if fk_coverage < 0.5 or pk_coverage < 0.001:
        return False

    fk_range_sim = range_sim(fk_profile, pk_profile)
    pk_range_sim = range_sim(pk_profile, fk_profile)
    if max(fk_range_sim, pk_range_sim) < 0.001:
        return False

    bi_range_sim_ = bi_range_sim(fk_profile, pk_profile)
    if bi_range_sim_ < 0.001:
        return False
    # Otherwise
    return True
