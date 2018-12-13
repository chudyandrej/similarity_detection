
import os
import random
import pickle
import enum
import pandas as pd
import numpy as np
from ast import literal_eval

from HLL import HyperLogLog
from evaluater.key_analyzer.profile_class import Profile

PROFILES_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/profiles.pkl"
FK_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/foreign_keys.pkl"

TARGET_PATH = os.environ['PYTHONPATH'].split(":")[0] + "/data/cvut/"


class SelectData(enum.Enum):
    profile_similarity_basic = 1
    profile_top100_last100_raw = 2
    load_key_analyzer = 3


class CvutDataset:
    def __init__(self, data_selectro):
        self.df = pd.DataFrame({"value": [], "type": ""})
        self.profiles = {}
        self.pk_fk = []
        self.name = ''

        if data_selectro.value == 1:
            self.load_for_similarity_case()
        elif data_selectro.value == 2:
            self.load_top100_last100()
        elif data_selectro.value == 3:
            self.load_key_analyzer()

    def load_top100_last100(self):
        if os.path.exists(TARGET_PATH + "top100_last100.csv"):
            print("Data loaded!")
            self.df = pd.read_csv(TARGET_PATH + "top100_last100.csv")
            return self.df

        if not os.path.exists(PROFILES_PATH):
            raise ValueError('Data not found: ' + PROFILES_PATH)

        self.profiles = pickle.load(open(PROFILES_PATH, "rb"))

        # Load data from profile
        for cel_name, profObj in list(self.profiles.items()):
            data1 = list(map(lambda x: x[0], profObj.most_frequent))
            data2 = list(map(lambda x: x[0], profObj.least_frequent))
            data = list(set(data1 + data2))
            if len(data) <= 1:
                continue

            if not digit_values(data):
                self.df = pd.concat([self.df, pd.DataFrame({"value": data, "type": str(cel_name)})])

        # generate number value by count of dental advice
        for i in range(1, 20):
            vals = []
            for k in range(1000):
                vals.append(generate_profile_numbers(i))

            vals = list(set(vals))[:200]
            name = "digit_" + str(i)
            self.df = pd.concat([self.df, pd.DataFrame({"value": vals, "type": name})])

            self.save_to_csv(self.name)
            print("Data loaded!")
            print("Data saved!")

    def load_for_similarity_case(self, count_of_data_in_class=50):
        self.name = "cvut_prof_numgen_split_column.csv"
        if os.path.exists(TARGET_PATH + self.name):
            print("Data loaded!")
            self.df = pd.read_csv(TARGET_PATH + self.name)
            return self.df

        if not os.path.exists(PROFILES_PATH):
            raise ValueError('Data not found: ' + PROFILES_PATH)

        self.profiles = pickle.load(open(PROFILES_PATH, "rb"))
        # Load data from profile
        for cel_name, profObj in list(self.profiles.items()):
            data1 = list(map(lambda x: x[0], profObj.most_frequent))
            data2 = list(map(lambda x: x[0], profObj.least_frequent))
            if len(data1) < count_of_data_in_class and len(data2) < count_of_data_in_class:
                continue

            self.df = pd.concat([self.df, pd.DataFrame({"value": data1, "type": str(cel_name)+"_1"})])
            self.df = pd.concat([self.df, pd.DataFrame({"value": data2, "type": str(cel_name)+"_2"})])

        # generate number value by count of dental advice
        for i in range(1, 20):
            vals = []
            for k in range(1000):
                vals.append(generate_profile_numbers(i))

            vals = list(set(vals))[:200]
            if len(vals) < 2*count_of_data_in_class:
                continue

            vals1 = vals[:len(vals)//2]
            vals2 = vals[len(vals) // 2:]
            name = "digit_" + str(i)
            self.df = pd.concat([self.df, pd.DataFrame({"value": vals1, "type": name+"_1"})])
            self.df = pd.concat([self.df, pd.DataFrame({"value": vals2, "type": name+"_2"})])
        self.save_to_csv(self.name)
        print("Data loaded!")
        print("Data saved!")

    def load_key_analyzer(self):
        self.name = CvutDataset.__name__+self.load_key_analyzer.__name__

        if not os.path.exists(PROFILES_PATH):
            raise ValueError('Data not found: ' + PROFILES_PATH)

        if not os.path.exists(FK_PATH):
            raise ValueError('Data not found: ' + FK_PATH)

        self.profiles = pickle.load(open(PROFILES_PATH, "rb"))

        self.pk_fk = pickle.load(open(FK_PATH, "rb"))
        self.pk_fk = dict(list(filter_keys(self.profiles, self.pk_fk)))

        if os.path.exists(TARGET_PATH + self.name+".csv"):
            print("Data loaded!")
            self.df = pd.read_csv(TARGET_PATH + self.name+".csv")
            return

        self.df = pd.DataFrame({"value": [], "type": ""})
        profiles_data = dict(list(map(lambda x: (x[0], x[1].most_frequent + x[1].least_frequent), self.profiles.items())))

        for key, prof_val in profiles_data.items():
            self.df = pd.concat([self.df, pd.DataFrame({"value": prof_val, "type": str(key)})], ignore_index=True)
        self.save_to_csv(self.name+".csv")

    def save_to_csv(self, name):
        self.df.to_csv(TARGET_PATH+name, index=False)


def filter_cand_key_by_prof_id(cfk, cpks, profiles):
    if len(profiles) == 0:
        assert "First must run load dataset"
        return

    cfk_profile = profiles[literal_eval(cfk)]
    cpk_filtered = []
    for cpk in cpks:
        profile_cpks = profiles[literal_eval(cpk)]
        if is_candidate_key(cfk_profile, profile_cpks):
            cpk_filtered.append(cpk)
    return cpk_filtered


def digit_values(values):
    for val in values:
        if str(val).isdigit():
            return True

    return False


def generate_profile_numbers(digit):
    return ''.join([str(random.randint(1, 9))]+["%s" % random.randint(0, 9) for num in range(0, digit-1)])


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


if __name__ == '__main__':
    dataclass = CvutDataset(SelectData.profile_similarity_basic)
    print(dataclass.df)

