
import os
import random
import pickle
import pandas as pd

from evaluater.key_analyzer.profile_class import Profile

PATH = "../data/cvut/profiles.pkl"
TARGET_PATH = "../data/cvut/"


class CvutDataset:
    def __init__(self):

        self.df = pd.DataFrame({"value": [], "type": ""})

    def load_top100_last100(self):
        if os.path.exists(TARGET_PATH + "top100_last100.csv"):
            print("Data loaded!")
            self.df = pd.read_csv(TARGET_PATH + "top100_last100.csv")
            return self.df

        if not os.path.exists(PATH):
            raise ValueError('Data not found: ' + PATH)

        profiles = pickle.load(open(PATH, "rb"))

        # Load data from profile
        for cel_name, profObj in list(profiles.items()):
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

            self.save_to_csv("top100_last100.csv")
            print("Data loaded!")
            print("Data saved!")
            return self.df

    def load_for_similarity_case(self, count_of_data_in_class=50):
        if os.path.exists(TARGET_PATH+"similarity_problem.csv"):
            print("Data loaded!")
            self.df = pd.read_csv(TARGET_PATH+"similarity_problem.csv")
            return self.df

        if not os.path.exists(PATH):
            raise ValueError('Data not found: ' + PATH)

        profiles = pickle.load(open(PATH, "rb"))

        # Load data from profile
        for cel_name, profObj in list(profiles.items()):
            data1 = list(map(lambda x: x[0], profObj.most_frequent))
            data2 = list(map(lambda x: x[0], profObj.least_frequent))
            if len(data1) < count_of_data_in_class and len(data2) < count_of_data_in_class:
                continue

            if not digit_values(data1) and not digit_values(data2):
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
            self.save_to_csv("similarity_problem.csv")
            print("Data loaded!")
            print("Data saved!")
            return self.df

    def save_to_csv(self, name):
        self.df.to_csv(TARGET_PATH+name, index=False)


def digit_values(values):
    for val in values:
        if str(val).isdigit():
            return True

    return False


def generate_profile_numbers(digit):
    return ''.join([str(random.randint(1, 9))]+["%s" % random.randint(0, 9) for num in range(0, digit-1)])


if __name__ == '__main__':
    dataclass = CvutDataset()
    dataclass.load_top100_last100()
