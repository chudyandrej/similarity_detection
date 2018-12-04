import re
import string
from HLL import HyperLogLog
from collections import Counter

import numpy as np
from markdown.odict import OrderedDict
import json
from unidecode import unidecode


class Profile:
    def __init__(self, database, table, column, data_type):
        self.uid = (database, table, column)
        self.data_type = data_type
        self.resolved_type = None

        # Resolve the simple type of the column
        if 'INT' in data_type:
            self.resolved_type = int
        elif 'CHAR' in data_type:
            self.resolved_type = str

    def profile_data(self, data, num_values=100, hyper_size=10):
        present_data = list(filter(None.__ne__, data))
        resolved_data = self.resolve_data(present_data)
        sorted_data = list(sorted(resolved_data))
        unique_data = list(OrderedDict.fromkeys(sorted_data))
        unique_bytes = list(map(self.value2bytes, unique_data))
        numeric_data = list(map(self.value2numerical, resolved_data))
        numeric_unique_data = list(OrderedDict.fromkeys(numeric_data))

        self.size = len(data)
        self.valued_count = len(sorted_data)
        self.unique_count = len(unique_data)
        if self.valued_count == 0:
            return

        self.mean_value = np.mean(numeric_data).item()
        self.min_value = np.min(numeric_data).item()
        self.max_value = np.max(numeric_data).item()

        number_range = self.max_value - self.min_value
        self.presence = self.valued_count / self.size
        self.uniqueness = self.unique_count / self.valued_count
        self.sparsity = len(numeric_unique_data) / (number_range + 1)
        self.monotonicity = self.monotonicity(resolved_data)

        value_counts = [x for x in Counter(resolved_data).items()]
        sorted_values = sorted(value_counts, key=lambda x: x[0], reverse=False)
        sorted_counts = sorted(sorted_values, key=lambda x: x[1], reverse=True)

        self.first_values = unique_data[:num_values]
        self.last_values = unique_data[-num_values:]
        self.sort_first_values = sorted_values[:num_values]
        self.sort_last_values = sorted_values[-num_values:]
        self.most_frequent = sorted_counts[:num_values]
        self.least_frequent = sorted_counts[-num_values:]

        quantile_idxs = np.linspace(0, 1, num=(num_values + 1)) * (self.valued_count - 1)
        self.quantiles = [sorted_data[idx] for idx in np.round(quantile_idxs).astype(int)]

        self.hyper = HyperLogLog(hyper_size, seed=42)
        for value in unique_bytes:
            self.hyper.add(value)

    def resolve_data(self, present_data):
        try:
            present_data = list(map(int, present_data))
            self.resolved_type = int  # Change the data type
        finally:
            return present_data

    def value2bytes(self, value):
        if self.resolved_type == int:
            return str(value).encode()
        elif self.resolved_type == str:
            return str(value).encode()
        else:
            raise ValueError

    def value2numerical(self, value):
        if self.resolved_type == int:
            return value
        elif self.resolved_type == str:
            return len(value)
        else:
            raise ValueError

    def monotonicity(self, data):
        # Trivial datauence, length ∈ [0, 1]
        if len(data) < 2:
            return 1

        max_adj_inversions = len(data) - 1
        num_adj_inversions = sum(data[i] > data[i + 1] for i in range(len(data) - 1))
        return 1 - (num_adj_inversions / max_adj_inversions)

    def is_pk_candidate(self):
        return self.presence == 1 and self.uniqueness == 1

    def is_fk_candidate(self):
        return self.presence > 0

    def get_registers(self):
        registers = self.hyper.registers()
        return np.array(registers) / 23

    def get_quantiles(self):
        def clean_chars(x): return ''.join(c for c in x if c in string.printable)

        def clean_spaces(x): return re.sub('\s+', ' ', x)

        quantiles = map(str, self.quantiles)
        quantiles = map(unidecode, quantiles)
        quantiles = map(clean_chars, quantiles)
        quantiles = map(clean_spaces, quantiles)
        quantiles = map(str.lower, quantiles)
        quantiles = map(str.strip, quantiles)
        return list(quantiles)

    def to_dict(self):
        return dict(
            uid=self.uid,
            data_type=self.data_type,
            resolved_type=self.resolved_type.__name__,
            size=self.size,
            valued_count=self.valued_count,
            unique_count=self.unique_count,
            min_value=self.min_value,
            mean_value=self.mean_value,
            max_value=self.max_value,
            presence=self.presence,
            uniqueness=self.uniqueness,
            sparsity=self.sparsity,
            monotonicity=self.monotonicity
        )

    def __str__(self):
        return json.dumps(self.to_dict())
