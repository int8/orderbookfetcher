import os
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random

from src.model import OrderBooksDataSequenceDatasetV1


class OrderBooksSequenceDatasetBase(Dataset, ABC):
    def __init__(self, data_directory, epoch_size: int,
                 files_to_merge: int = 10, time_length=100,
                 min_sequence_length_in_file=1000):
        self.files_to_merge = files_to_merge
        self.time_length = time_length
        self.data_directory = data_directory
        self.epoch_size = epoch_size
        self.files = os.listdir(data_directory)
        self.file_loaded_counts = dict(zip(self.files, [0] * len(self.files)))
        self.loaded_data = []
        self.last_file_read = 0
        self.min_sequence_length_in_file = min_sequence_length_in_file

    def load_to_memory(self):
        self.loaded_data = []

        def _r(_):
            return random.random()

        sorted_files = sorted(
            self.files,
            key=lambda k: (self.file_loaded_counts[k] + _r(k))
        )
        i = 0
        for f in sorted_files:
            self.mark_as_read(f)
            x, y, idx, metadata = OrderBooksDataSequenceDatasetV1.load(
                os.path.join(self.data_directory, f), remove_nans=True
            )

            if len(idx) >= self.min_sequence_length_in_file:
                self.loaded_data.append((x, y, idx, metadata))
                i += 1
                if i >= self.files_to_merge:
                    break

    @abstractmethod
    def x(self, sparse_matrices, idx):
        pass

    def mark_as_read(self, f):
        self.file_loaded_counts[f] += 1

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if idx == 0:
            self.load_to_memory()

        sparse_matrices, y, s3_keys, metadata = self.loaded_data[
            self.last_file_read]
        self.last_file_read = (self.last_file_read + 1) % len(self.loaded_data)
        i = random.randint(1, len(s3_keys) - self.time_length)
        return (
            self.x(sparse_matrices, i),
            y[:, i: (i + self.time_length)].astype(float)
        )


class OrderBooksSequenceDataset(OrderBooksSequenceDatasetBase):
    def __init__(self, data_directory, epoch_size: int,
                 files_to_merge: int = 10, time_length=100,
                 min_sequence_length_in_file=1000, x_shape=(64, 64),
                 flattened=False):
        super().__init__(data_directory, epoch_size, files_to_merge,
                         time_length, min_sequence_length_in_file)
        self.x_shape = x_shape
        self.flattened = flattened

    def x(self, sparse_matrices, idx):
        m = np.zeros(shape=(2,) + self.x_shape + (self.time_length,))
        for i in range(self.time_length):
            bids_sparse = sparse_matrices['bids_sparse'][idx + i]
            asks_sparse = sparse_matrices['asks_sparse'][idx + i]
            m[0, bids_sparse[0], bids_sparse[1], i] = bids_sparse[2]
            m[1, asks_sparse[0], asks_sparse[1], i] = asks_sparse[2]
        if self.flattened:
            return m.reshape(2 * self.x_shape[0] * self.x_shape[1],
                             self.time_length)
        else:
            return m


class OrderBooksDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size=batch_size, shuffle=False)
