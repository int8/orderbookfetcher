import os
from torch.utils.data import Dataset, DataLoader
import random

from src.model import OrderBooksDataSequenceDatasetV1


class OrderBooksDataset(Dataset):
    def __init__(self, data_directory, epoch_size: int, files_to_merge: int = 10, time_length=100):
        self.files_to_merge = files_to_merge
        self.time_length = time_length
        self.data_directory = data_directory
        self.epoch_size = epoch_size
        self.files = os.listdir(data_directory)
        self.file_loaded_counts = dict(zip(self.files, [0] * len(self.files)))
        self.loaded_data = []

        self.last_file_read = 0

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
                os.path.join(self.data_directory, f)
            )
            if len(idx) >= 1000:
                self.loaded_data.append((x, y, idx, metadata))
                i += 1
                if i >= self.files_to_merge:
                    break

    def mark_as_read(self, f):
        self.file_loaded_counts[f] += 1

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        if idx == 0:
            self.load_to_memory()

        x, y, s3_keys, metadata = self.loaded_data[self.last_file_read]
        self.last_file_read = (self.last_file_read + 1) % len(self.loaded_data)
        i = random.randint(1, len(s3_keys) - self.time_length)
        # TODO: return proper matrices
        return (
            y[:, i: (i + self.time_length)].astype(float),
            y[:, i: (i + self.time_length)].astype(float)
        )


class OrderBooksDataLoader(DataLoader):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset, batch_size=batch_size, shuffle=False)
