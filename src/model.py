from __future__ import annotations
import concurrent
import datetime
import json
import os
import shutil
import tarfile
import traceback
import zlib
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from hashlib import md5
from multiprocessing import Pool

from typing import List, Dict

import dataclasses

import boto3
import msgpack
import numpy as np
import pandas as pd
import scipy
from dateutil.parser import parse
from scipy import sparse

ACCEPTABLE_PRICE_DIFF = 0.03


class S3OrderBookDataSource:
    def __init__(self, bucket_name, s3_client: boto3.client):
        self.bucket_name = bucket_name
        self.s3_client = s3_client

    def fetch_order_book_by_key(
            self,
            key: str):
        s3_response_object = self.s3_client.get_object(
            Bucket=self.bucket_name,
            Key=key
        )
        object_content = s3_response_object['Body'].read()
        decompressed_bytes = zlib.decompress(object_content)
        raw_dict = msgpack.unpackb(
            decompressed_bytes
        )
        return OrderBook(
            s3_key=key,
            timestamp=int(raw_dict[b'timestamp']),
            bids=[
                Bid(price=bid[0] / 10., amount=bid[1]) for bid in
                raw_dict[b'bids']
            ],
            asks=[
                Ask(price=ask[0] / 10., amount=ask[1]) for ask in
                raw_dict[b'asks']
            ]
        )

    def get_keys_per_day_multithreaded(self, day: str, max_workers: int = 32):
        prefixes = self.__get_day_timestamp_range(day)
        future_results = dict()
        results_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for prefix in prefixes:
                future = executor.submit(
                    self.get_keys_per_prefix,
                    prefix
                )
                future_results[future] = prefix
            for future in concurrent.futures.as_completed(future_results):
                try:
                    results_dict[future_results[future]] = future.result()
                except Exception as e:
                    print(traceback.format_exc())

        return reduce(
            lambda x, y: x + y,
            list(results_dict.values()),
            list()
        )

    def get_keys_per_prefix(self, prefix):
        keys = []
        paginator = self.s3_client.get_paginator('list_objects')
        page_iterator = paginator.paginate(
            **{'Bucket': self.bucket_name,
               'Prefix': prefix}
        )
        for page in page_iterator:
            if 'Contents' in page:
                for elem in page['Contents']:
                    keys.append(elem['Key'])
        return keys

    def __get_day_timestamp_range(self, day: str, precision=10000):
        paginator = self.s3_client.get_paginator('list_objects')
        page_iterator = paginator.paginate(
            **{'Bucket': self.bucket_name,
               'Prefix': day}
        )
        for page in page_iterator:
            if 'Contents' in page:
                for elem in page['Contents']:
                    min = int(elem['Key'].split('/')[1])
                    return [
                        f'{day}/{v}'
                        for v in
                        list(
                            range(
                                min // precision,
                                (min + 60 * 60 * 30) // precision
                            )
                        )
                    ]
        return list()

    def get_all_order_books(
            self, keys: List[str], max_workers: int = 16) -> List[OrderBook]:
        future_results = dict()
        results_dict = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for key in keys:
                future = executor.submit(
                    self.fetch_order_book_by_key,
                    key
                )
                future_results[future] = key
            for future in concurrent.futures.as_completed(future_results):
                try:
                    results_dict[future_results[future]] = future.result()
                except Exception as e:
                    print(e)

        return sorted(results_dict.values(), key=lambda ob: ob.timestamp)


@dataclasses.dataclass
class Bid:
    price: float
    amount: float

    def __iter__(self):
        return iter((self.price, self.amount))

    @property
    def amount_usd(self):
        return self.price * self.amount


@dataclasses.dataclass
class Ask:
    price: float
    amount: float

    def __iter__(self):
        return iter((self.price, self.amount))

    @property
    def amount_usd(self):
        return self.price * self.amount


@dataclasses.dataclass
class OrderBooksChunkMetadata:
    start_timestamp: int
    end_timestamp: int
    size: int


@dataclasses.dataclass
class OrderBooksChunk:
    keys: List[str]

    @property
    def sorted_keys(self):
        return sorted(self.keys, key=lambda x: int(x[11:]))

    @property
    def size(self):
        return len(self.keys)

    @property
    def days(self):
        return np.unique([k[:10] for k in self.keys])

    def chunks(self, chunk_size=5000):
        chunks = []
        current_index = 0
        while current_index < self.size:
            chunks.append(
                OrderBooksChunk(
                    keys=self.sorted_keys[
                         current_index: (current_index + chunk_size)
                         ]
                )
            )
            current_index += chunk_size
        return chunks

    def meta_data(self) -> OrderBooksChunkMetadata:
        return OrderBooksChunkMetadata(
            start_timestamp=int(self.sorted_keys[0][11:]),
            end_timestamp=int(self.sorted_keys[-1][11:]),
            size=self.size
        )


@dataclasses.dataclass
class OrderBook:
    def __init__(self, s3_key, timestamp: int, bids: List[Bid],
                 asks: List[Ask]):
        self.s3_key = s3_key
        self.bids = bids
        self.timestamp = timestamp
        self.asks = asks

        self._current_price = (self.best_bid + self.best_ask) / 2.
        self._asks_df = None
        self._bids_df = None
        self._tensor = None
        self._sparse_matrix = None

    @property
    def current_price(self):
        return self._current_price

    @property
    def best_bid(self):
        return self.bids[0].price

    @property
    def best_ask(self):
        return self.asks[0].price

    def spread(self):
        return self.best_ask - self.best_bid

    def _get_asks_df(self,
                     acceptable_price_diff: float = ACCEPTABLE_PRICE_DIFF):
        if self._asks_df is not None:
            return self._asks_df
        asks = []
        amounts = []
        for ask_price, ask_amount in self.asks:
            if ask_price / self.current_price - 1.0 <= acceptable_price_diff:
                asks.append(ask_price / self.current_price - 1.0)
                amounts.append(ask_amount)
        asks_df = pd.DataFrame(
            dict(asks=pd.Series(asks).values,
                 amounts=amounts,
                 amounts_usd=[a * self.current_price for a in amounts]
                 )
        )
        self._asks_df = asks_df
        return asks_df

    def _get_bids_df(self,
                     acceptable_price_diff: float = ACCEPTABLE_PRICE_DIFF):
        if self._bids_df is not None:
            return self._bids_df
        bids = []
        amounts = []

        for bid_price, bid_amount in self.bids:

            if bid_price / self.current_price > (1 - acceptable_price_diff):
                bids.append(bid_price / self.current_price)
                amounts.append(bid_amount)

        bids_df = pd.DataFrame(
            dict(
                bids=pd.Series(bids).values, amounts=amounts,
                amounts_usd=[a * self.current_price for a in amounts]
            )
        )
        self._bids_df = bids_df
        return bids_df

    def get_asks_matrix(self, amount_indices, amount_bins, price_diff_bins,
                        price_diff_indices):
        asks_df = self._get_asks_df()

        x_ind = pd.cut(asks_df.amounts_usd, amount_bins).values
        y_ind = pd.cut(asks_df.asks, price_diff_bins).values

        m_asks = np.zeros((len(amount_indices), len(price_diff_indices)))
        for x, y in zip(x_ind, y_ind):
            ind_x = amount_indices[x]
            ind_y = price_diff_indices[y]
            m_asks[ind_x, ind_y] += 1
        return m_asks

    def get_bids_matrix(self, amount_indices, amount_bins, price_diff_bins,
                        price_diff_indices):
        bids_df = self._get_bids_df()

        x_ind = pd.cut(bids_df.amounts_usd, amount_bins).values
        y_ind = pd.cut(1 - bids_df.bids, price_diff_bins).values

        m_bids = np.zeros((len(amount_bins), len(price_diff_indices)))
        for x, y in zip(x_ind, y_ind):
            ind_x = amount_indices[x]
            ind_y = price_diff_indices[y]
            m_bids[ind_x, ind_y] += 1
        return m_bids

    def get_sparse_matrices(self, amount_indices, amount_bins, price_diff_bins,
                            price_diff_indices):
        m_bids = self.get_bids_matrix(
            amount_indices, amount_bins,
            price_diff_bins,
            price_diff_indices
        )

        m_asks = self.get_asks_matrix(
            amount_indices, amount_bins,
            price_diff_bins,
            price_diff_indices
        )

        return {'bids_sparse': sparse.csr_matrix(m_bids),
                'asks_sparse': sparse.csr_matrix(m_asks)}


def run_job(args):
    day, bucket_name = args
    s3 = boto3.client("s3")
    data_source = S3OrderBookDataSource(bucket_name=bucket_name,
                                        s3_client=s3)

    return data_source.get_keys_per_day_multithreaded(day)


class OrderBookChunksCollection:

    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name

    def get_all_keys(
            self, start_date: datetime.datetime,
            end_date: datetime.datetime, parallelism: int = 16) -> List[str]:

        days = [str(d.date()) for d in pd.date_range(start_date, end_date)]
        with Pool(processes=parallelism) as pool:
            results = pool.map(run_job,
                               [(day, self.bucket_name) for day in days])

        return reduce(lambda x, y: x + y, results, list())

    @staticmethod
    def get_chunks(keys, min_number_of_elements: int = 5000,
                   max_gap_s: int = 12):

        ts = [int(x[11:]) for x in keys]
        df = pd.DataFrame(dict(timestamp=ts, key=keys)).sort_values(
            by='timestamp')

        ts_values = df.timestamp.values
        keys_values = df.key.values
        chunks = []
        chunk = [keys_values[0]]
        for index in range(1, len(ts_values)):
            if (ts_values[index] - ts_values[index - 1]) < max_gap_s:
                chunk.append(keys_values[index])
            else:
                chunks.append(chunk)
                chunk = [keys_values[index]]
        if len(chunk) > 0:
            chunks.append(chunk)

        chunks = [chunk for chunk in chunks if
                  len(chunk) > min_number_of_elements]

        return [OrderBooksChunk(c) for c in chunks]


class OrderBooksDataSequenceBase(ABC):
    def __init__(self, order_books: List[OrderBook]):
        self.order_books = order_books

    def validate(self, max_percentage=2):
        bids_same = 0
        asks_same = 0
        for i, o in enumerate(self.order_books):
            if (i + 3) < len(self.order_books):
                bids_same += (self.order_books[i].bids == self.order_books[
                    i + 3].bids)
                asks_same += (self.order_books[i].asks == self.order_books[
                    i + 3].asks)

        return ((bids_same + asks_same) / len(self.order_books)) <= (
                max_percentage / 100.)

    @property
    def length(self):
        return len(self.order_books)

    def timestamps(self):
        return [o.timestamp for o in self.order_books]

    @property
    def days(self):
        return np.unique(
            [
                parse(o.s3_key[:10]).date() for o in self.order_books
            ]
        )

    def metadata(self):
        return {
            'start_day': str(min(self.days)),
            'length': len(self.order_books),
            'end_day': str(max(self.days)),
            'first_timestamp': min(self.timestamps()),
            'last_timestamp': max(self.timestamps())
        }

    @abstractmethod
    def x(self):
        pass

    @abstractmethod
    def y(self):
        pass

    @abstractmethod
    def save(self, file_path: str):
        pass

    def _y_time_shifted_entries(self, n_seconds, attr_name='current_price'):

        s = pd.Series(list(range(self.length)), index=self.timestamps())

        def __get_future_index(idx_, seconds):
            timestamp = s.index[idx_]
            future_idx = s.loc[timestamp: (timestamp + seconds)].iloc[-1]
            return future_idx

        indices = s.map(
            lambda x: __get_future_index(x, seconds=n_seconds)).values
        values = []
        for i, idx in enumerate(indices):
            if idx + 1 < self.length:
                ref_value = getattr(self.order_books[i], attr_name)
                values.append(
                    getattr(self.order_books[idx], attr_name) / ref_value
                )
            else:
                values.append(None)
        return values


class OrderBooksDataSequenceDatasetV1(OrderBooksDataSequenceBase):
    TMP_DATA_DIR = "/tmp/orderbooks-data/"

    def __init__(self, order_books: List[OrderBook], amount_best_bins,
                 amount_indices, price_diff_best_bins, price_diff_indices):
        super().__init__(order_books)
        self.amount_best_bins = amount_best_bins
        self.amount_indices = amount_indices
        self.price_diff_best_bins = price_diff_best_bins
        self.price_diff_indices = price_diff_indices

    def x(self):
        return [
            o.get_sparse_matrices(
                amount_bins=self.amount_best_bins,
                amount_indices=self.amount_indices,
                price_diff_bins=self.price_diff_best_bins,
                price_diff_indices=self.price_diff_indices
            ) for o in self.order_books
        ]

    def y(self):
        return np.array([
            self._y_time_shifted_entries(n_seconds=30),
            self._y_time_shifted_entries(n_seconds=60),
            self._y_time_shifted_entries(n_seconds=120),
            self._y_time_shifted_entries(n_seconds=300),
            self._y_time_shifted_entries(n_seconds=600),
            self._y_time_shifted_entries(n_seconds=1200),
            self._y_time_shifted_entries(n_seconds=3600),
        ])

    def save(self, filepath):

        unique_tmp_path = os.path.join(self.TMP_DATA_DIR,
                                       md5(filepath).hexdigest())
        if os.path.isdir(unique_tmp_path):
            shutil.rmtree(unique_tmp_path)

        os.makedirs(
            os.path.join(unique_tmp_path),
            exist_ok=True
        )

        x = self.x()
        y = self.y()

        for idx, elem in enumerate(x):
            scipy.sparse.save_npz(
                matrix=elem['bids_sparse'],
                file=os.path.join(unique_tmp_path, f"{idx}-bids-sparse"),
            )
            scipy.sparse.save_npz(
                matrix=elem['asks_sparse'],
                file=os.path.join(unique_tmp_path, f"{idx}-asks-sparse"),
            )
        np.savez_compressed(
            file=os.path.join(unique_tmp_path, "y"),
            y=y,
        )
        np.save(
            file=os.path.join(unique_tmp_path, "idx"),
            arr=np.array([o.s3_key for o in self.order_books]),
        )

        with open(os.path.join(unique_tmp_path, "metadata.json"), "w") as fp:
            json.dump(
                obj=self.metadata(),
                fp=fp
            )

        with tarfile.open(filepath, "w:gz") as tar:
            tar.add(
                unique_tmp_path,
                arcname=os.path.basename(unique_tmp_path)
            )

        shutil.rmtree(unique_tmp_path)

    @staticmethod
    def load(filepath):
        if os.path.isdir(OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR):
            shutil.rmtree(OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR)
        os.makedirs(OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR, exist_ok=True)

        with tarfile.open(filepath, "r:gz") as tar:
            tar.extractall(
                path=OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR
            )

        with np.load(
                file=os.path.join(
                    OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR, "y.npz"
                ), allow_pickle=True
        ) as f:
            y = f['y']

        idx = np.load(
            file=os.path.join(
                OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR,
                "idx.npy"
            ),
            allow_pickle=True
        )

        with open(os.path.join(OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR,
                               "metadata.json"), "r") as fp:
            metadata = json.load(
                fp=fp
            )

        x = []
        for index, _ in enumerate(idx):
            bids_sparse = scipy.sparse.load_npz(
                file=os.path.join(OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR,
                                  f"{index}-bids-sparse.npz")
            )

            asks_sparse = scipy.sparse.load_npz(
                file=os.path.join(OrderBooksDataSequenceDatasetV1.TMP_DATA_DIR,
                                  f"{index}-asks-sparse.npz")
            )

            x.append({'bids_sparse': bids_sparse, 'asks_sparse': asks_sparse})
        return x, y, idx, metadata
