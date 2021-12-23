import concurrent
import datetime
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from multiprocessing import Pool

from typing import List

import dataclasses

import boto3
import msgpack
import numpy as np
import pandas as pd

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
            timestamp=raw_dict[b'timestamp'],
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

    def get_all_order_books(self, keys: List[str], max_workers: int = 32):
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

        return results_dict


@dataclasses.dataclass
class Bid:
    price: float
    amount: float

    def __iter__(self):
        return iter((self.price, self.amount))


@dataclasses.dataclass
class Ask:
    price: float
    amount: float

    def __iter__(self):
        return iter((self.price, self.amount))


@dataclasses.dataclass
class OrderBooksChunk:
    keys: List[str]

    @property
    def size(self):
        return len(self.keys)

    @property
    def days(self):
        return np.unique([k[:10] for k in self.keys])


@dataclasses.dataclass
class OrderBook:
    def __init__(self, timestamp: datetime.datetime, bids: List[Bid],
                 asks: List[Ask]):
        self.bids = bids
        self.timestamp = timestamp
        self.asks = asks

        self._current_price = (self.best_bid + self.best_ask) / 2.

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
        asks = []
        amounts = []
        for ask_price, ask_amount in self.asks:
            if ask_price / self.current_price - 1.0 <= acceptable_price_diff:
                asks.append(ask_price / self.current_price - 1.0)
                amounts.append(ask_amount)
        asks_df = pd.DataFrame(
            dict(asks=pd.Series(asks).values, amounts=amounts))
        return asks_df

    def _get_bids_df(self,
                     acceptable_price_diff: float = ACCEPTABLE_PRICE_DIFF):
        bids = []
        amounts = []

        for bid_price, bid_amount in self.bids:

            if bid_price / self.current_price > (1 - acceptable_price_diff):
                bids.append(bid_price / self.current_price)
                amounts.append(bid_amount)

        bids_df = pd.DataFrame(
            dict(bids=pd.Series(bids).values, amounts=amounts))
        return bids_df

    def get_tensor(self, amount_indices, amount_bins, price_diff_bins,
                   price_diff_indices):
        bids_df = self._get_bids_df()
        asks_df = self._get_asks_df()

        x_ind = pd.cut(asks_df.amounts, amount_bins).values
        y_ind = pd.cut(asks_df.asks, price_diff_bins).values

        m_asks = np.zeros((len(amount_indices), len(price_diff_indices)))
        for x, y in zip(x_ind, y_ind):
            ind_x = amount_indices[x]
            ind_y = price_diff_indices[y]
            m_asks[ind_x, ind_y] += 1

        x_ind = pd.cut(bids_df.amounts, amount_bins).values
        y_ind = pd.cut(1 - bids_df.bids, price_diff_bins).values

        m_bids = np.zeros((len(amount_bins), len(price_diff_indices)))
        for x, y in zip(x_ind, y_ind):
            ind_x = amount_indices[x]
            ind_y = price_diff_indices[y]
            m_bids[ind_x, ind_y] += 1

        return np.array([m_bids, m_asks])


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
    def get_min_chunks(keys, min_number_of_elements: int = 5000,
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
