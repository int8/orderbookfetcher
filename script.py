import traceback

import msgpack
import zlib
import boto3
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import pandas as pd
from tqdm import tqdm
import numpy as np

import pickle
import sys

ACCEPTABLE_PRICE_DIFF = 0.03


def decrypt_single_order_book_file(filepath):
    with open(filepath, "rb") as f:
        compressed_bytes = f.read()
        decompressed_bytes = zlib.decompress(compressed_bytes)
        return msgpack.unpackb(
            decompressed_bytes
        )


def fetch_and_decrypt_order_book(s3_client, key):
    s3_response_object = s3_client.get_object(Bucket='btc-order-book', Key=key)
    object_content = s3_response_object['Body'].read()
    decompressed_bytes = zlib.decompress(object_content)
    return msgpack.unpackb(
        decompressed_bytes
    )


def get_keys_per_day(day, s3_client, max_keys=100000):
    keys = []
    paginator = s3_client.get_paginator('list_objects')
    page_iterator = paginator.paginate(
        **{'Bucket': 'btc-order-book',
           'Prefix': day}
    )
    for page in page_iterator:
        for elem in page['Contents']:
            keys.append(elem['Key'])
        if len(keys) > max_keys:
            return keys
    return keys


def get_all_order_books_dicts(s3_client, keys):
    future_results = dict()
    results_dict = {}
    with ThreadPoolExecutor(max_workers=32) as executor:
        for key in keys:
            future = executor.submit(fetch_and_decrypt_order_book, s3_client,
                                     key)
            future_results[future] = key
        for future in concurrent.futures.as_completed(future_results):
            try:
                results_dict[future_results[future]] = future.result()
            except Exception as e:
                print(e)

    return results_dict


def get_asks_df_for_key(order_books, key):
    moment_price = (order_books[key][b'asks'][0][0] / 10) + (
            order_books[key][b'bids'][0][0] / 10)
    moment_price /= 2

    asks = []
    amounts = []
    current_price = []
    for price, amount in order_books[key][b'asks']:
        ask = price / 10
        if ask / moment_price - 1.0 <= ACCEPTABLE_PRICE_DIFF:
            asks.append(ask / moment_price - 1.0)
            amounts.append(amount)
            current_price.append(moment_price)
    asks_df = pd.DataFrame(dict(asks=pd.Series(asks).values, amounts=amounts))
    return asks_df


def get_bids_df_for_key(order_books, key):
    moment_price = (order_books[key][b'asks'][0][0] / 10) + (
            order_books[key][b'bids'][0][0] / 10)
    moment_price /= 2

    bids = []
    amounts = []
    current_price = []
    for price, amount in order_books[key][b'bids']:
        bid = price / 10
        if bid / moment_price > (1 - ACCEPTABLE_PRICE_DIFF):
            bids.append(bid / moment_price)
            amounts.append(amount)
            current_price.append(moment_price)
    bids_df = pd.DataFrame(dict(bids=pd.Series(bids).values, amounts=amounts))
    return bids_df


def create_asks_matrix(asks_df):
    x_ind = pd.cut(asks_df.amounts, AMOUNT_BEST_BINS).values
    y_ind = pd.cut(asks_df.asks, PRICE_DIFF_BEST_BINS).values

    m_asks = np.zeros((len(AMOUNT_INDICES), len(PRICE_DIFF_INDICES)))
    for x, y in zip(x_ind, y_ind):
        ind_x = AMOUNT_INDICES[x]
        ind_y = PRICE_DIFF_INDICES[y]
        m_asks[ind_x, ind_y] += 1
    return m_asks


def create_bids_matrix(bids_df):
    x_ind = pd.cut(bids_df.amounts, AMOUNT_BEST_BINS).values
    y_ind = pd.cut(1 - bids_df.bids, PRICE_DIFF_BEST_BINS).values

    m_bids = np.zeros((len(AMOUNT_BEST_BINS), len(PRICE_DIFF_INDICES)))
    for x, y in zip(x_ind, y_ind):
        ind_x = AMOUNT_INDICES[x]
        ind_y = PRICE_DIFF_INDICES[y]
        m_bids[ind_x, ind_y] += 1

    return m_bids


def create_daily_tensor(asks_dfs, bids_dfs, keys):
    future_results = dict()

    daily_tensor = np.zeros(
        (2, len(keys), len(AMOUNT_INDICES), len(PRICE_DIFF_INDICES)),
        dtype=np.int16)
    with tqdm(total=len(keys) * 2) as pbar:
        with ThreadPoolExecutor(max_workers=1) as executor:
            for i, key in enumerate(keys):
                ask_df = asks_dfs[key]
                bid_df = bids_dfs[key]
                future = executor.submit(create_asks_matrix, ask_df)
                future_results[future] = (i, 'ask')
                future = executor.submit(create_bids_matrix, bid_df)
                future_results[future] = (i, 'bid')
            for future in concurrent.futures.as_completed(future_results):
                try:
                    i = future_results[future][0]
                    slice_type = future_results[future][1]
                    daily_tensor[0 if slice_type == 'ask' else 1, i,
                    :] = future.result()
                    pbar.update(1)
                except Exception as e:
                    print(e)
        return daily_tensor


s3 = boto3.client("s3")

PRICE_DIFF_BEST_BINS = pickle.load(open("price_diff_best_bins.bin", "rb"))
AMOUNT_BEST_BINS = pickle.load(open("amount_best_bins.bin", "rb"))
AMOUNT_INDICES = dict(zip(AMOUNT_BEST_BINS, list(range(len(AMOUNT_BEST_BINS)))))
PRICE_DIFF_INDICES = dict(
    zip(PRICE_DIFF_BEST_BINS, list(range(len(PRICE_DIFF_BEST_BINS)))))

try:
    day = sys.argv[1]
    d = str(day)[:10]
    keys = get_keys_per_day(d, s3)
    order_books = get_all_order_books_dicts(s3, keys)
    bids_dfs = dict()
    asks_dfs = dict()
    for i, key in tqdm(enumerate(keys)):
        asks_dfs[key] = get_asks_df_for_key(order_books, key)
        bids_dfs[key] = get_bids_df_for_key(order_books, key)
    daily_tensor = create_daily_tensor(asks_dfs, bids_dfs, keys)
    np.savez_compressed(f"/tmp/{d}.bin", daily_tensor)
    np.savez_compressed(f"/tmp/{d}_keys.bin", np.array(keys))
    s3.upload_file(Filename=f"/tmp/{d}.bin.npz", Bucket='btc-datasets',
                   Key=f'v1/{d}/order-book-tensor.npz')
    s3.upload_file(Filename=f"/tmp/{d}_keys.bin.npz", Bucket='btc-datasets',
                   Key=f'v1/{d}/keys.npz')
except Exception as e:
    print(traceback.format_exc())
