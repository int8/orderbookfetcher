import os
from multiprocessing import Pool

import boto3

from dateutil.parser import parse

from src.model import OrderBookChunksCollection, \
    S3OrderBookDataSource, OrderBooksDataSequenceDatasetV1, OrderBooksChunk
import pickle

OUTPUT_DIR = "output"
PRICE_DIFF_BEST_BINS_256 = pickle.load(
    open("data/price_diff_best_bins_256_003_handcrafted_including_negative.bin",
         "rb"))
AMOUNT_BEST_BINS_256 = pickle.load(
    open("data/amount_usd_best_bins_256_003.bin", "rb"))
AMOUNT_INDICES_256 = dict(
    zip(AMOUNT_BEST_BINS_256, list(range(len(AMOUNT_BEST_BINS_256)))))
PRICE_DIFF_INDICES_256 = dict(
    zip(PRICE_DIFF_BEST_BINS_256, list(range(len(PRICE_DIFF_BEST_BINS_256)))))

PRICE_DIFF_BEST_BINS_64 = pickle.load(
    open("data/price_diff_best_bins_64_003_handcrafted_including_negative.bin",
         "rb"))
AMOUNT_BEST_BINS_64 = pickle.load(
    open("data/amount_usd_best_bins_64_003.bin", "rb"))
AMOUNT_INDICES_64 = dict(
    zip(AMOUNT_BEST_BINS_64, list(range(len(AMOUNT_BEST_BINS_64)))))
PRICE_DIFF_INDICES_64 = dict(
    zip(PRICE_DIFF_BEST_BINS_64, list(range(len(PRICE_DIFF_BEST_BINS_64)))))

import sys

start_date, end_date = sys.argv[1:]

order_book_col = OrderBookChunksCollection(bucket_name="btc-order-book")

print("About to fetch all keys ")
all_keys = order_book_col.get_all_keys(
    start_date=parse(start_date),
    end_date=parse(end_date)
)

print("Keys fetched")
print("Chunking keys up")
all_chunks = OrderBookChunksCollection.get_chunks(keys=all_keys,
                                                  min_number_of_elements=10000)


def run_single_chunk(order_book_chunk: OrderBooksChunk):
    print(f"Running single chunk for {order_book_chunk.days}")

    s3_client = boto3.client("s3")
    source = S3OrderBookDataSource(
        bucket_name="btc-order-book",
        s3_client=s3_client
    )

    chunks = order_book_chunk.chunks(chunk_size=5000)
    for chunk in chunks:
        order_books = source.get_all_order_books(chunk.keys)

        dataset_64 = OrderBooksDataSequenceDatasetV1(
            order_books, amount_best_bins=AMOUNT_BEST_BINS_64,
            amount_indices=AMOUNT_INDICES_64,
            price_diff_best_bins=PRICE_DIFF_BEST_BINS_64,
            price_diff_indices=PRICE_DIFF_INDICES_64
        )
        if dataset_64.validate():
            meta_data = dataset_64.metadata()
            file_name = f'{meta_data["start_day"]}-{meta_data["first_timestamp"]}' \
                        f'-{meta_data["last_timestamp"]}_64.tar.gz'
            dataset_64.save(os.path.join(OUTPUT_DIR, file_name))
            print(f"{file_name} saved")

        dataset_256 = OrderBooksDataSequenceDatasetV1(
            order_books, amount_best_bins=AMOUNT_BEST_BINS_256,
            amount_indices=AMOUNT_INDICES_256,
            price_diff_best_bins=PRICE_DIFF_BEST_BINS_256,
            price_diff_indices=PRICE_DIFF_INDICES_256
        )

        if dataset_256.validate():
            meta_data = dataset_256.metadata()
            file_name = f'{meta_data["start_day"]}-{meta_data["first_timestamp"]}' \
                        f'-{meta_data["last_timestamp"]}_256.tar.gz'
            dataset_256.save(os.path.join(OUTPUT_DIR, file_name))
            print(f"{file_name} saved")


with Pool(processes=8) as pool:
    results = pool.map(run_single_chunk, all_chunks)
