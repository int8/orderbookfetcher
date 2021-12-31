from multiprocessing import Pool

import boto3

from dateutil.parser import parse

from src.model import OrderBookChunksCollection, \
    S3OrderBookDataSource, OrderBooksDataSequenceDatasetV1, OrderBooksChunk
import pickle

PRICE_DIFF_BEST_BINS = pickle.load(open("data/price_diff_best_bins.bin", "rb"))
AMOUNT_BEST_BINS = pickle.load(open("data/amount_usd_best_bins.bin", "rb"))
AMOUNT_INDICES = dict(zip(AMOUNT_BEST_BINS, list(range(len(AMOUNT_BEST_BINS)))))
PRICE_DIFF_INDICES = dict(
    zip(PRICE_DIFF_BEST_BINS, list(range(len(PRICE_DIFF_BEST_BINS)))))

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
                                                  min_number_of_elements=20000)


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
        dataset = OrderBooksDataSequenceDatasetV1(
            order_books, amount_best_bins=AMOUNT_BEST_BINS,
            amount_indices=AMOUNT_INDICES,
            price_diff_best_bins=PRICE_DIFF_BEST_BINS,
            price_diff_indices=PRICE_DIFF_INDICES
        )
        if dataset.validate():
            meta_data = dataset.metadata()
            file_name = f'{meta_data["start_day"]}-{meta_data["first_timestamp"]}' \
                        f'-{meta_data["last_timestamp"]}.tar.gz'
            dataset.save(file_name)
            print(f"{file_name} saved")
        else:
            print("Dataset is not valid")


with Pool(processes=8) as pool:
    results = pool.map(run_single_chunk, all_chunks)
