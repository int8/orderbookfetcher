import requests
import msgpack
import zlib
import traceback
import time
import schedule
import os


EXCEPTIONLESS_API_KEY = os.environ.get("EXCEPTIONLESS_API_KEY")
EXCEPTIONLESS_PROJECT_ID = os.environ.get('EXCEPTIONLESS_PROJECT_ID')
API_CALL_FREQUENCY_SECONDS = os.environ.get('API_CALL_FREQUENCY', 4)
CURRENCY_PAIR = os.environ.get('CURRENCY_PAIR', 'btcusd')

EXCHANGE_API_URL = f"https://www.bitstamp.net/api/v2/order_book/{CURRENCY_PAIR}/"
EXCEPTIONLESS_API_EVENT_URL = (
    'https://api.exceptionless.io/api/v2'
    f'/projects/{EXCEPTIONLESS_PROJECT_ID}/events'
)

ORDERBOOK_OUTPUT_DIR = "/app/data/"


def decrypt_single_order_book_file(filepath):
    with open(filepath, "rb") as f:
        compressed_bytes = f.read()
        decompressed_bytes = zlib.decompress(compressed_bytes)
        return msgpack.unpackb(
            decompressed_bytes
        )


def send_exception_to_exceptionless(token, msg):
    headers = {"Authorization": "Bearer {0}".format(token)}
    requests.post(
        EXCEPTIONLESS_API_EVENT_URL + '?userAgent=orderbookfetcher',
        json=dict(
            timestamp=str(time.time()),
            type='error',
            message=msg
        ),
        headers=headers
    )


def get_current_btc_usd_orderbook(url=EXCHANGE_API_URL):
    response = requests.get(url)
    if response.status_code != 200:
        # alert here
        raise ConnectionError("HTTP Response code != 200, please make sure "
                              "API is responsive ")
    json_result = response.json()
    timestamp = json_result['timestamp']
    bids = [
        (int(float(price) * 10), float(amount))
        for price, amount in json_result['bids']
    ]
    asks = [
        (int(float(price) * 10), float(amount))
        for price, amount in json_result['asks']
    ]
    price = (bids[0][0] / 10) + (asks[0][0] / 10)
    price /= 2
    asks_25_prct = list(
        filter(lambda x: x[0] / 10 < price * 1.25, asks)
    )
    bids_25_prct = list(
        filter(lambda x: x[0] / 10 > price * 0.75, bids)
    )
    return timestamp, bids_25_prct, asks_25_prct


def the_job():
    try:
        timestamp, bids, asks = get_current_btc_usd_orderbook()
        bytes = msgpack.packb(
            {'timestamp': timestamp, 'bids': bids, 'asks': asks},
        )

        compressed_bytes = zlib.compress(bytes, 9)
        output_filepath = os.path.join(ORDERBOOK_OUTPUT_DIR, str(timestamp))
        with open(output_filepath, "wb") as f:
            f.write(compressed_bytes)
    except:
        msg = traceback.format_exc()
        if EXCEPTIONLESS_API_KEY:
            send_exception_to_exceptionless(EXCEPTIONLESS_API_KEY, msg)


schedule.every(4).seconds.do(the_job)

while 1:
    schedule.run_pending()
    time.sleep(1)
