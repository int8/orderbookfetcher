### Cryptocurrency order book fetcher  


Simple script retrieving any cryptocurrency pair via bitstamp API


#### how to use 

- rename `.env.example` to `.env`
- edit `.env` 
    - if you do not want to use exceptionless, delete `EXCEPTIONLESS_*` entries
    - `ORDERBOOK_OUTPUT_DIR_HOST` is a directory path where you want to keep your 
    orderbook files 
    - `CURRENCY_PAIR` is a currency pair you want to monitor/fetch
    - `API_CALL_FREQUENCY_SECONDS` is frequency of api calls (time sampling)

    
- `docker-compose up --build -d` 
- wait 12 months 
- analyze the data 
- ?!?!?
- profit 


#### how to later read the order book: 
use `decrypt_single_order_book_file` on order book filepath 
and do your magic 