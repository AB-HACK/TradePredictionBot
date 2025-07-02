import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import fetch_live_data

df = fetch_live_data('AAPL', period='1d', interval='1m')
print(df.tail(20))

# List of 5 tickers to test
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Fetch and print top 5 rows for each stock
for ticker in tickers:
    print(f"\nFetching data for {ticker}...")
    df = fetch_live_data(ticker, period='1d', interval='1m')
    print(df.tail(5))