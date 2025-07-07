import sys
import os

from yfinance.scrapers.history import pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import fetch_live_data

# List of 5 tickers to test
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
all_data = {ticker: fetch_live_data(ticker, period='1d', interval='1m') for ticker in tickers}

# Fetch and print top 20 rows for each stock
for ticker in tickers:
    print(f"\nFetching data for {ticker}...")
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(df.tail(20))
    else:
        print(f'No data fetched for {ticker}')
      
# checking for missing data
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.isnull().sum())
    else:
        print(f"No data fetched for {ticker}")

# checking for duplicates
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.duplicated().sum())
    else:
        print(f"No data fetched for {ticker}")
        
# visualise the data
for ticker in tickers:
    df = all_data[ticker]
    import matplotlib.pyplot as plt
    if df is not None and not df.empty:
        df.plot(kind='line', y='Open', figsize=(8, 5), subplots=True)
        plt.show()
    else:
        print(f"No data fetched for {ticker}")

# this was a generated code to visualise the data "gpt"
plt.figure(figsize=(12, 6))
for ticker in all_data:
    df = all_data[ticker]
    if df is not None and not df.empty:
        df['Close'].plot(label=ticker)

plt.legend()
plt.title("Stock Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.show()
# 

# major indicies from yfinance
major_indices = pd.read_html("https://yfinance.com/world-indices")[0]
major_indices['Ticker'] = major_indices['Ticker'].str.replace('^', '')

# calculate the P/E ratio, earnings reports, dividends, and other financial metrics
# calculate the correlation between the stocks
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.corr())
    else:
        print(f"No data fetched for {ticker}")
