import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks
from src.analysis_template import analyze_multiple_stocks

# List of 5 tickers to test
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Fetch 3 years of data using the centralized function with caching
print("🚀 Fetching 3 years of data with caching...")
all_data = fetch_multiple_stocks(tickers, period='3y', interval='1mo', use_cache=True)

# Clean and analyze the data with caching
print("\n🧹 Cleaning and analyzing data with caching...")
cleaned_data = clean_multiple_stocks(all_data, use_cache=True, save_permanent=True)
analyzed_data = analyze_multiple_stocks(cleaned_data, use_cache=True)

# Show cache status
from src.cache_manager import get_cache_manager
cache = get_cache_manager()
cache.print_cache_status()
all_data = analyzed_data  # Use analyzed data for further work

# Fetch and print top 20 rows for each stock
for ticker in tickers:
    print(f"\nFetching data for {ticker}...")
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(df.tail(20))
    else:
        print(f'No data fetched for {ticker}')

# Checking for missing data
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.isnull().sum())
    else:
        print(f"No data fetched for {ticker}")

# validate column names and data types
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.dtypes)
    else:
        print(f"No data fetched for {ticker}")

# check for outliers
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.describe())
    else:
        print(f"No data fetched for {ticker}")

# Checking for duplicates
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.duplicated().sum())
    else:
        print(f"No data fetched for {ticker}")

# Visualise the data
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        df.plot(kind='line', y='Open', figsize=(8, 5), subplots=True)
        plt.show()
    else:
        print(f"No data fetched for {ticker}")

# Major indices from yfinance
major_indices = pd.read_html("https://yfinance.com/world-indices")[0]
major_indices['Ticker'] = major_indices['Ticker'].str.replace('^', '')

# Correlation between the stocks
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        print(f"{ticker} shape: {df.shape}")
        print(df.corr())
    else:
        print(f"No data fetched for {ticker}")

# RSI calculation for each stock and plot (Relative Strength Index)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD calculation for each stock and plot (Moving Average Convergence Divergence)
def compute_macd(series, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Add RSI and MACD to each DataFrame
for ticker in tickers:
    df = all_data[ticker]
    if df is not None and not df.empty:
        df['RSI'] = compute_rsi(df['Close'])
        macd, signal = compute_macd(df['Close'])
        df['MACD'] = macd
        df['MACD_Signal'] = signal

# Plot the RSI, MACD, and Close Price for each stock
ticker = 'AAPL'
df = all_data[ticker]

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
if df is not None and not df.empty and 'Close' in df.columns:
    df['Close'].plot(title=f"{ticker} Close Price")
else:
    plt.title(f"{ticker} Close Price (No Data)")

plt.subplot(3, 1, 2)
if df is not None and not df.empty and 'MACD' in df.columns and 'MACD_Signal' in df.columns:
    df['MACD'].plot(label='MACD')
    df['MACD_Signal'].plot(label='Signal')
else:
    plt.title("MACD (No Data)")
plt.legend()
plt.title("MACD")

plt.subplot(3, 1, 3)
if df is not None and not df.empty and 'RSI' in df.columns:
    df['RSI'].plot()
    plt.title("RSI")
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
else:
    plt.title("RSI (No Data)")

plt.tight_layout()
plt.show()