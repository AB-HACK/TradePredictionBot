# --- src/data/download_data.py ---
import numpy as np
import yfinance as yf
import pandas as pd

def fetch_live_data(ticker, period='1d', interval='1m'):
    """
    Fetch stock data using yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        df = yf.download(ticker, period=period, interval=interval)
        if df is None or df.empty:
            print(f"⚠️  No data found for {ticker}")
            return None
        return df
    except Exception as e:
        print(f"❌ Error fetching {ticker}: {e}")
        return None


def fetch_multiple_stocks(tickers, period='3y', interval='1d'):
    """
    Fetch data for multiple stocks at once
    
    Args:
        tickers (list): List of stock ticker symbols
        period (str): Time period
        interval (str): Data interval
    
    Returns:
        dict: Dictionary with ticker names as keys and DataFrames as values
    """
    stock_data = {}
    
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = fetch_live_data(ticker, period=period, interval=interval)
        if df is not None and not df.empty:
            stock_data[ticker] = df
            print(f"✅ {ticker}: {df.shape[0]} rows")
        else:
            print(f"❌ Failed to fetch {ticker}")
    
    return stock_data


# Example usage
if __name__ == "__main__":
    # Test fetching a single stock
    print("Testing data fetching...")
    df = fetch_live_data('AAPL', period='1mo', interval='1d')
    if df is not None:
        print(f"AAPL data shape: {df.shape}")
        print(df.head())