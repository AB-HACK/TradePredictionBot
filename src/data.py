# --- src/data/download_data.py ---
import numpy as np
import yfinance as yf
import pandas as pd
import logging
import re
from .cache_manager import get_cache_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_ticker(ticker):
    """Validate ticker symbol format"""
    if not isinstance(ticker, str):
        return False, "Ticker must be a string"
    if len(ticker) > 10:
        return False, "Ticker too long (max 10 characters)"
    if not re.match(r'^[A-Z0-9.]+$', ticker.upper()):
        return False, "Invalid ticker format (alphanumeric and dots only)"
    return True, "Valid"

def fetch_live_data(ticker, period='1d', interval='1m', use_cache=True):
    """
    Fetch stock data using yfinance with optional caching
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        use_cache (bool): Whether to use temporary caching
    
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        # Validate ticker
        is_valid, error_msg = validate_ticker(ticker)
        if not is_valid:
            logger.error(f"Invalid ticker {ticker}: {error_msg}")
            return None
        
        # Validate period and interval
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        
        if period not in valid_periods:
            logger.error(f"Invalid period: {period}. Must be one of {valid_periods}")
            return None
        
        if interval not in valid_intervals:
            logger.error(f"Invalid interval: {interval}. Must be one of {valid_intervals}")
            return None
        
        logger.info(f"Fetching data for {ticker} (period={period}, interval={interval})")
        
        # Check cache first if enabled
        if use_cache:
            cache = get_cache_manager()
            # Look for existing cached data with same parameters
            for filepath, metadata in cache.cache_metadata.items():
                if (metadata.get('ticker') == ticker and 
                    metadata.get('data_type') == 'raw' and
                    metadata.get('metadata', {}).get('period') == period and
                    metadata.get('metadata', {}).get('interval') == interval):
                    print(f"[CACHE] Loading {ticker} data from cache...")
                    return cache.load_dataframe(filepath)
        
        # Fetch new data
        print(f"[API] Fetching fresh {ticker} data from API...")
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        # Handle MultiIndex columns (yfinance returns MultiIndex for single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            # If single ticker, flatten the columns
            if len(df.columns.levels[1]) == 1:
                df.columns = df.columns.droplevel(1)
            else:
                # Multiple tickers - keep MultiIndex but select first ticker
                df = df.iloc[:, df.columns.get_level_values(1) == ticker]
                df.columns = df.columns.droplevel(1)
        
        if df is None or df.empty:
            logger.warning(f"No data found for {ticker}")
            return None
        
        # Validate data structure
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns for {ticker}: {missing_cols}")
            return None
        
        # Validate data quality
        if len(df) < 1:
            logger.error(f"Insufficient data for {ticker}: {len(df)} rows")
            return None
        
        logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
        
        # Cache the data if enabled
        if use_cache and df is not None:
            cache = get_cache_manager()
            metadata = {
                'period': period,
                'interval': interval,
                'source': 'yfinance',
                'fetch_timestamp': pd.Timestamp.now().isoformat()
            }
            cache.store_dataframe(df, 'raw', ticker, metadata)
        
        return df
    except yf.exceptions.YFinanceException as e:
        logger.error(f"YFinance API error for {ticker}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching {ticker}: {e}", exc_info=True)
        return None


def fetch_multiple_stocks(tickers, period='3y', interval='1d', use_cache=True):
    """
    Fetch data for multiple stocks at once with caching support
    
    Args:
        tickers (list): List of stock ticker symbols
        period (str): Time period
        interval (str): Data interval
        use_cache (bool): Whether to use temporary caching
    
    Returns:
        dict: Dictionary with ticker names as keys and DataFrames as values
    """
    stock_data = {}
    
    # Check if we have all data in cache already
    if use_cache:
        cache = get_cache_manager()
        cached_data = {}
        
        for ticker in tickers:
            for filepath, metadata in cache.cache_metadata.items():
                if (metadata.get('ticker') == ticker and 
                    metadata.get('data_type') == 'raw' and
                    metadata.get('metadata', {}).get('period') == period and
                    metadata.get('metadata', {}).get('interval') == interval):
                    cached_data[ticker] = cache.load_dataframe(filepath)
                    break
        
        # If we have all data cached, return it
        if len(cached_data) == len(tickers):
            print(f"[CACHE] Loaded all {len(tickers)} stocks from cache")
            return cached_data
    
    # Fetch missing data
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = fetch_live_data(ticker, period=period, interval=interval, use_cache=use_cache)
        if df is not None and not df.empty:
            stock_data[ticker] = df
            print(f"[SUCCESS] {ticker}: {df.shape[0]} rows")
        else:
            print(f"[FAILED] Failed to fetch {ticker}")
    
    return stock_data


# Example usage
if __name__ == "__main__":
    # Test fetching a single stock
    print("Testing data fetching...")
    df = fetch_live_data('AAPL', period='1mo', interval='1d')
    if df is not None:
        print(f"AAPL data shape: {df.shape}")
        print(df.head())