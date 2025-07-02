# --- src/data/download_data.py ---
import numpy as np
import yfinance as yf
import pandas as pd
# import tensorflow as tf

# Live data fetch using yfinance
print("Fetching live data...")
def fetch_live_data(ticker, period='1d', interval='1m'):
    print("Entering fetch_live_data function")
    df = yf.download(ticker, period=period, interval=interval)
    print("Leaving fetch_live_data function")
    return df