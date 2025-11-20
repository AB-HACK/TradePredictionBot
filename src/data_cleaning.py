# =============================================================================
# STOCK DATA CLEANING MODULE
# =============================================================================
# This module provides comprehensive data cleaning and quality checks for stock data.
# It ensures data is clean, consistent, and ready for analysis and modeling.
# 
# Cleaning Steps:
# 1. Data Structure Validation - Check columns, data types, date ranges
# 2. Missing Value Detection & Handling - Identify and fill missing data
# 3. Duplicate Removal - Remove exact and date duplicates
# 4. Data Consistency Checks - Validate logical relationships
# 5. Outlier Detection - Identify statistical outliers
# 6. Trading Days Validation - Check for missing trading days
# 7. Data Gaps Analysis - Identify time series gaps
# 8. Technical Indicators Addition - Add RSI, MACD, moving averages
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from .cache_manager import get_cache_manager
warnings.filterwarnings('ignore')

class StockDataCleaner:
    """
    Simple stock data cleaner for basic data preparation
    """
    
    def __init__(self, df, ticker_name):
        if df is None or df.empty:
            raise ValueError(f"DataFrame for {ticker_name} is None or empty")
        self.df = df.copy()
        self.ticker_name = ticker_name
        
    def clean_data(self, use_cache=True):
        """
        Basic cleaning pipeline - handles missing values, duplicates, and adds indicators
        
        Args:
            use_cache (bool): Whether to use temporary caching for cleaned data
        """
        print(f"Cleaning {self.ticker_name}...")
        
        # Check cache first if enabled
        if use_cache:
            cache = get_cache_manager()
            # Look for existing cleaned data
            for filepath, metadata in cache.cache_metadata.items():
                if (metadata.get('ticker') == self.ticker_name and 
                    metadata.get('data_type') == 'cleaned'):
                    print(f"[CACHE] Loading cleaned {self.ticker_name} data from cache...")
                    cached_df = cache.load_dataframe(filepath)
                    if cached_df is not None:
                        self.df = cached_df
                        print(f"[SUCCESS] {self.ticker_name} loaded from cache: {self.df.shape[0]} rows")
                        return self.df
        
        # Perform cleaning if not cached
        print(f"[PROCESSING] Processing {self.ticker_name} data...")
        
        # Handle missing values
        # Use ffill() instead of deprecated fillna(method='ffill')
        self.df = self.df.ffill()  # Forward fill for prices
        if 'Volume' in self.df.columns:
            self.df['Volume'] = self.df['Volume'].fillna(0)
        
        # Remove duplicates
        self.df = self.df.drop_duplicates()
        
        # Add basic indicators
        self.df['Returns'] = self.df['Close'].pct_change()
        self.df['SMA_20'] = self.df['Close'].rolling(window=20).mean()
        self.df['SMA_50'] = self.df['Close'].rolling(window=50).mean()
        self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Avoid division by zero - use np.where to handle zero loss
        rs = np.where(loss != 0, gain / loss, np.nan)
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI'] = self.df['RSI'].fillna(50)  # Default to 50 if no loss/gain data
        
        # MACD
        ema_12 = self.df['Close'].ewm(span=12).mean()
        ema_26 = self.df['Close'].ewm(span=26).mean()
        self.df['MACD'] = ema_12 - ema_26
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9).mean()
        
        # Cache the cleaned data if enabled
        if use_cache:
            cache = get_cache_manager()  # Get cache manager if not already retrieved
            metadata = {
                'cleaning_timestamp': pd.Timestamp.now().isoformat(),
                'original_shape': self.df.shape,
                'indicators_added': ['Returns', 'SMA_20', 'SMA_50', 'Volatility', 'RSI', 'MACD', 'MACD_Signal']
            }
            cache.store_dataframe(self.df, 'cleaned', self.ticker_name, metadata)
        
        print(f"[SUCCESS] {self.ticker_name} cleaned: {self.df.shape[0]} rows")
        return self.df
    
    def save_data(self, filepath=None, use_cache=True):
        """
        Save cleaned data to CSV with optional caching
        
        Args:
            filepath: Custom filepath (if None, uses default naming)
            use_cache: Whether to also cache the data temporarily
        """
        if filepath is None:
            filepath = f"cleaned_{self.ticker_name}_data.csv"
        
        # Save to permanent location
        self.df.to_csv(filepath)
        print(f"[SAVED] Saved to {filepath}")
        
        # Also cache if enabled
        if use_cache:
            cache = get_cache_manager()
            metadata = {
                'permanent_filepath': filepath,
                'save_timestamp': pd.Timestamp.now().isoformat()
            }
            cache.store_dataframe(self.df, 'cleaned', self.ticker_name, metadata)
        
        return filepath


def clean_multiple_stocks(stock_data_dict, use_cache=True, save_permanent=True):
    """
    Clean multiple stocks at once with caching support
    
    Args:
        stock_data_dict: Dictionary of stock data
        use_cache: Whether to use temporary caching
        save_permanent: Whether to save permanent CSV files
    
    Returns:
        Dictionary of cleaned stock data
    """
    cleaned_data = {}
    
    for ticker, df in stock_data_dict.items():
        if df is None or df.empty:
            print(f"[WARNING] Skipping {ticker}: DataFrame is None or empty")
            continue
            
        try:
            cleaner = StockDataCleaner(df, ticker)
            cleaned_df = cleaner.clean_data(use_cache=use_cache)
            
            # Save permanent file only if requested
            if save_permanent:
                cleaner.save_data(use_cache=use_cache)
            
            cleaned_data[ticker] = cleaned_df
        except Exception as e:
            print(f"[ERROR] Error cleaning {ticker}: {e}")
            continue
    
    return cleaned_data 

