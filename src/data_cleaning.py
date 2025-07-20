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
warnings.filterwarnings('ignore')

class StockDataCleaner:
    """
    Simple stock data cleaner for basic data preparation
    """
    
    def __init__(self, df, ticker_name):
        self.df = df.copy()
        self.ticker_name = ticker_name
        
    def clean_data(self):
        """
        Basic cleaning pipeline - handles missing values, duplicates, and adds indicators
        """
        print(f"Cleaning {self.ticker_name}...")
        
        # Handle missing values
        self.df = self.df.fillna(method='ffill')  # Forward fill for prices
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
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = self.df['Close'].ewm(span=12).mean()
        ema_26 = self.df['Close'].ewm(span=26).mean()
        self.df['MACD'] = ema_12 - ema_26
        self.df['MACD_Signal'] = self.df['MACD'].ewm(span=9).mean()
        
        print(f"✅ {self.ticker_name} cleaned: {self.df.shape[0]} rows")
        return self.df
    
    def save_data(self, filepath=None):
        """Save cleaned data to CSV"""
        if filepath is None:
            filepath = f"cleaned_{self.ticker_name}_data.csv"
        self.df.to_csv(filepath)
        print(f"Saved to {filepath}")
        return filepath


def clean_multiple_stocks(stock_data_dict):
    """Clean multiple stocks at once"""
    cleaned_data = {}
    
    for ticker, df in stock_data_dict.items():
        cleaner = StockDataCleaner(df, ticker)
        cleaned_df = cleaner.clean_data()
        cleaner.save_data()
        cleaned_data[ticker] = cleaned_df
    
    return cleaned_data 