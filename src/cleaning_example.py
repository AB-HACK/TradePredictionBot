 # =============================================================================
# STOCK DATA CLEANING EXAMPLE
# =============================================================================
# This script demonstrates how to use the data cleaning module to:
# 1. Fetch 3 years of stock data
# 2. Clean the data using the comprehensive cleaning pipeline
# 3. Save cleaned data to CSV files for further analysis
# 4. Provide a summary of the cleaning results
# =============================================================================

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks
from src.analysis_template import analyze_multiple_stocks

def main():
    """
    Simple workflow: fetch, clean, analyze, and save data
    """
    print("🚀 Starting Stock Data Pipeline")
    print("=" * 40)
    
    # Define tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    
    # Step 1: Fetch data using centralized function
    print("📊 Fetching 3 years of data...")
    stock_data = fetch_multiple_stocks(tickers, period='3y', interval='1d')
    
    # Step 2: Clean data
    print("\n🧹 Cleaning data...")
    cleaned_data = clean_multiple_stocks(stock_data)
    
    # Step 3: Analyze data
    print("\n📈 Analyzing data...")
    analyzed_data = analyze_multiple_stocks(cleaned_data)
    
    print("\n✅ Pipeline completed!")
    print("📁 Cleaned CSV files saved in current directory")
    
    return analyzed_data

if __name__ == "__main__":
    data = main() 