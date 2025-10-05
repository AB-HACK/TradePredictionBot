#!/usr/bin/env python3
"""
DEMO: Temporary Cache System for Trade Prediction Bot
====================================================

This demo shows how the temporary cache system works:
1. Data is fetched and stored temporarily in cache
2. Subsequent operations load from cache (faster)
3. All temporary files are automatically cleaned up on exit

Run this script to see the cache system in action!
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks
from src.analysis_template import analyze_multiple_stocks
from src.cache_manager import get_cache_manager

def main():
    print("DEMO: Temporary Cache System")
    print("=" * 50)
    print("This demo shows how data is cached temporarily and cleaned up automatically.")
    print()
    
    # Initialize cache
    cache = get_cache_manager()
    print(f"Cache location: {cache.cache_dir}")
    print()
    
    # Test with a few stocks
    tickers = ['AAPL', 'MSFT']
    
    print("1. FETCHING DATA (will be cached)")
    print("-" * 30)
    data = fetch_multiple_stocks(tickers, period='6mo', interval='1mo', use_cache=True)
    cache.print_cache_status()
    
    print("\n2. CLEANING DATA (will be cached)")
    print("-" * 30)
    cleaned = clean_multiple_stocks(data, use_cache=True, save_permanent=False)
    cache.print_cache_status()
    
    print("\n3. ANALYZING DATA (will be cached)")
    print("-" * 30)
    analyzed = analyze_multiple_stocks(cleaned, use_cache=True)
    cache.print_cache_status()
    
    print("\n4. RE-RUNNING ANALYSIS (loads from cache)")
    print("-" * 30)
    print("This should be much faster as it loads from cache...")
    analyzed_2 = analyze_multiple_stocks(cleaned, use_cache=True)
    
    print("\n[SUCCESS] Demo completed!")
    print("[INFO] Cache files will be automatically deleted when this script exits.")
    
    # Show some results
    print("\nSample Results:")
    for ticker in tickers:
        if ticker in analyzed_2:
            df = analyzed_2[ticker]
            print(f"  {ticker}: {df.shape[0]} rows, {len(df.columns)} columns")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Demo interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
    finally:
        print("\n[INFO] Cache cleanup will happen automatically...")
