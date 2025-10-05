# =============================================================================
# CACHE SYSTEM TESTING SCRIPT
# =============================================================================
# This script demonstrates the temporary cache system for stock data.
# It shows how data is cached during processing and automatically cleaned up.
# =============================================================================

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks
from src.analysis_template import analyze_multiple_stocks
from src.cache_manager import get_cache_manager, cleanup_all_cache

def test_cache_system():
    """
    Test the complete cache system with stock data processing
    """
    print("🚀 Testing Temporary Cache System")
    print("=" * 60)
    
    # Initialize cache manager
    cache = get_cache_manager()
    cache.print_cache_status()
    
    # List of tickers to test
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    print(f"\n📊 Processing {len(tickers)} stocks with caching enabled...")
    
    # Step 1: Fetch data (will be cached)
    print("\n1️⃣ FETCHING DATA")
    print("-" * 30)
    all_data = fetch_multiple_stocks(tickers, period='1y', interval='1mo', use_cache=True)
    
    # Print cache status after fetching
    cache.print_cache_status()
    
    # Step 2: Clean data (will be cached)
    print("\n2️⃣ CLEANING DATA")
    print("-" * 30)
    cleaned_data = clean_multiple_stocks(all_data, use_cache=True, save_permanent=False)
    
    # Print cache status after cleaning
    cache.print_cache_status()
    
    # Step 3: Analyze data (will be cached)
    print("\n3️⃣ ANALYZING DATA")
    print("-" * 30)
    analyzed_data = analyze_multiple_stocks(cleaned_data, use_cache=True)
    
    # Print final cache status
    cache.print_cache_status()
    
    # Step 4: Test cache retrieval (should load from cache)
    print("\n4️⃣ TESTING CACHE RETRIEVAL")
    print("-" * 30)
    print("Re-running analysis (should load from cache)...")
    analyzed_data_2 = analyze_multiple_stocks(cleaned_data, use_cache=True)
    
    # Step 5: Display sample results
    print("\n5️⃣ SAMPLE RESULTS")
    print("-" * 30)
    for ticker in tickers:
        if ticker in analyzed_data_2:
            df = analyzed_data_2[ticker]
            print(f"\n{ticker} Analysis Summary:")
            print(f"  • Data shape: {df.shape}")
            print(f"  • Columns: {list(df.columns)}")
            if 'Returns' in df.columns:
                print(f"  • Mean return: {df['Returns'].mean():.4f}")
            if 'RSI' in df.columns:
                current_rsi = df['RSI'].iloc[-1]
                print(f"  • Current RSI: {current_rsi:.2f}")
    
    # Step 6: Test cache without permanent saves
    print("\n6️⃣ TESTING CACHE-ONLY MODE")
    print("-" * 30)
    print("Processing with cache only (no permanent files)...")
    
    # Clean data without saving permanent files
    cleaned_data_no_save = clean_multiple_stocks(all_data, use_cache=True, save_permanent=False)
    print("✅ Data cleaned and cached (no permanent files created)")
    
    # Final cache status
    print("\n7️⃣ FINAL CACHE STATUS")
    print("-" * 30)
    cache.print_cache_status()
    
    print("\n🎉 Cache system test completed!")
    print("📝 Cache files will be automatically cleaned up when the program exits.")
    
    return analyzed_data_2

def test_cache_cleanup():
    """
    Test manual cache cleanup
    """
    print("\n🧹 Testing manual cache cleanup...")
    
    # Get cache info before cleanup
    cache = get_cache_manager()
    cache_info = cache.get_cache_info()
    
    print(f"Files before cleanup: {cache_info['total_files']}")
    print(f"Size before cleanup: {cache_info['total_size_mb']:.2f} MB")
    
    # Manual cleanup
    cleanup_all_cache()
    
    print("✅ Manual cleanup completed")

def demonstrate_cache_benefits():
    """
    Demonstrate the benefits of caching
    """
    print("\n💡 CACHE SYSTEM BENEFITS")
    print("=" * 40)
    
    benefits = [
        "🚀 Faster subsequent runs - data loaded from cache instead of API",
        "💾 Reduced API calls - saves rate limits and bandwidth",
        "🔄 Automatic cleanup - no leftover temporary files",
        "📊 Organized storage - metadata tracking for each cached file",
        "⚡ Memory efficient - data stored on disk, not in memory",
        "🛡️ Safe operations - handles errors gracefully",
        "📈 Scalable - works with multiple stocks and data types"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print(f"\n📁 Cache location: {get_cache_manager().cache_dir}")
    print("🗑️  Automatic cleanup on program exit")

if __name__ == "__main__":
    try:
        # Run the main test
        results = test_cache_system()
        
        # Demonstrate benefits
        demonstrate_cache_benefits()
        
        # Test manual cleanup (optional)
        # test_cache_cleanup()
        
        print("\n✨ All tests completed successfully!")
        print("💡 The cache system is now ready for use in your trading bot.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Note: Automatic cleanup will happen on program exit
        print("\n📝 Note: Cache will be automatically cleaned up on program exit.")
