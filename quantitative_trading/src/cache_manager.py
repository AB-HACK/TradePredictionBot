# =============================================================================
# TEMPORARY CACHE MANAGER FOR STOCK DATA
# =============================================================================
# This module provides a temporary cache system for stock data that:
# 1. Stores data temporarily while the program is active
# 2. Automatically cleans up cache files when the program terminates
# 3. Provides organized storage with timestamps and metadata
# 4. Handles both raw and processed data
# =============================================================================

import os
import tempfile
import shutil
import atexit
import time
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class TemporaryCacheManager:
    """
    Manages temporary cache files for stock data with automatic cleanup
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the cache manager
        
        Args:
            cache_dir: Custom cache directory. If None, uses system temp directory
        """
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cache_metadata = {}
        self.active_files = set()
        
        # Create cache directory
        if cache_dir is None:
            # Use system temp directory with project-specific folder
            base_temp = tempfile.gettempdir()
            self.cache_dir = os.path.join(base_temp, "TradePredictionBot_Cache", self.session_id)
        else:
            self.cache_dir = os.path.join(cache_dir, f"cache_{self.session_id}")
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Register cleanup function to run on program exit
        atexit.register(self.cleanup_cache)
        
        print(f"[CACHE] Cache initialized at: {self.cache_dir}")
    
    def _generate_cache_filename(self, data_type: str, ticker: str = None, suffix: str = "") -> str:
        """
        Generate a standardized cache filename
        
        Args:
            data_type: Type of data (raw, cleaned, analyzed, etc.)
            ticker: Stock ticker symbol (optional)
            suffix: Additional suffix for filename
        
        Returns:
            str: Generated filename
        """
        timestamp = datetime.now().strftime("%H%M%S")
        
        if ticker:
            filename = f"{data_type}_{ticker}_{timestamp}{suffix}"
        else:
            filename = f"{data_type}_{timestamp}{suffix}"
        
        return filename
    
    def store_dataframe(self, df: pd.DataFrame, data_type: str, ticker: str = None, 
                       metadata: Dict[str, Any] = None) -> str:
        """
        Store a DataFrame in the cache
        
        Args:
            df: DataFrame to store
            data_type: Type of data (raw, cleaned, analyzed, etc.)
            ticker: Stock ticker symbol
            metadata: Additional metadata to store
        
        Returns:
            str: Path to the cached file
        """
        if df is None or df.empty:
            print(f"⚠️  Warning: Attempting to cache empty DataFrame for {data_type}")
            return None
        
        # Generate filename and path
        filename = self._generate_cache_filename(data_type, ticker, ".csv")
        filepath = os.path.join(self.cache_dir, filename)
        
        try:
            # Save DataFrame
            df.to_csv(filepath, index=True)
            
            # Store metadata
            cache_info = {
                'filepath': filepath,
                'data_type': data_type,
                'ticker': ticker,
                'timestamp': datetime.now().isoformat(),
                'shape': df.shape,
                'columns': list(df.columns),
                'metadata': metadata or {}
            }
            
            self.cache_metadata[filepath] = cache_info
            self.active_files.add(filepath)
            
            print(f"[CACHE] Cached {data_type} data for {ticker or 'general'}: {df.shape[0]} rows")
            return filepath
            
        except Exception as e:
            print(f"[ERROR] Error caching data: {e}")
            return None
    
    def store_multiple_dataframes(self, data_dict: Dict[str, pd.DataFrame], 
                                data_type: str, metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Store multiple DataFrames in the cache
        
        Args:
            data_dict: Dictionary with tickers as keys and DataFrames as values
            data_type: Type of data (raw, cleaned, analyzed, etc.)
            metadata: Additional metadata to store
        
        Returns:
            Dict[str, str]: Dictionary mapping tickers to cache file paths
        """
        cached_paths = {}
        
        for ticker, df in data_dict.items():
            if df is not None and not df.empty:
                path = self.store_dataframe(df, data_type, ticker, metadata)
                if path:
                    cached_paths[ticker] = path
        
        return cached_paths
    
    def load_dataframe(self, filepath: str) -> Optional[pd.DataFrame]:
        """
        Load a DataFrame from cache
        
        Args:
            filepath: Path to the cached file
        
        Returns:
            pd.DataFrame or None if loading fails
        """
        try:
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                print(f"[CACHE] Loaded cached data: {df.shape[0]} rows")
                return df
            else:
                print(f"[WARNING] Cache file not found: {filepath}")
                return None
        except Exception as e:
            print(f"[ERROR] Error loading cached data: {e}")
            return None
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about cached files
        
        Returns:
            Dict containing cache information
        """
        cache_stats = {
            'cache_dir': self.cache_dir,
            'session_id': self.session_id,
            'total_files': len(self.active_files),
            'total_size_mb': 0,
            'files': []
        }
        
        for filepath in self.active_files:
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                cache_stats['total_size_mb'] += file_size
                
                file_info = {
                    'filename': os.path.basename(filepath),
                    'size_mb': round(file_size, 2),
                    'metadata': self.cache_metadata.get(filepath, {})
                }
                cache_stats['files'].append(file_info)
        
        return cache_stats
    
    def print_cache_status(self):
        """Print current cache status"""
        info = self.get_cache_info()
        
        print(f"\n[CACHE STATUS]")
        print(f"Directory: {info['cache_dir']}")
        print(f"Session ID: {info['session_id']}")
        print(f"Files cached: {info['total_files']}")
        print(f"Total size: {info['total_size_mb']:.2f} MB")
        
        if info['files']:
            print(f"\n[CACHED FILES]")
            for file_info in info['files']:
                print(f"  - {file_info['filename']} ({file_info['size_mb']} MB)")
    
    def cleanup_cache(self):
        """
        Clean up all cache files and directories
        This function is automatically called on program exit
        """
        try:
            if os.path.exists(self.cache_dir):
                # Calculate size before deletion
                total_size = 0
                for filepath in self.active_files:
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
                
                # Remove the entire cache directory
                shutil.rmtree(self.cache_dir)
                
                size_mb = total_size / (1024 * 1024)
                print(f"\n[CACHE] Cache cleaned up: {len(self.active_files)} files ({size_mb:.2f} MB)")
                print(f"[CACHE] Removed cache directory: {self.cache_dir}")
                
        except Exception as e:
            print(f"[WARNING] Error during cache cleanup: {e}")
    
    def manual_cleanup(self):
        """
        Manually trigger cache cleanup (useful for testing or explicit cleanup)
        """
        print("[CACHE] Manual cache cleanup triggered...")
        self.cleanup_cache()
        self.active_files.clear()
        self.cache_metadata.clear()


# Global cache manager instance
_cache_manager = None

def get_cache_manager(cache_dir: Optional[str] = None) -> TemporaryCacheManager:
    """
    Get or create the global cache manager instance
    
    Args:
        cache_dir: Custom cache directory
    
    Returns:
        TemporaryCacheManager: Global cache manager instance
    """
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = TemporaryCacheManager(cache_dir)
    
    return _cache_manager


def cleanup_all_cache():
    """Manually trigger cleanup of all cache files"""
    global _cache_manager
    
    if _cache_manager is not None:
        _cache_manager.manual_cleanup()
        _cache_manager = None


# Example usage and testing
if __name__ == "__main__":
    # Test the cache system
    import numpy as np
    
    print("Testing Temporary Cache Manager...")
    
    # Create cache manager
    cache = get_cache_manager()
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100),
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000, 10000, 100)
    })
    
    # Store data
    cache.store_dataframe(sample_data, 'raw', 'TEST')
    cache.store_dataframe(sample_data, 'cleaned', 'TEST')
    
    # Print cache status
    cache.print_cache_status()
    
    # Test loading
    loaded_data = cache.load_dataframe(cache.cache_metadata[list(cache.cache_metadata.keys())[0]]['filepath'])
    print(f"Loaded data shape: {loaded_data.shape}")
    
    print("Cache test completed. Files will be cleaned up on exit.")
