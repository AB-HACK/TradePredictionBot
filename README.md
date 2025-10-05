# Stock Prediction Bot - Data Pipeline

## 🆕 NEW: Temporary Cache System

The bot now includes a **temporary cache system** that:
- ✅ Stores data temporarily while the program is active
- ✅ Automatically cleans up all cache files when the program terminates
- ✅ Speeds up subsequent runs by loading from cache
- ✅ Reduces API calls and saves bandwidth
- ✅ Organizes data with metadata tracking

### Cache System Files
- `src/cache_manager.py` - Core cache management system
- `demo_cache.py` - Simple demo of the cache system
- `src/test_cache_system.py` - Comprehensive cache testing

## File Structure

### 📁 **Data Fetching** (Single Source)
- `src/data.py` - **ONLY file responsible for fetching data**
  - `fetch_live_data()` - Fetch single stock (with caching)
  - `fetch_multiple_stocks()` - Fetch multiple stocks (with caching)

### 📁 **Data Processing**
- `src/data_cleaning.py` - Clean and prepare data (with caching)
- `src/analysis_template.py` - Analyze data and find features (with caching)

### 📁 **Usage Examples**
- `src/test_fetch.py` - Your original test file (updated with caching)
- `src/cleaning_example.py` - Simple pipeline example
- `demo_cache.py` - **NEW: Cache system demo**

## 🚀 Quick Start

### Option 1: Demo the cache system (RECOMMENDED)
```bash
python demo_cache.py
```

### Option 2: Run your test file (now with caching)
```bash
python src/test_fetch.py
```

### Option 3: Run comprehensive cache tests
```bash
python src/test_cache_system.py
```

### Option 4: Run the example pipeline
```bash
python src/cleaning_example.py
```

## 📊 What You Get

1. **Cleaned CSV files** for each stock (permanent)
2. **Temporary cached data** (automatically cleaned up)
3. **Basic analysis** with key metrics
4. **Feature importance** for model building
5. **Ready data** for your prediction model
6. **Faster subsequent runs** (cache system)

## 🗂️ Cache System Benefits

- **🚀 Performance**: Subsequent runs load from cache instead of API
- **💾 Efficiency**: Reduces API calls and saves bandwidth
- **🧹 Clean**: Automatic cleanup - no leftover temporary files
- **📊 Organized**: Metadata tracking for each cached file
- **⚡ Memory**: Data stored on disk, not in memory
- **🛡️ Safe**: Handles errors gracefully
- **📈 Scalable**: Works with multiple stocks and data types

## 🎯 Next Steps

1. Run the pipeline to get your data
2. Review the analysis results
3. Start building your prediction model!

## 📝 File Responsibilities

- **`src/data.py`** - ONLY file for fetching data (with caching)
- **`src/data_cleaning.py`** - ONLY file for cleaning data (with caching)
- **`src/analysis_template.py`** - ONLY file for analysis (with caching)
- **`src/cache_manager.py`** - Temporary cache system management
- **Other files** - Usage examples and testing

## 🗑️ Cache Cleanup

The cache system automatically cleans up all temporary files when the program terminates. You don't need to worry about manual cleanup - it's all handled automatically!

**Cache location**: System temporary directory with project-specific folder
**Cleanup trigger**: Program exit (automatic via `atexit`)
**Manual cleanup**: Available if needed via `cleanup_all_cache()` 