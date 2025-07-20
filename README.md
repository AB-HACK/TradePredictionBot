# Stock Prediction Bot - Data Pipeline

## File Structure

### 📁 **Data Fetching** (Single Source)
- `src/data.py` - **ONLY file responsible for fetching data**
  - `fetch_live_data()` - Fetch single stock
  - `fetch_multiple_stocks()` - Fetch multiple stocks

### 📁 **Data Processing**
- `src/data_cleaning.py` - Clean and prepare data
- `src/analysis_template.py` - Analyze data and find features

### 📁 **Usage Examples**
- `src/test_fetch.py` - Your original test file (updated)
- `src/cleaning_example.py` - Simple pipeline example

## 🚀 Quick Start

### Option 1: Run your test file
```bash
python src/test_fetch.py
```

### Option 2: Run the example pipeline
```bash
python src/cleaning_example.py
```

## 📊 What You Get

1. **Cleaned CSV files** for each stock
2. **Basic analysis** with key metrics
3. **Feature importance** for model building
4. **Ready data** for your prediction model

## 🎯 Next Steps

1. Run the pipeline to get your data
2. Review the analysis results
3. Start building your prediction model!

## 📝 File Responsibilities

- **`src/data.py`** - ONLY file for fetching data
- **`src/data_cleaning.py`** - ONLY file for cleaning data
- **`src/analysis_template.py`** - ONLY file for analysis
- **Other files** - Usage examples and testing 