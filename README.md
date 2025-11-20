# Stock Prediction Bot - Data Pipeline

A comprehensive Python application for fetching, cleaning, analyzing, and preparing stock market data for predictive modeling. This project includes a temporary caching system for efficient data processing and provides a complete pipeline from raw data to analysis-ready datasets.

## 📋 Table of Contents

- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [Cache System](#-cache-system)
- [Troubleshooting](#-troubleshooting)

## ✨ Features

- **Data Fetching**: Fetch stock data from Yahoo Finance using `yfinance`
- **Data Cleaning**: Comprehensive data cleaning pipeline with missing value handling, duplicate removal, and outlier detection
- **Technical Indicators**: Automatic calculation of RSI, MACD, moving averages, and volatility metrics
- **Data Analysis**: Feature importance analysis and correlation studies
- **Temporary Caching**: Efficient caching system that automatically cleans up temporary files
- **Multiple Stocks**: Process multiple stocks simultaneously
- **Export to CSV**: Save cleaned and analyzed data to CSV files

## 🔧 Prerequisites

Before you begin, ensure you have:

- **Python 3.7 or higher** installed on your system
- **pip** (Python package installer)
- **Internet connection** (for fetching stock data)

To check your Python version:
```bash
python --version
# or
python3 --version
```

## 📦 Installation

### Step 1: Clone or Navigate to the Project

```bash
cd TradePredictionBot
```

### Step 2: Install Dependencies

Install all required packages using pip:

```bash
pip install -r requirements.txt
```

This will install:
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance data fetching
- `matplotlib` - Data visualization
- `seaborn` - Statistical visualization
- `scikit-learn` - Machine learning utilities
- `statsmodels` - Statistical modeling
- `streamlit` - Web interface (optional)
- `xgboost` - Gradient boosting (optional)

### Step 3: Verify Installation

Test that everything is installed correctly:

```bash
python -c "import pandas, numpy, yfinance; print('✅ All packages installed successfully!')"
```

## 🚀 Quick Start

### Option 1: Run the Cache Demo (Recommended for First Time)

This demonstrates the caching system with a simple example:

```bash
python demo_cache.py
```

**What it does:**
- Fetches stock data for AAPL and MSFT
- Demonstrates caching behavior
- Shows automatic cache cleanup

### Option 2: Run the Complete Pipeline Example

Run the full data pipeline (fetch → clean → analyze):

```bash
python src/cleaning_example.py
```

**What it does:**
- Fetches 3 years of data for 5 stocks (AAPL, MSFT, GOOGL, AMZN, TSLA)
- Cleans the data
- Performs analysis
- Saves cleaned CSV files to the current directory

### Option 3: Run Your Custom Test Script

```bash
python src/test_fetch.py
```

**What it does:**
- Fetches and processes multiple stocks
- Shows data validation
- Displays visualizations

## 📁 Project Structure

```
TradePredictionBot/
│
├── src/                          # Core source code
│   ├── data.py                   # Data fetching module (ONLY file for fetching)
│   ├── data_cleaning.py          # Data cleaning module
│   ├── analysis_template.py      # Data analysis module
│   ├── cache_manager.py          # Temporary cache management
│   ├── cleaning_example.py       # Example: Complete pipeline
│   ├── test_fetch.py             # Example: Test data fetching
│   └── test_cache_system.py      # Example: Test cache system
│
├── demo_cache.py                 # Cache system demonstration
│
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── cleaned_*.csv                 # Output: Cleaned data files (generated)
│
└── quantitative_trading/         # Advanced trading system (separate module)
    ├── README.md                 # Quantitative trading documentation
    └── ...
```

## 📖 Usage Guide

### Basic Usage: Fetching Data

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import fetch_live_data, fetch_multiple_stocks

# Fetch single stock
df = fetch_live_data('AAPL', period='1y', interval='1d', use_cache=True)

# Fetch multiple stocks
stocks = fetch_multiple_stocks(['AAPL', 'MSFT', 'GOOGL'], period='3y', interval='1d')
```

### Complete Pipeline: Fetch → Clean → Analyze

```python
from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks
from src.analysis_template import analyze_multiple_stocks

# Step 1: Fetch data
tickers = ['AAPL', 'MSFT', 'GOOGL']
stock_data = fetch_multiple_stocks(tickers, period='3y', interval='1d', use_cache=True)

# Step 2: Clean data
cleaned_data = clean_multiple_stocks(stock_data, use_cache=True, save_permanent=True)

# Step 3: Analyze data
analyzed_data = analyze_multiple_stocks(cleaned_data, use_cache=True)
```

### Parameters

#### Data Fetching (`fetch_live_data` / `fetch_multiple_stocks`)

- `ticker` / `tickers`: Stock ticker symbol(s) (e.g., 'AAPL', ['AAPL', 'MSFT'])
- `period`: Time period - `'1d'`, `'5d'`, `'1mo'`, `'3mo'`, `'6mo'`, `'1y'`, `'2y'`, `'5y'`, `'10y'`, `'ytd'`, `'max'`
- `interval`: Data interval - `'1m'`, `'2m'`, `'5m'`, `'15m'`, `'30m'`, `'60m'`, `'90m'`, `'1h'`, `'1d'`, `'5d'`, `'1wk'`, `'1mo'`, `'3mo'`
- `use_cache`: Enable/disable temporary caching (default: `True`)

#### Data Cleaning (`clean_multiple_stocks`)

- `stock_data_dict`: Dictionary of stock DataFrames
- `use_cache`: Enable/disable caching (default: `True`)
- `save_permanent`: Save CSV files to disk (default: `True`)

## 💾 Cache System

The project includes an intelligent temporary cache system that:

### Benefits

- ✅ **Performance**: Subsequent runs load from cache (faster)
- ✅ **Efficiency**: Reduces API calls and saves bandwidth
- ✅ **Automatic Cleanup**: All temporary files deleted on program exit
- ✅ **Organized**: Metadata tracking for each cached file
- ✅ **Memory Efficient**: Data stored on disk, not in memory
- ✅ **Error Handling**: Gracefully handles errors

### How It Works

1. **First Run**: Data is fetched from API and cached
2. **Subsequent Runs**: Data is loaded from cache (much faster)
3. **On Exit**: All cache files are automatically deleted

### Cache Location

Cache files are stored in your system's temporary directory:
- **Windows**: `C:\Users\<username>\AppData\Local\Temp\TradePredictionBot_Cache\<session_id>`
- **Linux/Mac**: `/tmp/TradePredictionBot_Cache/<session_id>`

### Manual Cache Management

```python
from src.cache_manager import get_cache_manager, cleanup_all_cache

# Get cache manager instance
cache = get_cache_manager()

# Print cache status
cache.print_cache_status()

# Manually cleanup (optional - happens automatically on exit)
cleanup_all_cache()
```

## 📊 Output Files

### Cleaned CSV Files

After running the cleaning pipeline, you'll find CSV files in the project root:
- `cleaned_AAPL_data.csv`
- `cleaned_MSFT_data.csv`
- `cleaned_GOOGL_data.csv`
- etc.

These files contain:
- Original OHLCV (Open, High, Low, Close, Volume) data
- Calculated returns
- Technical indicators (RSI, MACD, SMA, Volatility)
- Ready for machine learning models

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError` or `ImportError`

**Solution**:
```bash
# Make sure you're in the project root directory
cd TradePredictionBot

# Install dependencies
pip install -r requirements.txt

# For relative imports, ensure you're running from project root
python demo_cache.py  # ✅ Correct
python src/demo_cache.py  # ❌ Incorrect
```

#### 2. yfinance Connection Errors

**Problem**: `Failed to fetch data` or network timeouts

**Solution**:
- Check your internet connection
- Yahoo Finance may be temporarily unavailable - wait and retry
- Try a different period or interval
- Reduce the number of stocks being fetched at once

#### 3. Cache Not Working

**Problem**: Cache files not being created or loaded

**Solution**:
- Ensure you're using `use_cache=True` in function calls
- Check that you have write permissions in the temp directory
- Verify the cache manager is imported correctly:
  ```python
  from src.cache_manager import get_cache_manager
  ```

#### 4. Missing Data Columns

**Problem**: `KeyError` when accessing columns like 'Close', 'Volume'

**Solution**:
- Ensure data was fetched successfully (check return value is not None)
- Verify the DataFrame is not empty: `if df is not None and not df.empty:`
- Some intervals may not have all columns - use daily (`'1d'`) or weekly (`'1wk'`) intervals

#### 5. Deprecated Pandas Methods

**Problem**: `FutureWarning` about `fillna(method='ffill')`

**Solution**: This is a warning, not an error. The code still works, but for newer pandas versions:
```python
# Old (still works):
df.fillna(method='ffill')

# New (recommended):
df.ffill()
```

### Getting Help

1. **Check the logs**: The program prints detailed messages about what it's doing
2. **Run examples**: Start with `demo_cache.py` to verify your setup
3. **Check dependencies**: Ensure all packages in `requirements.txt` are installed
4. **Verify Python version**: Requires Python 3.7+

## 🎯 Next Steps

After successfully running the data pipeline:

1. **Review the Data**: Check the generated CSV files
2. **Explore Analysis**: Review the analysis output for insights
3. **Build Models**: Use the cleaned data for machine learning models
4. **Advanced Trading**: Explore the `quantitative_trading/` module for ML-based trading strategies

## 📊 How to Evaluate Model Performance

### Understanding Model Evaluation Metrics

After training your models, you need to know if they're performing well. Here's how to interpret the evaluation metrics:

#### For Regression Models (Predicting Returns/Values)

**Key Metrics:**
- **RMSE (Root Mean Squared Error)**: Lower is better. Measures prediction error magnitude.
- **MAE (Mean Absolute Error)**: Lower is better. Average prediction error.
- **R² Score**: 
  - `1.0` = Perfect predictions
  - `> 0.1` = Good (model explains variance)
  - `0` = No better than baseline (predicting the mean)
  - `< 0` = Worse than baseline
- **Correlation**: 
  - `1.0` = Perfect positive correlation
  - `0` = No correlation
  - `> 0.3` = Useful for trading
- **Directional Accuracy**: 
  - `> 55%` = Good (predicts up/down correctly)
  - `50%` = Random guessing
  - `< 50%` = Worse than random

**What Good Performance Looks Like:**
- R² > 0.1 (model explains at least 10% of variance)
- Directional Accuracy > 55% (useful for trading)
- Correlation > 0.3 (predictions align with actual movements)

#### For Classification Models (Predicting Direction Up/Down)

**Key Metrics:**
- **Accuracy**: Percentage of correct predictions
  - `> 60%` = Good for stock prediction
  - `55-60%` = Moderate (better than random)
  - `50-55%` = Slightly better than random
  - `< 50%` = Worse than random
- **Precision**: Of predicted ups, how many were actually up?
- **Recall**: Of actual ups, how many did we predict correctly?
- **F1-Score**: Balance between precision and recall

**What Good Performance Looks Like:**
- Accuracy > 55% (better than random)
- Accuracy > 60% (good for trading)
- Balanced precision and recall

### How to Evaluate Your Models

```python
from quantitative_trading.src.quantitative_models import QuantitativePredictor
from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks

# 1. Prepare your data
tickers = ['AAPL']
stock_data = fetch_multiple_stocks(tickers, period='3y', interval='1d')
cleaned_data = clean_multiple_stocks(stock_data)

# 2. Create and train predictor
predictor = QuantitativePredictor(cleaned_data['AAPL'], 'AAPL', target_type='direction')
predictor.prepare_data()
predictor.train_models()

# 3. Evaluate models (automatically prints detailed report)
results = predictor.evaluate_models()

# 4. Check the best model
print(f"Best model: {results['best_model']}")
print(f"Best score: {results['best_score']}")
```

### Real-World Performance: Backtesting

Model metrics alone aren't enough! Always backtest with a trading strategy:

```python
from quantitative_trading.src.trading_strategy import run_complete_strategy

# Run complete strategy with backtesting
results = run_complete_strategy(
    tickers=['AAPL'],
    target_type='direction',
    signal_type='direction'
)

# Check backtest performance
for ticker, result in results.items():
    backtest = result['backtest_results']
    print(f"\n{ticker} Performance:")
    print(f"  Total Return: {backtest['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {backtest['sharpe_ratio']:.3f}")
    print(f"  Win Rate: {backtest['win_rate_pct']:.1f}%")
    print(f"  Max Drawdown: {backtest['max_drawdown_pct']:.2f}%")
```

**Backtest Metrics to Watch:**
- **Sharpe Ratio**: `> 1.0` = Good, `> 2.0` = Excellent
- **Win Rate**: `> 50%` = Profitable strategy
- **Max Drawdown**: `< 20%` = Acceptable risk
- **Total Return**: Compare to buy-and-hold baseline

### Red Flags (Model Not Working Well)

❌ **Regression Models:**
- R² < 0 (worse than baseline)
- Directional Accuracy < 50% (worse than random)
- Correlation < 0.2 (weak relationship)

❌ **Classification Models:**
- Accuracy < 50% (worse than random)
- Very imbalanced precision/recall
- Confusion matrix shows random guessing pattern

❌ **Backtesting:**
- Negative Sharpe Ratio
- Win Rate < 45%
- Max Drawdown > 30%
- Underperforms buy-and-hold

### Tips for Improving Performance

1. **Try Different Models**: Test Random Forest, XGBoost, Linear Regression
2. **Adjust Features**: Use feature importance to focus on relevant features
3. **Change Target**: Try 'direction', 'returns', or 'volatility'
4. **Tune Parameters**: Adjust confidence thresholds, prediction horizons
5. **More Data**: Use longer time periods (3-5 years minimum)
6. **Risk Management**: Implement stop-loss and position sizing

### Quick Performance Check

Run this to quickly evaluate your model:

```python
# Quick evaluation
predictor.evaluate_models()  # Prints comprehensive report

# The report will show:
# ✅ = Good performance
# ⚠️  = Moderate performance  
# ❌ = Poor performance
```

## 📚 Additional Resources

- **Quantitative Trading Module**: See `quantitative_trading/README.md` for advanced ML trading strategies
- **yfinance Documentation**: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)
- **Pandas Documentation**: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)

## 🔐 Notes

- **Data Source**: This project uses Yahoo Finance (yfinance) for free stock data
- **Rate Limits**: Be mindful of API rate limits when fetching large amounts of data
- **Data Accuracy**: Yahoo Finance data is for educational/research purposes
- **Cache Cleanup**: Cache files are automatically deleted - don't rely on them for permanent storage

---

**Happy Trading! 📈**

For issues or questions, check the troubleshooting section above or review the code comments in each module.
