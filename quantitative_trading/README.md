# 🎯 Quantitative Trading System

A comprehensive machine learning-based quantitative trading framework for stock market prediction and strategy backtesting. This system provides advanced feature engineering, multiple ML models, trading signal generation, and comprehensive backtesting capabilities.

## 📋 Table of Contents

- [Overview](#-overview)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Usage Guide](#-usage-guide)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)

## 🎯 Overview

This quantitative trading system enables you to:

- **Engineer 100+ features** from raw market data
- **Train multiple ML models** (Random Forest, XGBoost, Linear Regression)
- **Generate trading signals** (direction, momentum, mean reversion)
- **Implement risk management** (stop loss, take profit, position limits)
- **Backtest strategies** with comprehensive performance metrics
- **Evaluate performance** with Sharpe ratio, drawdown, win rate, and more

## 🔧 Prerequisites

Before you begin, ensure you have:

- **Python 3.7 or higher** installed
- **pip** package manager
- **Successfully installed the main project** (see main `README.md`)
- **Internet connection** for data fetching

**Check Python version:**
```bash
python --version
# or
python3 --version
```

## 📦 Installation

### Step 1: Navigate to the Quantitative Trading Directory

```bash
cd quantitative_trading
```

### Step 2: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This installs:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Stock data fetching
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting
- `matplotlib` - Visualization
- `seaborn` - Statistical plots
- `statsmodels` - Statistical analysis
- `streamlit` - Web interface (optional)

### Step 3: Verify Installation

Run the setup script to test everything:

```bash
python setup.py
```

This will:
- Install dependencies
- Test all imports
- Run system tests
- Verify everything works

**Or manually verify:**
```bash
python -c "from src.quantitative_models import QuantitativePredictor; print('✅ Installation successful!')"
```

## 🚀 Quick Start

### Option 1: Quick Start (Recommended for First Time)

Run a simple example with sample data:

```bash
python quick_start.py
```

**What it does:**
- Creates sample stock data
- Engineers features
- Trains a simple model
- Generates trading signals
- Shows basic backtesting results

### Option 2: Full Demo

Run the comprehensive demo with real stock data:

```bash
python quantitative_demo.py
```

**What it does:**
- Fetches real stock data (AAPL, MSFT, etc.)
- Engineers 100+ features
- Trains multiple ML models
- Generates trading signals
- Runs complete backtesting
- Displays performance metrics

### Option 3: Setup and Test

Run the setup script for automated installation and testing:

```bash
python setup.py
```

## 📁 Project Structure

```
quantitative_trading/
│
├── src/
│   ├── quantitative_models.py      # ML models and feature engineering
│   ├── trading_strategy.py         # Trading signals and backtesting
│   └── cache_manager.py            # Caching system for performance
│
├── quantitative_demo.py            # Comprehensive demo script
├── quick_start.py                  # Simple example for beginners
├── setup.py                        # Installation and testing script
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── QUANTITATIVE_GUIDE.md          # Detailed usage guide
└── START_HERE.md                  # Getting started guide
```

## ✨ Features

### Advanced Feature Engineering

- **100+ engineered features** from raw market data
- Price ratios, momentum, gaps, volume analysis
- Technical indicators (Bollinger Bands, RSI, MACD, Stochastic, etc.)
- Lag features, rolling statistics, market regime detection
- Volatility clustering and GARCH-like features

### Machine Learning Models

- **Regression Models**: Linear Regression, Random Forest, XGBoost
- **Classification Models**: Logistic Regression, Random Forest, XGBoost
- Automatic model evaluation and feature importance analysis
- Time series cross-validation

### Trading Strategy Framework

- **Signal Generation**: Direction, momentum, mean reversion strategies
- **Position Sizing**: Kelly Criterion, fixed fraction, risk parity, volatility targeting
- **Risk Management**: Stop loss, take profit, position limits
- **Backtesting**: Comprehensive performance evaluation

### Performance Metrics

- Risk-adjusted returns (Sharpe, Sortino, Calmar ratios)
- Drawdown analysis and trade statistics
- Win rate, profit factor, average win/loss
- Portfolio value tracking over time

## 📖 Usage Guide

### Complete Strategy Pipeline (Easiest Way)

```python
from src.trading_strategy import run_complete_strategy

# Run complete quantitative strategy
results = run_complete_strategy(
    tickers=['AAPL', 'MSFT'],
    target_type='direction',  # 'direction', 'momentum', 'mean_reversion'
    signal_type='direction'   # 'direction', 'momentum', 'mean_reversion'
)

# Access results
print(f"Sharpe Ratio: {results['sharpe_ratio']}")
print(f"Win Rate: {results['win_rate']}")
print(f"Total Return: {results['total_return']}")
```

### Step-by-Step Approach

```python
import pandas as pd
from src.quantitative_models import QuantitativePredictor
from src.trading_strategy import TradingSignal, Backtester, PositionSizer, RiskManager

# Step 1: Prepare your data (or fetch it)
# df should have columns: Open, High, Low, Close, Volume
# df = your_stock_dataframe

# Step 2: Create predictor and engineer features
predictor = QuantitativePredictor(df, 'AAPL', target_type='direction')
predictor.prepare_data()
predictor.train_models()

# Step 3: Generate trading signals
signal_generator = TradingSignal(predictor, 'direction')
signals_df = signal_generator.generate_signals(
    model_name='Random_Forest',
    confidence_threshold=0.6
)

# Step 4: Set up position sizing and risk management
position_sizer = PositionSizer(
    method='kelly',  # or 'fixed_fraction', 'risk_parity', 'volatility_target'
    initial_capital=100000
)

risk_manager = RiskManager(
    stop_loss_pct=0.05,    # 5% stop loss
    take_profit_pct=0.10,  # 10% take profit
    max_position_pct=0.20  # Maximum 20% position
)

# Step 5: Run backtest
backtester = Backtester(initial_capital=100000)
results = backtester.run_backtest(
    signals_df, 
    df, 
    position_sizer, 
    risk_manager
)

# Step 6: Analyze results
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
print(f"Win Rate: {results['win_rate']:.2%}")
```

### Custom Feature Engineering

```python
from src.quantitative_models import QuantitativeFeatureEngineer

# Create custom features
engineer = QuantitativeFeatureEngineer(df, 'AAPL')
engineer.create_advanced_features(
    lookback_periods=[5, 10, 20, 50, 100, 200]
)

# Access engineered features
feature_df = engineer.df
feature_columns = engineer.feature_columns
```

## ⚙️ Configuration

### Target Types

- **`'direction'`**: Predict if price will go up or down (classification)
- **`'momentum'`**: Predict price momentum/trend (regression)
- **`'mean_reversion'`**: Predict mean reversion opportunities (regression)

### Signal Types

- **`'direction'`**: Buy/sell signals based on predicted direction
- **`'momentum'`**: Signals based on momentum strength
- **`'mean_reversion'`**: Signals for oversold/overbought conditions

### Position Sizing Methods

- **`'kelly'`**: Kelly Criterion for optimal position sizing
- **`'fixed_fraction'`**: Fixed percentage of capital per trade
- **`'risk_parity'`**: Equal risk allocation
- **`'volatility_target'`**: Size based on volatility

### Model Options

```python
# Train with custom test size
predictor.train_models(test_size=0.2)  # 20% for testing

# Generate signals with confidence threshold
signals_df = signal_generator.generate_signals(
    model_name='XGBoost',           # or 'Random_Forest', 'Linear_Regression'
    confidence_threshold=0.7        # Only trade if confidence > 70%
)
```

## 📊 Expected Performance

### Typical Results

- **Feature Count**: 100+ engineered features per stock
- **Model Accuracy**: 55-65% for direction prediction
- **Sharpe Ratio**: 0.5-2.0 depending on strategy
- **Win Rate**: 45-60% depending on market conditions

### Optimization Tips

1. **Start Simple**: Use direction prediction with basic features
2. **Iterate Quickly**: Test different parameters and models
3. **Focus on Risk**: Implement proper risk management
4. **Backtest Thoroughly**: Use out-of-sample testing
5. **Monitor Performance**: Track live performance vs backtests

## 🔍 Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Make sure you're in the quantitative_trading directory
cd quantitative_trading

# Install requirements
pip install -r requirements.txt

# Run from the quantitative_trading directory
python quantitative_demo.py  # ✅ Correct
python ../quantitative_demo.py  # ❌ Incorrect
```

#### 2. XGBoost Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'xgboost'`

**Solution**:
```bash
# Install XGBoost separately
pip install xgboost

# On some systems, you may need:
pip install --upgrade pip
pip install xgboost
```

#### 3. Memory Issues

**Problem**: Out of memory errors when processing many features

**Solution**:
- Reduce feature count by using fewer lookback periods
- Process fewer stocks at once
- Use smaller datasets for testing
- Increase system RAM or use cloud computing

#### 4. Slow Performance

**Problem**: Training takes too long

**Solution**:
- Use caching (automatic in the framework)
- Reduce number of models trained
- Use smaller datasets for testing
- Reduce feature count
- Use fewer stocks

#### 5. Poor Model Performance

**Problem**: Low accuracy or negative Sharpe ratio

**Solution**:
- Try different target types (direction, momentum, mean_reversion)
- Adjust confidence thresholds
- Experiment with different models
- Check data quality (ensure clean data)
- Try different time periods
- Implement better risk management

#### 6. Data Format Issues

**Problem**: Errors about missing columns or incorrect data types

**Solution**:
- Ensure DataFrame has columns: `Open`, `High`, `Low`, `Close`, `Volume`
- Verify data is numeric (no strings or NaN values in price columns)
- Check that index is datetime (for time series)
- Use the main project's data cleaning pipeline first

## 🚀 Next Steps

### 1. Real-Time Trading
- Integrate with broker APIs (Interactive Brokers, Alpaca)
- Implement live signal generation
- Add order management system

### 2. Portfolio Optimization
- Multi-asset portfolio construction
- Risk parity allocation
- Correlation-based diversification

### 3. Advanced Models
- LSTM neural networks
- Ensemble methods
- Deep learning models

### 4. Web Interface
- Streamlit dashboard
- Real-time monitoring
- Interactive backtesting

## 📚 Additional Documentation

- **START_HERE.md**: Getting started guide with recommended learning path
- **QUANTITATIVE_GUIDE.md**: Detailed usage guide with advanced examples
- **Main README.md**: Data pipeline documentation (parent directory)

## 📖 Recommended Reading

- "Quantitative Trading" by Ernest Chan
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Machine Learning for Trading" by Stefan Jansen

## ⚠️ Important Notes

- **Backtesting Limitations**: Past performance doesn't guarantee future results
- **Paper Trading First**: Always test strategies with paper trading before live trading
- **Risk Management**: Never risk more than you can afford to lose
- **Data Quality**: Ensure clean, accurate data for best results
- **Market Conditions**: Strategies may perform differently in different market conditions

---

## 🎉 Ready to Start!

This quantitative trading framework provides everything you need to:
- ✅ Generate sophisticated features from raw market data
- ✅ Train multiple ML models automatically
- ✅ Create intelligent trading signals
- ✅ Implement proper risk management
- ✅ Backtest strategies comprehensively
- ✅ Evaluate performance professionally

**Choose your path:**
- 🚀 **Quick Test**: `python quick_start.py`
- 🎪 **Full Demo**: `python quantitative_demo.py`
- 🔧 **Setup First**: `python setup.py`

**Happy Trading! 📈**

For detailed examples and advanced usage, see `QUANTITATIVE_GUIDE.md`.
