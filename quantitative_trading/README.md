# 🎯 Quantitative Trading System

A comprehensive machine learning-based quantitative trading framework for stock market prediction and strategy backtesting.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Demo
```bash
python quantitative_demo.py
```

## 📁 File Structure

```
quantitative_trading/
├── src/
│   ├── quantitative_models.py      # ML models and feature engineering
│   ├── trading_strategy.py         # Trading signals and backtesting
│   └── cache_manager.py           # Caching system for performance
├── quantitative_demo.py           # Comprehensive demo script
├── QUANTITATIVE_GUIDE.md         # Detailed usage guide
├── requirements.txt               # Dependencies
└── README.md                     # This file
```

## 🎪 Features

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

## 🎯 Usage Examples

### Complete Strategy Pipeline
```python
from src.trading_strategy import run_complete_strategy

# Run complete quantitative strategy
results = run_complete_strategy(
    tickers=['AAPL', 'MSFT'],
    target_type='direction',
    signal_type='direction'
)
```

### Step-by-Step Approach
```python
from src.quantitative_models import QuantitativePredictor
from src.trading_strategy import TradingSignal, Backtester

# 1. Create predictor and engineer features
predictor = QuantitativePredictor(df, 'AAPL', target_type='direction')
predictor.prepare_data()
predictor.train_models()

# 2. Generate trading signals
signal_generator = TradingSignal(predictor, 'direction')
signals_df = signal_generator.generate_signals()

# 3. Run backtest
backtester = Backtester(initial_capital=100000)
results = backtester.run_backtest(signals_df, df, position_sizer, risk_manager)
```

### Custom Feature Engineering
```python
from src.quantitative_models import QuantitativeFeatureEngineer

# Create custom features
engineer = QuantitativeFeatureEngineer(df, 'AAPL')
engineer.create_advanced_features(lookback_periods=[5, 10, 20, 50])
```

## ⚙️ Configuration Options

### Model Training
```python
predictor.train_models(test_size=0.2)  # 20% for testing
```

### Signal Generation
```python
signals_df = signal_generator.generate_signals(
    model_name='Random_Forest',
    confidence_threshold=0.6
)
```

### Position Sizing
```python
from src.trading_strategy import PositionSizer

sizer = PositionSizer(
    method='kelly',  # 'kelly', 'fixed_fraction', 'risk_parity', 'volatility_target'
    initial_capital=100000
)
```

### Risk Management
```python
from src.trading_strategy import RiskManager

risk_manager = RiskManager(
    stop_loss_pct=0.05,    # 5% stop loss
    take_profit_pct=0.10,  # 10% take profit
    max_position_pct=0.20  # Maximum 20% position
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

## 🎪 Demo Scripts

### Complete Demo
```bash
python quantitative_demo.py
```

### Individual Components
```python
# Test feature engineering only
python -c "from quantitative_demo import demo_feature_engineering; demo_feature_engineering()"

# Test signal generation only
python -c "from quantitative_demo import demo_signal_generation; demo_signal_generation()"
```

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

## 🛠️ Troubleshooting

### Common Issues
1. **Memory Issues**: Reduce feature count or use sampling
2. **Slow Training**: Use fewer models or smaller datasets
3. **Poor Performance**: Try different target types or signal strategies
4. **Import Errors**: Install missing dependencies (`pip install xgboost`)

### Performance Tips
1. **Use Caching**: The framework automatically caches processed data
2. **Parallel Processing**: Models train in parallel when possible
3. **Feature Selection**: Remove low-importance features
4. **Data Quality**: Ensure clean, consistent data

## 📚 Further Reading

- [QUANTITATIVE_GUIDE.md](QUANTITATIVE_GUIDE.md) - Detailed usage guide
- "Quantitative Trading" by Ernest Chan
- "Advances in Financial Machine Learning" by Marcos López de Prado
- "Machine Learning for Trading" by Stefan Jansen

---

## 🎉 Ready to Start!

This quantitative trading framework provides everything you need to:
- ✅ Generate sophisticated features from raw market data
- ✅ Train multiple ML models automatically
- ✅ Create intelligent trading signals
- ✅ Implement proper risk management
- ✅ Backtest strategies comprehensively
- ✅ Evaluate performance professionally

**Start with the demo and build your trading system!** 🚀
