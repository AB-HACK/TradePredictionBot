#!/usr/bin/env python3
"""
QUANTITATIVE TRADING DEMO
========================

This demo showcases the complete quantitative trading framework:
1. Advanced feature engineering
2. Multiple ML models (Random Forest, XGBoost, Linear Regression)
3. Trading signal generation
4. Position sizing and risk management
5. Comprehensive backtesting
6. Performance evaluation

Run this script to see the quantitative framework in action!
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.quantitative_models import QuantitativePredictor, create_quantitative_pipeline
from src.trading_strategy import TradingSignal, PositionSizer, RiskManager, Backtester, run_complete_strategy

def demo_feature_engineering():
    """Demonstrate advanced feature engineering"""
    print("🔧 DEMO: Advanced Feature Engineering")
    print("=" * 50)
    
    # Import existing modules
    from src.data import fetch_live_data
    from src.data_cleaning import StockDataCleaner
    
    # Fetch sample data
    print("Fetching AAPL data...")
    df = fetch_live_data('AAPL', period='2y', interval='1d')
    
    if df is None:
        print("[ERROR] Failed to fetch data")
        return None
    
    # Clean data
    cleaner = StockDataCleaner(df, 'AAPL')
    cleaned_df = cleaner.clean_data()
    
    # Create predictor and engineer features
    predictor = QuantitativePredictor(cleaned_df, 'AAPL', target_type='returns')
    predictor.prepare_data()
    
    # Show feature count
    feature_count = len(predictor.feature_engineer.feature_columns)
    print(f"✅ Created {feature_count} advanced features")
    
    # Show sample features
    print("\n📊 Sample Features Created:")
    sample_features = predictor.feature_engineer.feature_columns[:10]
    for feature in sample_features:
        print(f"  - {feature}")
    
    return predictor

def demo_model_training():
    """Demonstrate model training and evaluation"""
    print("\n🤖 DEMO: Model Training & Evaluation")
    print("=" * 50)
    
    # Create predictors for multiple stocks
    tickers = ['AAPL', 'MSFT']
    predictors = create_quantitative_pipeline(tickers, target_type='direction')
    
    # Show model performance for each stock
    for ticker, predictor in predictors.items():
        print(f"\n📈 {ticker} Model Performance:")
        
        # Evaluate models
        results = predictor.evaluate_models()
        
        # Show best model
        if results:
            best_model = max(results.keys(), key=lambda x: results[x].get('accuracy', results[x].get('rmse', 0)))
            print(f"  Best Model: {best_model}")
            
            if 'accuracy' in results[best_model]:
                print(f"  Accuracy: {results[best_model]['accuracy']:.4f}")
            else:
                print(f"  RMSE: {results[best_model]['rmse']:.6f}")
    
    return predictors

def demo_signal_generation():
    """Demonstrate trading signal generation"""
    print("\n📡 DEMO: Trading Signal Generation")
    print("=" * 50)
    
    # Create predictor
    from src.data import fetch_live_data
    from src.data_cleaning import StockDataCleaner
    
    df = fetch_live_data('AAPL', period='1y', interval='1d')
    cleaner = StockDataCleaner(df, 'AAPL')
    cleaned_df = cleaner.clean_data()
    
    predictor = QuantitativePredictor(cleaned_df, 'AAPL', target_type='direction')
    predictor.prepare_data()
    predictor.train_models()
    
    # Generate different types of signals
    signal_types = ['direction', 'momentum', 'mean_reversion']
    
    for signal_type in signal_types:
        print(f"\n🎯 {signal_type.title()} Signals:")
        
        signal_generator = TradingSignal(predictor, signal_type)
        signals_df = signal_generator.generate_signals(confidence_threshold=0.6)
        
        if len(signals_df) > 0:
            buy_signals = len(signals_df[signals_df['Signal'] == 1])
            sell_signals = len(signals_df[signals_df['Signal'] == -1])
            print(f"  Buy Signals: {buy_signals}")
            print(f"  Sell Signals: {sell_signals}")
            print(f"  Signal Success Rate: {(buy_signals + sell_signals) / len(signals_df) * 100:.1f}%")

def demo_position_sizing():
    """Demonstrate position sizing methods"""
    print("\n💰 DEMO: Position Sizing Methods")
    print("=" * 50)
    
    # Test different position sizing methods
    methods = ['kelly', 'fixed_fraction', 'risk_parity', 'volatility_target']
    signal_strength = 0.8
    confidence = 0.7
    price = 150.0
    volatility = 0.02
    
    for method in methods:
        sizer = PositionSizer(method=method)
        position_size = sizer.calculate_position_size(signal_strength, confidence, price, volatility)
        
        print(f"  {method.replace('_', ' ').title()}: {position_size:.3f} ({position_size*100:.1f}% of capital)")

def demo_backtesting():
    """Demonstrate comprehensive backtesting"""
    print("\n📊 DEMO: Backtesting Framework")
    print("=" * 50)
    
    # Run complete strategy for a single stock
    tickers = ['AAPL']
    results = run_complete_strategy(tickers, target_type='direction', signal_type='direction')
    
    if tickers[0] in results:
        print(f"\n✅ Backtest completed for {tickers[0]}")
        
        # Show key metrics
        backtest_results = results[tickers[0]]['backtest_results']
        print(f"  Total Return: {backtest_results['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {backtest_results['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate: {backtest_results['win_rate_pct']:.1f}%")
        print(f"  Total Trades: {backtest_results['total_trades']}")

def demo_risk_management():
    """Demonstrate risk management features"""
    print("\n🛡️ DEMO: Risk Management")
    print("=" * 50)
    
    # Test risk management rules
    risk_manager = RiskManager(stop_loss_pct=0.05, take_profit_pct=0.10, max_position_pct=0.20)
    
    # Test scenarios
    scenarios = [
        {'entry_price': 100, 'current_price': 95, 'position_type': 'long', 'expected': 'stop_loss'},
        {'entry_price': 100, 'current_price': 110, 'position_type': 'long', 'expected': 'take_profit'},
        {'entry_price': 100, 'current_price': 105, 'position_type': 'short', 'expected': 'stop_loss'},
        {'entry_price': 100, 'current_price': 90, 'position_type': 'short', 'expected': 'take_profit'},
        {'entry_price': 100, 'current_price': 102, 'position_type': 'long', 'expected': None},
    ]
    
    print("Risk Management Test Scenarios:")
    for i, scenario in enumerate(scenarios, 1):
        result = risk_manager.check_exit_conditions(
            scenario['entry_price'], 
            scenario['current_price'], 
            scenario['position_type']
        )
        status = "✅" if result == scenario['expected'] else "❌"
        print(f"  Scenario {i}: {status} Expected: {scenario['expected']}, Got: {result}")

def create_performance_visualization(results):
    """Create performance visualization"""
    print("\n📈 DEMO: Performance Visualization")
    print("=" * 50)
    
    if not results:
        print("No results to visualize")
        return
    
    # Get first ticker's results
    ticker = list(results.keys())[0]
    portfolio_history = results[ticker]['backtest_results']['portfolio_history']
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Quantitative Trading Performance - {ticker}', fontsize=16)
    
    # Portfolio value over time
    axes[0, 0].plot(portfolio_history.index, portfolio_history['portfolio_value'])
    axes[0, 0].set_title('Portfolio Value Over Time')
    axes[0, 0].set_ylabel('Portfolio Value ($)')
    axes[0, 0].grid(True)
    
    # Returns distribution
    returns = portfolio_history['returns'].dropna()
    axes[0, 1].hist(returns, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Returns Distribution')
    axes[0, 1].set_xlabel('Daily Returns')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True)
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    axes[1, 0].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
    axes[1, 0].set_title('Drawdown Over Time')
    axes[1, 0].set_ylabel('Drawdown')
    axes[1, 0].grid(True)
    
    # Rolling Sharpe Ratio
    rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(252)
    axes[1, 1].plot(rolling_sharpe.index, rolling_sharpe)
    axes[1, 1].set_title('30-Day Rolling Sharpe Ratio')
    axes[1, 1].set_ylabel('Sharpe Ratio')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("✅ Performance visualization created")

def main():
    """Run complete quantitative trading demo"""
    print("🚀 QUANTITATIVE TRADING FRAMEWORK DEMO")
    print("=" * 60)
    print("This demo showcases the complete quantitative trading system:")
    print("• Advanced feature engineering")
    print("• Machine learning models")
    print("• Trading signal generation")
    print("• Position sizing & risk management")
    print("• Backtesting & performance evaluation")
    print()
    
    try:
        # Run all demos
        predictor = demo_feature_engineering()
        if predictor is None:
            return
        
        predictors = demo_model_training()
        demo_signal_generation()
        demo_position_sizing()
        demo_risk_management()
        
        # Run backtesting
        print("\n" + "="*60)
        print("RUNNING COMPREHENSIVE BACKTEST")
        print("="*60)
        results = demo_backtesting()
        
        # Create visualizations if results available
        if results:
            create_performance_visualization(results)
        
        print(f"\n{'='*60}")
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Next steps:")
        print("1. Experiment with different target types (returns, direction, volatility)")
        print("2. Try different signal types (momentum, mean_reversion)")
        print("3. Adjust position sizing methods and risk parameters")
        print("4. Add more stocks to your portfolio")
        print("5. Implement real-time trading with live data feeds")
        
    except KeyboardInterrupt:
        print("\n\n[INFO] Demo interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n[INFO] Demo completed. Check the results above!")

if __name__ == "__main__":
    main()
