#!/usr/bin/env python3
"""
Quick Start Guide for Quantitative Trading System
================================================

This script demonstrates the basic usage of the quantitative trading framework.
Run this to see a simple example of how to use the system.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def quick_example():
    """Run a quick example of the quantitative trading system"""
    print("🚀 Quick Start: Quantitative Trading System")
    print("=" * 50)
    
    try:
        # Import the quantitative modules
        from src.quantitative_models import QuantitativePredictor, QuantitativeFeatureEngineer
        from src.trading_strategy import TradingSignal, PositionSizer, RiskManager
        
        print("✅ Successfully imported quantitative trading modules")
        
        # Create sample data (simulating AAPL-like data)
        print("\n📊 Creating sample market data...")
        dates = pd.date_range('2022-01-01', periods=252, freq='D')  # 1 year of data
        
        # Generate realistic stock price data
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0008, 0.02, 252)  # ~20% annual volatility
        prices = 150 * np.exp(np.cumsum(returns))  # Starting at $150
        
        sample_data = pd.DataFrame({
            'Close': prices,
            'Open': prices * (1 + np.random.normal(0, 0.005, 252)),
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 252))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 252))),
            'Volume': np.random.randint(10000000, 100000000, 252)
        }, index=dates)
        
        # Ensure High >= Low and proper OHLC relationships
        sample_data['High'] = np.maximum(sample_data['High'], sample_data[['Open', 'Close']].max(axis=1))
        sample_data['Low'] = np.minimum(sample_data['Low'], sample_data[['Open', 'Close']].min(axis=1))
        
        print(f"✅ Created sample data: {len(sample_data)} days of market data")
        
        # Step 1: Feature Engineering
        print("\n🔧 Step 1: Advanced Feature Engineering")
        engineer = QuantitativeFeatureEngineer(sample_data, 'SAMPLE')
        engineer.create_advanced_features()
        
        feature_count = len(engineer.feature_columns)
        print(f"✅ Created {feature_count} advanced features")
        print(f"   Sample features: {engineer.feature_columns[:5]}")
        
        # Step 2: Create Predictor and Prepare Data
        print("\n🤖 Step 2: Machine Learning Setup")
        predictor = QuantitativePredictor(sample_data, 'SAMPLE', target_type='direction')
        predictor.prepare_data()
        
        print(f"✅ Data prepared for ML with {len(predictor.feature_engineer.feature_columns)} features")
        
        # Step 3: Train Models
        print("\n🎯 Step 3: Training Machine Learning Models")
        predictor.train_models(test_size=0.2)
        
        print("✅ Models trained successfully")
        print("   Available models:", list(predictor.models.keys()))
        
        # Step 4: Generate Trading Signals
        print("\n📡 Step 4: Trading Signal Generation")
        signal_generator = TradingSignal(predictor, 'direction')
        signals_df = signal_generator.generate_signals(confidence_threshold=0.6)
        
        if len(signals_df) > 0:
            buy_signals = len(signals_df[signals_df['Signal'] == 1])
            sell_signals = len(signals_df[signals_df['Signal'] == -1])
            print(f"✅ Generated {buy_signals} buy signals and {sell_signals} sell signals")
        else:
            print("⚠️  No signals generated (try lowering confidence threshold)")
        
        # Step 5: Position Sizing
        print("\n💰 Step 5: Position Sizing")
        sizer = PositionSizer(method='kelly', initial_capital=100000)
        
        # Example position sizing
        position_size = sizer.calculate_position_size(
            signal_strength=0.8,
            confidence=0.7,
            price=150.0,
            volatility=0.02
        )
        
        print(f"✅ Position sizing example: {position_size:.3f} ({position_size*100:.1f}% of capital)")
        
        # Step 6: Risk Management
        print("\n🛡️ Step 6: Risk Management")
        risk_manager = RiskManager(stop_loss_pct=0.05, take_profit_pct=0.10)
        
        # Test risk management scenarios
        exit_signal = risk_manager.check_exit_conditions(100, 95, 'long')
        print(f"✅ Risk management test: Entry $100, Current $95, Signal: {exit_signal}")
        
        # Step 7: Model Evaluation
        print("\n📊 Step 7: Model Performance")
        results = predictor.evaluate_models()
        
        if results:
            best_model = max(results.keys(), key=lambda x: results[x].get('accuracy', results[x].get('rmse', 0)))
            print(f"✅ Best performing model: {best_model}")
            
            if 'accuracy' in results[best_model]:
                print(f"   Accuracy: {results[best_model]['accuracy']:.4f}")
            else:
                print(f"   RMSE: {results[best_model]['rmse']:.6f}")
        
        print("\n" + "=" * 50)
        print("🎉 QUICK START COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        print("\n📚 Next Steps:")
        print("1. Run the full demo: python quantitative_demo.py")
        print("2. Read the detailed guide: QUANTITATIVE_GUIDE.md")
        print("3. Try with real data using your existing data pipeline")
        print("4. Experiment with different parameters and strategies")
        
        print("\n💡 Tips:")
        print("- Start with 'direction' target type for easier interpretation")
        print("- Use Random Forest models for good balance of performance/speed")
        print("- Implement proper risk management before live trading")
        print("- Backtest thoroughly with out-of-sample data")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you've installed all requirements: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that all files are in the correct location")
        return False

def show_file_structure():
    """Show the file structure"""
    print("\n📁 File Structure:")
    print("quantitative_trading/")
    print("├── src/")
    print("│   ├── quantitative_models.py      # ML models and feature engineering")
    print("│   ├── trading_strategy.py         # Trading signals and backtesting")
    print("│   └── cache_manager.py           # Caching system")
    print("├── quantitative_demo.py           # Comprehensive demo")
    print("├── QUANTITATIVE_GUIDE.md         # Detailed usage guide")
    print("├── requirements.txt               # Dependencies")
    print("├── setup.py                      # Setup and testing script")
    print("├── quick_start.py                # This file")
    print("└── README.md                     # Overview and documentation")

if __name__ == "__main__":
    show_file_structure()
    success = quick_example()
    
    if success:
        print("\n🚀 Ready to start quantitative trading!")
    else:
        print("\n❌ Please check the setup and try again")
        print("Run: python setup.py")
