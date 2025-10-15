#!/usr/bin/env python3
"""
Setup script for the Quantitative Trading System
This script helps install dependencies and test the system
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("🔧 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing imports...")
    
    required_modules = [
        'pandas',
        'numpy',
        'yfinance',
        'sklearn',
        'matplotlib',
        'seaborn',
        'xgboost'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Try installing missing packages manually:")
        for module in failed_imports:
            print(f"  pip install {module}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_system():
    """Test the quantitative trading system"""
    print("\n🎯 Testing quantitative trading system...")
    
    try:
        # Test basic functionality
        from src.quantitative_models import QuantitativeFeatureEngineer
        from src.trading_strategy import TradingSignal, PositionSizer, RiskManager, Backtester
        
        print("  ✅ Core modules imported successfully")
        
        # Test feature engineer
        import pandas as pd
        import numpy as np
        
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = pd.DataFrame({
            'Close': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'Open': 100 + np.cumsum(np.random.randn(100) * 0.02),
            'High': 100 + np.cumsum(np.random.randn(100) * 0.02) + 1,
            'Low': 100 + np.cumsum(np.random.randn(100) * 0.02) - 1,
            'Volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        engineer = QuantitativeFeatureEngineer(sample_data, 'TEST')
        engineer.create_advanced_features()
        
        print(f"  ✅ Feature engineering: {len(engineer.feature_columns)} features created")
        
        # Test position sizer
        sizer = PositionSizer()
        position_size = sizer.calculate_position_size(0.8, 0.7, 100.0, 0.02)
        print(f"  ✅ Position sizing: {position_size:.3f}")
        
        # Test risk manager
        risk_manager = RiskManager()
        exit_signal = risk_manager.check_exit_conditions(100, 95, 'long')
        print(f"  ✅ Risk management: {exit_signal}")
        
        print("\n🎉 System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Quantitative Trading System Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('requirements.txt'):
        print("❌ Please run this script from the quantitative_trading directory")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at requirements installation")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed at import testing")
        return
    
    # Test system
    if not test_system():
        print("❌ Setup failed at system testing")
        return
    
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run the demo: python quantitative_demo.py")
    print("2. Read the guide: QUANTITATIVE_GUIDE.md")
    print("3. Start building your trading strategies!")
    
    print("\n📚 Available files:")
    print("  - quantitative_demo.py     : Comprehensive demo")
    print("  - QUANTITATIVE_GUIDE.md    : Detailed usage guide")
    print("  - src/quantitative_models.py : ML models and features")
    print("  - src/trading_strategy.py   : Trading and backtesting")

if __name__ == "__main__":
    main()
