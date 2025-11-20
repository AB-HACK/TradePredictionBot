"""
Simple script to demonstrate how to evaluate model performance
Run this to see if your models are working properly
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data import fetch_multiple_stocks
from src.data_cleaning import clean_multiple_stocks
from quantitative_trading.src.quantitative_models import QuantitativePredictor

def evaluate_model_performance():
    """Evaluate model performance for a stock"""
    
    print("="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Step 1: Fetch and clean data
    print("\n1. Fetching and cleaning data...")
    ticker = 'AAPL'
    stock_data = fetch_multiple_stocks([ticker], period='3y', interval='1d', use_cache=True)
    cleaned_data = clean_multiple_stocks(stock_data, use_cache=True, save_permanent=False)
    
    if ticker not in cleaned_data:
        print(f"❌ Failed to get data for {ticker}")
        return
    
    # Step 2: Train models for direction prediction
    print(f"\n2. Training models for {ticker} (Direction Prediction)...")
    predictor = QuantitativePredictor(cleaned_data[ticker], ticker, target_type='direction')
    predictor.prepare_data(prediction_horizon=1, use_cache=True)
    predictor.train_models(test_size=0.2, use_cache=True)
    
    # Step 3: Evaluate models
    print(f"\n3. Evaluating models...")
    results = predictor.evaluate_models(verbose=True)
    
    # Step 4: Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.4f}")
    
    # Step 5: Interpretation
    print(f"\n{'='*60}")
    print("HOW TO INTERPRET RESULTS")
    print(f"{'='*60}")
    
    if 'best_model' in results:
        best_model_name = results['best_model']
        if 'Classifier' in best_model_name:
            # Classification model
            if 'accuracy' in results[best_model_name]:
                acc = results[best_model_name]['accuracy']
                if acc > 0.60:
                    print("✅ EXCELLENT: Model accuracy > 60% - Good for trading!")
                elif acc > 0.55:
                    print("✅ GOOD: Model accuracy > 55% - Better than random")
                elif acc > 0.50:
                    print("⚠️  MODERATE: Model accuracy > 50% - Slightly better than random")
                else:
                    print("❌ POOR: Model accuracy < 50% - Worse than random")
        else:
            # Regression model
            if 'r2' in results[best_model_name]:
                r2 = results[best_model_name]['r2']
                dir_acc = results[best_model_name].get('directional_accuracy', 0)
                
                if r2 > 0.1 and dir_acc > 0.55:
                    print("✅ EXCELLENT: Good R² and directional accuracy")
                elif r2 > 0 or dir_acc > 0.50:
                    print("✅ GOOD: Model shows some predictive power")
                else:
                    print("❌ POOR: Model not performing well")
    
    print(f"\n💡 Next Steps:")
    print("  1. If performance is good, try backtesting with trading strategy")
    print("  2. If performance is poor, try:")
    print("     - Different models (Random Forest, XGBoost)")
    print("     - Different target types (returns, volatility)")
    print("     - More data (longer time period)")
    print("     - Feature engineering improvements")
    
    return results

if __name__ == "__main__":
    results = evaluate_model_performance()

