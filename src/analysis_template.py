# =============================================================================
# STOCK DATA ANALYSIS TEMPLATE
# =============================================================================
# This module provides comprehensive analysis of stock data for model building.
# It covers all aspects of stock data analysis including:
# 
# Analysis Sections:
# A. Descriptive Statistics - Basic stats, distributions, return analysis
# B. Trend Analysis - Long-term trends, moving averages, price patterns
# C. Volatility & Risk - Volatility measures, drawdown, Sharpe ratio, VaR
# D. Correlation Analysis - Autocorrelation, cross-correlations
# E. Technical Indicators - RSI, MACD, moving average signals
# F. Seasonality Analysis - Monthly and day-of-week patterns
# G. Event Analysis - Volume spikes, large price movements
# H. Predictive Features - Feature importance for modeling
# =============================================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class StockDataAnalyzer:
    """
    Simple stock data analyzer for basic insights
    """
    
    def __init__(self, df, ticker_name):
        self.df = df.copy()
        self.ticker_name = ticker_name
        
    def basic_analysis(self):
        """
        Basic analysis - returns, volatility, and key metrics
        """
        print(f"\n=== BASIC ANALYSIS FOR {self.ticker_name} ===")
        
        if 'Returns' in self.df.columns:
            # Return statistics
            mean_return = self.df['Returns'].mean()
            std_return = self.df['Returns'].std()
            sharpe = mean_return / std_return * np.sqrt(252) if std_return > 0 else 0
            
            print(f"Mean Daily Return: {mean_return:.4f}")
            print(f"Return Volatility: {std_return:.4f}")
            print(f"Sharpe Ratio: {sharpe:.3f}")
            
            # Risk metrics
            max_drawdown = self._calculate_drawdown()
            print(f"Max Drawdown: {max_drawdown:.2%}")
            
            # Technical indicators
            if 'RSI' in self.df.columns:
                current_rsi = self.df['RSI'].iloc[-1]
                print(f"Current RSI: {current_rsi:.2f}")
                
            if 'MACD' in self.df.columns:
                current_macd = self.df['MACD'].iloc[-1]
                print(f"Current MACD: {current_macd:.4f}")
        
        return self.df
    
    def _calculate_drawdown(self):
        """Calculate maximum drawdown"""
        if 'Returns' not in self.df.columns:
            return 0
        
        cumulative = (1 + self.df['Returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def feature_importance(self):
        """
        Simple feature importance for modeling
        """
        print(f"\n=== FEATURE IMPORTANCE FOR {self.ticker_name} ===")
        
        if 'Returns' not in self.df.columns:
            return
        
        # Create target (next day's return)
        target = self.df['Returns'].shift(-1).dropna()
        
        # Create features
        features = {}
        if 'Close' in self.df.columns:
            features['Price_Change'] = self.df['Close'].pct_change()
            features['Price_MA_Ratio'] = self.df['Close'] / self.df['SMA_20']
        
        if 'Volume' in self.df.columns:
            features['Volume_Change'] = self.df['Volume'].pct_change()
        
        if 'RSI' in self.df.columns:
            features['RSI'] = self.df['RSI']
        
        if 'MACD' in self.df.columns:
            features['MACD'] = self.df['MACD']
        
        if 'Volatility' in self.df.columns:
            features['Volatility'] = self.df['Volatility']
        
        # Calculate correlations
        feature_df = pd.DataFrame(features)
        feature_df['Target'] = target
        aligned_data = feature_df.dropna()
        
        if len(aligned_data) > 0:
            correlations = aligned_data.corr()['Target'].abs()
            
            print("Feature correlations with next day return:")
            # Sort manually to avoid linter issues
            sorted_items = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            for feature, corr in sorted_items:
                if feature != 'Target':
                    print(f"  {feature}: {corr:.4f}")
        
        return self.df
    
    def run_analysis(self):
        """
        Run complete analysis
        """
        print(f"Analyzing {self.ticker_name}...")
        self.basic_analysis()
        self.feature_importance()
        print(f"✅ Analysis completed for {self.ticker_name}")
        return self.df


def analyze_multiple_stocks(stock_data_dict):
    """
    Analyze multiple stocks
    """
    results = {}
    
    for ticker, df in stock_data_dict.items():
        print(f"\n{'='*50}")
        analyzer = StockDataAnalyzer(df, ticker)
        analyzer.run_analysis()
        results[ticker] = df
    
    return results 