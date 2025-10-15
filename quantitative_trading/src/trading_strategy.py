# =============================================================================
# QUANTITATIVE TRADING STRATEGY & BACKTESTING FRAMEWORK
# =============================================================================
# This module provides comprehensive trading strategy implementation and backtesting.
# It includes signal generation, position sizing, risk management, and performance evaluation.
# 
# Features:
# 1. Signal Generation - Based on ML model predictions
# 2. Position Sizing - Kelly Criterion, Fixed Fraction, Risk Parity
# 3. Risk Management - Stop Loss, Take Profit, Portfolio Limits
# 4. Backtesting - Historical strategy performance evaluation
# 5. Performance Metrics - Sharpe, Sortino, Calmar, Max Drawdown
# =============================================================================

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from .cache_manager import get_cache_manager
warnings.filterwarnings('ignore')

class TradingSignal:
    """
    Generate trading signals based on model predictions
    """
    
    def __init__(self, predictor, signal_type='direction'):
        """
        Initialize signal generator
        
        Args:
            predictor: Trained QuantitativePredictor instance
            signal_type: Type of signal ('direction', 'momentum', 'mean_reversion')
        """
        self.predictor = predictor
        self.signal_type = signal_type
        self.signals = pd.DataFrame()
        
    def generate_signals(self, model_name='Random_Forest', confidence_threshold=0.6):
        """
        Generate trading signals based on model predictions
        
        Args:
            model_name: Name of model to use for predictions
            confidence_threshold: Minimum confidence for signal generation
        
        Returns:
            DataFrame with signals
        """
        print(f"Generating {self.signal_type} signals for {self.predictor.ticker_name}...")
        
        df = self.predictor.df.copy()
        feature_cols = self.predictor.feature_engineer.feature_columns
        
        # Prepare data for prediction
        X = df[feature_cols].dropna()
        
        if len(X) == 0:
            print(f"[ERROR] No valid data for signal generation")
            return pd.DataFrame()
        
        # Get model and scaler
        if model_name not in self.predictor.models:
            print(f"[ERROR] Model {model_name} not found")
            return pd.DataFrame()
        
        model = self.predictor.models[model_name]['model']
        scaler = self.predictor.scalers.get('standard')
        
        # Scale features
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Generate predictions
        if hasattr(model, 'predict_proba'):
            # Classification model - get probabilities
            probabilities = model.predict_proba(X_scaled)
            predictions = model.predict(X_scaled)
            
            # Use maximum probability as confidence
            confidence = np.max(probabilities, axis=1)
            signal_strength = probabilities[:, 1] - probabilities[:, 0]  # Bullish - Bearish
        else:
            # Regression model - predict returns
            predictions = model.predict(X_scaled)
            confidence = np.abs(predictions) / np.std(predictions)  # Normalized confidence
            signal_strength = predictions
        
        # Create signals DataFrame
        signals_df = pd.DataFrame(index=X.index)
        signals_df['Prediction'] = predictions
        signals_df['Confidence'] = confidence
        signals_df['Signal_Strength'] = signal_strength
        
        # Generate trading signals based on type
        if self.signal_type == 'direction':
            signals_df = self._generate_direction_signals(signals_df, confidence_threshold)
        elif self.signal_type == 'momentum':
            signals_df = self._generate_momentum_signals(signals_df, confidence_threshold)
        elif self.signal_type == 'mean_reversion':
            signals_df = self._generate_mean_reversion_signals(signals_df, confidence_threshold)
        
        # Add price data for context
        signals_df['Close'] = df['Close'].loc[signals_df.index]
        signals_df['Returns'] = df['Returns'].loc[signals_df.index]
        
        self.signals = signals_df
        print(f"[SUCCESS] Generated {len(signals_df[signals_df['Signal'] != 0])} signals")
        
        return signals_df
    
    def _generate_direction_signals(self, signals_df, threshold):
        """Generate directional signals (buy/sell)"""
        # High confidence predictions
        high_conf_mask = signals_df['Confidence'] >= threshold
        
        # Buy signals: positive prediction with high confidence
        buy_signals = (signals_df['Signal_Strength'] > 0) & high_conf_mask
        signals_df.loc[buy_signals, 'Signal'] = 1
        
        # Sell signals: negative prediction with high confidence
        sell_signals = (signals_df['Signal_Strength'] < 0) & high_conf_mask
        signals_df.loc[sell_signals, 'Signal'] = -1
        
        # No signal for low confidence
        signals_df.loc[~high_conf_mask, 'Signal'] = 0
        
        return signals_df
    
    def _generate_momentum_signals(self, signals_df, threshold):
        """Generate momentum signals"""
        # Rolling momentum
        momentum = signals_df['Signal_Strength'].rolling(5).mean()
        
        # Buy: strong positive momentum
        buy_signals = (momentum > threshold) & (signals_df['Signal_Strength'] > momentum)
        signals_df.loc[buy_signals, 'Signal'] = 1
        
        # Sell: strong negative momentum
        sell_signals = (momentum < -threshold) & (signals_df['Signal_Strength'] < momentum)
        signals_df.loc[sell_signals, 'Signal'] = -1
        
        signals_df.loc[~(buy_signals | sell_signals), 'Signal'] = 0
        
        return signals_df
    
    def _generate_mean_reversion_signals(self, signals_df, threshold):
        """Generate mean reversion signals"""
        # Z-score of signal strength
        z_score = (signals_df['Signal_Strength'] - signals_df['Signal_Strength'].rolling(20).mean()) / signals_df['Signal_Strength'].rolling(20).std()
        
        # Buy: oversold (negative z-score)
        buy_signals = (z_score < -threshold) & (signals_df['Signal_Strength'] < 0)
        signals_df.loc[buy_signals, 'Signal'] = 1
        
        # Sell: overbought (positive z-score)
        sell_signals = (z_score > threshold) & (signals_df['Signal_Strength'] > 0)
        signals_df.loc[sell_signals, 'Signal'] = -1
        
        signals_df.loc[~(buy_signals | sell_signals), 'Signal'] = 0
        
        return signals_df


class PositionSizer:
    """
    Calculate position sizes based on various methods
    """
    
    def __init__(self, method='kelly', initial_capital=100000):
        """
        Initialize position sizer
        
        Args:
            method: Sizing method ('kelly', 'fixed_fraction', 'risk_parity', 'volatility_target')
            initial_capital: Starting capital
        """
        self.method = method
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
    def calculate_position_size(self, signal_strength, confidence, price, volatility=None):
        """
        Calculate position size based on signal and risk parameters
        
        Args:
            signal_strength: Strength of trading signal (-1 to 1)
            confidence: Confidence in signal (0 to 1)
            price: Current price
            volatility: Volatility estimate
        
        Returns:
            Position size as fraction of capital
        """
        if self.method == 'kelly':
            return self._kelly_sizing(signal_strength, confidence, volatility)
        elif self.method == 'fixed_fraction':
            return self._fixed_fraction_sizing(confidence)
        elif self.method == 'risk_parity':
            return self._risk_parity_sizing(signal_strength, volatility)
        elif self.method == 'volatility_target':
            return self._volatility_target_sizing(signal_strength, volatility)
        else:
            return 0.01  # Default 1% position
    
    def _kelly_sizing(self, signal_strength, confidence, volatility):
        """Kelly Criterion position sizing"""
        if volatility is None or volatility == 0:
            return 0.01
        
        # Simplified Kelly: (win_prob * avg_win - loss_prob * avg_loss) / avg_loss
        win_prob = confidence
        avg_win = abs(signal_strength) * volatility * 2  # Estimated win size
        avg_loss = volatility * 1.5  # Estimated loss size
        
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_loss
        
        # Cap at 25% and ensure positive
        return max(0, min(0.25, kelly_fraction))
    
    def _fixed_fraction_sizing(self, confidence):
        """Fixed fraction based on confidence"""
        base_fraction = 0.05  # 5% base position
        return base_fraction * confidence
    
    def _risk_parity_sizing(self, signal_strength, volatility):
        """Risk parity position sizing"""
        if volatility is None or volatility == 0:
            return 0.01
        
        target_risk = 0.02  # 2% target risk
        position_size = target_risk / (volatility * abs(signal_strength))
        
        return max(0, min(0.20, position_size))
    
    def _volatility_target_sizing(self, signal_strength, volatility):
        """Volatility targeting position sizing"""
        if volatility is None or volatility == 0:
            return 0.01
        
        target_vol = 0.15  # 15% target volatility
        position_size = (target_vol / volatility) * abs(signal_strength)
        
        return max(0, min(0.30, position_size))


class RiskManager:
    """
    Implement risk management rules
    """
    
    def __init__(self, stop_loss_pct=0.05, take_profit_pct=0.10, max_position_pct=0.20):
        """
        Initialize risk manager
        
        Args:
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_position_pct: Maximum position size as percentage of capital
        """
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_pct = max_position_pct
        
    def check_exit_conditions(self, entry_price, current_price, position_type):
        """
        Check if exit conditions are met
        
        Args:
            entry_price: Price when position was entered
            current_price: Current price
            position_type: 'long' or 'short'
        
        Returns:
            Exit signal: 'stop_loss', 'take_profit', or None
        """
        if position_type == 'long':
            # Long position
            if current_price <= entry_price * (1 - self.stop_loss_pct):
                return 'stop_loss'
            elif current_price >= entry_price * (1 + self.take_profit_pct):
                return 'take_profit'
        else:
            # Short position
            if current_price >= entry_price * (1 + self.stop_loss_pct):
                return 'stop_loss'
            elif current_price <= entry_price * (1 - self.take_profit_pct):
                return 'take_profit'
        
        return None


class Backtester:
    """
    Comprehensive backtesting framework
    """
    
    def __init__(self, initial_capital=100000, commission=0.001):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.trades = []
        self.portfolio_history = []
        
    def run_backtest(self, signals_df, price_data, position_sizer, risk_manager):
        """
        Run complete backtesting simulation
        
        Args:
            signals_df: DataFrame with trading signals
            price_data: Price data DataFrame
            position_sizer: PositionSizer instance
            risk_manager: RiskManager instance
        
        Returns:
            Dictionary with backtest results
        """
        print("Running backtest simulation...")
        
        capital = self.initial_capital
        position = 0  # Current position size
        entry_price = 0
        position_type = None
        
        portfolio_values = []
        trade_log = []
        
        for date, row in signals_df.iterrows():
            current_price = row['Close']
            signal = row.get('Signal', 0)
            confidence = row.get('Confidence', 0)
            signal_strength = row.get('Signal_Strength', 0)
            
            # Calculate volatility for position sizing
            volatility = price_data.loc[date, 'Volatility'] if 'Volatility' in price_data.columns else 0.02
            
            # Check exit conditions for existing position
            if position != 0:
                exit_reason = risk_manager.check_exit_conditions(entry_price, current_price, position_type)
                
                if exit_reason or (position_type == 'long' and signal < 0) or (position_type == 'short' and signal > 0):
                    # Close position
                    pnl = self._calculate_pnl(position, entry_price, current_price, position_type)
                    capital += pnl
                    
                    # Log trade
                    trade_log.append({
                        'entry_date': entry_price,
                        'exit_date': date,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'position_type': position_type,
                        'position_size': abs(position),
                        'pnl': pnl,
                        'exit_reason': exit_reason or 'signal_reversal'
                    })
                    
                    position = 0
                    entry_price = 0
                    position_type = None
            
            # Enter new position if signal is strong enough
            if signal != 0 and position == 0:
                position_fraction = position_sizer.calculate_position_size(
                    signal_strength, confidence, current_price, volatility
                )
                
                # Ensure position doesn't exceed maximum
                position_fraction = min(position_fraction, risk_manager.max_position_pct)
                
                if position_fraction > 0.01:  # Minimum 1% position
                    position = (capital * position_fraction) / current_price
                    
                    if signal > 0:
                        position_type = 'long'
                    else:
                        position_type = 'short'
                        position = -position  # Negative for short
                    
                    entry_price = current_price
            
            # Calculate portfolio value
            if position != 0:
                if position_type == 'long':
                    portfolio_value = capital + position * (current_price - entry_price)
                else:
                    portfolio_value = capital + abs(position) * (entry_price - current_price)
            else:
                portfolio_value = capital
            
            portfolio_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'capital': capital,
                'position': position,
                'price': current_price
            })
        
        # Close any remaining position at the end
        if position != 0:
            final_price = signals_df['Close'].iloc[-1]
            pnl = self._calculate_pnl(position, entry_price, final_price, position_type)
            capital += pnl
            
            trade_log.append({
                'entry_date': entry_price,
                'exit_date': signals_df.index[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'position_type': position_type,
                'position_size': abs(position),
                'pnl': pnl,
                'exit_reason': 'end_of_period'
            })
        
        # Store results
        self.trades = trade_log
        self.portfolio_history = portfolio_values
        
        # Calculate performance metrics
        results = self._calculate_performance_metrics(portfolio_values, trade_log)
        
        print(f"[SUCCESS] Backtest completed: {len(trade_log)} trades, Final Value: ${capital:,.2f}")
        
        return results
    
    def _calculate_pnl(self, position, entry_price, exit_price, position_type):
        """Calculate profit/loss for a trade"""
        if position_type == 'long':
            gross_pnl = position * (exit_price - entry_price)
        else:
            gross_pnl = abs(position) * (entry_price - exit_price)
        
        # Subtract commission
        commission_cost = abs(position) * (entry_price + exit_price) * self.commission
        net_pnl = gross_pnl - commission_cost
        
        return net_pnl
    
    def _calculate_performance_metrics(self, portfolio_values, trades):
        """Calculate comprehensive performance metrics"""
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate returns
        portfolio_df['returns'] = portfolio_df['portfolio_value'].pct_change()
        
        # Basic metrics
        total_return = (portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        annualized_return = ((portfolio_df['portfolio_value'].iloc[-1] / self.initial_capital) ** (252 / len(portfolio_df)) - 1) * 100
        
        # Risk metrics
        volatility = portfolio_df['returns'].std() * np.sqrt(252) * 100
        sharpe_ratio = (portfolio_df['returns'].mean() * 252) / (portfolio_df['returns'].std() * np.sqrt(252)) if portfolio_df['returns'].std() > 0 else 0
        
        # Drawdown
        cumulative = (1 + portfolio_df['returns']).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        
        # Trade statistics
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).mean() * 100 if len(trades_df) > 0 else 0
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] > 0]) > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_df['returns'][portfolio_df['returns'] < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (portfolio_df['returns'].mean() * 252) / downside_std if downside_std > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        results = {
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'volatility_pct': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown_pct': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'win_rate_pct': win_rate,
            'total_trades': len(trades_df),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'portfolio_history': portfolio_df,
            'trade_log': trades_df
        }
        
        return results
    
    def print_performance_report(self, results):
        """Print comprehensive performance report"""
        print(f"\n{'='*60}")
        print(f"BACKTEST PERFORMANCE REPORT")
        print(f"{'='*60}")
        
        print(f"\n📊 RETURNS:")
        print(f"  Total Return: {results['total_return_pct']:.2f}%")
        print(f"  Annualized Return: {results['annualized_return_pct']:.2f}%")
        
        print(f"\n📈 RISK METRICS:")
        print(f"  Volatility: {results['volatility_pct']:.2f}%")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"  Calmar Ratio: {results['calmar_ratio']:.3f}")
        
        print(f"\n💰 TRADE STATISTICS:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Win Rate: {results['win_rate_pct']:.1f}%")
        print(f"  Average Win: ${results['avg_win']:.2f}")
        print(f"  Average Loss: ${results['avg_loss']:.2f}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")


def run_complete_strategy(tickers, target_type='direction', signal_type='direction'):
    """
    Run complete quantitative trading strategy
    
    Args:
        tickers: List of stock tickers
        target_type: Model target type ('returns', 'direction', 'volatility')
        signal_type: Signal generation type ('direction', 'momentum', 'mean_reversion')
    
    Returns:
        Dictionary with results for each ticker
    """
    print(f"Running complete quantitative strategy for {tickers}...")
    
    # Import and create predictors
    from .quantitative_models import create_quantitative_pipeline
    
    # Create predictors
    predictors = create_quantitative_pipeline(tickers, target_type)
    
    results = {}
    
    for ticker, predictor in predictors.items():
        print(f"\n{'='*50}")
        print(f"PROCESSING {ticker}")
        print(f"{'='*50}")
        
        # Generate signals
        signal_generator = TradingSignal(predictor, signal_type)
        signals_df = signal_generator.generate_signals()
        
        if len(signals_df) == 0:
            print(f"[SKIP] No signals generated for {ticker}")
            continue
        
        # Set up position sizing and risk management
        position_sizer = PositionSizer(method='kelly', initial_capital=100000)
        risk_manager = RiskManager(stop_loss_pct=0.05, take_profit_pct=0.10)
        
        # Run backtest
        backtester = Backtester(initial_capital=100000)
        backtest_results = backtester.run_backtest(
            signals_df, predictor.df, position_sizer, risk_manager
        )
        
        # Print performance report
        backtester.print_performance_report(backtest_results)
        
        results[ticker] = {
            'predictor': predictor,
            'signals': signals_df,
            'backtest_results': backtest_results
        }
    
    return results


# Example usage
if __name__ == "__main__":
    # Run complete strategy
    tickers = ['AAPL', 'MSFT']
    results = run_complete_strategy(tickers, target_type='direction', signal_type='direction')
    
    print(f"\n{'='*60}")
    print(f"STRATEGY SUMMARY")
    print(f"{'='*60}")
    
    for ticker, result in results.items():
        backtest = result['backtest_results']
        print(f"{ticker}: {backtest['total_return_pct']:.2f}% return, {backtest['sharpe_ratio']:.3f} Sharpe")
