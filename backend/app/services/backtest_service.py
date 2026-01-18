"""
Backtest Service
Handles backtesting logic
"""

import asyncio
from typing import List, Optional, Dict
import logging

from app.utils.path_setup import setup_paths
from app.utils.exceptions import BacktestError, handle_service_exception

# Setup paths for imports
setup_paths()

logger = logging.getLogger(__name__)

class BacktestService:
    """Service for handling backtests"""
    
    def __init__(self):
        self.results_cache = {}
    
    async def run_backtest(
        self,
        tickers: List[str],
        target_type: str = "direction",
        signal_type: str = "direction",
        initial_capital: float = 100000
    ) -> Dict:
        """
        Run backtest for given tickers
        
        Args:
            tickers: List of ticker symbols
            target_type: Target type for model
            signal_type: Signal type for strategy
            initial_capital: Starting capital
        
        Returns:
            Backtest results dictionary
        """
        # Run in thread pool
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._run_backtest_sync,
            tickers, target_type, signal_type, initial_capital
        )
        return result
    
    def _run_backtest_sync(
        self,
        tickers: List[str],
        target_type: str,
        signal_type: str,
        initial_capital: float
    ) -> Dict:
        """Synchronous backtest execution"""
        try:
            from trading_strategy import run_complete_strategy
            
            results = run_complete_strategy(
                tickers=tickers,
                target_type=target_type,
                signal_type=signal_type,
                initial_capital=initial_capital
            )
            
            # Convert results to API format
            formatted_results = {}
            for ticker, result_data in results.items():
                if 'backtest_results' in result_data:
                    bt = result_data['backtest_results']
                    
                    # Convert portfolio history to list of dicts
                    portfolio_history = []
                    if 'portfolio_history' in bt:
                        ph_df = bt['portfolio_history']
                        for idx, row in ph_df.iterrows():
                            portfolio_history.append({
                                "date": str(idx),
                                "portfolio_value": float(row.get('portfolio_value', 0)),
                                "returns": float(row.get('returns', 0))
                            })
                    
                    formatted_results[ticker] = {
                        "total_return_pct": float(bt.get('total_return_pct', 0)),
                        "annualized_return_pct": float(bt.get('annualized_return_pct', 0)),
                        "volatility_pct": float(bt.get('volatility_pct', 0)),
                        "sharpe_ratio": float(bt.get('sharpe_ratio', 0)),
                        "sortino_ratio": float(bt.get('sortino_ratio', 0)),
                        "max_drawdown_pct": float(bt.get('max_drawdown_pct', 0)),
                        "calmar_ratio": float(bt.get('calmar_ratio', 0)),
                        "win_rate_pct": float(bt.get('win_rate_pct', 0)),
                        "total_trades": int(bt.get('total_trades', 0)),
                        "avg_win": float(bt.get('avg_win', 0)),
                        "avg_loss": float(bt.get('avg_loss', 0)),
                        "profit_factor": float(bt.get('profit_factor', 0)),
                        "portfolio_history": portfolio_history
                    }
            
            # Cache results
            for ticker in tickers:
                if ticker in formatted_results:
                    self.results_cache[ticker] = formatted_results[ticker]
            
            return {
                "status": "completed",
                "results": formatted_results
            }
            
        except (ValueError, FileNotFoundError, KeyError) as e:
            raise BacktestError(f"Backtest failed: {str(e)}")
        except Exception as e:
            raise handle_service_exception(e, "BacktestError", {"tickers": tickers})
    
    async def get_cached_results(self, ticker: str) -> Optional[Dict]:
        """Get cached backtest results"""
        return self.results_cache.get(ticker)
