"""
Backtest API Routes
Handles backtesting requests
"""

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional

from app.utils.path_setup import setup_paths
from app.utils.exceptions import handle_api_exception, BacktestError
from app.services.backtest_service import BacktestService

# Setup paths for imports
setup_paths()

router = APIRouter()
backtest_service = BacktestService()

class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    tickers: List[str]
    target_type: str = "direction"
    signal_type: str = "direction"
    initial_capital: Optional[float] = 100000

@router.post("/run")
async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
    """
    Run backtest for given tickers
    
    Args:
        request: Backtest request with tickers and parameters
    
    Returns:
        Backtest results
    """
    try:
        result = await backtest_service.run_backtest(
            tickers=request.tickers,
            target_type=request.target_type,
            signal_type=request.signal_type,
            initial_capital=request.initial_capital
        )
        return result
    except (ValueError, BacktestError) as e:
        raise handle_api_exception(e)
    except Exception as e:
        raise handle_api_exception(e, default_message="Backtest error occurred")

@router.get("/results/{ticker}")
async def get_backtest_results(ticker: str):
    """Get cached backtest results for a ticker"""
    try:
        results = await backtest_service.get_cached_results(ticker)
        if results:
            return results
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Backtest results not found")
    except HTTPException:
        raise
    except Exception as e:
        raise handle_api_exception(e, default_message="Error retrieving backtest results")
