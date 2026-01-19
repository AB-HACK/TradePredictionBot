"""
Data API Routes
Handles stock data requests
"""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel

from app.utils.path_setup import setup_paths
from app.utils.exceptions import handle_api_exception, DataError

# Setup paths for imports
setup_paths()

from data import fetch_live_data, fetch_multiple_stocks

router = APIRouter()

class BatchStockDataRequest(BaseModel):
    """Request model for batch stock data"""
    tickers: List[str]
    period: str = "1y"
    interval: str = "1d"

@router.get("/ticker/{ticker}")
async def get_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d"
):
    """
    Get stock data for a ticker
    
    Args:
        ticker: Stock ticker symbol
        period: Time period (1d, 1mo, 1y, etc.)
        interval: Data interval (1d, 1h, etc.)
    
    Returns:
        Stock data as JSON
    """
    try:
        df = fetch_live_data(ticker, period=period, interval=interval, use_cache=True)
        if df is None or df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {ticker}")
        
        # Convert DataFrame to JSON
        df_dict = df.reset_index().to_dict(orient='records')
        return {
            "ticker": ticker,
            "period": period,
            "interval": interval,
            "rows": len(df),
            "data": df_dict
        }
    except HTTPException:
        raise
    except (ValueError, DataError) as e:
        raise handle_api_exception(e)
    except Exception as e:
        raise handle_api_exception(e, default_message=f"Error fetching data for {ticker}")

@router.post("/batch")
async def get_batch_stock_data(request: BatchStockDataRequest):
    """
    Get stock data for multiple tickers
    
    Args:
        request: Batch stock data request with tickers, period, and interval
    
    Returns:
        Dictionary of stock data
    """
    try:
        stock_data = fetch_multiple_stocks(
            request.tickers, 
            period=request.period, 
            interval=request.interval, 
            use_cache=True
        )
        
        # Convert DataFrames to JSON
        result = {}
        for ticker, df in stock_data.items():
            if df is not None and not df.empty:
                result[ticker] = {
                    "rows": len(df),
                    "data": df.reset_index().to_dict(orient='records')
                }
        
        return {
            "tickers": list(result.keys()),
            "data": result
        }
    except (ValueError, DataError) as e:
        raise handle_api_exception(e)
    except Exception as e:
        raise handle_api_exception(e, default_message="Error fetching batch data")
