"""
Prediction API Routes
Handles prediction requests and responses
"""

from fastapi import APIRouter, BackgroundTasks
from typing import List, Optional
from pydantic import BaseModel

from app.utils.path_setup import setup_paths
from app.utils.exceptions import handle_api_exception, PredictionError, DataError, ModelError
from app.services.prediction_service import PredictionService

# Setup paths for imports
setup_paths()

router = APIRouter()
prediction_service = PredictionService()

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    ticker: str
    model_name: Optional[str] = "Random_Forest_Classifier"
    target_type: Optional[str] = "direction"
    return_confidence: Optional[bool] = True

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    ticker: str
    prediction: float
    confidence: Optional[float]
    direction: str  # "UP" or "DOWN"
    timestamp: str

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    tickers: List[str]
    model_name: Optional[str] = None

@router.post("/predict", response_model=PredictionResponse)
async def get_prediction(request: PredictionRequest):
    """
    Get prediction for a stock ticker
    
    Args:
        request: Prediction request with ticker and model info
    
    Returns:
        Prediction response with prediction, confidence, and direction
    """
    try:
        result = await prediction_service.get_prediction(
            ticker=request.ticker,
            model_name=request.model_name,
            target_type=request.target_type,
            return_confidence=request.return_confidence
        )
        return result
    except (ValueError, PredictionError, DataError, ModelError) as e:
        raise handle_api_exception(e)
    except Exception as e:
        raise handle_api_exception(e, default_message="Prediction error occurred")

@router.post("/predict/batch")
async def get_batch_predictions(request: BatchPredictionRequest):
    """
    Get predictions for multiple tickers
    
    Args:
        request: Batch prediction request with list of tickers
    
    Returns:
        List of predictions for each ticker
    """
    try:
        results = await prediction_service.get_batch_predictions(
            request.tickers, 
            request.model_name
        )
        return {"predictions": results}
    except Exception as e:
        raise handle_api_exception(e, default_message="Batch prediction error occurred")

@router.get("/models/{ticker}")
async def get_available_models(ticker: str):
    """
    Get available models for a ticker
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        List of available models for the ticker
    """
    try:
        models = await prediction_service.get_available_models(ticker)
        return {"ticker": ticker, "models": models}
    except Exception as e:
        raise handle_api_exception(e, default_message="Error retrieving available models")
