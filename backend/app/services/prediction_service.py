"""
Prediction Service
Handles prediction logic and model management
"""

import asyncio
from typing import Optional, List, Dict
from datetime import datetime
import logging

from app.utils.path_setup import setup_paths
from app.utils.exceptions import PredictionError, ModelError, DataError, handle_service_exception

# Setup paths for imports
setup_paths()

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling predictions"""
    
    def __init__(self):
        self.predictors_cache = {}
    
    async def get_prediction(
        self, 
        ticker: str, 
        model_name: str = "Random_Forest_Classifier",
        target_type: str = "direction",
        return_confidence: bool = True
    ) -> Dict:
        """
        Get prediction for a ticker (async wrapper)
        
        Args:
            ticker: Stock ticker symbol
            model_name: Name of model to use
            target_type: Target type ('direction', 'returns', 'volatility')
            return_confidence: Whether to return confidence score
        
        Returns:
            Dictionary with prediction results
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._get_prediction_sync,
            ticker, model_name, target_type, return_confidence
        )
        return result
    
    def _get_prediction_sync(
        self, 
        ticker: str, 
        model_name: str,
        target_type: str,
        return_confidence: bool
    ) -> Dict:
        """
        Synchronous prediction logic
        
        Args:
            ticker: Stock ticker symbol
            model_name: Model name
            target_type: Target type
            return_confidence: Return confidence flag
        
        Returns:
            Prediction dictionary
        """
        try:
            from quantitative_models import QuantitativePredictor
            from data import fetch_multiple_stocks
            from data_cleaning import clean_multiple_stocks
            
            # Check cache
            cache_key = f"{ticker}_{model_name}_{target_type}"
            
            if cache_key not in self.predictors_cache:
                # Try to load saved model first
                try:
                    loaded = QuantitativePredictor.load_model(
                        ticker, model_name, target_type, model_dir="models"
                    )
                    if loaded:
                        # Model loaded successfully - would need predictor reconstruction
                        # For now, create new predictor
                        logger.debug(f"Model loaded for {ticker}, but creating new predictor")
                except (FileNotFoundError, ValueError) as e:
                    logger.debug(f"Could not load saved model for {ticker}: {e}")
                except Exception as e:
                    logger.warning(f"Unexpected error loading model for {ticker}: {e}")
                
                # Create new predictor if not loaded
                stock_data = fetch_multiple_stocks([ticker], period='3y', interval='1d', use_cache=True)
                cleaned_data = clean_multiple_stocks(stock_data, use_cache=True, save_permanent=False)
                
                if ticker not in cleaned_data:
                    raise DataError(f"No data available for {ticker}")
                
                predictor = QuantitativePredictor(cleaned_data[ticker], ticker, target_type)
                predictor.prepare_data()
                predictor.train_models()
                self.predictors_cache[cache_key] = predictor
            
            predictor = self.predictors_cache[cache_key]
            
            # Make prediction
            if return_confidence:
                prediction, confidence = predictor.predict(model_name, return_confidence=True)
            else:
                prediction = predictor.predict(model_name)
                confidence = None
            
            # Determine direction
            if target_type == "direction":
                direction = "UP" if prediction > 0.5 else "DOWN"
            else:
                direction = "UP" if prediction > 0 else "DOWN"
            
            return {
                "ticker": ticker,
                "prediction": float(prediction),
                "confidence": float(confidence) if confidence is not None else None,
                "direction": direction,
                "timestamp": datetime.now().isoformat()
            }
            
        except (ValueError, DataError, ModelError) as e:
            raise
        except Exception as e:
            raise handle_service_exception(e, "PredictionError", {"ticker": ticker, "model": model_name})
    
    async def get_batch_predictions(self, tickers: List[str], model_name: Optional[str] = None) -> List[Dict]:
        """
        Get predictions for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            model_name: Optional model name
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for ticker in tickers:
            try:
                result = await self.get_prediction(
                    ticker, 
                    model_name or "Random_Forest_Classifier"
                )
                results.append(result)
            except (ValueError, PredictionError, DataError, ModelError) as e:
                logger.warning(f"Failed to get prediction for {ticker}: {e}")
                results.append({
                    "ticker": ticker, 
                    "error": str(e),
                    "prediction": None
                })
            except Exception as e:
                logger.error(f"Unexpected error getting prediction for {ticker}: {e}", exc_info=True)
                results.append({
                    "ticker": ticker, 
                    "error": f"Unexpected error: {str(e)}",
                    "prediction": None
                })
        return results
    
    async def get_available_models(self, ticker: str) -> List[str]:
        """
        Get list of available models for a ticker
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            List of available model names
        """
        # Check models directory or return default list
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models')
        available_models = []
        
        if os.path.exists(models_dir):
            import glob
            pattern = f"{ticker}_*_{'direction'}.joblib"
            files = glob.glob(os.path.join(models_dir, pattern))
            # Extract model names from filenames
            for f in files:
                parts = os.path.basename(f).split('_')
                if len(parts) >= 3:
                    available_models.append(parts[1])
        
        # Default models if none found
        if not available_models:
            available_models = [
                "Random_Forest_Classifier",
                "XGBoost_Classifier",
                "Logistic_Regression"
            ]
        
        return available_models
