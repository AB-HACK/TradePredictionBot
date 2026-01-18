"""
Model Management API Routes
Handles model operations
"""

from fastapi import APIRouter, HTTPException
from typing import Optional
import os

from app.utils.path_setup import setup_paths
from app.utils.exceptions import handle_api_exception, ModelError

# Setup paths for imports
setup_paths()

router = APIRouter()

@router.get("/list")
async def list_models(ticker: Optional[str] = None):
    """List available models"""
    # This would check the models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models')
    if not os.path.exists(models_dir):
        return {"models": []}
    
    # Scan for model files
    models = []
    if ticker:
        # Filter by ticker
        pass
    else:
        # Return all models
        pass
    
    return {"models": models}

@router.get("/info/{ticker}/{model_name}")
async def get_model_info(ticker: str, model_name: str, target_type: str = "direction"):
    """Get information about a specific model"""
    try:
        from quantitative_models import QuantitativePredictor
        loaded = QuantitativePredictor.load_model(ticker, model_name, target_type)
        if loaded:
            return {
                "ticker": ticker,
                "model_name": model_name,
                "metadata": loaded.get("metadata", {}),
                "version": loaded.get("version")
            }
        raise HTTPException(status_code=404, detail="Model not found")
    except HTTPException:
        raise
    except (FileNotFoundError, ValueError, ModelError) as e:
        raise handle_api_exception(e)
    except Exception as e:
        raise handle_api_exception(e, default_message="Error retrieving model information")
