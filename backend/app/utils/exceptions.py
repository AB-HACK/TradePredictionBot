"""
Exception Utilities
Custom exceptions and exception handling utilities
"""

from typing import Optional, Type, Tuple
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass

class DataError(Exception):
    """Custom exception for data fetching/processing errors"""
    pass

class ModelError(Exception):
    """Custom exception for model operations"""
    pass

class BacktestError(Exception):
    """Custom exception for backtest operations"""
    pass

def handle_api_exception(
    exception: Exception,
    default_status: int = 500,
    default_message: str = "An error occurred",
    log_error: bool = True
) -> HTTPException:
    """
    Convert exception to HTTPException with appropriate status code
    
    Args:
        exception: The exception to handle
        default_status: Default HTTP status code
        default_message: Default error message
        log_error: Whether to log the error
    
    Returns:
        HTTPException with appropriate status and message
    """
    # Map exception types to status codes
    status_map = {
        ValueError: 400,
        FileNotFoundError: 404,
        KeyError: 404,
        PredictionError: 400,
        DataError: 400,
        ModelError: 404,
        BacktestError: 500,
    }
    
    status_code = status_map.get(type(exception), default_status)
    error_message = str(exception) or default_message
    
    if log_error:
        logger.error(f"API Error [{status_code}]: {error_message}", exc_info=True)
    
    return HTTPException(status_code=status_code, detail=error_message)

def handle_service_exception(
    exception: Exception,
    error_type: str = "ServiceError",
    context: Optional[dict] = None
) -> ValueError:
    """
    Convert service exception to ValueError for API layer
    
    Args:
        exception: The exception to handle
        error_type: Type of error for logging
        context: Additional context for logging
    
    Returns:
        ValueError with formatted message
    """
    error_message = f"{error_type}: {str(exception)}"
    logger.error(error_message, exc_info=True, extra=context or {})
    return ValueError(error_message)
