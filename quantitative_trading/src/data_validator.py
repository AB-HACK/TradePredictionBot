"""
Data Validation Module
Handles all data validation and security checks
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


class DataValidator:
    """Validate data inputs and model data"""
    
    def __init__(self, config=None):
        """Initialize validator with optional config"""
        self.config = config
    
    def validate_ticker(self, ticker: str) -> bool:
        """Validate ticker symbol"""
        if not isinstance(ticker, str):
            logger.error(f"Invalid ticker type: {type(ticker)}")
            return False
        
        max_length = self.config.get('security.max_ticker_length', 10) if self.config else 10
        if len(ticker) > max_length:
            logger.error(f"Ticker too long: {ticker}")
            return False
        
        # Sanitize: only alphanumeric and dots
        sanitize = self.config.get('security.sanitize_inputs', True) if self.config else True
        if sanitize and not ticker.replace('.', '').isalnum():
            logger.error(f"Invalid ticker format: {ticker}")
            return False
        
        return True
    
    def validate_dataframe(self, df: pd.DataFrame, min_rows: Optional[int] = None, required_cols: Optional[List[str]] = None, max_missing_pct: Optional[float] = None) -> Tuple[bool, str]:
        """Validate DataFrame structure and content"""
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        min_rows = min_rows or (self.config.get('data.min_data_points', 100) if self.config else 100)
        if len(df) < min_rows:
            return False, f"Insufficient data: {len(df)} rows (minimum: {min_rows})"
        
        required_cols = required_cols or ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for excessive missing values
        max_missing = max_missing_pct or (self.config.get('data.max_missing_pct', 0.1) if self.config else 0.1)
        for col in required_cols:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > max_missing:
                return False, f"Too many missing values in {col}: {missing_pct:.1%}"
        
        return True, "Valid"
    
    def validate_features(self, features: np.ndarray, expected_shape: Optional[Tuple] = None) -> Tuple[bool, str]:
        """Validate feature array"""
        if features is None:
            return False, "Features are None"
        
        if not isinstance(features, np.ndarray):
            return False, f"Features must be numpy array, got {type(features)}"
        
        if features.size == 0:
            return False, "Features array is empty"
        
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return False, "Features contain NaN or Inf values"
        
        if expected_shape and features.shape != expected_shape:
            return False, f"Shape mismatch: expected {expected_shape}, got {features.shape}"
        
        return True, "Valid"

