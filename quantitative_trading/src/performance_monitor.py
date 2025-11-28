"""
Performance Monitoring Module
Handles logging and monitoring of model performance
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor model performance and predictions"""
    
    def __init__(self, log_file='logs/performance.log', config=None):
        self.log_file = log_file
        self.config = config
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        self.predictions_log = []
        self.log_enabled = True
    
    def log_prediction(self, ticker: str, model_name: str, prediction: Any, confidence: Optional[float] = None):
        """Log prediction for monitoring"""
        log_enabled = self.config.get('monitoring.log_predictions', True) if self.config else True
        if not self.log_enabled or not log_enabled:
            return
        
        entry = {
            'timestamp': datetime.now().isoformat(),
            'ticker': ticker,
            'model': model_name,
            'prediction': float(prediction) if isinstance(prediction, (int, float, np.number)) else str(prediction),
            'confidence': float(confidence) if confidence else None
        }
        self.predictions_log.append(entry)
        
        # Write to file periodically
        if len(self.predictions_log) >= 10:
            self._flush_logs()
    
    def _flush_logs(self):
        """Write logs to file"""
        try:
            with open(self.log_file, 'a') as f:
                for entry in self.predictions_log:
                    f.write(json.dumps(entry) + '\n')
            self.predictions_log = []
        except Exception as e:
            logger.error(f"Error writing performance logs: {e}")
    
    def log_error(self, error_type: str, message: str, context: Optional[Dict] = None):
        """Log errors for monitoring"""
        log_enabled = self.config.get('monitoring.log_errors', True) if self.config else True
        if self.log_enabled and log_enabled:
            logger.error(f"{error_type}: {message}", extra=context or {})

