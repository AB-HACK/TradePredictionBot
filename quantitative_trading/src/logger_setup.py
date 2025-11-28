"""
Logging Setup Module
Centralized logging configuration
"""

import os
import logging


def setup_logger(name='quantitative_models', log_file='logs/model.log', level=logging.INFO):
    """Setup logging configuration"""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger

