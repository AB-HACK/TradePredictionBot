"""
Configuration Management Module
Handles all configuration loading and management
"""

import os
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Config:
    """Configuration management for model settings"""
    
    def __init__(self, config_file='config.json'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            'model': {
                'test_size': 0.2,
                'random_state': 42,
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2
            },
            'data': {
                'min_data_points': 100,
                'max_missing_pct': 0.1,
                'validation_enabled': True
            },
            'security': {
                'validate_inputs': True,
                'max_ticker_length': 10,
                'sanitize_inputs': True
            },
            'monitoring': {
                'log_predictions': True,
                'log_errors': True,
                'performance_tracking': True
            },
            'model_versioning': {
                'enabled': True,
                'version_format': 'v{timestamp}_{hash}',
                'keep_versions': 5
            }
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    for key, value in user_config.items():
                        if key in default_config:
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Error loading config: {e}. Using defaults.")
        else:
            logger.info("Using default configuration")
        
        return default_config
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def save(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving config: {e}")

