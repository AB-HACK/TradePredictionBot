# =============================================================================
# QUANTITATIVE MACHINE LEARNING MODELS
# =============================================================================
# This module provides comprehensive machine learning models for stock prediction.
# It includes feature engineering, model training, and evaluation frameworks.
# 
# Models Included:
# 1. Regression Models - Linear, Random Forest, XGBoost
# 2. Classification Models - Direction prediction (up/down)
# 3. Time Series Models - LSTM, ARIMA
# 4. Ensemble Methods - Voting, Stacking
# 5. Advanced Features - Lag features, rolling statistics, volatility
# =============================================================================

import pandas as pd
import numpy as np
import warnings
import os
import sys
import joblib
import logging
import json
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb

# Use parent cache_manager instead of duplicate
parent_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
if parent_src not in sys.path:
    sys.path.insert(0, parent_src)
from cache_manager import get_cache_manager
warnings.filterwarnings('ignore')

# =============================================================================
# PRODUCTION FEATURES: Logging, Configuration, Validation, Monitoring
# =============================================================================

# Setup logging
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

logger = setup_logger()

# Configuration management
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

# Global config instance
config = Config()

# Data validation
class DataValidator:
    """Validate data inputs and model data"""
    
    @staticmethod
    def validate_ticker(ticker: str) -> bool:
        """Validate ticker symbol"""
        if not isinstance(ticker, str):
            logger.error(f"Invalid ticker type: {type(ticker)}")
            return False
        
        max_length = config.get('security.max_ticker_length', 10)
        if len(ticker) > max_length:
            logger.error(f"Ticker too long: {ticker}")
            return False
        
        # Sanitize: only alphanumeric and dots
        if config.get('security.sanitize_inputs', True):
            if not ticker.replace('.', '').isalnum():
                logger.error(f"Invalid ticker format: {ticker}")
                return False
        
        return True
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, min_rows: int = None, required_cols: List[str] = None) -> Tuple[bool, str]:
        """Validate DataFrame structure and content"""
        if df is None:
            return False, "DataFrame is None"
        
        if df.empty:
            return False, "DataFrame is empty"
        
        min_rows = min_rows or config.get('data.min_data_points', 100)
        if len(df) < min_rows:
            return False, f"Insufficient data: {len(df)} rows (minimum: {min_rows})"
        
        required_cols = required_cols or ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check for excessive missing values
        max_missing = config.get('data.max_missing_pct', 0.1)
        for col in required_cols:
            missing_pct = df[col].isna().sum() / len(df)
            if missing_pct > max_missing:
                return False, f"Too many missing values in {col}: {missing_pct:.1%}"
        
        return True, "Valid"
    
    @staticmethod
    def validate_features(features: np.ndarray, expected_shape: Tuple = None) -> Tuple[bool, str]:
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

# Model versioning
class ModelVersionManager:
    """Manage model versions and metadata"""
    
    @staticmethod
    def generate_version(model_name: str, ticker: str, metadata: Dict) -> str:
        """Generate version string for model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Create hash from metadata
        metadata_str = json.dumps(metadata, sort_keys=True)
        hash_obj = hashlib.md5(metadata_str.encode())
        hash_short = hash_obj.hexdigest()[:8]
        return f"{ticker}_{model_name}_{timestamp}_{hash_short}"
    
    @staticmethod
    def save_version_info(version: str, metadata: Dict, save_dir: str):
        """Save version information"""
        version_file = os.path.join(save_dir, 'versions.json')
        versions = []
        
        if os.path.exists(version_file):
            try:
                with open(version_file, 'r') as f:
                    versions = json.load(f)
            except:
                versions = []
        
        versions.append({
            'version': version,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata
        })
        
        # Keep only last N versions
        keep_versions = config.get('model_versioning.keep_versions', 5)
        versions = versions[-keep_versions:]
        
        with open(version_file, 'w') as f:
            json.dump(versions, f, indent=2)
        
        logger.info(f"Saved version info: {version}")

# Performance monitoring
class PerformanceMonitor:
    """Monitor model performance and predictions"""
    
    def __init__(self, log_file='logs/performance.log'):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
        self.predictions_log = []
    
    def log_prediction(self, ticker: str, model_name: str, prediction: Any, confidence: float = None):
        """Log prediction for monitoring"""
        if config.get('monitoring.log_predictions', True):
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
    
    def log_error(self, error_type: str, message: str, context: Dict = None):
        """Log errors for monitoring"""
        if config.get('monitoring.log_errors', True):
            logger.error(f"{error_type}: {message}", extra=context or {})

# Global monitor instance
monitor = PerformanceMonitor()

class QuantitativeFeatureEngineer:
    """
    Advanced feature engineering for quantitative trading models
    """
    
    def __init__(self, df, ticker_name):
        self.df = df.copy()
        self.ticker_name = ticker_name
        self.feature_columns = []
        
    def create_advanced_features(self, lookback_periods=[5, 10, 20, 50]):
        """
        Create advanced features for machine learning models
        
        Args:
            lookback_periods: List of periods for rolling calculations
        """
        print(f"Creating advanced features for {self.ticker_name}...")
        
        # Price-based features
        self._create_price_features(lookback_periods)
        
        # Volume-based features
        self._create_volume_features(lookback_periods)
        
        # Technical indicators
        self._create_technical_features()
        
        # Lag features
        self._create_lag_features()
        
        # Rolling statistics
        self._create_rolling_features(lookback_periods)
        
        # Market regime features
        self._create_regime_features()
        
        # Volatility features
        self._create_volatility_features(lookback_periods)
        
        print(f"[SUCCESS] Created {len(self.feature_columns)} features for {self.ticker_name}")
        return self.df
    
    def _create_price_features(self, periods):
        """Create price-based features"""
        # Price ratios
        for period in periods:
            self.df[f'Price_SMA_{period}_Ratio'] = self.df['Close'] / self.df['Close'].rolling(period).mean()
            self.df[f'Price_High_{period}_Ratio'] = self.df['Close'] / self.df['High'].rolling(period).max()
            self.df[f'Price_Low_{period}_Ratio'] = self.df['Close'] / self.df['Low'].rolling(period).min()
            
        # Price momentum
        for period in [1, 2, 3, 5, 10]:
            self.df[f'Returns_{period}d'] = self.df['Close'].pct_change(period)
            self.df[f'Log_Returns_{period}d'] = np.log(self.df['Close'] / self.df['Close'].shift(period))
            
        # Price volatility
        for period in periods:
            self.df[f'Price_Volatility_{period}'] = self.df['Returns'].rolling(period).std()
            
        # Gap features
        self.df['Gap'] = (self.df['Open'] - self.df['Close'].shift(1)) / self.df['Close'].shift(1)
        self.df['Gap_Size'] = abs(self.df['Gap'])
        
        # Update feature columns
        price_features = [col for col in self.df.columns if 'Price_' in col or 'Returns_' in col or 'Log_Returns_' in col or 'Gap' in col]
        self.feature_columns.extend(price_features)
    
    def _create_volume_features(self, periods):
        """Create volume-based features"""
        # Volume ratios
        for period in periods:
            self.df[f'Volume_SMA_{period}_Ratio'] = self.df['Volume'] / self.df['Volume'].rolling(period).mean()
            
        # Volume-price relationship
        self.df['Volume_Price_Trend'] = self.df['Volume'] * self.df['Returns']
        self.df['On_Balance_Volume'] = (self.df['Volume'] * np.sign(self.df['Returns'])).cumsum()
        
        # Volume volatility
        for period in periods:
            self.df[f'Volume_Volatility_{period}'] = self.df['Volume'].rolling(period).std()
            
        # Update feature columns
        volume_features = [col for col in self.df.columns if 'Volume_' in col or 'On_Balance_Volume' in col]
        self.feature_columns.extend(volume_features)
    
    def _create_technical_features(self):
        """Create additional technical indicators"""
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        sma = self.df['Close'].rolling(bb_period).mean()
        std = self.df['Close'].rolling(bb_period).std()
        
        self.df['BB_Upper'] = sma + (std * bb_std)
        self.df['BB_Lower'] = sma - (std * bb_std)
        self.df['BB_Width'] = (self.df['BB_Upper'] - self.df['BB_Lower']) / sma
        self.df['BB_Position'] = (self.df['Close'] - self.df['BB_Lower']) / (self.df['BB_Upper'] - self.df['BB_Lower'])
        
        # Stochastic Oscillator
        k_period = 14
        d_period = 3
        
        low_min = self.df['Low'].rolling(k_period).min()
        high_max = self.df['High'].rolling(k_period).max()
        
        self.df['Stoch_K'] = 100 * ((self.df['Close'] - low_min) / (high_max - low_min))
        self.df['Stoch_D'] = self.df['Stoch_K'].rolling(d_period).mean()
        
        # Williams %R
        self.df['Williams_R'] = -100 * ((high_max - self.df['Close']) / (high_max - low_min))
        
        # Commodity Channel Index (CCI)
        tp = (self.df['High'] + self.df['Low'] + self.df['Close']) / 3
        tp_sma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        self.df['CCI'] = (tp - tp_sma) / (0.015 * mad)
        
        # Update feature columns
        tech_features = [col for col in self.df.columns if col.startswith(('BB_', 'Stoch_', 'Williams_', 'CCI'))]
        self.feature_columns.extend(tech_features)
    
    def _create_lag_features(self):
        """Create lag features for time series modeling"""
        lag_periods = [1, 2, 3, 5, 10, 20]
        
        for lag in lag_periods:
            self.df[f'Close_Lag_{lag}'] = self.df['Close'].shift(lag)
            self.df[f'Returns_Lag_{lag}'] = self.df['Returns'].shift(lag)
            self.df[f'Volume_Lag_{lag}'] = self.df['Volume'].shift(lag)
            
        # Update feature columns
        lag_features = [col for col in self.df.columns if 'Lag_' in col]
        self.feature_columns.extend(lag_features)
    
    def _create_rolling_features(self, periods):
        """Create rolling statistical features"""
        for period in periods:
            # Rolling statistics for returns
            self.df[f'Returns_Mean_{period}'] = self.df['Returns'].rolling(period).mean()
            self.df[f'Returns_Std_{period}'] = self.df['Returns'].rolling(period).std()
            self.df[f'Returns_Skew_{period}'] = self.df['Returns'].rolling(period).skew()
            self.df[f'Returns_Kurt_{period}'] = self.df['Returns'].rolling(period).kurt()
            
            # Rolling statistics for volume
            self.df[f'Volume_Mean_{period}'] = self.df['Volume'].rolling(period).mean()
            self.df[f'Volume_Std_{period}'] = self.df['Volume'].rolling(period).std()
            
        # Update feature columns
        rolling_features = [col for col in self.df.columns if any(x in col for x in ['Mean_', 'Std_', 'Skew_', 'Kurt_'])]
        self.feature_columns.extend(rolling_features)
    
    def _create_regime_features(self):
        """Create market regime features"""
        # Trend detection
        self.df['Trend_20'] = np.where(self.df['Close'] > self.df['Close'].rolling(20).mean(), 1, -1)
        self.df['Trend_50'] = np.where(self.df['Close'] > self.df['Close'].rolling(50).mean(), 1, -1)
        
        # Volatility regime
        vol_20 = self.df['Returns'].rolling(20).std()
        vol_median = vol_20.rolling(252).median()
        self.df['High_Vol_Regime'] = np.where(vol_20 > vol_median, 1, 0)
        
        # Volume regime
        vol_ratio = self.df['Volume'] / self.df['Volume'].rolling(20).mean()
        self.df['High_Volume_Regime'] = np.where(vol_ratio > 1.5, 1, 0)
        
        # Update feature columns
        regime_features = [col for col in self.df.columns if 'Trend_' in col or 'Regime' in col]
        self.feature_columns.extend(regime_features)
    
    def _create_volatility_features(self, periods):
        """Create volatility-based features"""
        # GARCH-like features
        for period in periods:
            # Realized volatility
            self.df[f'Realized_Vol_{period}'] = np.sqrt(self.df['Returns'].rolling(period).var() * 252)
            
            # Parkinson volatility (using high-low)
            self.df[f'Parkinson_Vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) * np.log(self.df['High'] / self.df['Low']).rolling(period).mean() * 252
            )
            
        # Volatility clustering
        self.df['Vol_Clustering'] = (self.df['Returns'] * self.df['Returns'].shift(1)).rolling(20).mean()
        
        # Update feature columns
        vol_features = [col for col in self.df.columns if 'Vol_' in col or 'Vol_Clustering' in col]
        self.feature_columns.extend(vol_features)


class QuantitativePredictor:
    """
    Main class for quantitative prediction models
    
    Features:
    - Error handling with comprehensive logging
    - Data validation at each step
    - Configuration management
    - Performance monitoring
    - Model versioning
    - Security validation
    """
    
    def __init__(self, df, ticker_name, target_type='returns'):
        """
        Initialize predictor with validation and error handling
        
        Args:
            df: DataFrame with features
            ticker_name: Stock ticker
            target_type: 'returns', 'direction', or 'volatility'
        
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Security: Validate ticker
            if not DataValidator.validate_ticker(ticker_name):
                raise ValueError(f"Invalid ticker: {ticker_name}")
            
            # Data validation
            is_valid, error_msg = DataValidator.validate_dataframe(df)
            if not is_valid:
                raise ValueError(f"Invalid DataFrame: {error_msg}")
            
            # Validate target type
            valid_targets = ['returns', 'direction', 'volatility']
            if target_type not in valid_targets:
                raise ValueError(f"Invalid target_type: {target_type}. Must be one of {valid_targets}")
            
            self.df = df.copy()
            self.ticker_name = ticker_name
            self.target_type = target_type
            self.feature_engineer = QuantitativeFeatureEngineer(df, ticker_name)
            self.models = {}
            self.scalers = {}
            self.feature_importance = {}
            self.version = None
            self.training_metadata = {}
            
            logger.info(f"Initialized predictor for {ticker_name} with target_type={target_type}")
            
        except Exception as e:
            logger.error(f"Error initializing predictor for {ticker_name}: {e}", exc_info=True)
            monitor.log_error('InitializationError', str(e), {'ticker': ticker_name, 'target_type': target_type})
            raise
        
    def prepare_data(self, prediction_horizon=1, use_cache=True):
        """
        Prepare data for machine learning
        
        Args:
            prediction_horizon: Days ahead to predict
            use_cache: Whether to use caching
        """
        print(f"Preparing data for {self.ticker_name}...")
        
        # Check cache first
        if use_cache:
            cache = get_cache_manager()
            for filepath, metadata in cache.cache_metadata.items():
                if (metadata.get('ticker') == self.ticker_name and 
                    metadata.get('data_type') == 'ml_ready'):
                    print(f"[CACHE] Loading ML-ready data for {self.ticker_name}...")
                    cached_df = cache.load_dataframe(filepath)
                    if cached_df is not None:
                        self.df = cached_df
                        return self.df
        
        # Create advanced features
        self.feature_engineer.create_advanced_features()
        self.df = self.feature_engineer.df
        
        # Create targets based on prediction type
        if self.target_type == 'returns':
            self.df['Target'] = self.df['Returns'].shift(-prediction_horizon)
        elif self.target_type == 'direction':
            future_returns = self.df['Returns'].shift(-prediction_horizon)
            self.df['Target'] = np.where(future_returns > 0, 1, 0)
        elif self.target_type == 'volatility':
            self.df['Target'] = self.df['Returns'].shift(-prediction_horizon).rolling(5).std()
        
        # Cache the prepared data
        if use_cache:
            metadata = {
                'prediction_horizon': prediction_horizon,
                'target_type': self.target_type,
                'features_count': len(self.feature_engineer.feature_columns),
                'preparation_timestamp': pd.Timestamp.now().isoformat()
            }
            cache.store_dataframe(self.df, 'ml_ready', self.ticker_name, metadata)
        
        print(f"[SUCCESS] Data prepared for {self.ticker_name}: {len(self.feature_engineer.feature_columns)} features")
        return self.df
    
    def train_models(self, test_size=0.2, use_cache=True):
        """
        Train multiple machine learning models with validation and error handling
        
        Args:
            test_size: Proportion of data for testing (from config if not provided)
            use_cache: Whether to use caching
        
        Returns:
            Dictionary of trained models
        
        Raises:
            ValueError: If training fails
        """
        try:
            # Get test_size from config if not provided
            test_size = test_size or config.get('model.test_size', 0.2)
            
            # Validate test_size
            if not 0 < test_size < 1:
                raise ValueError(f"Invalid test_size: {test_size}. Must be between 0 and 1")
            
            logger.info(f"Training models for {self.ticker_name} (test_size={test_size})")
            
            # Prepare features and target
            feature_cols = self.feature_engineer.feature_columns
            if len(feature_cols) == 0:
                raise ValueError("No features available. Run prepare_data() first.")
            
            X = self.df[feature_cols].dropna()
            y = self.df['Target'].loc[X.index].dropna()
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            # Validate data
            if len(X) == 0:
                raise ValueError(f"No valid data for training {self.ticker_name}")
            
            # Validate features
            is_valid, error_msg = DataValidator.validate_features(X.values)
            if not is_valid:
                raise ValueError(f"Invalid features: {error_msg}")
            
            # Time series split
            split_point = int(len(X) * (1 - test_size))
            if split_point < 10:
                raise ValueError(f"Insufficient data for training: {split_point} samples")
            
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Store training metadata
            self.training_metadata = {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': len(feature_cols),
                'test_size': test_size,
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Validate scaled features
            is_valid, error_msg = DataValidator.validate_features(X_train_scaled)
            if not is_valid:
                raise ValueError(f"Invalid scaled features: {error_msg}")
            
            self.scalers['standard'] = scaler
            
            # Train models based on target type
            if self.target_type in ['returns', 'volatility']:
                self._train_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)
            else:
                self._train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
            
            if len(self.models) == 0:
                raise ValueError("No models were successfully trained")
            
            logger.info(f"Successfully trained {len(self.models)} models for {self.ticker_name}")
            return self.models
            
        except Exception as e:
            logger.error(f"Error training models for {self.ticker_name}: {e}", exc_info=True)
            monitor.log_error('TrainingError', str(e), {'ticker': self.ticker_name})
            raise
    
    def _train_regression_models(self, X_train, X_test, y_train, y_test):
        """Train regression models"""
        # Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
        
        self.models['Linear_Regression'] = {
            'model': lr,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred,
            'y_test': y_test.values
        }
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        correlation = np.corrcoef(y_test, y_pred)[0, 1]
        directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
        
        self.models['Random_Forest'] = {
            'model': rf,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae,
            'r2': r2,
            'correlation': correlation,
            'directional_accuracy': directional_accuracy,
            'predictions': y_pred,
            'y_test': y_test.values,
            'feature_importance': dict(zip(self.feature_engineer.feature_columns, rf.feature_importances_))
        }
        
        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            correlation = np.corrcoef(y_test, y_pred)[0, 1]
            directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
            
            self.models['XGBoost'] = {
                'model': xgb_model,
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mae,
                'r2': r2,
                'correlation': correlation,
                'directional_accuracy': directional_accuracy,
                'predictions': y_pred,
                'y_test': y_test.values,
                'feature_importance': dict(zip(self.feature_engineer.feature_columns, xgb_model.feature_importances_))
            }
        except ImportError:
            print("[WARNING] XGBoost not available. Install with: pip install xgboost")
    
    def _train_classification_models(self, X_train, X_test, y_train, y_test):
        """Train classification models"""
        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        self.models['Logistic_Regression'] = {
            'model': lr,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_test': y_test.values,
            'classification_report': classification_report(y_test, y_pred)
        }
        
        # Random Forest Classifier
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        self.models['Random_Forest_Classifier'] = {
            'model': rf,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'y_test': y_test.values,
            'feature_importance': dict(zip(self.feature_engineer.feature_columns, rf.feature_importances_))
        }
        
        # XGBoost Classifier
        try:
            xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            y_pred = xgb_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)
            
            self.models['XGBoost_Classifier'] = {
                'model': xgb_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'predictions': y_pred,
                'y_test': y_test.values,
                'feature_importance': dict(zip(self.feature_engineer.feature_columns, xgb_model.feature_importances_))
            }
        except ImportError:
            print("[WARNING] XGBoost not available. Install with: pip install xgboost")
    
    def evaluate_models(self, verbose=True):
        """
        Evaluate trained models and return comprehensive performance metrics
        
        Args:
            verbose: If True, print detailed evaluation report
        
        Returns:
            Dictionary with evaluation results for each model
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"MODEL EVALUATION FOR {self.ticker_name}")
            print(f"{'='*60}")
        
        results = {}
        best_model = None
        best_score = -np.inf
        
        for model_name, model_info in self.models.items():
            if verbose:
                print(f"\n📊 {model_name}:")
            
            # Regression models
            if 'mse' in model_info:
                rmse = model_info['rmse']
                mae = model_info.get('mae', 0)
                r2 = model_info.get('r2', 0)
                correlation = model_info.get('correlation', 0)
                directional_acc = model_info.get('directional_accuracy', 0)
                
                if verbose:
                    print(f"  RMSE: {rmse:.6f} (lower is better)")
                    print(f"  MAE: {mae:.6f} (lower is better)")
                    print(f"  R² Score: {r2:.4f} (1.0 = perfect, 0 = baseline, <0 = worse than baseline)")
                    print(f"  Correlation: {correlation:.4f} (1.0 = perfect, 0 = no correlation)")
                    print(f"  Directional Accuracy: {directional_acc:.2%} (predicts up/down correctly)")
                    
                    # Performance interpretation
                    if r2 > 0.1:
                        print(f"  ✅ Model explains {r2:.1%} of variance (Good)")
                    elif r2 > 0:
                        print(f"  ⚠️  Model explains {r2:.1%} of variance (Weak)")
                    else:
                        print(f"  ❌ Model worse than baseline (Poor)")
                    
                    if directional_acc > 0.55:
                        print(f"  ✅ Directional accuracy {directional_acc:.1%} (Good for trading)")
                    elif directional_acc > 0.50:
                        print(f"  ⚠️  Directional accuracy {directional_acc:.1%} (Slightly better than random)")
                    else:
                        print(f"  ❌ Directional accuracy {directional_acc:.1%} (Worse than random)")
                
                results[model_name] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'correlation': correlation,
                    'directional_accuracy': directional_acc
                }
                
                # Use R² as primary score for regression
                score = r2
                
            # Classification models
            elif 'accuracy' in model_info:
                accuracy = model_info['accuracy']
                precision = model_info.get('precision', 0)
                recall = model_info.get('recall', 0)
                f1 = model_info.get('f1_score', 0)
                cm = model_info.get('confusion_matrix', None)
                
                if verbose:
                    print(f"  Accuracy: {accuracy:.4f} ({accuracy:.2%})")
                    print(f"  Precision: {precision:.4f}")
                    print(f"  Recall: {recall:.4f}")
                    print(f"  F1-Score: {f1:.4f}")
                    
                    if cm is not None:
                        print(f"  Confusion Matrix:")
                        print(f"    {cm}")
                    
                    # Performance interpretation
                    if accuracy > 0.60:
                        print(f"  ✅ Accuracy {accuracy:.1%} (Good for stock prediction)")
                    elif accuracy > 0.55:
                        print(f"  ⚠️  Accuracy {accuracy:.1%} (Moderate - better than random)")
                    elif accuracy > 0.50:
                        print(f"  ⚠️  Accuracy {accuracy:.1%} (Slightly better than random)")
                    else:
                        print(f"  ❌ Accuracy {accuracy:.1%} (Worse than random)")
                
                results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'confusion_matrix': cm.tolist() if cm is not None else None
                }
                
                # Use accuracy as primary score for classification
                score = accuracy
            
            # Feature importance
            if 'feature_importance' in model_info:
                importance = model_info['feature_importance']
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                if verbose:
                    print("  Top 10 Most Important Features:")
                    for i, (feature, score) in enumerate(top_features, 1):
                        print(f"    {i}. {feature}: {score:.4f}")
            
            # Track best model
            if score > best_score:
                best_score = score
                best_model = model_name
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🏆 BEST MODEL: {best_model} (Score: {best_score:.4f})")
            print(f"{'='*60}")
            print("\n💡 INTERPRETATION GUIDE:")
            print("  • For Regression: R² > 0.1 is good, Directional Accuracy > 55% is useful")
            print("  • For Classification: Accuracy > 55% is better than random, >60% is good")
            print("  • Always backtest with trading strategy to see real performance")
            print("  • Compare model performance to buy-and-hold baseline")
        
        results['best_model'] = best_model
        results['best_score'] = best_score
        
        return results
    
    def predict(self, model_name='Random_Forest', days_ahead=1, return_confidence=False):
        """
        Make predictions using trained model with validation and monitoring
        
        Args:
            model_name: Name of model to use
            days_ahead: Number of days to predict ahead
            return_confidence: Whether to return confidence score
        
        Returns:
            Prediction value (and confidence if requested)
        
        Raises:
            ValueError: If prediction fails
        """
        try:
            # Validate model exists
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Get latest features
            feature_cols = self.feature_engineer.feature_columns
            if len(feature_cols) == 0:
                raise ValueError("No features available. Run prepare_data() first.")
            
            latest_features = self.df[feature_cols].iloc[-1:].values
            
            # Validate features
            is_valid, error_msg = DataValidator.validate_features(latest_features)
            if not is_valid:
                raise ValueError(f"Invalid features for prediction: {error_msg}")
            
            # Scale features
            if 'standard' in self.scalers:
                latest_features_scaled = self.scalers['standard'].transform(latest_features)
            else:
                latest_features_scaled = latest_features
            
            # Make prediction
            prediction = model.predict(latest_features_scaled)[0]
            
            # Get confidence if available
            confidence = None
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(latest_features_scaled)[0]
                confidence = float(np.max(proba))
            elif hasattr(model, 'feature_importances_'):
                # For tree models, use feature importance as proxy
                confidence = 0.7  # Default confidence
            
            # Log prediction for monitoring
            monitor.log_prediction(self.ticker_name, model_name, prediction, confidence)
            
            logger.info(f"Prediction for {self.ticker_name}: {prediction:.6f} (confidence: {confidence:.2f})")
            
            if return_confidence:
                return prediction, confidence
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction for {self.ticker_name}: {e}", exc_info=True)
            monitor.log_error('PredictionError', str(e), {'ticker': self.ticker_name, 'model': model_name})
            raise
    
    def save_model(self, model_name='Random_Forest', save_dir='models', version=None):
        """
        Save trained model with versioning, validation, and error handling
        
        Args:
            model_name: Name of model to save
            save_dir: Directory to save model files
            version: Optional version string (auto-generated if not provided)
        
        Returns:
            str: Version string if successful, None otherwise
        """
        try:
            # Validate model exists
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
            
            # Validate save directory
            if not isinstance(save_dir, str) or len(save_dir) == 0:
                raise ValueError(f"Invalid save_dir: {save_dir}")
            
            os.makedirs(save_dir, exist_ok=True)
            
            model_info = self.models[model_name]
            model = model_info['model']
            
            # Generate version if not provided
            if version is None and config.get('model_versioning.enabled', True):
                metadata_for_version = {
                    'model_name': model_name,
                    'target_type': self.target_type,
                    'accuracy': model_info.get('accuracy'),
                    'rmse': model_info.get('rmse'),
                    'r2': model_info.get('r2'),
                    **self.training_metadata
                }
                version = ModelVersionManager.generate_version(model_name, self.ticker_name, metadata_for_version)
                self.version = version
                ModelVersionManager.save_version_info(version, metadata_for_version, save_dir)
            
            # Save model
            model_filename = f"{self.ticker_name}_{model_name}_{self.target_type}"
            if version:
                model_filename = f"{model_filename}_{version}"
            model_path = os.path.join(save_dir, f"{model_filename}.joblib")
            
            joblib.dump(model, model_path)
            logger.info(f"Saved model to {model_path}")
            
            # Save scaler
            if 'standard' in self.scalers:
                scaler_filename = f"{self.ticker_name}_{model_name}_scaler"
                if version:
                    scaler_filename = f"{scaler_filename}_{version}"
                scaler_path = os.path.join(save_dir, f"{scaler_filename}.joblib")
                joblib.dump(self.scalers['standard'], scaler_path)
                logger.info(f"Saved scaler to {scaler_path}")
            
            # Save comprehensive metadata
            metadata = {
                'ticker': self.ticker_name,
                'model_name': model_name,
                'target_type': self.target_type,
                'version': version,
                'feature_columns': self.feature_engineer.feature_columns,
                'accuracy': model_info.get('accuracy', None),
                'rmse': model_info.get('rmse', None),
                'r2': model_info.get('r2', None),
                'precision': model_info.get('precision', None),
                'recall': model_info.get('recall', None),
                'f1_score': model_info.get('f1_score', None),
                'training_metadata': self.training_metadata,
                'saved_at': datetime.now().isoformat(),
                'python_version': sys.version,
                'dependencies': {
                    'pandas': pd.__version__,
                    'numpy': np.__version__,
                    'sklearn': None,  # Would need to import sklearn.__version__
                    'xgboost': None
                }
            }
            
            metadata_filename = f"{self.ticker_name}_{model_name}_metadata"
            if version:
                metadata_filename = f"{metadata_filename}_{version}"
            metadata_path = os.path.join(save_dir, f"{metadata_filename}.json")
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved successfully: {version or 'no version'}")
            return version
            
        except Exception as e:
            logger.error(f"Error saving model for {self.ticker_name}: {e}", exc_info=True)
            monitor.log_error('ModelSaveError', str(e), {'ticker': self.ticker_name, 'model': model_name})
            return None
    
    @staticmethod
    def load_model(ticker, model_name, target_type, model_dir='models', version=None):
        """
        Load saved model with validation and error handling
        
        Args:
            ticker: Stock ticker symbol
            model_name: Name of model to load
            target_type: Target type ('direction', 'returns', 'volatility')
            model_dir: Directory where models are saved
            version: Optional version string (loads latest if not provided)
        
        Returns:
            Dictionary with model, scaler, and metadata
        
        Raises:
            FileNotFoundError: If model file not found
            ValueError: If model loading fails
        """
        try:
            # Validate inputs
            if not DataValidator.validate_ticker(ticker):
                raise ValueError(f"Invalid ticker: {ticker}")
            
            valid_targets = ['returns', 'direction', 'volatility']
            if target_type not in valid_targets:
                raise ValueError(f"Invalid target_type: {target_type}")
            
            # Construct file paths
            if version:
                model_filename = f"{ticker}_{model_name}_{target_type}_{version}"
            else:
                model_filename = f"{ticker}_{model_name}_{target_type}"
            
            model_path = os.path.join(model_dir, f"{model_filename}.joblib")
            scaler_filename = f"{ticker}_{model_name}_scaler"
            if version:
                scaler_filename = f"{scaler_filename}_{version}"
            scaler_path = os.path.join(model_dir, f"{scaler_filename}.joblib")
            metadata_filename = f"{ticker}_{model_name}_metadata"
            if version:
                metadata_filename = f"{metadata_filename}_{version}"
            metadata_path = os.path.join(model_dir, f"{metadata_filename}.json")
            
            # Check if model exists
            if not os.path.exists(model_path):
                # Try to find latest version
                if not version:
                    versions_file = os.path.join(model_dir, 'versions.json')
                    if os.path.exists(versions_file):
                        with open(versions_file, 'r') as f:
                            versions = json.load(f)
                            # Find matching version
                            for v in reversed(versions):
                                if (v['metadata'].get('ticker') == ticker and 
                                    v['metadata'].get('model_name') == model_name):
                                    version = v['version']
                                    return QuantitativePredictor.load_model(ticker, model_name, target_type, model_dir, version)
                
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model
            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            
            # Load scaler
            scaler = None
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.warning(f"Scaler file not found: {scaler_path}")
            
            # Load metadata
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                logger.info(f"Loaded metadata: version={metadata.get('version')}")
            else:
                logger.warning(f"Metadata file not found: {metadata_path}")
            
            # Validate loaded model
            if model is None:
                raise ValueError("Loaded model is None")
            
            logger.info(f"Successfully loaded model for {ticker}")
            return {
                'model': model,
                'scaler': scaler,
                'metadata': metadata,
                'version': version or metadata.get('version')
            }
            
        except Exception as e:
            logger.error(f"Error loading model for {ticker}: {e}", exc_info=True)
            monitor.log_error('ModelLoadError', str(e), {'ticker': ticker, 'model': model_name})
            raise


def create_quantitative_pipeline(tickers, target_type='returns', prediction_horizon=1):
    """
    Create a complete quantitative prediction pipeline for multiple stocks
    
    Args:
        tickers: List of stock tickers
        target_type: 'returns', 'direction', or 'volatility'
        prediction_horizon: Days ahead to predict
    
    Returns:
        Dictionary of predictors for each ticker
    """
    print(f"Creating quantitative pipeline for {len(tickers)} stocks...")
    
    # Import your existing modules from parent src directory
    import sys
    import os
    parent_src = os.path.join(os.path.dirname(__file__), '..', '..', 'src')
    if parent_src not in sys.path:
        sys.path.insert(0, parent_src)
    from data import fetch_multiple_stocks
    from data_cleaning import clean_multiple_stocks
    
    # Fetch and clean data
    print("Fetching stock data...")
    stock_data = fetch_multiple_stocks(tickers, period='3y', interval='1d', use_cache=True)
    
    print("Cleaning stock data...")
    cleaned_data = clean_multiple_stocks(stock_data, use_cache=True, save_permanent=False)
    
    # Create predictors
    predictors = {}
    for ticker in tickers:
        if ticker in cleaned_data:
            print(f"\nSetting up predictor for {ticker}...")
            predictor = QuantitativePredictor(cleaned_data[ticker], ticker, target_type)
            predictor.prepare_data(prediction_horizon)
            predictor.train_models()
            predictor.evaluate_models()
            predictors[ticker] = predictor
    
    return predictors


# Example usage
if __name__ == "__main__":
    # Test with a single stock
    tickers = ['AAPL']
    predictors = create_quantitative_pipeline(tickers, target_type='direction')
    
    # Evaluate models and make predictions
    for ticker, predictor in predictors.items():
        # Evaluate model performance
        results = predictor.evaluate_models()
        
        # Make predictions
        prediction = predictor.predict(model_name='Random_Forest_Classifier')
        print(f"{ticker} direction prediction: {'UP' if prediction > 0.5 else 'DOWN'}")
