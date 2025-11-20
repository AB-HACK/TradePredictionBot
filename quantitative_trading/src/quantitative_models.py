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
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost as xgb
from .cache_manager import get_cache_manager
warnings.filterwarnings('ignore')

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
    """
    
    def __init__(self, df, ticker_name, target_type='returns'):
        """
        Initialize predictor
        
        Args:
            df: DataFrame with features
            ticker_name: Stock ticker
            target_type: 'returns', 'direction', or 'volatility'
        """
        self.df = df.copy()
        self.ticker_name = ticker_name
        self.target_type = target_type
        self.feature_engineer = QuantitativeFeatureEngineer(df, ticker_name)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        
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
        Train multiple machine learning models
        
        Args:
            test_size: Proportion of data for testing
            use_cache: Whether to use caching
        """
        print(f"Training models for {self.ticker_name}...")
        
        # Prepare features and target
        feature_cols = self.feature_engineer.feature_columns
        X = self.df[feature_cols].dropna()
        y = self.df['Target'].loc[X.index].dropna()
        
        # Align X and y
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]
        
        if len(X) == 0:
            print(f"[ERROR] No valid data for training {self.ticker_name}")
            return None
        
        # Time series split
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['standard'] = scaler
        
        # Train models based on target type
        if self.target_type in ['returns', 'volatility']:
            self._train_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)
        else:
            self._train_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        print(f"[SUCCESS] Models trained for {self.ticker_name}")
        return self.models
    
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
    
    def predict(self, model_name='Random_Forest', days_ahead=1):
        """
        Make predictions using trained model
        
        Args:
            model_name: Name of model to use
            days_ahead: Number of days to predict ahead
        """
        if model_name not in self.models:
            print(f"[ERROR] Model {model_name} not found")
            return None
        
        model_info = self.models[model_name]
        model = model_info['model']
        
        # Get latest features
        feature_cols = self.feature_engineer.feature_columns
        latest_features = self.df[feature_cols].iloc[-1:].values
        
        # Scale features
        if 'standard' in self.scalers:
            latest_features_scaled = self.scalers['standard'].transform(latest_features)
        else:
            latest_features_scaled = latest_features
        
        # Make prediction
        prediction = model.predict(latest_features_scaled)[0]
        
        print(f"[PREDICTION] {self.ticker_name} {self.target_type} prediction: {prediction:.6f}")
        return prediction


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
    
    # Import your existing modules
    from .data import fetch_multiple_stocks
    from .data_cleaning import clean_multiple_stocks
    
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
    
    # Make predictions
    for ticker, predictor in predictors.items():
        prediction = predictor.predict(model_name='Random_Forest_Classifier')
        print(f"{ticker} direction prediction: {'UP' if prediction > 0.5 else 'DOWN'}")
