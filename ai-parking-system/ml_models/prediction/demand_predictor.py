"""
Predictive analytics for parking demand forecasting
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ParkingDemandPredictor:
    """
    ML-powered parking demand prediction system
    """
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
        # Model parameters
        self.model_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for demand prediction
        """
        try:
            df = data.copy()
            
            # Time-based features
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['month'] = pd.to_datetime(df['timestamp']).dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_holiday'] = self._is_holiday(df['timestamp'])
            
            # Weather features (if available)
            if 'temperature' in df.columns:
                df['temp_category'] = pd.cut(df['temperature'], 
                                           bins=[-np.inf, 0, 15, 25, np.inf], 
                                           labels=['cold', 'cool', 'warm', 'hot'])
            
            # Location features
            if 'zone' in df.columns:
                df['zone_encoded'] = self._encode_categorical(df['zone'], 'zone')
            
            # Historical demand features
            df['demand_1h_ago'] = df['demand'].shift(1)
            df['demand_24h_ago'] = df['demand'].shift(24)
            df['demand_7d_ago'] = df['demand'].shift(24*7)
            
            # Rolling statistics
            df['demand_avg_1h'] = df['demand'].rolling(window=4).mean()
            df['demand_avg_24h'] = df['demand'].rolling(window=24).mean()
            df['demand_std_24h'] = df['demand'].rolling(window=24).std()
            
            # Fill NaN values
            df = df.fillna(method='bfill').fillna(0)
            
            # Select numerical features
            numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
            self.feature_names = [f for f in numerical_features if f != 'demand']
            
            return df[self.feature_names]
            
        except Exception as e:
            logger.error(f"Error in feature preparation: {e}")
            return pd.DataFrame()
    
    def _encode_categorical(self, series: pd.Series, column_name: str) -> pd.Series:
        """Encode categorical variables"""
        if column_name not in self.label_encoders:
            self.label_encoders[column_name] = LabelEncoder()
            return self.label_encoders[column_name].fit_transform(series)
        else:
            return self.label_encoders[column_name].transform(series)
    
    def _is_holiday(self, timestamps: pd.Series) -> pd.Series:
        """Check if dates are holidays (simplified implementation)"""
        # This is a simplified holiday detection
        # In production, you would use a proper holiday calendar
        holidays = ['2024-01-01', '2024-07-04', '2024-12-25']  # Example holidays
        return timestamps.dt.date.astype(str).isin(holidays).astype(int)
    
    def train_model(self, data: pd.DataFrame, target_column: str = 'demand') -> Dict[str, Any]:
        """
        Train the demand prediction model
        """
        try:
            logger.info("Starting model training...")
            
            # Prepare features
            X = self.prepare_features(data)
            y = data[target_column]
            
            if X.empty or y.empty:
                raise ValueError("No valid features or target data")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = RandomForestRegressor(**self.model_params)
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, 
                cv=5, scoring='r2'
            )
            metrics['cv_r2_mean'] = cv_scores.mean()
            metrics['cv_r2_std'] = cv_scores.std()
            
            # Feature importance
            feature_importance = dict(zip(self.feature_names, 
                                        self.model.feature_importances_))
            metrics['feature_importance'] = feature_importance
            
            self.is_trained = True
            logger.info(f"Model training completed. R² Score: {metrics['r2']:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {}
    
    def predict_demand(self, features: pd.DataFrame, hours_ahead: int = 24) -> List[float]:
        """
        Predict parking demand for future time periods
        """
        try:
            if not self.is_trained or self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Prepare features
            X = self.prepare_features(features)
            
            if X.empty:
                raise ValueError("No valid features for prediction")
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            
            # Ensure predictions are non-negative
            predictions = np.maximum(predictions, 0)
            
            return predictions.tolist()
            
        except Exception as e:
            logger.error(f"Error in demand prediction: {e}")
            return []
    
    def detect_anomalies(self, data: pd.DataFrame, contamination: float = 0.1) -> List[bool]:
        """
        Detect anomalies in parking demand using Isolation Forest
        """
        try:
            # Prepare features
            X = self.prepare_features(data)
            
            if X.empty:
                return []
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Initialize and fit Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42
            )
            
            # Fit and predict
            iso_forest.fit(X_scaled)
            anomaly_labels = iso_forest.predict(X_scaled)
            
            # Convert to boolean (True for anomalies)
            return (anomaly_labels == -1).tolist()
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return []
    
    def get_prediction_confidence(self, features: pd.DataFrame) -> List[float]:
        """
        Get prediction confidence intervals using model ensemble
        """
        try:
            if not self.is_trained or self.model is None:
                return []
            
            # Prepare features
            X = self.prepare_features(features)
            
            if X.empty:
                return []
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all trees
            predictions = []
            for estimator in self.model.estimators_:
                pred = estimator.predict(X_scaled)
                predictions.append(pred)
            
            # Calculate confidence intervals
            predictions_array = np.array(predictions)
            confidence_intervals = []
            
            for i in range(predictions_array.shape[1]):
                preds = predictions_array[:, i]
                confidence = 1.0 - (np.std(preds) / (np.mean(preds) + 1e-8))
                confidence_intervals.append(max(0.0, min(1.0, confidence)))
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error in confidence calculation: {e}")
            return []
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        try:
            if self.is_trained:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'label_encoders': self.label_encoders,
                    'feature_names': self.feature_names,
                    'model_params': self.model_params
                }
                joblib.dump(model_data, filepath)
                logger.info(f"Model saved to {filepath}")
            else:
                logger.warning("No trained model to save")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data["scaler"]
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.model_params = model_data['model_params']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def generate_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate insights from the data and model
        """
        try:
            insights = {
                'peak_hours': [],
                'peak_days': [],
                'seasonal_patterns': [],
                'anomalies': [],
                'recommendations': []
            }
            
            if 'timestamp' in data.columns and 'demand' in data.columns:
                # Peak hours analysis
                hourly_avg = data.groupby(data['timestamp'].dt.hour)['demand'].mean()
                peak_hours = hourly_avg.nlargest(3).index.tolist()
                insights['peak_hours'] = peak_hours
                
                # Peak days analysis
                daily_avg = data.groupby(data['timestamp'].dt.dayofweek)['demand'].mean()
                peak_days = daily_avg.nlargest(3).index.tolist()
                insights['peak_days'] = peak_days
                
                # Anomaly detection
                anomalies = self.detect_anomalies(data)
                insights['anomalies'] = anomalies
                
                # Recommendations
                if peak_hours:
                    insights['recommendations'].append(
                        f"Peak parking demand occurs at {peak_hours[0]}:00 hours"
                    )
                
                if peak_days:
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    insights['recommendations'].append(
                        f"Highest demand on {day_names[peak_days[0]]}"
                    )
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {}
