"""
LSTM Dengue Predictor for Time-Series Outbreak Prediction
Integrates weather data + historical cases to predict future risk
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import joblib
import json
from typing import Dict, List, Any

class DengueLSTMPredictor:
    """Advanced LSTM model for dengue outbreak prediction"""
    
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.window_size = 12  # 12 weeks lookback
        self.feature_columns = ['temperature', 'rainfall', 'humidity', 'dengue_cases']
        self.model_path = 'models/dengue_lstm_model.h5'
        self.scaler_path = 'models/dengue_lstm_scaler.pkl'
        
    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare time-series data for LSTM training"""
        # Ensure we have all required columns
        required_cols = ['date', 'temperature', 'rainfall', 'humidity', 'dengue_cases']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Sort by date
        df = df.sort_values('date').copy()
        
        # Create features array
        features = df[self.feature_columns].values
        
        # Normalize features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_features) - self.window_size):
            # Input: 12 weeks of [temperature, rainfall, humidity, previous_cases]
            X.append(scaled_features[i:i+self.window_size])
            # Target: next week's dengue cases (index 3 = dengue_cases)
            y.append(scaled_features[i+self.window_size, 3])  
            
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Build advanced LSTM model for dengue prediction"""
        self.model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.window_size, len(self.feature_columns))),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.1),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')  # Predicting case count
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        print("✅ LSTM Model built successfully")
        print(f"   - Input shape: ({self.window_size}, {len(self.feature_columns)})")
        print(f"   - Output: Dengue cases prediction")
        
    def train(self, training_data: pd.DataFrame, epochs=100, validation_split=0.2):
        """Train the LSTM model"""
        print("🏋️ Training LSTM Dengue Predictor...")
        
        X, y = self.prepare_training_data(training_data)
        
        print(f"   - Training samples: {X.shape[0]}")
        print(f"   - Features per sample: {X.shape[1]} weeks × {X.shape[2]} features")
        
        # Train model
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=32,
            validation_split=validation_split,
            verbose=1,
            shuffle=False  # Don't shuffle time series data
        )
        
        # Save model and scaler
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        print("✅ LSTM Model trained and saved")
        return history
        
    def load_model(self):
        """Load trained model and scaler"""
        try:
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("✅ LSTM Model loaded successfully")
            return True
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            return False
    
    def predict_next_weeks(self, recent_data: Dict[str, List], weeks_ahead=4):
        """Predict dengue cases for next N weeks"""
        if self.model is None:
            if not self.load_model():
                return self._fallback_prediction(weeks_ahead)
        
        try:
            # Prepare input data
            input_features = []
            for i in range(self.window_size):
                week_data = [
                    recent_data['temperature'][i],
                    recent_data['rainfall'][i],
                    recent_data['humidity'][i],
                    recent_data['dengue_cases'][i]
                ]
                input_features.append(week_data)
            
            # Scale input
            input_array = np.array(input_features)
            input_scaled = self.scaler.transform(input_array)
            input_scaled = input_scaled.reshape(1, self.window_size, len(self.feature_columns))
            
            predictions = []
            current_input = input_scaled.copy()
            
            # Predict multiple weeks ahead
            for week in range(weeks_ahead):
                # Predict next week
                prediction = self.model.predict(current_input, verbose=0)
                predicted_cases = self.scaler.inverse_transform(
                    np.hstack([np.zeros((1, 3)), prediction])
                )[0, 3]
                
                predictions.append({
                    'week': week + 1,
                    'predicted_cases': max(0, int(predicted_cases)),
                    'risk_level': self._calculate_risk_level(predicted_cases),
                    'confidence': self._calculate_confidence(prediction[0][0])
                })
                
                # Update input for next prediction
                # Add predicted value to input sequence
                next_features = np.array([[
                    recent_data['temperature'][-1],  # Use latest weather
                    recent_data['rainfall'][-1],
                    recent_data['humidity'][-1],
                    prediction[0][0]  # Use predicted cases
                ]])
                next_scaled = self.scaler.transform(next_features)
                
                # Shift input window
                current_input = np.concatenate([
                    current_input[:, 1:, :],
                    next_scaled.reshape(1, 1, len(self.feature_columns))
                ], axis=1)
                
            return {
                'predictions': predictions,
                'model_info': {
                    'type': 'LSTM',
                    'window_size': self.window_size,
                    'features_used': self.feature_columns
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ LSTM prediction error: {e}")
            return self._fallback_prediction(weeks_ahead)
    
    def _calculate_risk_level(self, predicted_cases: float) -> str:
        """Convert case prediction to risk level"""
        if predicted_cases < 10:
            return 'LOW'
        elif predicted_cases < 30:
            return 'MODERATE'
        elif predicted_cases < 70:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _calculate_confidence(self, prediction_value: float) -> float:
        """Calculate prediction confidence based on model certainty"""
        # Simple confidence based on prediction magnitude
        # In practice, this would use prediction intervals
        base_confidence = 0.75
        
        # Higher confidence for moderate predictions
        if 10 <= abs(prediction_value) <= 100:
            base_confidence += 0.15
        
        return min(0.95, base_confidence)
    
    def _fallback_prediction(self, weeks_ahead: int) -> Dict:
        """Fallback prediction when model unavailable"""
        print("⚠️ Using fallback prediction method")
        
        predictions = []
        for week in range(weeks_ahead):
            # Simple trend-based prediction
            base_cases = 25 + (week * 5)  # Increasing trend
            
            predictions.append({
                'week': week + 1,
                'predicted_cases': base_cases,
                'risk_level': self._calculate_risk_level(base_cases),
                'confidence': 0.65  # Lower confidence for fallback
            })
        
        return {
            'predictions': predictions,
            'model_info': {
                'type': 'Fallback',
                'note': 'LSTM model not available'
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_sample_training_data(self) -> pd.DataFrame:
        """Generate sample training data for demonstration"""
        print("📊 Generating sample training data...")
        
        # Create 2 years of weekly data
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='W')
        np.random.seed(42)  # Reproducible results
        
        data = []
        base_cases = 20
        
        for i, date in enumerate(dates):
            # Simulate seasonal patterns
            week_of_year = date.isocalendar()[1]
            
            # Temperature: higher in mid-year (Malaysian dry season)
            temp = 28 + 4 * np.sin(2 * np.pi * week_of_year / 52) + np.random.normal(0, 1)
            
            # Rainfall: higher during monsoon seasons (Nov-Mar, May-Sep)
            rainfall_pattern = (
                np.sin(2 * np.pi * week_of_year / 52) + 
                0.5 * np.sin(4 * np.pi * week_of_year / 52)
            )
            rainfall = max(0, 150 + 100 * rainfall_pattern + np.random.normal(0, 30))
            
            # Humidity: correlated with rainfall
            humidity = 75 + 0.1 * rainfall + np.random.normal(0, 3)
            humidity = max(60, min(95, humidity))
            
            # Dengue cases: influenced by weather with lag
            weather_effect = (
                0.5 * (temp - 28) +  # Temperature effect
                0.002 * rainfall +    # Rainfall effect (delayed)
                0.1 * (humidity - 75) # Humidity effect
            )
            
            # Add seasonal trend and random noise
            seasonal_effect = 5 * np.sin(2 * np.pi * (week_of_year + 8) / 52)  # Peak around April
            cases = base_cases + weather_effect + seasonal_effect + np.random.normal(0, 5)
            cases = max(0, int(cases))
            
            data.append({
                'date': date,
                'temperature': round(temp, 1),
                'rainfall': round(rainfall, 1),
                'humidity': round(humidity, 1),
                'dengue_cases': cases
            })
            
            # Slight trend over time
            base_cases += np.random.normal(0, 0.1)
        
        df = pd.DataFrame(data)
        print(f"✅ Generated {len(df)} weeks of training data")
        print(f"   - Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   - Avg dengue cases: {df['dengue_cases'].mean():.1f}")
        
        return df

# Test the LSTM predictor
if __name__ == "__main__":
    predictor = DengueLSTMPredictor()
    
    # Generate sample data
    training_data = predictor.generate_sample_training_data()
    
    # Build and train model
    predictor.build_model()
    history = predictor.train(training_data, epochs=50)
    
    # Test prediction
    recent_data = {
        'temperature': [30.2, 31.1, 29.8, 30.5, 31.0, 29.7, 30.3, 29.9, 30.8, 31.2, 30.1, 29.6],
        'rainfall': [120, 80, 200, 150, 90, 250, 180, 160, 110, 140, 190, 170],
        'humidity': [78, 75, 82, 80, 76, 85, 81, 79, 77, 78, 83, 82],
        'dengue_cases': [25, 30, 35, 28, 32, 45, 38, 33, 29, 35, 42, 37]
    }
    
    predictions = predictor.predict_next_weeks(recent_data, weeks_ahead=4)
    
    print("\n🔮 LSTM Predictions for next 4 weeks:")
    for pred in predictions['predictions']:
        print(f"   Week {pred['week']}: {pred['predicted_cases']} cases ({pred['risk_level']} risk, {pred['confidence']:.2f} confidence)")