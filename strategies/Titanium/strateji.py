import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress convergence warnings since partial online training doesn't always need to fully converge
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

class KalmanFilter1D:
    """A lightweight 1D Kalman Filter to estimate true price by stripping artificial noise."""
    def __init__(self, process_variance=1e-4, estimated_measurement_variance=1e-2):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        
    def filter_series(self, prices):
        """Processes a price series and returns the smoothed, denoised underlying state."""
        filtered_prices = []
        posteri_estimate = prices.iloc[0]
        posteri_error_estimate = 1.0
        
        for price in prices:
            # Predict
            priori_estimate = posteri_estimate
            priori_error_estimate = posteri_error_estimate + self.process_variance

            # Correct
            blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
            posteri_estimate = priori_estimate + blending_factor * (price - priori_estimate)
            posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
            
            filtered_prices.append(posteri_estimate)
            
        return pd.Series(filtered_prices, index=prices.index)


class NeuralRiskStrategy(BaseStrategy):
    """
    A quantitative strategy that uses a Kalman Filter to denoise data, 
    a Neural Network (MLP) to predict direction on the clean data, 
    and scales leverage dynamically based on conviction and historical volatility.
    """
    def __init__(self, window=100, vol_window=10):
        super().__init__()
        self.window = window
        self.vol_window = vol_window
        
        # Initialize Kalman Filter
        self.kf = KalmanFilter1D(process_variance=1e-4, estimated_measurement_variance=1e-2)
        
        # Track active positions
        self.current_positions = {}
        
        # ML Components
        self.models = {}
        self.scalers = {}
        self.last_train_len = {}
        
        # We need a longer MIN_LOOKBACK because of the SMAs + shift
        self.MIN_LOOKBACK = 100 

    def _build_features(self, df: pd.DataFrame, training=False):
        """Constructs features for the Neural Network using DENOISED data."""
        features = pd.DataFrame(index=df.index)
        
        # ALL ML FEATURES MUST USE THE CLEAN PRICE
        clean_close = df["Clean_Close"]
        
        # Returns and Volatility (Denoised)
        returns = clean_close.pct_change()
        features["returns"] = returns
        features["volatility"] = returns.rolling(self.vol_window).std()
        
        # Moving Averages
        features["sma_7"] = clean_close.rolling(7).mean() / clean_close - 1
        features["sma_30"] = clean_close.rolling(30).mean() / clean_close - 1
        features["sma_90"] = clean_close.rolling(90).mean() / clean_close - 1
        
        # Momentum
        features["momentum_14"] = clean_close / clean_close.shift(14) - 1
        
        if training:
            # Target is whether the *next* clean candle's close is higher than this one
            features["target"] = (clean_close.shift(-1) > clean_close).astype(int)
            
        return features.dropna()

    def _train_coin(self, coin: str, df: pd.DataFrame):
        """Trains or updates the Neural Network for a specific coin online."""
        features_df = self._build_features(df, training=True)
        
        # Ensure we have enough data points after dropping NaNs
        if len(features_df) < 50:
            return
            
        X = features_df.drop(columns=["target"])
        y = features_df["target"]
        
        # Fit a new scaler on the current history
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[coin] = scaler
        
        if coin not in self.models:
            # warm_start=True allows the model to adapt perfectly on the go
            model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, warm_start=True, random_state=42)
        else:
            model = self.models[coin]
            
        model.fit(X_scaled, y)
        self.models[coin] = model
        self.last_train_len[coin] = len(df)

    def _calculate_neural_risk(self, coin, df):
        """Calculates prediction using Neural Network and market risk."""
        # 1. Volatility Risk Factor Logic (Using Clean Data for structural risk)
        clean_returns = df["Clean_Close"].pct_change().dropna()
        
        # Calculate actual raw volatility for setting safe Stop-Losses later
        actual_vol = df["Close"].pct_change().rolling(self.vol_window).std().iloc[-1]
        
        if len(clean_returns) < self.vol_window:
            return 0, 0.0, 1.0, actual_vol
            
        rolling_clean_vol = clean_returns.rolling(self.vol_window).std()
        current_clean_vol = rolling_clean_vol.iloc[-1]
        
        lookback = min(len(rolling_clean_vol), 90)
        max_vol_90d = rolling_clean_vol.iloc[-lookback:].max()
        risk_factor = (current_clean_vol / max_vol_90d) if max_vol_90d > 0 else 1.0
        
        # 2. Neural Network Prediction
        if coin not in self.models:
            return 0, 0.0, 1.0, actual_vol
            
        features_df = self._build_features(df, training=False)
        if len(features_df) == 0:
            return 0, 0.0, 1.0, actual_vol
            
        X_latest = features_df.iloc[-1:]
        X_scaled = self.scalers[coin].transform(X_latest)
        
        probs = self.models[coin].predict_proba(X_scaled)[0]
        prob_down, prob_up = probs[0], probs[1]
        
        # Trend filter (using clean data)
        uptrend = features_df["sma_30"].iloc[-1] < 0
        
        # Volume Filter: Calculate 20-day volume ratio
        avg_vol_20 = df["Volume"].rolling(20).mean().iloc[-1]
        current_vol_raw = df["Volume"].iloc[-1]
        volume_ratio = current_vol_raw / avg_vol_20 if avg_vol_20 > 0 else 0
        volume_confirmed = volume_ratio > 1.0
        
        if prob_up > prob_down and prob_up > 0.65 and uptrend and volume_confirmed:
            predicted_signal = 1
            confidence = prob_up
        elif prob_down > prob_up and prob_down > 0.65 and not uptrend and volume_confirmed:
            predicted_signal = -1
            confidence = prob_down
        else:
            predicted_signal = 0
            confidence = max(prob_up, prob_down)
            
        # Notice we return the actual_vol to ensure real-world stop losses
        return predicted_signal, confidence, risk_factor, actual_vol

    def _calculate_dynamic_leverage(self, confidence, risk_factor):
        score = confidence * (1.0 - risk_factor)
        if score > 0.70: return 10
        if score > 0.55: return 5
        if score > 0.40: return 3
        if score > 0.25: return 2
        return 1

    def predict(self, data: dict) -> list[dict]:
        decisions = []
        expected_coins = list(data.keys())
        
        for coin in expected_coins:
            if coin not in self.current_positions:
                self.current_positions[coin] = 0
                
            if coin not in data or len(data[coin]) < self.MIN_LOOKBACK:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue

            # Create a copy to avoid SettingWithCopyWarning and inject the Kalman Denoised series
            df = data[coin].copy()
            df["Clean_Close"] = self.kf.filter_series(df["Close"])
            
            # ONLINE TRAINING LOGIC
            last_len = self.last_train_len.get(coin, 0)
            if len(df) - last_len >= 7 or last_len == 0:
                self._train_coin(coin, df)
            
            target_signal, confidence, risk_factor, actual_vol = self._calculate_neural_risk(coin, df)
            leverage = self._calculate_dynamic_leverage(confidence, risk_factor)
            
            # Actual execution price MUST be the raw noisy price
            current_price = df["Close"].iloc[-1] 
            
            base_allocation = round(0.99 / len(expected_coins), 4) if target_signal != 0 else 0.0
            current_signal = self.current_positions[coin]
            
            # --- RULE ENFORCEMENT: Close Before Open ---
            if current_signal != 0 and current_signal != target_signal:
                final_signal = 0
                final_allocation = 0.0
                final_leverage = 1
            else:
                final_signal = target_signal
                final_allocation = base_allocation
                final_leverage = leverage if final_signal != 0 else 1

            self.current_positions[coin] = final_signal
            
            decision = {
                "coin": coin,
                "signal": final_signal,
                "allocation": final_allocation,
                "leverage": final_leverage
            }
            
            if final_signal != 0:
                # Use the actual_vol to scale the stop-loss to survive the artificial noise
                sl_pct = max(0.03, actual_vol * 3) 
                tp_pct = sl_pct * 2.5
                
                if final_signal == 1:
                    decision["take_profit"] = current_price * (1 + tp_pct)
                    decision["stop_loss"] = current_price * (1 - sl_pct)
                elif final_signal == -1:
                    short_tp_pct = min(tp_pct, 0.99)
                    decision["take_profit"] = current_price * (1 - short_tp_pct)
                    decision["stop_loss"] = current_price * (1 + sl_pct)
            
            decisions.append(decision)

        return decisions

# ==========================================
# Execution Block
# ==========================================
if __name__ == "__main__":
    strategy = NeuralRiskStrategy(window=100, vol_window=10)
    print("Initiating Denoised Deep Learning Backtest...")
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()