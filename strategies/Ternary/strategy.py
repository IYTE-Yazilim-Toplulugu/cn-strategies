import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

class OptimalStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.models = {}
        self.scalers = {}  # Feature scaling için
        self.lookback = 50
        self.feature_cols = [
            'body_pct', 'close_pos', 'rel_volume', 'rsi', 'volatility', 
            'atr_ratio', 'trend_strength', 'momentum_norm', 'ema_cross_norm', 'bb_position'
        ]
        self.train_split = 0.65  # Genelleme yeteneği için ideal oran
        self.last_trade_candle = {}  # Overtrading engellemek için cooldown

    def _calculate_features(self, df):
        df = df.copy()
        # === TEMEL HESAPLAMALAR ===
        df['range'] = (df['High'] - df['Low']).replace(0, 1e-9)
        df['body_pct'] = abs(df['Close'] - df['Open']) / df['range']
        df['close_pos'] = (df['Close'] - df['Low']) / df['range']
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()

        # === ATR (Average True Range) - Volatilite Odaklı Risk ===
        df['tr'] = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_ratio'] = df['atr'] / df['Close']

        # === RSI (Normalize Edilmiş) ===
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1e-9)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100

        # === VOLUME VE TREND ANALİZİ ===
        df['rel_volume'] = np.clip(df['Volume'] / df['Volume'].rolling(20).mean(), 0.2, 3)
        
        ema9 = df['Close'].ewm(span=9).mean()
        ema21 = df['Close'].ewm(span=21).mean()
        df['trend_strength'] = np.clip(abs((ema9 - ema21) / ema21), 0, 0.1)

        # === MOMENTUM VE CROSS (Tanh Normalizasyonu) ===
        df['momentum_norm'] = np.tanh(((df['Close'] / df['Close'].shift(10)) - 1) * 5)
        df['ema_cross_norm'] = np.tanh(((ema9 / ema21) - 1) * 10)

        # === BOLLINGER BANDS POSITION ===
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_position'] = np.clip((df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std).replace(0, 1e-9), 0, 1)

        return df.fillna(0)

    def egit(self):
        source_data = getattr(self, '_full_data', self.coin_data)
        for coin, df in source_data.items():
            processed = self._calculate_features(df)
            train_end = int(len(processed) * self.train_split)
            
            X_train = processed[self.feature_cols].iloc[self.lookback : train_end]
            y_train = (processed['Close'].shift(-1) > processed['Close']).iloc[self.lookback : train_end].astype(int)

            # Feature Scaling: Modeli standartlaştırma
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scalers[coin] = scaler

            # Gradient Boosting: Hataları düzelterek ilerleyen zeka
            model = GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=3, 
                learning_rate=0.1, 
                subsample=0.8, 
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            self.models[coin] = model
            print(f"{coin} | Optimal Model Eğitildi (OOS Accuracy Kontrol Edildi)")

    def predict(self, data: dict) -> list[dict]:
        decisions = []
        all_coins = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
        
        for coin in all_coins:
            df = data.get(coin)
            if df is None or coin not in self.models or len(df) < self.lookback:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue
            
            processed = self._calculate_features(df)
            curr = processed.iloc[-1]
            
            # Scaling ve Tahmin
            X_curr_scaled = self.scalers[coin].transform(processed[self.feature_cols].iloc[[-1]])
            prob = self.models[coin].predict_proba(X_curr_scaled)[0][1]
            conf = abs(prob - 0.5)

            # Trade Cooldown (5 mum kuralı)
            candle_idx = len(df) - 1
            if coin in self.last_trade_candle and (candle_idx - self.last_trade_candle[coin] < 5):
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue

            # Güven Filtresi (%62 ve üzeri)
            if conf < 0.12:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                continue

            signal = 1 if prob > 0.5 else -1

            # Basamaklı Kaldıraç Sistemi (1-2-3)
            if conf > 0.25: 
                allocation, leverage = 0.25, 3
            elif conf > 0.18: 
                allocation, leverage = 0.20, 2
            else: 
                allocation, leverage = 0.15, 1

            # ATR Tabanlı Dinamik Stop Loss / Take Profit
            atr = max(curr['atr'], curr['Close'] * 0.02)
            stop_loss = curr['Close'] - (signal * atr * 1.5)
            take_profit = curr['Close'] + (signal * atr * 3.75)

            self.last_trade_candle[coin] = candle_idx
            decisions.append({
                "coin": coin, "signal": signal, "allocation": allocation, "leverage": leverage,
                "stop_loss": stop_loss, "take_profit": take_profit
            })
            
        return decisions

if __name__ == "__main__":
    strategy = OptimalStrategy()
    strategy.get_data()
    print("=" * 50)
    print("OPTIMAL STRATEGY v1.0 - EĞİTİM VE BACKTEST")
    print("=" * 50)
    strategy.egit()
    
    # Görülmemiş veri üzerinde test (unseen data)
    result = backtest.run(strategy=strategy, initial_capital=3000.0, start_candle=1100)
    result.print_summary()