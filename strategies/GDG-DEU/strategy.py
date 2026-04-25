import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ==========================================
# 1. CONFIGURATION
# ==========================================
COINS = ['kapcoin-usd_train', 'metucoin-usd_train', 'tamcoin-usd_train']
EMA_FAST, EMA_MID, EMA_SLOW = 9, 21, 55
RSI_PERIOD, BB_WINDOW, BB_STD = 14, 20, 2.0
ATR_PERIOD = 14

# Risk Parametreleri
MAX_RISK_PER_TRADE   = 0.06   
MAX_PORTFOLIO_HEAT   = 0.35   
SL_ATR_MULT          = 2.0
TP_ATR_MULT          = 5.5    
MAX_ALLOCATION       = 0.90   
INITIAL_CAPITAL      = 3000.0
MAX_DRAWDOWN_HALT    = 0.45   

# ==========================================
# 2. FEATURE ENGINE
# ==========================================
def compute_atr(df, period=ATR_PERIOD):
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    close = df['Close'].squeeze()
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=RSI_PERIOD):
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(span=period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def kalman_filter(prices, R=0.01, Q=0.001):
    n = len(prices)
    x, P = np.zeros(n), np.zeros(n)
    if n == 0: return x
    x[0], P[0] = prices[0], 1.0
    for i in range(1, n):
        P_pred = P[i-1] + Q
        K = P_pred / (P_pred + R)
        x[i] = x[i-1] + K * (prices[i] - x[i-1])
        P[i] = (1 - K) * P_pred
    return x

def compute_features(df):
    close = df['Close'].squeeze()
    high  = df['High'].squeeze()
    low   = df['Low'].squeeze()
    vol   = df['Volume'].squeeze()
    feat = {'kalman': pd.Series(kalman_filter(close.values), index=close.index)}
    feat['ema_fast'] = close.ewm(span=EMA_FAST, adjust=False).mean()
    feat['ema_mid'] = close.ewm(span=EMA_MID, adjust=False).mean()
    feat['ema_slow'] = close.ewm(span=EMA_SLOW, adjust=False).mean()
    feat['rsi'] = compute_rsi(close)
    feat['atr'] = compute_atr(df)
    
    feat['obv'] = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    tp = (high + low + close) / 3
    feat['vwap'] = (tp * vol).rolling(window=20).sum() / vol.rolling(window=20).sum()
    price_range = (high - low).replace(0, np.nan)
    v_delta = vol * ((close - low) - (high - close)) / price_range
    feat['cvd'] = v_delta.fillna(0).cumsum()
    kc_ema = close.ewm(span=20, adjust=False).mean()
    feat['kc_upper'] = kc_ema + (2.0 * feat['atr'])
    feat['kc_lower'] = kc_ema - (2.0 * feat['atr'])
    
    feat['bb_ma'] = close.rolling(BB_WINDOW).mean()
    std = close.rolling(BB_WINDOW).std().replace(0, np.nan)
    feat['bb_upper'], feat['bb_lower'] = feat['bb_ma'] + BB_STD * std, feat['bb_ma'] - BB_STD * std
    feat['z_score'] = (close - feat['bb_ma']) / std
    macd = feat['ema_fast'] - feat['ema_mid']
    feat['macd_hist'] = macd - macd.ewm(span=9, adjust=False).mean()
    return feat

# ==========================================
# 3. RL ENVIRONMENT FOR PPO
# ==========================================
class TradingEnv(gym.Env):
    def __init__(self, features_df, prices):
        super(TradingEnv, self).__init__()
        self.df = features_df
        self.prices = prices.values
        self.n_steps = len(self.df)
        self.current_step = 0
        
        # Action: Strong Short, Short, Neutral, Long, Strong Long
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.df.shape[1],), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return self.df.iloc[self.current_step].values.astype(np.float32), {}

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # Action Map: 0: Strong Short (-2), 1: Short (-1), 2: Neutral (0), 3: Long (1), 4: Strong Long (2)
        signal_map = {0: -2.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 2.0}
        act_val = signal_map[action]
        
        price_change = (self.prices[self.current_step] / self.prices[self.current_step-1]) - 1
        reward = act_val * price_change
        
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return obs, reward, done, False, {}

# ==========================================
# 4. RISK MANAGER
# ==========================================
class RiskManager:
    def __init__(self):
        self.peak_value = INITIAL_CAPITAL
        self.halted, self.halt_counter = False, 0

    def update_peak(self, val): self.peak_value = max(self.peak_value, val)

    def check_drawdown_halt(self, val):
        if (self.peak_value - val) / self.peak_value >= MAX_DRAWDOWN_HALT: self.halted, self.halt_counter = True, 10
        if self.halted:
            self.halt_counter -= 1
            if self.halt_counter <= 0: self.halted = False
        return self.halted

    def compute_allocation(self, portfolio_value, atr, entry_price, leverage, risk_multiplier=1.0):
        sl_dist = SL_ATR_MULT * atr
        if sl_dist == 0: return 0.0
        risk_cap = portfolio_value * (MAX_RISK_PER_TRADE * risk_multiplier)
        coins = risk_cap / sl_dist
        pos_val = coins * entry_price / leverage
        alloc = pos_val / portfolio_value
        return round(min(alloc, MAX_ALLOCATION), 2)

    def compute_stops(self, entry_price, atr, signal):
        sl_dist, tp_dist = SL_ATR_MULT * atr, TP_ATR_MULT * atr
        if signal == 1: 
            sl, tp = entry_price - sl_dist, entry_price + tp_dist
        else:
            sl, tp = entry_price + sl_dist, entry_price - tp_dist
        return max(0.000001, sl), max(0.000001, tp)

# ==========================================
# 5. PPO STRATEGY
# ==========================================
class TradeBotStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.risk = RiskManager()
        self.capital = INITIAL_CAPITAL
        self.active_positions, self.ppo_model, self.is_trained = {}, None, False
        self.scaler = StandardScaler()
        self.get_data()
        self.egit()

    def egit(self):
        print(f"PPO Reinforcement Learning modeli eğitiliyor (%80 eğitim / %20 test ayrımı)...")
        import cnlib
        from pathlib import Path
        data_dir = Path(cnlib.__file__).parent / "data"
        
        all_X, all_prices = [], []
        
        for coin in COINS:
            file_path = data_dir / f"{coin}.parquet"
            if not file_path.exists():
                print(f"Uyarı: {file_path} bulunamadı!")
                continue
            
            df = pd.read_parquet(file_path)
            if len(df) < 100: continue
            
            # Verinin %80'ini eğitim için ayırıyoruz
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx]
            
            feat_df = self._prepare_ml_features(train_df)
            # Son satırı (label oluşamayacağı için) hariç tutuyoruz
            if len(feat_df) > 1:
                all_X.append(feat_df[:-1])
                all_prices.append(train_df['Close'].iloc[:-1])
        
        if not all_X:
            print("Hata: Eğitim için veri bulunamadı!")
            return

        X_combined = pd.concat(all_X)
        # Scaler sadece eğitim verisiyle fit edilir (Data Leakage önlemi)
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_combined), columns=X_combined.columns)
        prices_combined = pd.concat(all_prices)

        env = DummyVecEnv([lambda: TradingEnv(X_scaled, prices_combined)])
        self.ppo_model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=512)
        print(f"Model {X_scaled.shape[0]} satır veri ile eğitiliyor...")
        self.ppo_model.learn(total_timesteps=15000)
        self.is_trained = True

    def _prepare_ml_features(self, df):
        feat = compute_features(df)
        close = df['Close']
        return pd.DataFrame({
            'rsi': feat['rsi'], 'z': feat['z_score'], 'm': feat['macd_hist'], 'a': feat['atr'] / close,
            'momentum_3': close.pct_change(3), 'momentum_5': close.pct_change(5),
            'obv': feat['obv'] / feat['obv'].abs().rolling(20).max(),
            'vwap_dist': (close - feat['vwap']) / close,
            'cvd': feat['cvd'] / feat['cvd'].abs().rolling(20).max(),
            'kc_dist': (close - (feat['kc_upper'] + feat['kc_lower'])/2) / close
        }).fillna(0)

    def predict(self, data: dict) -> list[dict]:
        self.risk.update_peak(self.capital)
        if self.risk.check_drawdown_halt(self.capital):
            self.active_positions = {}
            return [{'coin': c, 'signal': 0, 'allocation': 0.0, 'leverage': 1} for c in COINS]

        decisions, signals_found = [], []
        for coin in COINS:
            df = data.get(coin)
            if df is None or len(df) < 60 or not self.is_trained:
                decisions.append({'coin': coin, 'signal': 0, 'allocation': 0.0, 'leverage': 1})
                continue

            ml_feat = self._prepare_ml_features(df).iloc[[-1]]
            ml_feat_scaled = self.scaler.transform(ml_feat)
            action, _ = self.ppo_model.predict(ml_feat_scaled)
            action = int(np.asarray(action).flat[0])  # SB3 bazen (1,) shape döndürür
            
            # Action Mapping
            # 0: Strong Short, 1: Short, 2: Neutral, 3: Long, 4: Strong Long
            sig, leverage, risk_mult = 0, 1, 0.0
            if action == 0: sig, leverage, risk_mult = -1, 10, 3.0
            elif action == 1: sig, leverage, risk_mult = -1, 3, 1.0
            elif action == 3: sig, leverage, risk_mult = 1, 3, 1.0
            elif action == 4: sig, leverage, risk_mult = 1, 10, 3.0

            if sig != 0:
                atr_v = float(np.asarray(compute_atr(df).iloc[-1]).flat[0])
                signals_found.append((coin, sig, atr_v, leverage, risk_mult, df))
            else:
                if coin in self.active_positions: del self.active_positions[coin]
                decisions.append({'coin': coin, 'signal': 0, 'allocation': 0.0, 'leverage': 1})

        total_alloc = 0.0
        for coin, sig, atr_v, leverage, risk_mult, df in signals_found:
            entry = float(np.asarray(df['Close'].squeeze().iloc[-1]).flat[0])
            if coin in self.active_positions:
                pos = self.active_positions[coin]
                if sig == 1:
                    pos['max_p'] = max(pos.get('max_p', entry), entry)
                    pos['sl'] = max(pos.get('sl', 0), pos['max_p'] - (SL_ATR_MULT * atr_v))
                else:
                    pos['min_p'] = min(pos.get('min_p', entry), entry)
                    pos['sl'] = min(pos.get('sl', 999999), pos['min_p'] + (SL_ATR_MULT * atr_v))
                sl, tp = pos['sl'], pos['tp']
            else:
                sl, tp = self.risk.compute_stops(entry, atr_v, sig)
                self.active_positions[coin] = {'max_p': entry, 'min_p': entry, 'sl': sl, 'tp': tp}

            alloc = self.risk.compute_allocation(self.capital, atr_v, entry, leverage, risk_mult)
            if len(signals_found) > 1: alloc *= 0.8
            alloc = min(alloc, 0.98 - total_alloc)
            
            if alloc <= 0.01:
                decisions.append({'coin': coin, 'signal': 0, 'allocation': 0.0, 'leverage': 1})
                continue
                
            total_alloc += alloc
            decisions.append({
                'coin': coin, 'signal': sig, 'allocation': round(alloc, 2), 
                'leverage': leverage, 'stop_loss': round(sl, 6), 'take_profit': round(tp, 6)
            })

        decided = {d['coin'] for d in decisions}
        for c in COINS:
            if c not in decided: decisions.append({'coin': c, 'signal': 0, 'allocation': 0.0, 'leverage': 1})
        return decisions

if __name__ == '__main__':
    print("\n--- Testing PPO Strategy on Local Dataset ---")
    bot = TradeBotStrategy()
    #bot.get_data()
    #bot.egit()
    backtest.run(strategy=bot, initial_capital=INITIAL_CAPITAL).print_summary()
