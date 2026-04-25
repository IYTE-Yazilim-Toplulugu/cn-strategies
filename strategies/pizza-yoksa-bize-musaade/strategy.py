import numpy as np
import pandas as pd
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest

class CompetitionBot(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.lookback = 200
        self.equity_hwm = 0.0
        self.freeze_candles = 0

    def predict(self, data: dict) -> list[dict]:
        coins = list(data.keys())
        if len(data[coins[0]]) < self.lookback:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coins]

        equity = getattr(self, "portfolio_value", 3000.0)
        self.equity_hwm = max(self.equity_hwm, equity)
        drawdown = (self.equity_hwm - equity) / self.equity_hwm if self.equity_hwm > 0 else 0

        # Dynamic freeze based on drawdown intensity
        if drawdown > 0.15:
            self.freeze_candles = 3
        if drawdown > 0.25:
            self.freeze_candles = 7

        if self.freeze_candles > 0:
            self.freeze_candles -= 1
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coins]

        # Adaptive ER lookback (10% of main lookback)
        er_lookback = max(10, self.lookback // 10)
        
        # Calculate systemic risk via correlation
        rets = pd.DataFrame({c: data[c]["Close"].pct_change().tail(60) for c in coins}).fillna(0)
        corr_matrix = rets.corr().fillna(0).values
        systemic_risk = float(np.max(np.real(np.linalg.eigvals(corr_matrix)))) / len(coins)

        scores = {}
        signals = {}
        vols = {}

        for coin in coins:
            df = data[coin]
            close = df["Close"]
            
            # Efficiency Ratio (ER) - Adaptive
            diff = abs(close.iloc[-1] - close.iloc[-er_lookback])
            noise = np.sum(np.abs(close.diff().tail(er_lookback)))
            er = diff / (noise + 1e-10)
            
            # Adaptive smoothing (Fastest 2, Slowest 30)
            sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
            ema_fast = close.ewm(alpha=sc).mean()
            ema_slow = close.ewm(span=26).mean()
            
            # Hysteresis: Require a minimum separation to flip signal
            diff_pct = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (close.iloc[-1] + 1e-10)
            
            # We use a small threshold (0.1%) to avoid noise-flipping
            if diff_pct > 0.001:
                direction = 1
            elif diff_pct < -0.001:
                direction = -1
            else:
                direction = 0 # Neutral in choppy zones
            
            # Volatility (Relative to its own history)
            vol_series = close.pct_change().tail(100)
            vol = vol_series.std()
            vols[coin] = vol
            
            # Score Calculation with Regime Filtering
            trend_strength = min(1.0, er * 2.0) 
            accel = (ema_fast.iloc[-1] - ema_fast.iloc[-3]) / (abs(ema_fast.iloc[-3]) + 1e-10)
            
            score = (1.0 + max(0.0, accel)) * trend_strength
            score *= (1.0 - systemic_risk)
            
            scores[coin] = score if direction != 0 else 0
            signals[coin] = direction

        # Weight distribution with power scaling
        raw_vals = np.array([scores[c] for c in coins])
        max_score = np.max(raw_vals)
        
        if max_score <= 1e-5:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coins]

        norm_scores = raw_vals / (max_score + 1e-10)
        weights = (norm_scores ** 1.8) / (np.sum(norm_scores ** 1.8) + 1e-10)

        used_alloc = 0.0
        sorted_indices = np.argsort(-raw_vals)
        results_map = {}

        for i in sorted_indices:
            coin = coins[i]
            w = float(weights[i])
            sig = signals[coin]
            
            if w < 0.05 or sig == 0:
                results_map[coin] = {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
                continue

            # Volatility-Adjusted Leverage (Dynamic thresholds)
            lev = 1
            if vols[coin] < 0.010: lev = 2
            if vols[coin] < 0.005: lev = 3
            
            # Risk reduction
            if systemic_risk > 0.55: lev = 1 
            if drawdown > 0.08: lev = min(lev, 1)

            alloc = round(min(w, 1.0 - used_alloc), 4)
            used_alloc += alloc
            
            price = data[coin]["Close"].iloc[-1]
            sl_dist = 4.5 * vols[coin] # Increased slightly
            sl = float(price * (1 - sig * sl_dist))

            results_map[coin] = {
                "coin": coin,
                "signal": int(sig),
                "allocation": float(alloc),
                "leverage": int(lev),
                "stop_loss": sl
            }

        # Final assembly
        final_output = []
        for coin in coins:
            res = results_map.get(coin, {"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
            final_output.append(res)

        return final_output

if __name__ == "__main__":
    strategy = CompetitionBot()
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()
