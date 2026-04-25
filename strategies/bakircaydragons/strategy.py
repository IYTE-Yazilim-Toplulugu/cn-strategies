from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
import pandas as pd
import numpy as np


class MomentumHunter(BaseStrategy):
    """
    Adaptive Raw Momentum Strategy.
    
    Measures 20-day rolling autocorrelation to detect momentum regime.
    - AC > 0.10: Strong/moderate momentum → full aggression (5x, 100% allocation)
    - AC <= 0.10: Weak/no momentum → defensive (1x leverage, 20% allocation, 4% threshold)
    
    Signal: 1-day raw price momentum on strongest coin.
    """
    
    def __init__(self):
        super().__init__()
        self.lookback = 2
    
    def predict(self, data: dict) -> list[dict]:
        coin_names = list(data.keys())
        min_len = min(len(df) for df in data.values())
        
        if min_len < 25:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coin_names]
        
        # Step 1: Measure rolling autocorrelation across ALL coins
        autocorrs = []
        for coin_name, df in data.items():
            closes = df["Close"]
            recent_returns = closes.pct_change().iloc[-21:-1]
            if len(recent_returns) >= 20:
                ac = recent_returns.autocorr(lag=1)
                if not pd.isna(ac):
                    autocorrs.append(ac)
        
        avg_ac = np.mean(autocorrs) if autocorrs else 0.0
        
        # Step 2: Set parameters based on regime
        if avg_ac > 0.10:
            min_strength = 0.008
            leverage = 5
            allocation = 1.0
        else:
            min_strength = 0.04
            leverage = 1
            allocation = 0.20
        
        # Step 3: Compute raw momentum for each coin
        coin_analysis = []
        for coin_name, df in data.items():
            closes = df["Close"]
            current_price = closes.iloc[-1]
            lookback_price = closes.iloc[-self.lookback]
            
            if (pd.isna(current_price) or pd.isna(lookback_price) or 
                lookback_price == 0 or current_price == 0):
                coin_analysis.append({"coin": coin_name, "signal": 0, "strength": 0.0})
                continue
            
            momentum = (current_price / lookback_price) - 1
            signal = 1 if momentum > 0 else -1
            strength = abs(momentum)
            
            coin_analysis.append({
                "coin": coin_name,
                "signal": signal,
                "strength": strength
            })
        
        # Step 4: Pick strongest coin
        coin_analysis.sort(key=lambda x: x["strength"], reverse=True)
        best = coin_analysis[0]
        
        if best["strength"] < min_strength:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coin_names]
        
        # Step 5: Build decisions
        decisions = []
        for coin_name in coin_names:
            if coin_name == best["coin"]:
                decisions.append({
                    "coin": coin_name,
                    "signal": best["signal"],
                    "allocation": allocation,
                    "leverage": leverage
                })
            else:
                decisions.append({
                    "coin": coin_name,
                    "signal": 0,
                    "allocation": 0.0,
                    "leverage": 1
                })
        
        return decisions


if __name__ == "__main__":
    strategy = MomentumHunter()
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()
from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
import pandas as pd
import numpy as np


class MomentumHunter(BaseStrategy):
    """
    Adaptive Raw Momentum Strategy.
    
    Measures 20-day rolling autocorrelation to detect momentum regime.
    - AC > 0.10: Strong/moderate momentum → full aggression (5x, 100% allocation)
    - AC <= 0.10: Weak/no momentum → defensive (1x leverage, 20% allocation, 4% threshold)
    
    Signal: 1-day raw price momentum on strongest coin.
    """
    
    def __init__(self):
        super().__init__()
        self.lookback = 2
    
    def predict(self, data: dict) -> list[dict]:
        coin_names = list(data.keys())
        min_len = min(len(df) for df in data.values())
        
        if min_len < 25:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coin_names]
        
        # Step 1: Measure rolling autocorrelation across ALL coins
        autocorrs = []
        for coin_name, df in data.items():
            closes = df["Close"]
            recent_returns = closes.pct_change().iloc[-21:-1]
            if len(recent_returns) >= 20:
                ac = recent_returns.autocorr(lag=1)
                if not pd.isna(ac):
                    autocorrs.append(ac)
        
        avg_ac = np.mean(autocorrs) if autocorrs else 0.0
        
        # Step 2: Set parameters based on regime
        if avg_ac > 0.10:
            min_strength = 0.008
            leverage = 5
            allocation = 1.0
        else:
            min_strength = 0.04
            leverage = 1
            allocation = 0.20
        
        # Step 3: Compute raw momentum for each coin
        coin_analysis = []
        for coin_name, df in data.items():
            closes = df["Close"]
            current_price = closes.iloc[-1]
            lookback_price = closes.iloc[-self.lookback]
            
            if (pd.isna(current_price) or pd.isna(lookback_price) or 
                lookback_price == 0 or current_price == 0):
                coin_analysis.append({"coin": coin_name, "signal": 0, "strength": 0.0})
                continue
            
            momentum = (current_price / lookback_price) - 1
            signal = 1 if momentum > 0 else -1
            strength = abs(momentum)
            
            coin_analysis.append({
                "coin": coin_name,
                "signal": signal,
                "strength": strength
            })
        
        # Step 4: Pick strongest coin
        coin_analysis.sort(key=lambda x: x["strength"], reverse=True)
        best = coin_analysis[0]
        
        if best["strength"] < min_strength:
            return [{"coin": c, "signal": 0, "allocation": 0.0, "leverage": 1} for c in coin_names]
        
        # Step 5: Build decisions
        decisions = []
        for coin_name in coin_names:
            if coin_name == best["coin"]:
                decisions.append({
                    "coin": coin_name,
                    "signal": best["signal"],
                    "allocation": allocation,
                    "leverage": leverage
                })
            else:
                decisions.append({
                    "coin": coin_name,
                    "signal": 0,
                    "allocation": 0.0,
                    "leverage": 1
                })
        
        return decisions


if __name__ == "__main__":
    strategy = MomentumHunter()
    result = backtest.run(strategy=strategy, initial_capital=3000.0)
    result.print_summary()
