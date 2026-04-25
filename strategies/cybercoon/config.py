from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List


COINS: List[str] = [
    "kapcoin-usd_train",
    "metucoin-usd_train",
    "tamcoin-usd_train",
]

INITIAL_CAPITAL: float = 3000.0


@dataclass
class StrategyConfig:
    # --- Warmup ---
    min_bars: int = 80          # ema_slow + buffer (1 yıllık veri için yarıya indirildi)

    # --- Indicators ---
    ema_fast: int = 10           # 20 → 10
    ema_mid: int = 25            # 50 → 25
    ema_slow: int = 100          # 200 → 100
    rsi_period: int = 14
    atr_period: int = 14
    lookback: int = 10           # 20 → 10

    # --- Skor eşiği ---
    score_threshold: int = 4    # sinyal için gereken min skor (4 veya 5)

    # --- RSI aralıkları ---
    rsi_long_low: float = 50.0
    rsi_long_high: float = 68.0
    rsi_short_low: float = 32.0
    rsi_short_high: float = 50.0

    # --- Breakout/breakdown yakınlık payı ---
    breakout_pct: float = 0.995   # close >= breakout_pct * hh10
    breakdown_pct: float = 1.005  # close <= breakdown_pct * ll10

    # --- Volatilite eşikleri (coin bazlı) ---
    atr_high: Dict[str, float] = field(default_factory=lambda: {
        "kapcoin-usd_train":  0.10,
        "metucoin-usd_train": 0.08,
        "tamcoin-usd_train":  0.08,
    })
    # ATR bu çarpanın altındaysa güçlü trend kaldıracı verilir
    atr_low_multiplier: float = 0.7

    # --- Kaldıraç seviyeleri ---
    lev_strong: int = 5    # güçlü trend + düşük ATR
    lev_normal: int = 3    # normal trend
    lev_weak: int = 1      # yüksek vol / sideways

    # --- Hard risk-off ---
    shock_exit_pct: float = 0.06   # günlük ters hareket eşiği

    # --- Time-stop ---
    time_stop_days: int = 7            # 15 → 7 (kısa veri setinde hızlı rotate)
    time_stop_momentum: float = 0.01   # ret_3d bu eşiğin altındaysa kapat

    # --- Allocation ağırlıkları ---
    alloc_single: float = 0.50
    alloc_first: float = 0.45
    alloc_second: float = 0.35   # 2 aktif coin varken 2. coin
    alloc_second_of_three: float = 0.30
    alloc_third: float = 0.20


# Strateji tarafından kullanılan varsayılan config
DEFAULT_CONFIG = StrategyConfig()
