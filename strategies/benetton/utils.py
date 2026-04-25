"""
Bu dosya tekrar eden küçük yardımcı fonksiyonları tutar.
Strateji dosyasını gereksiz uzatmamak için burada saklıyoruz.
"""

from typing import Iterable


def has_nan(values: Iterable) -> bool:
    """
    İçeride NaN veya bozuk değer var mı kontrol eder.
    Rolling indikatörlerin ilk satırlarında bu çok işe yarar.
    """
    for value in values:
        if value is None or value != value:  # NaN != NaN olduğu için pratik kontrol
            return True
    return False


def _volatility_guard_active(volatility: float, volatility_avg: float, multiplier: float) -> bool:
    if has_nan([volatility, volatility_avg]) or volatility_avg == 0:
        return False
    return volatility > volatility_avg * multiplier


def _flat_decision(coin: str) -> dict:
    return {
        "coin": coin,
        "signal": 0,
        "allocation": 0.0,
        "leverage": 1,
    }


def compute_max_drawdown(portfolio_series) -> float:
    import pandas as pd
    values = []

    for item in portfolio_series:
        if isinstance(item, (int, float)):
            values.append(float(item))
            continue

        if isinstance(item, dict):
            for key in ["portfolio_value", "value", "equity", "portfolio", "cash"]:
                if key in item and isinstance(item[key], (int, float)):
                    values.append(float(item[key]))
                    break

    if not values:
        return 0.0

    series = pd.Series(values, dtype="float64")
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max * 100.0
    return abs(drawdown.min())



def choose_ma_columns(use_ema: bool, short_ma: int, long_ma: int) -> tuple[str, str]:
    """
    Trend filtresinde hangi kolonların kullanılacağını döndürür.
    """
    if use_ema:
        return f"EMA_{short_ma}", f"EMA_{long_ma}"
    return f"SMA_{short_ma}", f"SMA_{long_ma}"


def build_tp_sl(entry: float, direction: int, stop_loss_pct: float, take_profit_pct: float) -> tuple[float, float]:
    """
    Giriş fiyatına göre TP ve SL seviyelerini hesaplar.

    direction:
    1  -> long
    -1 -> short
    """
    if direction == 1:
        stop_loss = max(1e-8, entry * (1 - stop_loss_pct))
        take_profit = max(1e-8, entry * (1 + take_profit_pct))
        return take_profit, stop_loss

    if direction == -1:
        stop_loss = max(1e-8, entry * (1 + stop_loss_pct))
        take_profit = max(1e-8, entry * (1 - take_profit_pct))
        return take_profit, stop_loss

    raise ValueError("direction sadece 1 veya -1 olabilir.")