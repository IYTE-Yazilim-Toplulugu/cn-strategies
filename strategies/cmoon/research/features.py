"""
Shared indicator library — import from here, never reimplement elsewhere.

All functions take a pd.Series or pd.DataFrame and return a pd.Series.
Every function handles len(series) < window gracefully (returns NaN, not raises).
"""
import numpy as np
import pandas as pd


COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
OHLCV_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume")


def _validate_window(n: int, name: str = "n") -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError(f"{name} must be a positive integer, got {n!r}")


def _validate_band_multiplier(k: float) -> None:
    if isinstance(k, bool) or not isinstance(k, (int, float)) or k <= 0:
        raise ValueError(f"k must be a positive number, got {k!r}")


def validate_ohlcv(df: pd.DataFrame, name: str = "df") -> None:
    """Validate that a DataFrame has the OHLCV columns used by indicators."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")

    missing = set(OHLCV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing OHLCV column(s): {sorted(missing)}")


# ---------------------------------------------------------------------------
# Trend indicators
# ---------------------------------------------------------------------------

def ema(series: pd.Series, n: int) -> pd.Series:
    _validate_window(n)
    return series.ewm(span=n, adjust=False).mean()


def sma(series: pd.Series, n: int) -> pd.Series:
    _validate_window(n)
    return series.rolling(n).mean()


def ema_cross_signal(series: pd.Series, fast: int = 20, slow: int = 50) -> pd.Series:
    """Returns +1 when fast EMA crosses above slow, -1 when crosses below, 0 otherwise."""
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    above = fast_ema > slow_ema
    cross_up = above & ~above.shift(1).fillna(above)
    cross_dn = ~above & above.shift(1).fillna(~above)
    result = pd.Series(0, index=series.index, dtype=int)
    result[cross_up] = 1
    result[cross_dn] = -1
    return result


# ---------------------------------------------------------------------------
# Volatility indicators
# ---------------------------------------------------------------------------

def true_range(df: pd.DataFrame) -> pd.Series:
    """Single-candle True Range: max(H-L, |H-prev_C|, |L-prev_C|)."""
    validate_ohlcv(df)
    prev_close = df["Close"].shift(1)
    ranges = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1)
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder's Average True Range — EWM with alpha=1/n, NaN for first n-1 rows."""
    _validate_window(n)
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def atr_pct(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR as a fraction of Close — normalized volatility, comparable across coins."""
    close = df["Close"].replace(0, np.nan)
    return atr(df, n) / close


def bb_bands(series: pd.Series, n: int = 20, k: float = 2.0):
    """Returns (upper, mid, lower) Bollinger Bands. Uses population std (ddof=0)."""
    _validate_window(n)
    _validate_band_multiplier(k)
    mid   = sma(series, n)
    std   = series.rolling(n).std(ddof=0)
    upper = mid + k * std
    lower = mid - k * std
    return upper, mid, lower


def bb_width(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """(upper - lower) / mid — measures band squeeze. High = trending, low = ranging."""
    upper, mid, lower = bb_bands(series, n, k)
    return (upper - lower) / mid.replace(0, np.nan)


def bb_pct(series: pd.Series, n: int = 20, k: float = 2.0) -> pd.Series:
    """(Close - lower) / (upper - lower) — position within bands [0, 1]."""
    upper, _, lower = bb_bands(series, n, k)
    width = (upper - lower).replace(0, np.nan)
    return (series - lower) / width


# ---------------------------------------------------------------------------
# Momentum / oscillators
# ---------------------------------------------------------------------------

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI using EWM smoothing. NaN for first n-1 rows."""
    _validate_window(n)
    delta    = series.diff()
    gains    = delta.clip(lower=0)
    losses   = (-delta).clip(lower=0)
    avg_gain = gains.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = losses.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    return result


def momentum(series: pd.Series, n: int = 10) -> pd.Series:
    """Rate of change: (close / close[n] - 1)."""
    _validate_window(n)
    return series.pct_change(n)


def volume_ratio(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """Current volume vs n-period average — spike = breakout confirmation."""
    validate_ohlcv(df)
    _validate_window(n)
    avg_volume = df["Volume"].rolling(n).mean().replace(0, np.nan)
    return df["Volume"] / avg_volume


# ---------------------------------------------------------------------------
# Cross-coin features
# ---------------------------------------------------------------------------

def rolling_correlation(a: pd.Series, b: pd.Series, n: int = 30) -> pd.Series:
    """Rolling Pearson correlation between two return series."""
    _validate_window(n)
    return a.pct_change().rolling(n).corr(b.pct_change())


def lead_lag_signal(leader: pd.Series, follower: pd.Series, lag: int = 1) -> pd.Series:
    """
    Returns the leader's return lagged by `lag` candles.
    Use as a cross-coin feature: if kap went up yesterday, does met go up today?
    """
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 0:
        raise ValueError(f"lag must be a non-negative integer, got {lag!r}")
    return leader.pct_change().shift(lag).reindex(follower.index)


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def is_trending(df: pd.DataFrame, bb_threshold: float = 0.08) -> pd.Series:
    """
    Boolean series: True when market is in a trending regime.
    Uses BB width — wide bands = trending, narrow = ranging.
    bb_threshold of 0.08 suits ~8% average daily range in this dataset.
    """
    validate_ohlcv(df)
    return bb_width(df["Close"]) > bb_threshold
