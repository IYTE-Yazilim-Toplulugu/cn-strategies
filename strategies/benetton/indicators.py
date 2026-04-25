"""
Bu dosya ortak indikatörleri hesaplar.

Kullandıklarımız:
- SMA / EMA: trend yönü
- OBV: hacim teyidi
- Bollinger Bands: aşırı uzama filtresi
- Ortalama hacim ve volume spike
- Basit volatilite
"""

import pandas as pd
import numpy as np

from config import (
    SHORT_MA,
    LONG_MA,
    FAST_EMA_SHORT,
    FAST_EMA_LONG,
    BB_WINDOW,
    BB_STD,
    VOLUME_AVG_WINDOW,
    VOLUME_SPIKE_MULTIPLIER,
    VOLATILITY_WINDOW,
)


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume hesaplar (vektörel — hızlı).
    """
    direction = np.sign(df["Close"].diff())
    direction.iloc[0] = 0
    return (direction * df["Volume"]).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gelen DataFrame'e ortak indikatör kolonlarını ekler.
    """
    df = df.copy()

    # Safe strateji için SMA
    df[f"SMA_{SHORT_MA}"] = df["Close"].rolling(SHORT_MA).mean()
    df[f"SMA_{LONG_MA}"] = df["Close"].rolling(LONG_MA).mean()

    # Safe strateji için EMA opsiyonu
    df[f"EMA_{SHORT_MA}"] = df["Close"].ewm(span=SHORT_MA, adjust=False).mean()
    df[f"EMA_{LONG_MA}"] = df["Close"].ewm(span=LONG_MA, adjust=False).mean()

    # Fast strateji için EMA
    df[f"EMA_{FAST_EMA_SHORT}"] = df["Close"].ewm(span=FAST_EMA_SHORT, adjust=False).mean()
    df[f"EMA_{FAST_EMA_LONG}"] = df["Close"].ewm(span=FAST_EMA_LONG, adjust=False).mean()

    # OBV
    df["OBV"] = calculate_obv(df)

    # Bollinger Bands
    rolling_mean = df["Close"].rolling(BB_WINDOW).mean()
    rolling_std = df["Close"].rolling(BB_WINDOW).std()

    df["BB_MID"] = rolling_mean
    df["BB_UPPER"] = rolling_mean + BB_STD * rolling_std
    df["BB_LOWER"] = rolling_mean - BB_STD * rolling_std
    bb_diff = df["BB_UPPER"] - df["BB_LOWER"]
    df["BB_WIDTH"] = bb_diff / df["BB_MID"].replace(0, 1e-10) # Prevent zero division

    # RSI (14 period)
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10) # Prevent zero division
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # ATR (14 period)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df["ATR_14"] = true_range.rolling(14).mean()

    # Hacim teyidi
    df["VOLUME_AVG"] = df["Volume"].rolling(VOLUME_AVG_WINDOW).mean()
    df["VOLUME_CONFIRMED"] = df["Volume"] >= df["VOLUME_AVG"]
    df["VOLUME_SPIKE"] = df["Volume"] >= (df["VOLUME_AVG"] * VOLUME_SPIKE_MULTIPLIER)

    # Basit volatilite
    df["RETURNS"] = df["Close"].pct_change()
    df["VOLATILITY"] = df["RETURNS"].rolling(VOLATILITY_WINDOW).std()
    df["VOLATILITY_AVG"] = df["VOLATILITY"].rolling(VOLATILITY_WINDOW).mean()

    # ML için yardımcı özellikler
    df["MA_DIFF_SAFE"] = df[f"SMA_{SHORT_MA}"] - df[f"SMA_{LONG_MA}"]
    df["MA_DIFF_FAST"] = df[f"EMA_{FAST_EMA_SHORT}"] - df[f"EMA_{FAST_EMA_LONG}"]
    df["OBV_DIFF"] = df["OBV"].diff()
    df["BB_POSITION"] = (df["Close"] - df["BB_LOWER"]) / bb_diff.replace(0, 1e-10)

    return df