"""
Feature matrix builder for ML models — Person 3's primary deliverable.

build_X_y() is the main entry point: it takes full coin_data and returns
(X, y, dates) ready for sklearn/XGBoost training.

Keep this file free of model code — just feature engineering.
"""
import numpy as np
import pandas as pd

from research.features import (
    ema, atr_pct, rsi, bb_pct, bb_width,
    momentum, volume_ratio, lead_lag_signal, rolling_correlation,
)

# Baseline forecast horizon used by train_models.py and strategy.py.
TARGET_HORIZON = 3

# Minimum rows needed before the first valid feature row
MIN_ROWS = 55


def build_features_single(df: pd.DataFrame, leader_close: pd.Series | None = None) -> pd.DataFrame:
    """
    Build a feature DataFrame for one coin.

    leader_close: optional Close series of another coin to use as a lead-lag feature
                  (e.g. pass kapcoin Close when computing features for metucoin)

    Returns a DataFrame aligned with df's index. Rows with NaN are included —
    caller should drop them after aligning across all coins.

    The column order is part of the training/inference contract. If features
    are changed later, retrain models and regenerate saved bundles.
    """
    close = df["Close"]
    feat = pd.DataFrame(index=df.index)

    # --- Returns ---
    feat["ret_1"] = close.pct_change(1)
    feat["ret_3"] = close.pct_change(3)
    feat["ret_5"] = close.pct_change(5)
    feat["ret_10"] = close.pct_change(10)
    feat["ret_20"] = close.pct_change(20)

    # --- Trend ---
    ema_20 = ema(close, 20)
    ema_50 = ema(close, 50)
    feat["ema_diff_20_50"] = (ema_20 - ema_50) / close
    feat["close_vs_ema_20"] = (close - ema_20) / close
    feat["close_vs_ema_50"] = (close - ema_50) / close

    # --- Momentum / oscillators ---
    feat["rsi_14"] = rsi(close, 14)
    feat["bb_pct_20"] = bb_pct(close, 20)
    feat["bb_width_20"] = bb_width(close, 20)
    feat["mom_10"] = momentum(close, 10)

    # --- Volatility / participation ---
    feat["atr_pct_14"] = atr_pct(df, 14)
    feat["vol_ratio_20"] = volume_ratio(df, 20)

    # --- Cross-coin (lead-lag) ---
    if leader_close is not None:
        feat["leader_ret_1"] = lead_lag_signal(leader_close, close, lag=1)
        feat["leader_ret_3"] = lead_lag_signal(leader_close, close, lag=3)
        feat["leader_corr_30"] = rolling_correlation(leader_close, close, n=30)

    # --- Price action / candle structure (appended last — column order is a contract) ---
    high   = df["High"]
    low    = df["Low"]
    open_  = df["Open"]
    hl_range = (high - low).replace(0, np.nan)

    feat["body_pct"]         = (close - open_) / hl_range
    feat["upper_shadow_pct"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / hl_range
    feat["lower_shadow_pct"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / hl_range
    feat["hl_range_pct"]     = hl_range / close.replace(0, np.nan)

    return feat


def build_X_y(
    coin_data: dict[str, pd.DataFrame],
    target_coin: str,
    horizon: int = TARGET_HORIZON,
) -> tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Build (X, y, valid_index) for a single coin's model.

    target_coin: which coin to predict
    X: 2D float array, shape (n_samples, n_features)
    y: 1D int array, 1 if price rises over horizon candles, 0 if falls
    valid_index: original DataFrame index for the valid rows (for time-series split)

    The baseline target is binary: 1 if price rises over the horizon, else 0.
    """
    df = coin_data[target_coin]

    # Use kapcoin as leader for non-kap coins (BTC leads alts historically)
    leader = coin_data["kapcoin-usd_train"]["Close"] if target_coin != "kapcoin-usd_train" else None

    feat = build_features_single(df, leader_close=leader)

    # Target: did price go up over the next `horizon` candles?
    future_return = df["Close"].shift(-horizon) / df["Close"] - 1.0
    y_raw = (future_return > 0).astype(int)

    # Align and drop NaN rows
    combined = pd.concat([feat, future_return.rename("future_return"), y_raw.rename("target")], axis=1)
    combined = combined.dropna()

    combined = combined.iloc[MIN_ROWS:]

    X = combined.drop(columns=["future_return", "target"]).values.astype(np.float32)
    y = combined["target"].values.astype(int)
    valid_index = combined.index

    return X, y, valid_index


def feature_names(coin_data: dict, target_coin: str) -> list[str]:
    """Returns column names in the same order as build_X_y() — useful for importance plots."""
    df = coin_data[target_coin]
    leader = coin_data["kapcoin-usd_train"]["Close"] if target_coin != "kapcoin-usd_train" else None
    feat = build_features_single(df, leader_close=leader)
    return list(feat.columns)
