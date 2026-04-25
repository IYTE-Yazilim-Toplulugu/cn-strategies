import pandas as pd
import pandas_ta as ta


def calculate_indicators(
    ohlcv: pd.DataFrame,
    ma_period: int = 20,
    bb_period: int = 20,
    bb_std: float = 2.0,
    rsi_period: int = 14,
    adx_period: int = 14,
) -> pd.DataFrame:
    """Calculate MA, MACD, RSI, Bollinger Bands, and ADX for one coin OHLCV dataframe."""
    required_cols = {"High", "Low", "Close"}
    missing = required_cols - set(ohlcv.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(f"OHLCV dataframe is missing required columns: {missing_cols}")

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)

    out = pd.DataFrame(index=ohlcv.index)

    out["close"] = close
    out["ma"] = ta.sma(close=close, length=ma_period)

    bbands = ta.bbands(close=close, length=bb_period, std=bb_std)
    if bbands is None or bbands.empty:
        raise ValueError("pandas-ta failed to compute Bollinger Bands.")
    out["bb_lower"] = bbands.iloc[:, 0]
    out["bb_mid"] = bbands.iloc[:, 1]
    out["bb_upper"] = bbands.iloc[:, 2]

    macd_df = ta.macd(close=close, fast=12, slow=26, signal=9)
    if macd_df is None or macd_df.empty:
        raise ValueError("pandas-ta failed to compute MACD.")
    out["macd"] = macd_df.iloc[:, 0]
    out["macd_hist"] = macd_df.iloc[:, 1]
    out["macd_signal"] = macd_df.iloc[:, 2]

    out["rsi"] = ta.rsi(close=close, length=rsi_period)

    adx_df = ta.adx(high=high, low=low, close=close, length=adx_period)
    if adx_df is None or adx_df.empty:
        raise ValueError("pandas-ta failed to compute ADX.")
    out["adx"] = adx_df.iloc[:, 0]
    out["plus_di"] = adx_df.iloc[:, 1]
    out["minus_di"] = adx_df.iloc[:, 2]

    out = out.bfill().ffill()

    return out
