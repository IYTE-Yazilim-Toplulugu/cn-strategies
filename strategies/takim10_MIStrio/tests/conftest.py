import numpy as np
import pandas as pd
import pytest
from pathlib import Path

DATA_DIR = Path("C:/Users/yusuf/AppData/Local/Programs/Python/Python312/Lib/site-packages/cnlib/data")
COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]


def _make_ohlcv(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({
        "Date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "Open": closes * 0.999,
        "High": closes * 1.01,
        "Low": closes * 0.99,
        "Close": closes,
        "Volume": np.random.randint(100, 10000, size=n),
    })

@pytest.fixture
def uptrend_df():
    """Güçlü yükseliş trendi — EMA(5) > EMA(20) > EMA(50) ve onay ema'ları sağlanmalı."""
    base = 100.0
    prices = [base * (1.02 ** i) for i in range(100)] # daha hızlı büyüme
    return _make_ohlcv(prices)

@pytest.fixture
def weak_uptrend_df():
    """Zayıf yükseliş trendi — Ana trend uptrend ama onay ema'ları bozmuş olabilir."""
    base = 100.0
    prices = [base * (1.005 ** i) for i in range(95)]
    # son günlerde ani düşüş
    prices.extend([prices[-1] * 0.95 for _ in range(5)])
    return _make_ohlcv(prices)

@pytest.fixture
def downtrend_df():
    """Güçlü düşüş trendi — EMA(5) < EMA(20) < EMA(50) + short onay."""
    base = 200.0
    prices = [base * (0.98 ** i) for i in range(100)]
    return _make_ohlcv(prices)

@pytest.fixture
def high_vol_df():
    """Yüksek volatilite dönemi (fiyatlar çok dalgalanır)."""
    np.random.seed(42)
    prices = [100.0 * (1 + np.random.uniform(-0.10, 0.10)) for _ in range(100)]
    return _make_ohlcv(prices)

@pytest.fixture
def short_df():
    """Warmup dönemi için yetersiz veri (< 55)."""
    prices = [100.0 + i * 0.5 for i in range(40)]
    return _make_ohlcv(prices)
