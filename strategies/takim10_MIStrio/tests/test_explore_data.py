import pytest
import pandas as pd
from pathlib import Path

DATA_DIR = Path("C:/Users/yusuf/AppData/Local/Programs/Python/Python312/Lib/site-packages/cnlib/data")
COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]
REQUIRED_COLUMNS = {"Date", "Open", "High", "Low", "Close", "Volume"}

class TestDataFiles:
    @pytest.mark.parametrize("coin", COINS)
    def test_data_file_exists(self, coin):
        path = DATA_DIR / f"{coin}.parquet"
        assert path.exists(), f"Dosya bulunamadı: {path}"

    @pytest.mark.parametrize("coin", COINS)
    def test_data_columns_present(self, coin):
        df = pd.read_parquet(DATA_DIR / f"{coin}.parquet")
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"{coin}: eksik sütunlar: {missing}"

class TestDataConsistency:
    def test_coin_data_equal_length(self):
        lengths = {coin: len(pd.read_parquet(DATA_DIR / f"{coin}.parquet")) for coin in COINS}
        assert len(set(lengths.values())) == 1, f"Farklı uzunluklar: {lengths}"

    @pytest.mark.parametrize("coin", COINS)
    def test_prices_positive(self, coin):
        df = pd.read_parquet(DATA_DIR / f"{coin}.parquet")
        assert (df["Close"] <= 0).sum() == 0

    @pytest.mark.parametrize("coin", COINS)
    def test_ohlc_consistency(self, coin):
        df = pd.read_parquet(DATA_DIR / f"{coin}.parquet")
        assert (df["Low"] > df[["Open", "Close"]].min(axis=1)).sum() == 0
        assert (df["High"] < df[["Open", "Close"]].max(axis=1)).sum() == 0
