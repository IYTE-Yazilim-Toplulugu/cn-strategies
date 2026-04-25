import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy import FinalStrategy
from cnlib.validator import validate, ValidationError

COINS = ["kapcoin-usd_train", "metucoin-usd_train", "tamcoin-usd_train"]

def _make_data(closes_func, n=100):
    import conftest
    closes = closes_func(n)
    return {coin: conftest._make_ohlcv(closes) for coin in COINS}

def _uptrend(n):
    return [100.0 * (1.02 ** i) for i in range(n)]

class TestFinalStrategySignals:
    """EMA ve Trailing Stop sinyalleri testi."""

    def test_rebalance_triggers_on_interval(self):
        strat = FinalStrategy()
        strat.candle_index = 20 # REBALANCE_EVERY == 20
        data = _make_data(_uptrend)
        
        decisions = strat.predict(data)
        # Rebalance candle'sında tüm allocationlar 0 olmalı.
        for dec in decisions:
            assert dec["signal"] == 0
            assert dec["allocation"] == 0.0

    def test_trailing_stop_is_triggered(self, uptrend_df):
        strat = FinalStrategy()
        
        # manuel bir high->low hareketi simüle et
        prices = list(uptrend_df["Close"].values)
        prices.append(prices[-1] * 0.84) # 1 candle'da %16 düşüş (Trailing pct %15)
        
        import conftest
        df = conftest._make_ohlcv(prices)
        
        signal = strat._coin_signal("test_coin", df)
        assert signal == 0, "Trailing stop devreye girip pozisyonu kapatmalı"
        assert strat.trailed_out["test_coin"] == 1, "trailed_out bayrağı 1 olmalı"

    def test_no_reentry_when_trailed_out(self):
        strat = FinalStrategy()
        strat.trailed_out["test_coin"] = 1 # zaten long stop edildi
        
        import conftest
        base_df = conftest._make_ohlcv([100.0 * (1.05 ** i) for i in range(60)])
        
        # uptrend hala geçerli
        signal = strat._coin_signal("test_coin", base_df)
        assert signal == 0, "Bayrak 1 iken tekrar long girilmemeli"

class TestPredictOutput:
    """Tahmin sonuç formatlarını cnlib ile uyum testleri."""

    def test_predict_format_valid(self):
        strat = FinalStrategy()
        strat.candle_index = 5
        data = _make_data(_uptrend)
        decisions = strat.predict(data)
        
        try:
            validate(decisions)
        except ValidationError as e:
            pytest.fail(f"Validation hatası: {e}")

    def test_allocation_and_leverage(self):
        strat = FinalStrategy()
        strat.candle_index = 5
        data = _make_data(_uptrend)
        
        # her iki coin de long alacak
        decisions = strat.predict(data)
        
        for dec in decisions:
            if dec["signal"] != 0:
                assert dec["allocation"] == 0.252, "ALLOC 0.252 olmalı"
                assert dec["leverage"] == 3, "LEVERAGE 3 olmalı"
