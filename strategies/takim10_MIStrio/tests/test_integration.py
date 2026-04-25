import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from strategy import FinalStrategy
from cnlib import backtest

class TestFullIntegration:
    @pytest.fixture(scope="class")
    def result(self):
        strategy = FinalStrategy()
        return backtest.run(strategy=strategy, initial_capital=3000.0, silent=True)

    def test_backtest_runs_with_no_errors(self, result):
        assert result.validation_errors == 0
        assert result.strategy_errors == 0

    def test_backtest_makes_profit(self, result):
        assert result.final_portfolio_value > 3000, "Strateji kâr ettirmeli"

    def test_positivity(self, result):
        import pandas as pd
        df = pd.DataFrame(result.portfolio_series)
        assert (df['portfolio_value'] >= 0).all()

    def test_has_sufficient_trade_activity(self, result):
        # 1570 candle'ın 1570'inde her 20 barda rebalance ettiği için çok fazla trade çıkacak
        assert result.total_trades > 100, "Rebalance sebebiyle ciddi trade sayısı olmalı"
