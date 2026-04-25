# cmoon — Code Night Trading Strategy

Leveraged long/short backtest strategy for the [Code Night](https://github.com/IYTE-Yazilim-Toplulugu/code-night-lib) algorithmic trading competition.

## Setup

```bash
git clone <repo-url>
cd cmoon
uv sync
```

## Running a backtest

```bash
# Test the current submission strategy
.venv/bin/python run.py

# Test individual research strategies
.venv/bin/python run.py --strategy trend
.venv/bin/python run.py --strategy meanrevert

# Options
.venv/bin/python run.py --strategy trend --start 50 --end 500 --plot --capital 3000
```

| Flag | Default | Description |
|---|---|---|
| `--strategy` | `main` | `main`, `ensemble`, `trend`, `meanrevert` |
| `--start` | `0` | First candle to run, inclusive |
| `--end` | last candle | Last candle to run, inclusive |
| `--capital` | `3000` | Starting capital |
| `--data-dir` | bundled `cnlib` data | Directory containing the three coin parquet files |
| `--list-strategies` | off | Print strategy registry status and exit |
| `--plot` | off | Save equity curve to `results/` (requires matplotlib) |
| `--silent` | off | Suppress per-candle progress output |

`run.py` uses the same bounded backtest helper as walk-forward validation, so
`--start 10 --end 20` runs exactly 11 candles. `main` uses the final
`MyStrategy` path and loads its tracked model bundles from `models/`.

## cnlib 0.1.4 compatibility

The project is now locked to `cnlib` 0.1.4 in `uv.lock`. This version changed
how strategy data is exposed:

- Inside `predict(data)`, keep using the `data` argument. It contains history up
  to the current candle and is the safe competition-facing API.
- `strategy.coin_data` is now also sliced to the current `candle_index`; after
  `get_data()` it may only expose candle 0 until the backtest advances.
- Offline research code that needs the full dataset must use `strategy._full_data`
  after `get_data()`.
- `research.backtest_window.run_backtest_window()` already handles both the old
  and new data models, and mirrors the new `failed_opens` result fields.

Example for full-history research scripts:

```python
from cnlib.base_strategy import BaseStrategy

class Loader(BaseStrategy):
    def predict(self, data):
        return []

loader = Loader()
loader.get_data()
coin_data = loader._full_data  # full 1570-candle history in cnlib 0.1.4
```

## Training ML models

Tracked ML model bundles live in `models/` and are loaded by `strategy.py` at
startup. Regenerate them after changing features, labels, model parameters, or
the train/holdout split:

```bash
.venv/bin/python research/train_models.py
# → saves model bundles to models/model_*.pkl
# → saves results/feature_importance.csv
```

`research/train_models.py` returns full history from `_full_data` under
`cnlib` 0.1.4, splits by original candle index, purges the target-horizon gap
before holdout, and saves bundled metadata next to each estimator for
inference. `results/` is generated and gitignored; `models/` contains the
portable inference artifacts.

## Project structure

```
cmoon/
├── strategy.py                   # Submission file — final ensemble
├── run.py                        # Backtest runner (see usage above)
├── models/                       # Tracked model bundles used by strategy.py
│   └── model_*.pkl
│
├── research/
│   ├── features.py               # Shared indicator library (EMA, ATR, RSI, BB, …)
│   ├── backtest_window.py         # Bounded inclusive start/end backtest helper
│   ├── risk.py                   # Leverage selection, stop/TP placement, sizing
│   ├── trend_strategy.py         # EMA crossover strategy (trending regime)
│   ├── mean_revert_strategy.py   # RSI + Bollinger Bands (ranging regime)
│   ├── ml_features.py            # Feature matrix builder for ML models
│   ├── train_models.py           # Train models to models/, reports to results/
│   ├── ensemble.py               # Combines trend signal + ML probability
│   └── walk_forward.py           # Validation: walk-forward + holdout test
│
└── results/                      # Generated reports/plots — gitignored
    ├── feature_importance.csv    # Feature importance from train_models.py
    └── *_equity.png              # Equity curve plots from --plot
```

## Strategy architecture

```
predict(data)
    │
    ├── features.py          compute EMA, ATR, RSI, BB width per coin
    │
    ├── trend_strategy.py    EMA crossover signal  (+1 / -1 / 0)
    │   └── risk.py          ATR-based leverage + stop loss price
    │
    ├── ml_features.py       build feature vector for current candle
    │   └── model.predict_proba()   P(price goes up over next N candles)
    │
    └── ensemble.py          only trade when trend + ML agree → final decision dict
```

### Regime switching

| BB Width | Regime | Active strategy |
|---|---|---|
| > 0.08 | Trending | `trend_strategy` — EMA crossover, leverage 3–5x |
| < 0.06 | Ranging | `mean_revert_strategy` — RSI + BB bands, leverage 2x max |
| Between | Ambiguous | Stay flat |

### Risk rules

- Stop loss set at position open via `stop_loss_price()` — never rely on `predict()` catching a reversal in time
- Leverage scales with ATR%: low volatility → 5x, high volatility → 1–2x
- Max 2 coins active simultaneously; total allocation ≤ 0.9
- 10x leverage is never used without a stop loss tighter than the liquidation threshold

### Liquidation thresholds (from engine)

| Leverage | Liquidation distance from entry |
|---|---|
| 10x | 10% |
| 5x | 20% |
| 3x | 33% |
| 2x | 50% |

Average daily range across all coins is ~8%, so **10x without a stop loss is effectively a coin flip on liquidation per candle.**

## Validation

Never tune parameters against the holdout set. Use walk-forward on the train window:

```python
from research.walk_forward import walk_forward, holdout_test
from research.trend_strategy import TrendStrategy

# Cross-validate on candles 0–1099 (train only)
walk_forward(TrendStrategy, n_splits=4)

# Run once at the very end
holdout_test(TrendStrategy())
```

| Window | Candles | Purpose |
|---|---|---|
| Train | 0 – 1099 | Parameter tuning, walk-forward CV |
| Holdout | 1100 – 1569 | Final evaluation only — do not peek |

Walk-forward folds use `research.backtest_window.run_backtest_window()`, which
stops execution at each fold's `test_end` instead of running future candles and
slicing results afterward. Future ML workflows can pass
`retrain_hook(train_end_candle)` to retrain before each fold.

## Tests

```bash
.venv/bin/python -B -m unittest discover -s tests
```

The test suite covers Person 1 infrastructure plus Person 3 ensemble plumbing:
indicators, bounded backtest windows, strategy registry behavior, walk-forward
fold bounds, holdout starts, ML feature-order contracts, ensemble gating, and
CLI smoke behavior.

## Team responsibilities

| Person | Owns | Files |
|---|---|---|
| Person 1 | Data & infrastructure | `features.py`, `backtest_window.py`, `walk_forward.py`, `run.py` |
| Person 2 | Quant strategies & risk | `risk.py`, `trend_strategy.py`, `mean_revert_strategy.py` |
| Person 3 | ML & ensemble | `ml_features.py`, `train_models.py`, `ensemble.py` |

Final `strategy.py` is assembled together by inlining the research pieces.
