from cnlib.base_strategy import BaseStrategy
from cnlib import backtest
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


HOLDOUT_START = 1205


class RiskManager:
    def __init__(self, trailing_atr_mult=1.5, sl_atr_mult=2.5, tp_atr_mult=4.0,
                 max_hold_days=10, vol_target=0.020, max_total_alloc=0.90):
        self.recent_pnl = []
        self.drawdown_window = 10
        self.drawdown_multiplier = 1.0
        self.positions = {}
        self.max_hold_days = max_hold_days
        self.trailing_atr_mult = trailing_atr_mult
        self.sl_atr_mult = sl_atr_mult
        self.tp_atr_mult = tp_atr_mult
        self.vol_target = vol_target
        self.max_total_alloc = max_total_alloc

    def update_drawdown(self, data, active_positions):
        daily_pnl = 0.0
        for coin, df in data.items():
            if len(df) < 2:
                continue
            if coin in active_positions:
                pos = active_positions[coin]
                if pos["signal"] != 0:
                    day_return = df["Close"].iloc[-1] / df["Close"].iloc[-2] - 1
                    daily_pnl += day_return * pos["signal"] * 0.20
        self.recent_pnl.append(daily_pnl)
        if len(self.recent_pnl) > self.drawdown_window:
            self.recent_pnl.pop(0)
        total_recent = sum(self.recent_pnl)

        if total_recent < -0.10:
            self.drawdown_multiplier = 0.0
        elif total_recent < -0.05:
            self.drawdown_multiplier = 0.5
        else:
            self.drawdown_multiplier = 1.0
        return self.drawdown_multiplier

    def vol_scale(self, current_vol):
        if current_vol <= 0 or pd.isna(current_vol):
            return 1.0
        raw_scale = self.vol_target / current_vol
        return max(0.5, min(1.5, raw_scale))

    def update_position(self, coin, raw_signal, current_price, current_atr):
        if raw_signal == 0:
            if coin in self.positions:
                del self.positions[coin]
            return 0
        if coin in self.positions:
            pos = self.positions[coin]
            pos["days_held"] += 1
            if pos["signal"] != raw_signal:
                self.positions[coin] = {
                    "signal": raw_signal, "entry_price": current_price,
                    "peak_price": current_price, "days_held": 0
                }
                return raw_signal
            if pos["signal"] == 1:
                pos["peak_price"] = max(pos["peak_price"], current_price)
                pullback = pos["peak_price"] - current_price
            else:
                pos["peak_price"] = min(pos["peak_price"], current_price)
                pullback = current_price - pos["peak_price"]
            if current_atr > 0 and pullback > current_atr * self.trailing_atr_mult:
                del self.positions[coin]
                return 0
            if pos["days_held"] >= self.max_hold_days:
                pnl = (current_price / pos["entry_price"] - 1) * pos["signal"]
                if pnl <= 0:
                    del self.positions[coin]
                    return 0
            return raw_signal
        else:
            self.positions[coin] = {
                "signal": raw_signal, "entry_price": current_price,
                "peak_price": current_price, "days_held": 0
            }
            return raw_signal

    def calc_sl_tp(self, signal, entry_price, current_atr):
        if current_atr <= 0 or pd.isna(current_atr):
            return None, None
        sl_distance = current_atr * self.sl_atr_mult
        tp_distance = current_atr * self.tp_atr_mult
        if signal == 1:
            sl = round(entry_price - sl_distance, 2)
            tp = round(entry_price + tp_distance, 2)
        else:
            sl = round(entry_price + sl_distance, 2)
            tp = round(entry_price - tp_distance, 2)
        return sl, tp


class MLStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self.modeller = {}
        self.risk = RiskManager(max_total_alloc=0.90)

        self.conf_th = 0.45
        self.vol_filter = 0.035
        self.label_threshold = 0.005

        self.max_alloc = 0.80
        self.max_coin_alloc = 0.35
        self.max_effective_exposure = 1.0

        self.active_allocations = {}
        self.candle_counter = 0

    def egit(self):
        for coin, df in self._full_data.items():
            df_train = df.iloc[:HOLDOUT_START].copy()

            closes = df_train["Close"]
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            bb_mid = closes.rolling(20).mean()
            bb_std = closes.rolling(20).std()
            z_score = (closes - bb_mid) / bb_std

            X = pd.DataFrame({
                "return_5": closes.pct_change(5),
                "ma_farki": (closes.rolling(5).mean() - closes.rolling(20).mean()) / closes.rolling(20).mean(),
                "volatility_10": closes.pct_change().rolling(10).std(),
                "rsi": rsi,
                "volume_change": df_train["Volume"].pct_change(5),
                "z_score": z_score
            })
            future_return = closes.shift(-1) / closes - 1
            y = pd.Series(0, index=df_train.index, dtype=int)
            y[future_return > self.label_threshold] = 1
            y[future_return < -self.label_threshold] = -1
            mask = X.notna().all(axis=1) & y.notna()
            model = RandomForestClassifier(
                n_estimators=200, max_depth=4, min_samples_leaf=40,
                max_features='sqrt', random_state=42
            )
            model.fit(X[mask].values, y[mask].values)
            self.modeller[coin] = model
            print(f"  {coin}: egitildi (ilk {HOLDOUT_START} candle), n={mask.sum()}")

    def _belirle_leverage(self, confidence, volatility, dd_mult):
        if dd_mult < 1.0:
            return 1
        if volatility > 0.025:
            return 1
        elif volatility > 0.014:
            if confidence > 0.60:
                return 2
            else:
                return 1
        else:
            if confidence > 0.60:
                return 3
            elif confidence > 0.50:
                return 2
            else:
                return 1

    def predict(self, data: dict) -> list[dict]:
        self.candle_counter += 1

        # İlk 1205 candle: işlem yapma (eğitim verisi)
        if self.candle_counter <= HOLDOUT_START:
            return [{"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1}
                    for coin in data.keys()]

        # Candle 1206+: out-of-sample trading
        dd_mult = self.risk.update_drawdown(data, self.risk.positions)
        if dd_mult == 0.0:
            decisions = []
            for coin in data.keys():
                self.risk.update_position(coin, 0, 0, 0)
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
            self.active_allocations = {}
            return decisions

        signals = {}
        for coin, df in data.items():
            current_price = df["Close"].iloc[-1] if len(df) > 0 else 0
            atr_series = (df["High"] - df["Low"]).rolling(14).mean()
            current_atr = atr_series.iloc[-1] if len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]) else 0

            if len(df) < 30 or coin not in self.modeller:
                self.risk.update_position(coin, 0, current_price, current_atr)
                signals[coin] = {"signal": 0, "confidence": 0.0, "volatility": 0.0,
                                 "current_price": current_price, "current_atr": current_atr}
                continue

            closes = df["Close"]
            delta = closes.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss_s = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss_s
            rsi = 100 - (100 / (1 + rs))
            bb_mid = closes.rolling(20).mean()
            bb_std = closes.rolling(20).std()
            z_score = (closes - bb_mid) / bb_std

            X = pd.DataFrame({
                "return_5": closes.pct_change(5),
                "ma_farki": (closes.rolling(5).mean() - closes.rolling(20).mean()) / closes.rolling(20).mean(),
                "volatility_10": closes.pct_change().rolling(10).std(),
                "rsi": rsi,
                "volume_change": df["Volume"].pct_change(5),
                "z_score": z_score
            })

            if X.iloc[-1].isna().any():
                self.risk.update_position(coin, 0, current_price, current_atr)
                signals[coin] = {"signal": 0, "confidence": 0.0, "volatility": 0.0,
                                 "current_price": current_price, "current_atr": current_atr}
                continue

            son_features = X.iloc[-1].values.reshape(1, -1)
            model = self.modeller[coin]
            proba = model.predict_proba(son_features)[0]
            classes = list(model.classes_)
            p_long = proba[classes.index(1)] if 1 in classes else 0
            p_short = proba[classes.index(-1)] if -1 in classes else 0

            volatility = closes.pct_change().rolling(10).std().iloc[-1]
            if pd.isna(volatility):
                volatility = 0.0

            if volatility > self.vol_filter:
                raw_signal = 0
                confidence = 0.0
            elif p_long > self.conf_th and p_long > p_short:
                raw_signal = 1
                confidence = p_long
            elif p_short > self.conf_th and p_short > p_long:
                raw_signal = -1
                confidence = p_short
            else:
                raw_signal = 0
                confidence = 0.0

            adjusted_signal = self.risk.update_position(coin, raw_signal, current_price, current_atr)
            signals[coin] = {
                "signal": adjusted_signal,
                "confidence": confidence if adjusted_signal != 0 else 0.0,
                "volatility": volatility,
                "current_price": current_price,
                "current_atr": current_atr
            }

        return self._dagit_allocation(signals, dd_mult)

    def _dagit_allocation(self, signals, dd_mult):
        decisions = []
        sirali = sorted(signals.items(), key=lambda x: x[1]["confidence"], reverse=True)

        toplam_alloc = 0.0
        toplam_exposure = 0.0
        for coin, info in sirali:
            if info["signal"] != 0 and coin in self.active_allocations:
                prev = self.active_allocations[coin]
                toplam_alloc += prev["allocation"]
                toplam_exposure += prev["allocation"] * prev["leverage"]

        new_active = {}

        for coin, info in sirali:
            if info["signal"] == 0:
                decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
            else:
                c = info["confidence"]
                v = info["volatility"]

                if c >= 0.65:
                    base_alloc = 0.38
                elif c >= 0.55:
                    base_alloc = 0.28
                elif c >= 0.45:
                    base_alloc = 0.18
                else:
                    base_alloc = 0.0

                vol_mult = self.risk.vol_scale(v)
                alloc = base_alloc * vol_mult * dd_mult
                alloc = round(alloc, 2)

                lev = self._belirle_leverage(c, v, dd_mult)

                if coin in self.active_allocations:
                    prev = self.active_allocations[coin]
                    alloc = prev["allocation"]
                    lev = prev["leverage"]
                else:
                    if toplam_alloc + alloc > self.max_alloc:
                        alloc = max(0.0, round(self.max_alloc - toplam_alloc, 2))
                    alloc = min(alloc, self.max_coin_alloc)

                    efektif = alloc * lev
                    if toplam_exposure + efektif > self.max_effective_exposure:
                        kalan_exposure = self.max_effective_exposure - toplam_exposure
                        if kalan_exposure <= 0:
                            alloc = 0.0
                        else:
                            alloc = round(kalan_exposure / lev, 2)
                            alloc = min(alloc, self.max_coin_alloc)

                if alloc <= 0.01:
                    decisions.append({"coin": coin, "signal": 0, "allocation": 0.0, "leverage": 1})
                    self.risk.update_position(coin, 0, info["current_price"], info["current_atr"])
                else:
                    if coin not in self.active_allocations:
                        toplam_alloc += alloc
                        toplam_exposure += alloc * lev

                    new_active[coin] = {"allocation": alloc, "leverage": lev}

                    sl, tp = self.risk.calc_sl_tp(info["signal"], info["current_price"], info["current_atr"])
                    decision = {"coin": coin, "signal": info["signal"],
                                "allocation": alloc, "leverage": lev}
                    if sl is not None:
                        decision["stop_loss"] = sl
                    if tp is not None:
                        decision["take_profit"] = tp
                    decisions.append(decision)

        self.active_allocations = new_active
        return decisions


if __name__ == "__main__":
    strategy = MLStrategy()
    strategy.get_data()
    strategy.egit()

    # Eğitim ilk 1205 candle ile yapıldı.
    # Backtest sadece son 365 candle üzerinde başlatılıyor.
    strategy.candle_counter = HOLDOUT_START

    result = backtest.run(
        strategy=strategy,
        initial_capital=3000.0,
        start_candle=HOLDOUT_START
    )

    result.print_summary()