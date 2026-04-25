import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from cnlib.base_strategy import BaseStrategy


FEATURES = [
    "return_1",
    "return_5",
    "ma_ratio",
    "ma50_ratio",
    "momentum_10",
    "volatility_20",
    "volume_change",
]

MODEL_PATH = "model.pkl"

MAX_SINGLE_ALLOC = 0.30
MAX_PORTFOLIO_RISK = 0.10
MIN_VOLUME = 1_000
CRASH_RETURN_LIMIT = -0.08


def make_features(df):
    df = df.copy()

    df["return_1"] = df["Close"].pct_change()
    df["return_5"] = df["Close"].pct_change(5)

    df["ma5"] = df["Close"].rolling(5).mean()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()

    df["ma_ratio"] = df["ma5"] / df["ma20"] - 1
    df["ma50_ratio"] = df["Close"] / df["ma50"] - 1
    df["momentum_10"] = df["Close"] / df["Close"].shift(10) - 1

    df["volatility_20"] = df["return_1"].rolling(20).std()

    df["safe_volume"] = df["Volume"].replace(0, np.nan)
    df["volume_change"] = df["safe_volume"].pct_change()

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_latest_features(df):
    df = make_features(df)
    df = df.dropna(subset=FEATURES)

    if len(df) == 0:
        return None

    return df[FEATURES].iloc[[-1]]


def latest_volatility(df):
    vol = df["Close"].pct_change().rolling(20).std().iloc[-1]

    if np.isnan(vol) or vol <= 0:
        return 0.05

    return vol


def liquidity_ok(df):
    if "Volume" not in df.columns:
        return True

    recent_volume = df["Volume"].rolling(20).mean().iloc[-1]

    if np.isnan(recent_volume):
        return False

    return recent_volume >= MIN_VOLUME


def crash_filter(df):
    ret = df["Close"].pct_change().iloc[-1]

    if np.isnan(ret):
        return False

    return ret < CRASH_RETURN_LIMIT


def allocation_from_edge(direction_prob, risk_prob, df):
    vol = latest_volatility(df)

    if risk_prob > 0.65:
        return 0.0

    if vol > 0.08:
        vol_factor = 0.25
    elif vol > 0.05:
        vol_factor = 0.50
    elif vol > 0.03:
        vol_factor = 0.75
    else:
        vol_factor = 1.00

    edge = max(direction_prob - 0.50, 0.0)
    allocation = edge * 2.0 * vol_factor

    if risk_prob > 0.55:
        allocation *= 0.35
    elif risk_prob > 0.40:
        allocation *= 0.60
    elif risk_prob > 0.25:
        allocation *= 0.80

    return max(0.0, min(allocation, MAX_SINGLE_ALLOC))


def dynamic_stop_loss(df, risk_prob, direction_prob):
    vol = latest_volatility(df)

    if risk_prob > 0.60:
        stop = vol * 1.5
    elif risk_prob > 0.45:
        stop = vol * 2.0
    else:
        stop = vol * 2.8

    if direction_prob > 0.70:
        stop *= 1.15

    return min(max(stop, 0.025), 0.12)


def cap_portfolio_risk(decisions, data):
    total_risk = 0.0

    for d in decisions:
        if d["allocation"] <= 0:
            continue

        coin = d["coin"]
        vol = latest_volatility(data[coin])
        total_risk += d["allocation"] * vol

    if total_risk > MAX_PORTFOLIO_RISK and total_risk > 0:
        scale = MAX_PORTFOLIO_RISK / total_risk
        for d in decisions:
            d["allocation"] *= scale

    return decisions


def normalize_allocations(decisions):
    total_alloc = sum(d["allocation"] for d in decisions)

    if total_alloc > 1:
        for d in decisions:
            d["allocation"] /= total_alloc

    return decisions


class CoinRiskStrategy(BaseStrategy):
    """
    Submission strategy.

    Bu dosya eğitim yapmaz. Modeli model.pkl dosyasından yükler.
    model.pkl, ilk 1570 günlük veriyle repo gönderilmeden önce üretilmelidir.
    """

    def __init__(self, model_path=MODEL_PATH):
        super().__init__()
        self.model_path = model_path
        self.direction_models = {}
        self.risk_models = {}
        self.is_loaded = False
        self.load_model()

    def load_model(self):
        path = self.model_path

        if not os.path.exists(path):
            alt_path = os.path.join(os.path.dirname(__file__), self.model_path)
            if os.path.exists(alt_path):
                path = alt_path
            else:
                raise FileNotFoundError(
                    "model.pkl bulunamadı. strategy.py ile aynı klasöre model.pkl koymalısın."
                )

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self.direction_models = payload["direction_models"]
        self.risk_models = payload["risk_models"]
        self.is_loaded = True

    def flat_decision(self, coin):
        return {
            "coin": coin,
            "signal": 0,
            "allocation": 0.0,
            "leverage": 1,
            "stop_loss": 0.05,
            "take_profit": 0.10,
        }

    def predict(self, data: dict):
        decisions = []

        for coin, df in data.items():
            if coin not in self.direction_models or coin not in self.risk_models:
                decisions.append(self.flat_decision(coin))
                continue

            if len(df) < 60:
                decisions.append(self.flat_decision(coin))
                continue

            if not liquidity_ok(df):
                decisions.append(self.flat_decision(coin))
                continue

            if crash_filter(df):
                decisions.append(self.flat_decision(coin))
                continue

            latest_x = get_latest_features(df)

            if latest_x is None:
                decisions.append(self.flat_decision(coin))
                continue

            direction_prob = self.direction_models[coin].predict_proba(latest_x)[0][1]
            risk_prob = self.risk_models[coin].predict_proba(latest_x)[0][1]

            if risk_prob > 0.65 or direction_prob < 0.60:
                decisions.append(self.flat_decision(coin))
                continue

            allocation = allocation_from_edge(direction_prob, risk_prob, df)

            if allocation <= 0:
                decisions.append(self.flat_decision(coin))
                continue

            stop = dynamic_stop_loss(df, risk_prob, direction_prob)

            decisions.append({
                "coin": coin,
                "signal": 1,
                "allocation": allocation,
                "leverage": 1,
                "stop_loss": stop,
                "take_profit": stop * 2.2,
            })

        decisions = cap_portfolio_risk(decisions, data)
        decisions = normalize_allocations(decisions)

        return decisions


strategy = CoinRiskStrategy()
