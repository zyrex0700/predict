from __future__ import annotations

import numpy as np
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_3"] = out["close"].pct_change(3)
    out["ret_5"] = out["close"].pct_change(5)

    out["ema_10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["ema_30"] = out["close"].ewm(span=30, adjust=False).mean()
    out["dist_ema_10"] = (out["close"] - out["ema_10"]) / out["close"]
    out["dist_ema_30"] = (out["close"] - out["ema_30"]) / out["close"]

    high_low = out["high"] - out["low"]
    high_close_prev = (out["high"] - out["close"].shift(1)).abs()
    low_close_prev = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean()
    out["atr_norm"] = out["atr_14"] / out["close"]

    out["roll_mean_20"] = out["close"].rolling(20).mean()
    out["roll_std_20"] = out["close"].rolling(20).std()
    out["zscore_20"] = (out["close"] - out["roll_mean_20"]) / out["roll_std_20"]

    return out


def make_labels(df: pd.DataFrame, horizon: int, target_return: float) -> pd.DataFrame:
    out = df.copy()
    future_return = out["close"].shift(-horizon) / out["close"] - 1.0
    out["target"] = (future_return > target_return).astype(int)
    return out


def build_dataset(df: pd.DataFrame, horizon: int, target_return: float):
    feat = add_features(df)
    labeled = make_labels(feat, horizon=horizon, target_return=target_return)

    feature_cols = [
        "ret_1",
        "ret_3",
        "ret_5",
        "dist_ema_10",
        "dist_ema_30",
        "atr_norm",
        "zscore_20",
    ]

    clean = labeled.dropna(subset=feature_cols + ["target"]).copy()
    X = clean[feature_cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = clean.loc[X.index, "target"]

    meta = clean.loc[X.index, ["timestamp", "open", "high", "low", "close"]]
    return X, y, meta, feature_cols
