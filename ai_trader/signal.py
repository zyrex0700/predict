from __future__ import annotations

import argparse
import json

import joblib

from ai_trader.data import fetch_rates
from ai_trader.features import add_features


def parse_args():
    parser = argparse.ArgumentParser(description="Generate XAUUSD signal from latest candles")
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.55)
    return parser.parse_args()


def main():
    args = parse_args()

    payload = joblib.load(args.model_path)
    model = payload["model"]
    feature_cols = payload["feature_cols"]
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]

    df = fetch_rates(symbol=symbol, timeframe=timeframe, months=args.months)
    df = df.tail(max(args.lookback, 60)).copy()
    feat = add_features(df).dropna()

    if feat.empty:
        raise RuntimeError("Not enough candles to create features.")

    latest = feat.iloc[-1]
    X_latest = latest[feature_cols].to_frame().T

    prob_up = float(model.predict_proba(X_latest)[0, 1])
    signal = "BUY" if prob_up >= args.threshold else "NO_TRADE"

    out = {
        "timestamp": str(latest["timestamp"]),
        "symbol": symbol,
        "signal": signal,
        "prob_up": round(prob_up, 4),
        "threshold": args.threshold,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
