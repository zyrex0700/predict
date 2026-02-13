from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from ai_trader.config import DEFAULT_CONFIG
from ai_trader.data import fetch_rates
from ai_trader.features import build_dataset


@dataclass
class TimeframeResult:
    timeframe: str
    rows: int
    roc_auc: float
    positive_rate: float


def evaluate_timeframe(
    symbol: str,
    timeframe: str,
    months: int,
    horizon: int,
    target_return: float,
) -> TimeframeResult:
    df = fetch_rates(symbol=symbol, timeframe=timeframe, months=months)
    X, y, _, _ = build_dataset(df, horizon=horizon, target_return=target_return)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, prob))

    return TimeframeResult(
        timeframe=timeframe,
        rows=int(len(X)),
        roc_auc=auc,
        positive_rate=float(np.mean(y)),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple timeframes for XAUUSD model quality")
    parser.add_argument("--symbol", default=DEFAULT_CONFIG.symbol)
    parser.add_argument("--months", type=int, default=DEFAULT_CONFIG.months)
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.horizon)
    parser.add_argument("--target-return", type=float, default=DEFAULT_CONFIG.target_return)
    parser.add_argument("--timeframes", nargs="+", default=["M15", "M30", "H1", "H4"])
    return parser.parse_args()


def main():
    args = parse_args()

    results: list[TimeframeResult] = []
    for tf in args.timeframes:
        result = evaluate_timeframe(
            symbol=args.symbol,
            timeframe=tf,
            months=args.months,
            horizon=args.horizon,
            target_return=args.target_return,
        )
        results.append(result)

    results = sorted(results, key=lambda x: x.roc_auc, reverse=True)

    payload = {
        "symbol": args.symbol,
        "months": args.months,
        "horizon": args.horizon,
        "target_return": args.target_return,
        "ranking": [asdict(r) for r in results],
        "best_timeframe": results[0].timeframe if results else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
