from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score

from ai_trader.config import DEFAULT_CONFIG
from ai_trader.data import fetch_rates
from ai_trader.features import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train XAUUSD model from MT5 data")
    parser.add_argument("--symbol", default=DEFAULT_CONFIG.symbol)
    parser.add_argument("--timeframe", default=DEFAULT_CONFIG.timeframe)
    parser.add_argument("--months", type=int, default=DEFAULT_CONFIG.months)
    parser.add_argument("--horizon", type=int, default=DEFAULT_CONFIG.horizon)
    parser.add_argument("--target-return", type=float, default=DEFAULT_CONFIG.target_return)
    parser.add_argument("--artifacts-dir", default="artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    df = fetch_rates(symbol=args.symbol, timeframe=args.timeframe, months=args.months)
    X, y, _, feature_cols = build_dataset(df, horizon=args.horizon, target_return=args.target_return)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "report": classification_report(y_test, pred, output_dict=True),
        "rows_total": int(len(X)),
        "rows_train": int(len(X_train)),
        "rows_test": int(len(X_test)),
        "feature_cols": feature_cols,
    }

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "horizon": args.horizon,
            "target_return": args.target_return,
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
