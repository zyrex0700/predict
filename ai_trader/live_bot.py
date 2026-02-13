from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib

from ai_trader.data import fetch_rates
from ai_trader.execution import (
    TradeConfig,
    has_open_position,
    initialize_mt5,
    send_market_order,
    shutdown_mt5,
)
from ai_trader.features import add_features


def parse_args():
    parser = argparse.ArgumentParser(description="Run continuous live bot per timeframe candle close")
    parser.add_argument("--model-path", default="artifacts/model.joblib")
    parser.add_argument("--lookback", type=int, default=100)
    parser.add_argument("--months", type=int, default=1)
    parser.add_argument("--buy-threshold", type=float, default=0.58)
    parser.add_argument("--sell-threshold", type=float, default=0.42)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--lot", type=float, default=0.01)
    parser.add_argument("--sl-points", type=int, default=400)
    parser.add_argument("--tp-points", type=int, default=800)
    parser.add_argument("--allow-trading", action="store_true")
    return parser.parse_args()


def decide_signal(prob_up: float, buy_threshold: float, sell_threshold: float) -> str:
    if prob_up >= buy_threshold:
        return "BUY"
    if prob_up <= sell_threshold:
        return "SELL"
    return "NO_TRADE"


def main():
    args = parse_args()
    payload = joblib.load(args.model_path)

    model = payload["model"]
    feature_cols = payload["feature_cols"]
    symbol = payload["symbol"]
    timeframe = payload["timeframe"]

    cfg = TradeConfig(lot=args.lot, stop_loss_points=args.sl_points, take_profit_points=args.tp_points)

    print(f"[START] symbol={symbol} timeframe={timeframe} allow_trading={args.allow_trading}")

    initialize_mt5()
    last_candle_time = None

    try:
        while True:
            df = fetch_rates(symbol=symbol, timeframe=timeframe, months=args.months, initialized=True)
            df = df.tail(max(args.lookback, 60)).copy()
            feat = add_features(df).dropna()

            if feat.empty:
                time.sleep(args.poll_seconds)
                continue

            latest = feat.iloc[-1]
            candle_time = latest["timestamp"]

            if last_candle_time is not None and candle_time <= last_candle_time:
                time.sleep(args.poll_seconds)
                continue

            last_candle_time = candle_time
            X_latest = latest[feature_cols].to_frame().T
            prob_up = float(model.predict_proba(X_latest)[0, 1])
            signal = decide_signal(prob_up, args.buy_threshold, args.sell_threshold)

            event = {
                "event_time": datetime.utcnow().isoformat(),
                "candle_time": str(candle_time),
                "symbol": symbol,
                "timeframe": timeframe,
                "signal": signal,
                "prob_up": round(prob_up, 4),
            }

            if args.allow_trading and signal in {"BUY", "SELL"}:
                if has_open_position(symbol):
                    event["trade"] = "skipped_open_position_exists"
                else:
                    try:
                        result = send_market_order(symbol=symbol, side=signal, cfg=cfg)
                        event["trade"] = {"status": "opened", "result": result}
                    except Exception as exc:  # noqa: BLE001
                        event["trade"] = {"status": "error", "error": str(exc)}

            print(json.dumps(event, ensure_ascii=False))
            time.sleep(args.poll_seconds)

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
