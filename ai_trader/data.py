from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception:  # noqa: BLE001
    mt5 = None


MT5_TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _require_mt5() -> None:
    if mt5 is None:
        raise RuntimeError(
            "MetaTrader5 package is not available. Install it with `pip install MetaTrader5`."
        )


def get_timeframe_constant(timeframe: str):
    _require_mt5()
    tf_name = MT5_TIMEFRAME_MAP.get(timeframe)
    if tf_name is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return getattr(mt5, tf_name)


def fetch_rates(symbol: str, timeframe: str, months: int) -> pd.DataFrame:
    _require_mt5()

    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

    end = datetime.utcnow()
    start = end - timedelta(days=30 * months)

    tf = get_timeframe_constant(timeframe)
    rates = mt5.copy_rates_range(symbol, tf, start, end)

    mt5.shutdown()

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            "No data received from MT5. Ensure symbol exists and is visible in Market Watch."
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"time": "timestamp"})
    return df
